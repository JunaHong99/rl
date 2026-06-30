"""
Per-arm Single-arm Impedance Controller for dual-arm cooperative manipulation.

Cooperative wrench split을 제거하고, 각 arm이 자기 grasp point의 target을 독립 추종.
연구 contribution은 그대로 유지:
  - RL이 (rod_target, K_arm1, K_arm2) 동시 출력
  - DA-VIL: shared K vs. 우리: per-arm K
  - 실시간 target 생성 + 장애물 회피 (future work)

전체 흐름:
  RL → rod_target_pose
       ↓ (기하학적 변환)
  ee1_target = rod_target + R_target × (-L/2, 0, +TCP) — 왼쪽 grasp panda_hand position
  ee2_target = rod_target + R_target × (+L/2, 0, +TCP) — 오른쪽 grasp
  ee_target_quat = quat_mul(rod_target_quat, Rx(π))      — grasp_roll=π
       ↓
  각 arm 독립 impedance:
    f = K_pos·(ee_target_pos - ee_pos) + D_pos·(0 - ee_vel)
    m = K_rot·rot_err_aa + D_rot·(0 - ee_ang_vel)
    τ = J^T × [f, m]

Geometry assumptions (env3 fixed joint setup과 일치):
  - rod local frame: +X = rod's long axis
  - grasp points at rod ends: ±0.4m from rod center
  - panda_hand R_world = R_rod × Rx(π) (gripper points down at rod)
  - TCP는 panda_hand local +Z 방향 0.1034m
"""
from __future__ import annotations
import torch
import isaaclab.utils.math as math_utils


class PerArmImpedanceController:
    """
    Per-arm single-arm impedance for dual-arm cooperative manipulation.

    각 arm은 자기 grasp point의 target을 추종.
    cooperative wrench split, force couple, M_abs/M_rel 분해 없음.
    """

    # Geometry constants (env3 fixed joint setup과 일치)
    TCP_OFFSET = 0.1034
    ROD_HALF_LENGTH = 0.4  # rod 0.8m / 2

    def __init__(
        self,
        env,
        K_pos: float = 200.0,
        D_pos: float = 60.0,
        K_rot: float = 20.0,
        D_rot: float = 8.0,
    ):
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs

        self.robot_1 = env.robot_1
        self.robot_2 = env.robot_2

        self.ee_idx_1 = env.ee_body_idx_1
        self.ee_idx_2 = env.ee_body_idx_2
        self.joint_ids_1 = env.robot_1_joint_ids
        self.joint_ids_2 = env.robot_2_joint_ids

        # Impedance gains — 단일 set, 양 arm 동일.
        # 향후 GNN extension: K_arm1, K_arm2를 RL 출력으로
        self.K_pos = K_pos
        self.D_pos = D_pos
        self.K_rot = K_rot
        self.D_rot = D_rot

        # 회전 grasp constants
        # Rx(π) quat: w=0, x=1, y=0, z=0
        self._rx_pi = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(self.num_envs, 4).contiguous()

        # Constant local offsets in rod frame
        self._left_offset_local = torch.tensor(
            [-self.ROD_HALF_LENGTH, 0.0, self.TCP_OFFSET], device=self.device
        ).unsqueeze(0).expand(self.num_envs, 3).contiguous()
        self._right_offset_local = torch.tensor(
            [+self.ROD_HALF_LENGTH, 0.0, self.TCP_OFFSET], device=self.device
        ).unsqueeze(0).expand(self.num_envs, 3).contiguous()

        self._last_info: dict = {}
        # 중력 보상 (gravity ON): τ += grav_sign·G(q). gravity_test.py로 sign=+1 검증됨(2026-06-15).
        self.gravity_comp = True
        self.grav_sign = 1.0

        # ── Nullspace 모델기반 장애물 회피 (2026-06-19, control 기법 A) ──
        # 7-DoF 팔이 6-DoF EE를 잡으면 1-DoF redundancy(팔꿈치 swing). **RL이 아니라 컨트롤러가**
        # 여분 DoF로 팔 링크를 장애물에서 밀어냄(potential field). τ += N·τ_rep (EE=rod 안 흔듦).
        self.D_null = 3.0          # nullspace 댐핑 (안정)
        self._null_lambda = 1e-3   # projector damping (1e-3로 낮춤 → EE leak 더 작게)
        self.null_d0 = 0.15        # 반발 영향 거리 [m] (링크-장애물 표면거리 < d0면 밀어냄)
        self.null_krep = 8.0       # 반발 게인
        self.null_link_r = 0.06    # 팔 링크 반경 근사 [m]
        self.null_tau_cap = 30.0   # 링크당 반발 토크 크기 제한
        self._arm_link_ids = None  # lazy: 팔 링크 body 인덱스
        self._swivel_ids = None    # lazy: (shoulder, elbow, wrist) body 인덱스 (ψ_des handle)
        # 2026-06-19: 당장은 nullspace 팔-회피 OFF. RL(GNN, obstacle→arm 엣지)이 거시 회피를
        # 먼저 학습하게 두고, task 수행이 안 되면(미세 팔 충돌 잔여) 그때 True로 재활성.
        self.use_nullspace_avoidance = False

    # ──────────────────────────────────────────────────────────────────
    # Quaternion utilities (Isaac Lab convention: w, x, y, z)
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _quat_conj(q):
        return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)

    @staticmethod
    def _quat_mul(q1, q2):
        return math_utils.quat_mul(q1, q2)

    @staticmethod
    def _quat_apply(q, v):
        """Rotate vector v (B,3) by quat q (B,4)."""
        qv = q[:, 1:4]
        qw = q[:, :1]
        t = 2.0 * torch.cross(qv, v, dim=-1)
        return v + qw * t + torch.cross(qv, t, dim=-1)

    @classmethod
    def _quat_to_axis_angle(cls, q):
        """Quat → axis-angle (3D). Double-cover handling for shortest path."""
        w = q[:, 0:1]
        sign = torch.sign(w)
        sign[sign == 0] = 1.0
        q_signed = q * sign

        v = q_signed[:, 1:4]
        w_pos = q_signed[:, 0]

        v_norm = torch.norm(v, dim=-1, keepdim=True)
        angle = 2.0 * torch.atan2(v_norm.squeeze(-1), w_pos.clamp(min=-1.0, max=1.0))

        eps = 1e-8
        axis = v / (v_norm + eps)
        return axis * angle.unsqueeze(-1)

    # ──────────────────────────────────────────────────────────────────
    # Jacobian
    # ──────────────────────────────────────────────────────────────────
    def _get_jacobians(self):
        J1_full = self.robot_1.root_physx_view.get_jacobians()
        J1 = J1_full[:, self.ee_idx_1, :, :][:, :, self.joint_ids_1]  # (B, 6, 7)
        J2_full = self.robot_2.root_physx_view.get_jacobians()
        J2 = J2_full[:, self.ee_idx_2, :, :][:, :, self.joint_ids_2]
        return J1, J2

    # ──────────────────────────────────────────────────────────────────
    # Target geometry: rod target → ee target
    # ──────────────────────────────────────────────────────────────────
    def _compute_ee_targets(self, target_obj_pos, target_obj_quat):
        """
        rod target pose에서 양 panda_hand target pose 계산.

        Args:
            target_obj_pos: (B, 3) rod 중심 target (world)
            target_obj_quat: (B, 4) rod 자세 target (world)

        Returns:
            ee1_target_pos, ee1_target_quat, ee2_target_pos, ee2_target_quat
        """
        # 왼쪽 panda_hand position = rod_target_pos + R_target × (-L/2, 0, +TCP)
        ee1_target_pos = target_obj_pos + self._quat_apply(target_obj_quat, self._left_offset_local)
        ee2_target_pos = target_obj_pos + self._quat_apply(target_obj_quat, self._right_offset_local)

        # panda_hand quaternion = quat_mul(target_obj_quat, Rx(π))
        ee_target_quat = self._quat_mul(target_obj_quat, self._rx_pi)
        # Normalize
        ee_target_quat = ee_target_quat / torch.norm(ee_target_quat, dim=-1, keepdim=True)

        return ee1_target_pos, ee_target_quat, ee2_target_pos, ee_target_quat

    # ──────────────────────────────────────────────────────────────────
    # Single-arm impedance
    # ──────────────────────────────────────────────────────────────────
    def _single_arm_impedance(self, robot, ee_idx, target_pos, target_quat, K_pos=None, D_pos=None):
        """단일 arm task-space impedance. tau (B, 7) 반환.
        K_pos/D_pos: None이면 고정 self.K_pos/self.D_pos, 아니면 per-env (B,) 텐서 (Stage 1 per-arm K)."""
        ee_pos = robot.data.body_pos_w[:, ee_idx, :]
        ee_quat = robot.data.body_quat_w[:, ee_idx, :]
        ee_lin = robot.data.body_lin_vel_w[:, ee_idx, :]
        ee_ang = robot.data.body_ang_vel_w[:, ee_idx, :]

        # Position error
        pos_err = target_pos - ee_pos
        # Velocity error (target velocity = 0)
        lin_err = -ee_lin

        # Orientation error (axis-angle)
        q_err = self._quat_mul(target_quat, self._quat_conj(ee_quat))
        rot_err = self._quat_to_axis_angle(q_err)
        ang_err = -ee_ang

        # Impedance law (K_pos/D_pos가 per-env 텐서면 (B,1)로 broadcast)
        Kp = self.K_pos if K_pos is None else K_pos
        Dp = self.D_pos if D_pos is None else D_pos
        if torch.is_tensor(Kp):
            Kp = Kp.unsqueeze(-1)
        if torch.is_tensor(Dp):
            Dp = Dp.unsqueeze(-1)
        force = Kp * pos_err + Dp * lin_err
        torque = self.K_rot * rot_err + self.D_rot * ang_err
        wrench = torch.cat([force, torque], dim=-1)  # (B, 6)

        return wrench

    # ──────────────────────────────────────────────────────────────────
    # Nullspace torque (redundancy → 팔꿈치 회피)
    # ──────────────────────────────────────────────────────────────────
    def _nullspace_obstacle_torque(self, robot, J, joint_ids, obs_pos_w, obs_active):
        """모델기반 팔-장애물 회피 (control 기법 A): 팔 링크를 장애물에서 밀어내는 반발 토크를
        nullspace로 투영 → EE(=rod)는 안 흔들며 팔꿈치/링크만 장애물 회피.
        J: EE Jacobian (B,6,7), obs_pos_w: (B,Nobs,3) world, obs_active: (B,Nobs) bool. Returns (B,7)."""
        B = J.shape[0]
        if self._arm_link_ids is None:
            names = ["panda_link4", "panda_link5", "panda_link6"]   # 팔꿈치/팔뚝/손목
            self._arm_link_ids = [robot.body_names.index(n) for n in names]
        qd = robot.data.joint_vel[:, joint_ids]

        # nullspace projector N = I - Jᵀ(JJᵀ+λI)⁻¹J
        JT = J.transpose(-1, -2)
        eye6 = torch.eye(6, device=self.device).expand(B, 6, 6)
        JJt_inv = torch.linalg.inv(torch.bmm(J, JT) + self._null_lambda * eye6)
        eye7 = torch.eye(7, device=self.device).expand(B, 7, 7)
        N = eye7 - torch.bmm(JT, torch.bmm(JJt_inv, J))    # (B,7,7)

        # 팔 링크별 반발력 → joint torque (link Jacobian Jᵀ·f)
        jac_full = robot.root_physx_view.get_jacobians()   # (B, nbody, 6, ndof)
        tau_rep = torch.zeros(B, 7, device=self.device)
        near_any = torch.zeros(B, dtype=torch.bool, device=self.device)  # 장애물 근처 env 게이트
        d0, krep, lr, cap = self.null_d0, self.null_krep, self.null_link_r, self.null_tau_cap
        for l in self._arm_link_ids:
            p_l = robot.data.body_pos_w[:, l, :]                       # (B,3)
            diff = p_l.unsqueeze(1) - obs_pos_w                        # (B,Nobs,3)
            dctr = diff.norm(dim=-1)                                   # (B,Nobs) 중심거리
            surf = dctr - self.env.cfg.obstacle_radius - lr           # 표면거리
            within = obs_active & (surf < d0)
            near_any = near_any | within.any(dim=1)
            surf_safe = surf.clamp_min(1e-2)
            mag = (krep * (1.0 / surf_safe - 1.0 / d0) / (surf_safe ** 2))
            mag = torch.where(within, mag.clamp(max=cap), torch.zeros_like(mag))  # (B,Nobs)
            dirn = diff / dctr.unsqueeze(-1).clamp_min(1e-6)           # (B,Nobs,3)
            f_l = (mag.unsqueeze(-1) * dirn).sum(dim=1)                # (B,3) 합 반발력
            J_l = jac_full[:, l, :3, :][:, :, joint_ids]              # (B,3,7) 링크 위치 Jacobian
            tau_rep = tau_rep + torch.bmm(J_l.transpose(-1, -2), f_l.unsqueeze(-1)).squeeze(-1)

        tau_sec = tau_rep - self.D_null * qd                          # 반발 + 댐핑
        tau_null = torch.bmm(N, tau_sec.unsqueeze(-1)).squeeze(-1)    # nullspace 투영
        # ★ 근접-게이트: 어떤 팔 링크도 장애물 d0 이내 없으면 nullspace 토크 0 (운반에 간섭 X).
        return torch.where(near_any.unsqueeze(-1), tau_null, torch.zeros_like(tau_null))

    def _nullspace_rl_torque(self, robot, J, joint_ids, alpha):
        """RL이 출력한 α(B,)로 팔꿈치 nullspace self-motion 구동 (control: RL이 여분 DoF 직접 제어).
        τ_null = N·(α·gain·e − D_null·qd), e=ones. N=I−Jᵀ(JJᵀ+λI)⁻¹J로 EE(=rod) 안 흔들게 투영.
        α∈[-1,1] → 팔꿈치 swing 방향·크기. 모델기반(_nullspace_obstacle_torque)과 달리 RL이 언제·얼마나 결정."""
        B = J.shape[0]
        JT = J.transpose(-1, -2)
        eye6 = torch.eye(6, device=self.device).expand(B, 6, 6)
        JJt_inv = torch.linalg.inv(torch.bmm(J, JT) + self._null_lambda * eye6)
        eye7 = torch.eye(7, device=self.device).expand(B, 7, 7)
        N = eye7 - torch.bmm(JT, torch.bmm(JJt_inv, J))               # (B,7,7)
        qd = robot.data.joint_vel[:, joint_ids]
        gain = getattr(self.env.cfg, "null_gain", 5.0)
        e = torch.ones(B, 7, device=self.device)
        tau_sec = alpha.unsqueeze(-1) * gain * e - self.D_null * qd    # α 구동 + 댐핑
        return torch.bmm(N, tau_sec.unsqueeze(-1)).squeeze(-1)        # nullspace 투영

    def _nullspace_swivel_torque(self, robot, J, joint_ids, psi_des):
        """RL이 출력한 목표 swivel각 ψ_des(B,)로 팔꿈치를 nullspace로 *servo* (전 범위 position setpoint).
        swivel각 = 어깨(link2)–손목(link6) 축 둘레 팔꿈치(link4) 원 위 위치. ψ_des∈[-1,1] → [-π,π].
        τ_null = N·(Jₑᵀ·K·(E_des−E_cur) − D·qd). N으로 EE(=rod) 불간섭. α 핸들(e=ones, 1방향 blunt push)과
        달리 *전 원 도달* + position setpoint라 rod 끌림에 self-correct (예측 부담 없음)."""
        B = J.shape[0]
        if self._swivel_ids is None:
            bn = list(robot.body_names)
            self._swivel_ids = (bn.index("panda_link2"), bn.index("panda_link4"), bn.index("panda_link6"))
        sh_i, el_i, wr_i = self._swivel_ids
        # ★ 동역학적으로 일관된 nullspace projector (2026-06-30):
        # kinematic N=I−Jᵀ(JJᵀ)⁻¹J는 *속도*만 EE 고정 → 토크로 쓰면 M⁻¹ 커플링으로 EE 가속 leak →
        # 닫힌사슬 rod 200mm 드리프트(K_sw 무관). N_dyn=I−Jᵀ(JM⁻¹Jᵀ)⁻¹JM⁻¹ → EE 가속 0 보장.
        JT = J.transpose(-1, -2)
        eye6 = torch.eye(6, device=self.device).expand(B, 6, 6)
        eye7 = torch.eye(7, device=self.device).expand(B, 7, 7)
        M_full = robot.root_physx_view.get_mass_matrices()        # (B, ndof, ndof)
        M = M_full[:, joint_ids][:, :, joint_ids]                 # (B,7,7) 팔 관절만
        Minv = torch.linalg.inv(M)
        JMinv = torch.bmm(J, Minv)                                # (B,6,7)
        Lam = torch.linalg.inv(torch.bmm(JMinv, JT) + self._null_lambda * eye6)   # (B,6,6)
        N = eye7 - torch.bmm(JT, torch.bmm(Lam, JMinv))           # (B,7,7) dynamically consistent
        # 어깨/팔꿈치/손목 world 위치 → SW 축 + 팔꿈치 원(중심·반지름)
        sh = robot.data.body_pos_w[:, sh_i]; el = robot.data.body_pos_w[:, el_i]; wr = robot.data.body_pos_w[:, wr_i]
        axis = wr - sh; axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        rel = el - sh
        center = sh + (rel * axis).sum(-1, keepdim=True) * axis
        radius = (el - center).norm(dim=-1, keepdim=True)
        # 원 평면 기준축 (world up를 축에 수직 투영; 축이 거의 수직이면 x로 fallback)
        up = torch.tensor([0., 0., 1.], device=self.device).expand(B, 3)
        ref = up - (up * axis).sum(-1, keepdim=True) * axis
        fb = torch.tensor([1., 0., 0.], device=self.device).expand(B, 3)
        ref = torch.where(ref.norm(dim=-1, keepdim=True) < 1e-3,
                          fb - (fb * axis).sum(-1, keepdim=True) * axis, ref)
        ref = ref / ref.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        ref2 = torch.cross(axis, ref, dim=-1)
        # ψ_des → 목표 팔꿈치 점 E_des (원 위)
        psi = psi_des.clamp(-1., 1.) * torch.pi
        E_des = center + radius * (torch.cos(psi).unsqueeze(-1) * ref + torch.sin(psi).unsqueeze(-1) * ref2)
        # 팔꿈치 position Jacobian (3×7) — body index로 직접 인덱싱(EE와 동일 규약)
        Je = robot.root_physx_view.get_jacobians()[:, el_i, :3, :][:, :, joint_ids]   # (B,3,7)
        qd = robot.data.joint_vel[:, joint_ids]
        K_sw = getattr(self.env.cfg, "swivel_gain", 60.0)
        f = K_sw * (E_des - el)                                                       # (B,3) 목표로 끄는 힘
        tau_sec = torch.bmm(Je.transpose(-1, -2), f.unsqueeze(-1)).squeeze(-1) - self.D_null * qd
        return torch.bmm(N, tau_sec.unsqueeze(-1)).squeeze(-1)                        # nullspace 투영

    def _hard_safety_torque(self, robot, joint_ids, obs_pos_w, obs_active):
        """Hard 안전 토크 (RoboBallet velocity-zeroing 근사). 팔 링크(link4/5/6)가 장애물 임박 시
        접근속도 제동(kbrake·v_approach) + 강한 barrier(krep/surf) 를 link Jᵀ로 인가.
        **nullspace 투영 안 함** = 충돌 직전엔 EE(rod)도 양보(안전>task). → 충돌 0 지향."""
        B = obs_pos_w.shape[0]
        if self._arm_link_ids is None:
            names = ["panda_link4", "panda_link5", "panda_link6"]
            self._arm_link_ids = [robot.body_names.index(n) for n in names]
        d0 = getattr(self.env.cfg, "hard_d0", 0.08)
        krep = getattr(self.env.cfg, "hard_krep", 30.0)
        kbrake = getattr(self.env.cfg, "hard_kbrake", 40.0)
        lr, r_obs, cap = self.null_link_r, self.env.cfg.obstacle_radius, 80.0
        jac_full = robot.root_physx_view.get_jacobians()              # (B,nbody,6,ndof)
        tau = torch.zeros(B, 7, device=self.device)
        for l in self._arm_link_ids:
            p_l = robot.data.body_pos_w[:, l, :]                      # (B,3)
            v_l = robot.data.body_lin_vel_w[:, l, :]                  # (B,3)
            diff = p_l.unsqueeze(1) - obs_pos_w                       # (B,Nobs,3)
            dctr = diff.norm(dim=-1)                                  # (B,Nobs)
            surf = dctr - r_obs - lr                                  # 표면거리
            within = obs_active & (surf < d0)
            away = diff / dctr.unsqueeze(-1).clamp_min(1e-6)          # obstacle→link (밀 방향)
            v_app = -(v_l.unsqueeze(1) * away).sum(-1)               # (B,Nobs) >0 = 접근중
            surf_safe = surf.clamp_min(5e-3)
            mag = krep * (1.0 / surf_safe - 1.0 / d0) + kbrake * v_app.clamp_min(0.0)
            mag = torch.where(within, mag.clamp(max=cap), torch.zeros_like(mag))   # (B,Nobs)
            f = (mag.unsqueeze(-1) * away).sum(dim=1)                 # (B,3) 합력
            J_l = jac_full[:, l, :3, :][:, :, joint_ids]             # (B,3,7) 링크 위치 Jacobian
            tau = tau + torch.bmm(J_l.transpose(-1, -2), f.unsqueeze(-1)).squeeze(-1)
        return tau

    def _hard_safety_rod_torque(self, J1, J2, obs_pos_w, obs_active):
        """Rod hard 안전 (velocity-zeroing 근사): rod 선분이 장애물 임박 시 접근속도 제동+barrier.
        rod에 가할 힘을 양 EE Jᵀ로 분배 인가 → rod가 장애물 직전 stall. 팔과 같은 원리(torque-level)."""
        B = obs_pos_w.shape[0]
        HALF_W, ROD_R, cap = 0.4, 0.02, 80.0
        d0 = getattr(self.env.cfg, "hard_d0", 0.08)
        krep = getattr(self.env.cfg, "hard_krep", 30.0)
        kbrake = getattr(self.env.cfg, "hard_kbrake", 40.0)
        r_obs = self.env.cfg.obstacle_radius
        rod_pos = self.env.rod.data.root_pos_w                       # (B,3) world
        rod_quat = self.env.rod.data.root_quat_w
        rod_vel = self.env.rod.data.root_lin_vel_w                   # (B,3)
        axis = self._quat_apply(rod_quat, torch.tensor([HALF_W, 0., 0.], device=self.device).expand(B, 3))
        end1 = rod_pos - axis; seg = 2.0 * axis
        seg_len2 = (seg * seg).sum(-1, keepdim=True).clamp_min(1e-8)
        AP = obs_pos_w - end1.unsqueeze(1)                           # (B,N,3)
        u = ((AP * seg.unsqueeze(1)).sum(-1) / seg_len2).clamp(0., 1.).unsqueeze(-1)
        closest = end1.unsqueeze(1) + u * seg.unsqueeze(1)          # (B,N,3) rod 선분상 최근접점
        diff = closest - obs_pos_w                                  # obstacle→rod (밀 방향)
        dctr = diff.norm(dim=-1)
        surf = dctr - r_obs - ROD_R
        within = obs_active & (surf < d0)
        away = diff / dctr.unsqueeze(-1).clamp_min(1e-6)
        v_app = -(rod_vel.unsqueeze(1) * away).sum(-1)              # (B,N) >0=접근
        surf_safe = surf.clamp_min(5e-3)
        mag = krep * (1.0 / surf_safe - 1.0 / d0) + kbrake * v_app.clamp_min(0.0)
        mag = torch.where(within, mag.clamp(max=cap), torch.zeros_like(mag))
        F = (mag.unsqueeze(-1) * away).sum(dim=1)                   # (B,3) rod에 가할 총 힘
        wrench = torch.cat([0.5 * F, torch.zeros(B, 3, device=self.device)], dim=-1)  # 양 EE에 절반씩
        tau1 = torch.bmm(J1.transpose(-1, -2), wrench.unsqueeze(-1)).squeeze(-1)
        tau2 = torch.bmm(J2.transpose(-1, -2), wrench.unsqueeze(-1)).squeeze(-1)
        return tau1, tau2

    # ──────────────────────────────────────────────────────────────────
    # Main: target → both arms' joint torques
    # ──────────────────────────────────────────────────────────────────
    def compute_torques(self, target_obj_pos, target_obj_quat, target_x_rel=None,
                        K_arm1=None, K_arm2=None, null_alpha1=None, null_alpha2=None):
        """
        Args:
            target_obj_pos: (B, 3) rod target position (world)
            target_obj_quat: (B, 4) rod target quaternion (world)
            target_x_rel: ignored (interface 호환용)
            K_arm1, K_arm2: (B,) per-arm positional stiffness (Stage 1). None이면 고정 self.K_pos.
            null_alpha1, null_alpha2: (B,) per-arm nullspace 명령 (8-DoF action). None이면 nullspace 미사용.

        Returns:
            tau_1, tau_2: (B, 7) joint torques
            info: diagnostic dict
        """
        # 1. Compute per-arm targets
        ee1_target_pos, ee1_target_quat, ee2_target_pos, ee2_target_quat = self._compute_ee_targets(
            target_obj_pos, target_obj_quat
        )

        # ── Stage 1 per-arm K: D는 damping ratio 유지 위해 √(K/K_base) 스케일 ──
        if K_arm1 is not None:
            D_arm1 = self.D_pos * torch.sqrt(K_arm1 / self.K_pos)
            D_arm2 = self.D_pos * torch.sqrt(K_arm2 / self.K_pos)
        else:
            D_arm1 = D_arm2 = None

        # 2. Single-arm impedance for each arm (per-arm K 적용)
        wrench_1 = self._single_arm_impedance(self.robot_1, self.ee_idx_1, ee1_target_pos, ee1_target_quat,
                                              K_pos=K_arm1, D_pos=D_arm1)
        wrench_2 = self._single_arm_impedance(self.robot_2, self.ee_idx_2, ee2_target_pos, ee2_target_quat,
                                              K_pos=K_arm2, D_pos=D_arm2)

        # 3. Jacobian transpose → joint torque
        J1, J2 = self._get_jacobians()
        tau_1 = torch.bmm(J1.transpose(-1, -2), wrench_1.unsqueeze(-1)).squeeze(-1)
        tau_2 = torch.bmm(J2.transpose(-1, -2), wrench_2.unsqueeze(-1)).squeeze(-1)

        # 3a. 모델기반 nullspace 팔-장애물 회피 (control A): 컨트롤러가 여분 DoF로 팔을 장애물에서 밀어냄.
        # use_nullspace_avoidance=False면 OFF (당장은 RL이 거시 회피 학습. 필요시 재활성).
        if (self.use_nullspace_avoidance and getattr(self.env.cfg, "n_obstacles", 0) > 0
                and getattr(self.env, "obstacle_active", None) is not None):
            obs_pos_w = torch.stack([o.data.root_pos_w for o in self.env.obstacles], dim=1)  # (B,Nobs,3)
            obs_active = self.env.obstacle_active
            tau_1 = tau_1 + self._nullspace_obstacle_torque(self.robot_1, J1, self.joint_ids_1, obs_pos_w, obs_active)
            tau_2 = tau_2 + self._nullspace_obstacle_torque(self.robot_2, J2, self.joint_ids_2, obs_pos_w, obs_active)

        # 3a'. RL 구동 nullspace (control: 8-DoF action의 α로 팔꿈치 제어 → 팔-장애물 회피 학습).
        if null_alpha1 is not None:
            if getattr(self.env.cfg, "use_swivel_nullspace", False):
                # ψ_des handle: action[6:8]=목표 swivel각 → 팔꿈치 servo (전 범위·setpoint).
                tau_1 = tau_1 + self._nullspace_swivel_torque(self.robot_1, J1, self.joint_ids_1, null_alpha1)
                tau_2 = tau_2 + self._nullspace_swivel_torque(self.robot_2, J2, self.joint_ids_2, null_alpha2)
            else:
                tau_1 = tau_1 + self._nullspace_rl_torque(self.robot_1, J1, self.joint_ids_1, null_alpha1)
                tau_2 = tau_2 + self._nullspace_rl_torque(self.robot_2, J2, self.joint_ids_2, null_alpha2)

        # 3a''. Hard 안전 필터 (충돌 0 지향): 팔 링크 장애물 임박 시 제동+반발 (nullspace 투영 X).
        if (getattr(self.env.cfg, "use_hard_safety", False) and getattr(self.env.cfg, "n_obstacles", 0) > 0
                and getattr(self.env, "obstacle_active", None) is not None):
            obs_pos_w = torch.stack([o.data.root_pos_w for o in self.env.obstacles], dim=1)
            obs_active = self.env.obstacle_active
            tau_1 = tau_1 + self._hard_safety_torque(self.robot_1, self.joint_ids_1, obs_pos_w, obs_active)
            tau_2 = tau_2 + self._hard_safety_torque(self.robot_2, self.joint_ids_2, obs_pos_w, obs_active)
            rt1, rt2 = self._hard_safety_rod_torque(J1, J2, obs_pos_w, obs_active)   # rod도 hard
            tau_1 = tau_1 + rt1; tau_2 = tau_2 + rt2

        # 3b. Gravity compensation (flag-gated). τ += sign·G(q).
        # EOM: M q̈ + C + G = τ → 정적 유지엔 τ=G 필요. sign은 API convention 따라 부호 테스트로 결정.
        if getattr(self, "gravity_comp", False):
            g1 = self.robot_1.root_physx_view.get_generalized_gravity_forces()[:, self.joint_ids_1]
            g2 = self.robot_2.root_physx_view.get_generalized_gravity_forces()[:, self.joint_ids_2]
            tau_1 = tau_1 + self.grav_sign * g1
            tau_2 = tau_2 + self.grav_sign * g2

        # 4. Effort clamp
        tau_1 = torch.clamp(tau_1, min=-50.0, max=50.0)
        tau_2 = torch.clamp(tau_2, min=-50.0, max=50.0)

        # 5. Diagnostics
        self._last_info = {
            "ee1_pos_err_norm": torch.norm(ee1_target_pos - self.robot_1.data.body_pos_w[:, self.ee_idx_1, :], dim=-1).mean().item(),
            "ee2_pos_err_norm": torch.norm(ee2_target_pos - self.robot_2.data.body_pos_w[:, self.ee_idx_2, :], dim=-1).mean().item(),
            "wrench_1_force_norm": torch.norm(wrench_1[:, :3], dim=-1).mean().item(),
            "wrench_2_force_norm": torch.norm(wrench_2[:, :3], dim=-1).mean().item(),
            "tau_1_max": tau_1.abs().max().item(),
            "tau_2_max": tau_2.abs().max().item(),
            "K_arm1_mean": (K_arm1.mean().item() if K_arm1 is not None else self.K_pos),
            "K_arm2_mean": (K_arm2.mean().item() if K_arm2 is not None else self.K_pos),
        }

        return tau_1, tau_2, self._last_info
