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

    # ──────────────────────────────────────────────────────────────────
    # Main: target → both arms' joint torques
    # ──────────────────────────────────────────────────────────────────
    def compute_torques(self, target_obj_pos, target_obj_quat, target_x_rel=None,
                        K_arm1=None, K_arm2=None):
        """
        Args:
            target_obj_pos: (B, 3) rod target position (world)
            target_obj_quat: (B, 4) rod target quaternion (world)
            target_x_rel: ignored (interface 호환용)
            K_arm1, K_arm2: (B,) per-arm positional stiffness (Stage 1). None이면 고정 self.K_pos.

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
