"""
Cooperative Impedance Controller for Dual-Arm Rigid Transport (Phase 2 refactor)

이전 버전 (Phase 3.1)은 rod root pose를 직접 추종하는 object-centric.
본 refactor는 **cooperative task space** (x_abs, x_rel)로 분해해 두 채널을 독립 제어:

    x_abs = (ee_1 + ee_2) / 2       ← 객체가 어디 있는가 (절대 운동)
    x_rel = ee_1 − ee_2              ← 두 팔이 얼마나 벌어져 있는가 (상대 운동)

    f_abs = K_abs·(x_abs_d − x_abs) − D_abs·v_abs    ← 객체를 끌어당기는 힘
    f_rel = K_rel·(x_rel_d − x_rel) − D_rel·v_rel    ← 두 팔 사이를 유지하는 힘

    F_grasp_1 = f_abs / 2 + f_rel                    ← grasp 1 에 적용 (world)
    F_grasp_2 = f_abs / 2 − f_rel                    ← grasp 2 에 적용 (world)

회전은 rod 자세를 직접 사용 (fixed-joint sim에서 두 EE 회전 == rod 회전):
    M_abs = K_abs_rot·rot_err_aa − D_abs_rot·ω_rod
    M_grasp_1 = M_grasp_2 = 0.5·M_abs

장점:
- K_abs, K_rel 분리 노출 → 향후 RL이 두 채널을 독립 출력 가능 (Variable Impedance)
- DA-VIL 등 cooperative manipulation 문헌과 동일한 분해 → fair comparison

한계:
- Fixed joint sim에선 x_rel이 물리적으로 강제됨 → f_rel ≈ 0 (학습 신호 약함)
- 그러나 transient 응답·sim-to-real 시 K_rel이 의미를 가짐
- 회전 cooperative 분해는 v1에서 생략 (rod 자세 직접 추종)

Gain default:
    K_abs_pos = 200 N/m,  D_abs_pos = 60  (≈ critical for 5kg eff. mass)
    K_abs_rot = 20 Nm/rad, D_abs_rot = 8
    K_rel     = 200 N/m,  D_rel     = 60
"""

import math
import torch


class CooperativeImpedanceController:
    # Franka panda_hand origin(wrist)에서 TCP까지 +Z 오프셋 (sampler·env3와 일치)
    TCP_OFFSET = 0.1034

    def __init__(
        self,
        env,
        # ── Absolute (object motion) channel ──
        K_abs_pos: float = 200.0,
        D_abs_pos: float = 60.0,
        K_abs_rot: float = 20.0,
        D_abs_rot: float = 8.0,
        # ── Relative (inter-arm) channel ──
        # NOTE: Fixed-joint sim에선 x_rel이 physics constraint로 이미 고정됨.
        # 0이 아니면 numerical noise × K_rel 또는 × D_rel feedback이 oscillation 유발.
        # 따라서 default 0 (cooperative 분해 구조는 유지 — Phase 4 RL이 비-zero 출력 가능).
        # Deformable grasp(향후 sim-to-real)에선 의미 있는 값으로 활성화.
        K_rel: float = 0.0,
        D_rel: float = 0.0,
    ):
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs

        self.robot_1 = env.robot_1
        self.robot_2 = env.robot_2
        self.rod = env.rod

        self.ee_idx_1 = env.ee_body_idx_1
        self.ee_idx_2 = env.ee_body_idx_2
        self.joint_ids_1 = env.robot_1_joint_ids
        self.joint_ids_2 = env.robot_2_joint_ids

        self.K_abs_pos = K_abs_pos
        self.D_abs_pos = D_abs_pos
        self.K_abs_rot = K_abs_rot
        self.D_abs_rot = D_abs_rot
        self.K_rel = K_rel
        self.D_rel = D_rel

    # ──────────────────────────────────────────────────────────────────
    # Quaternion utilities (wxyz convention, Isaac Lab과 일치)
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _quat_conj(q):
        return torch.cat([q[:, :1], -q[:, 1:]], dim=-1)

    @staticmethod
    def _quat_mul(q1, q2):
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack([w, x, y, z], dim=-1)

    @staticmethod
    def _quat_apply(q, v):
        """Apply quaternion rotation to vector (B,3)→(B,3). q is wxyz."""
        qv = q[:, 1:]
        qw = q[:, :1]
        t = 2.0 * torch.cross(qv, v, dim=-1)
        return v + qw * t + torch.cross(qv, t, dim=-1)

    @classmethod
    def _quat_to_axis_angle(cls, q):
        """Quat (B,4) → axis-angle vector (B,3). 작은 회전엔 ≈ 2·xyz."""
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
    # Jacobian — panda_hand body 기준 arm 7 joint Jacobian
    # ──────────────────────────────────────────────────────────────────
    def _get_jacobians(self):
        J1_full = self.robot_1.root_physx_view.get_jacobians()
        J1 = J1_full[:, self.ee_idx_1, :, :][:, :, self.joint_ids_1]  # (B, 6, 7)
        J2_full = self.robot_2.root_physx_view.get_jacobians()
        J2 = J2_full[:, self.ee_idx_2, :, :][:, :, self.joint_ids_2]
        return J1, J2

    # ──────────────────────────────────────────────────────────────────
    # 메인: target → 양 팔 joint torque
    # ──────────────────────────────────────────────────────────────────
    def compute_torques(self, target_obj_pos, target_obj_quat, target_x_rel):
        """
        Args:
            target_obj_pos:  (B, 3) 객체 목표 위치 (world frame).
                             RL/test에서 들어오는 "객체를 어디에 두고 싶은가".
            target_obj_quat: (B, 4) 객체 목표 자세 wxyz (world frame).
            target_x_rel:    (B, 3) 두 EE 사이 desired 분리 벡터 (world frame).
                             reset 시 캡처된 초기 값을 그대로 사용 (fixed-joint 가정).

        Returns:
            tau_1, tau_2: (B, 7) 양 팔 joint torque (clamped to ±50 N·m)
            info: dict diagnostics
        """
        # ── 1. EE state 추출 ──
        ee1 = self.robot_1.data.body_state_w[:, self.ee_idx_1]  # (B, 13) [pos(3); quat(4); lin(3); ang(3)]
        ee2 = self.robot_2.data.body_state_w[:, self.ee_idx_2]
        ee1_pos, ee1_quat, ee1_lin = ee1[:, 0:3], ee1[:, 3:7], ee1[:, 7:10]
        ee2_pos, ee2_quat, ee2_lin = ee2[:, 0:3], ee2[:, 3:7], ee2[:, 7:10]

        # ── 2. Cooperative task space 분해 ──
        # x_abs: rod 실제 pose 직접 사용 (translation-rotation cross-coupling 제거).
        # 이전: 0.5*(ee1+ee2) — rod 회전 시 EE 중점이 이동하여 spurious translation error 유발.
        rod_pos = self.rod.data.root_pos_w
        rod_quat_for_offset = self.rod.data.root_quat_w
        # rod center + TCP offset (panda_hand 방향, world frame) → EE midpoint와 정렬되는 reference
        tcp_local_pre = torch.tensor(
            [0.0, 0.0, self.TCP_OFFSET], device=self.device
        ).expand(self.num_envs, 3).contiguous()
        x_abs = rod_pos + self._quat_apply(rod_quat_for_offset, tcp_local_pre)  # (B, 3) rod 기준
        x_rel = ee1_pos - ee2_pos                # (B, 3) EE 분리 벡터 (그대로)
        v_rel = ee1_lin - ee2_lin

        # ── 3. Absolute (객체 운동) target:
        #     target_obj_pos는 rod_pos 기준 → x_abs는 rod_pos + R_rod·(0,0,TCP) 이므로 offset 필요
        tcp_local = torch.tensor(
            [0.0, 0.0, self.TCP_OFFSET], device=self.device
        ).expand(self.num_envs, 3).contiguous()
        target_x_abs = target_obj_pos + self._quat_apply(target_obj_quat, tcp_local)  # (B, 3)

        # ── 4. 채널별 impedance ──
        # Absolute translation
        # NOTE: damping은 rod_lin (단일 rigid body 측정, less noisy)을 사용. EE 기반 v_abs는
        # TCP offset × ω 효과로 회전 transient가 translation damping에 누적 contamination됨.
        # rod_lin 사용이 task 1 (refactor 전) 거동과 일치 + 실용적.
        rod_lin = self.rod.data.root_lin_vel_w                                          # (B, 3)
        f_abs_lin = self.K_abs_pos * (target_x_abs - x_abs) - self.D_abs_pos * rod_lin  # (B, 3)

        # Relative translation
        f_rel = self.K_rel * (target_x_rel - x_rel) - self.D_rel * v_rel                # (B, 3)

        # Absolute rotation (rod 자세 직접, two EE rotation == rod rotation in fixed-joint sim)
        rod_quat = self.rod.data.root_quat_w
        rod_ang = self.rod.data.root_ang_vel_w
        q_err = self._quat_mul(target_obj_quat, self._quat_conj(rod_quat))
        rot_err_aa = self._quat_to_axis_angle(q_err)
        # Angle clamp: overshoot이 커져도 controller 복원력 포화 → energy injection 방지.
        # max 1 rad (~57°). 큰 회전은 다단계로 따라가도록 강제.
        rot_err_norm = torch.norm(rot_err_aa, dim=-1, keepdim=True)
        rot_clamp_factor = torch.clamp(1.0 / (rot_err_norm + 1e-8), max=1.0)
        rot_err_aa_clamped = rot_err_aa * rot_clamp_factor
        m_abs_ang = self.K_abs_rot * rot_err_aa_clamped - self.D_abs_rot * rod_ang      # (B, 3)

        # ── 5. Grasp point 분배 (cooperative wrench distribution) ──
        # F_grasp_i = absolute/2 ± f_rel,   M_grasp_i = M_abs/2
        F_grasp_1 = 0.5 * f_abs_lin + f_rel
        F_grasp_2 = 0.5 * f_abs_lin - f_rel
        M_grasp_1 = 0.5 * m_abs_ang
        M_grasp_2 = 0.5 * m_abs_ang

        # ── 6. Grasp → panda_hand origin 평행 이동 ──
        #     panda_hand가 grasp point에서 panda local +Z 방향 TCP_OFFSET 떨어져 있음
        #     ⇒ delta_panda_to_grasp (world) = R_panda · (0, 0, +TCP_OFFSET)
        #     M_panda = M_grasp + delta × F_grasp
        delta_pg_1 = self._quat_apply(ee1_quat, tcp_local)   # (B, 3) world
        delta_pg_2 = self._quat_apply(ee2_quat, tcp_local)

        M_panda_1 = M_grasp_1 + torch.cross(delta_pg_1, F_grasp_1, dim=-1)
        M_panda_2 = M_grasp_2 + torch.cross(delta_pg_2, F_grasp_2, dim=-1)

        wrench_1 = torch.cat([F_grasp_1, M_panda_1], dim=-1)   # (B, 6)
        wrench_2 = torch.cat([F_grasp_2, M_panda_2], dim=-1)

        # ── 7. Jacobian transpose → joint torque (task-space wrench → joint torque) ──
        # NOTE: Phase 3.1 검증 상태 (gravity OFF). Gravity compensation은 두 번 시도(+g, -g) 모두
        # 발산. 원인 미해결 (`get_generalized_gravity_forces()`의 frame/sign convention 또는
        # Robot_2 base yaw=π에 따른 비대칭 가능성). 추후 별도 디버깅 필요. 현재 cfg에서 robot
        # gravity OFF로 유지하면 본 컨트롤러는 정확한 task-space tracking 수행.
        J1, J2 = self._get_jacobians()
        tau_1 = torch.bmm(J1.transpose(-1, -2), wrench_1.unsqueeze(-1)).squeeze(-1)
        tau_2 = torch.bmm(J2.transpose(-1, -2), wrench_2.unsqueeze(-1)).squeeze(-1)

        # ── 8. Effort 안전 클램핑 ──
        tau_1 = torch.clamp(tau_1, min=-50.0, max=50.0)
        tau_2 = torch.clamp(tau_2, min=-50.0, max=50.0)

        # ── 9. 진단 ──
        pos_err = target_x_abs - x_abs
        rel_err = target_x_rel - x_rel
        info = {
            # Absolute channel
            "pos_err_norm": torch.norm(pos_err, dim=-1).mean().item(),
            "rot_err_deg": (torch.norm(rot_err_aa, dim=-1) * 180.0 / math.pi).mean().item(),
            "f_abs_lin_norm": torch.norm(f_abs_lin, dim=-1).mean().item(),
            "m_abs_ang_norm": torch.norm(m_abs_ang, dim=-1).mean().item(),
            # Relative channel
            "rel_err_norm": torch.norm(rel_err, dim=-1).mean().item(),
            "f_rel_norm": torch.norm(f_rel, dim=-1).mean().item(),
            # Joint
            "tau_1_max": tau_1.abs().max().item(),
            "tau_2_max": tau_2.abs().max().item(),
            # Legacy keys (back-compat for test script)
            "f_ext_lin_norm": torch.norm(f_abs_lin, dim=-1).mean().item(),
            "f_ext_ang_norm": torch.norm(m_abs_ang, dim=-1).mean().item(),
        }
        return tau_1, tau_2, info
