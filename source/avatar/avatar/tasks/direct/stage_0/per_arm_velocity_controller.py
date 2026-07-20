"""
Per-arm Velocity Controller for dual-arm cooperative manipulation.

Torque control 대신 velocity control 사용 — closed-chain instability 해결.

전체 흐름:
  RL (10Hz, 100ms)
    출력: desired_rod_lin_vel (3D), desired_rod_ang_vel (3D)
         ↓
  Per-arm 기하학적 변환 (매 physics step, 240Hz)
    rod_velocity → ee1_vel, ee2_vel (rigid body 관계 v_ee = v_rod + ω × r)
         ↓
  Jacobian damped pseudoinverse
    ee_vel → joint velocity (dq_target)
         ↓
  Isaac Lab ImplicitActuator (stiffness=0, damping>0 → velocity mode)
    set_joint_velocity_target(dq_target) → PhysX가 토크 자동 계산

장점:
  - Closed-chain 안정성: joint-level PD가 진동 흡수
  - Singularity 안전: damped pseudoinverse + velocity_limit
  - Wrench transport 부호 오류 위험 없음
  - Inertia matching 자동 처리

Geometry assumptions (env3 fixed joint setup):
  - Rod local frame: +X = rod's long axis
  - Grasp at rod ends: ±0.4m from rod center
  - panda_hand R_world = R_rod × Rx(π)
  - TCP_OFFSET = 0.1034m
"""
from __future__ import annotations
import torch
import isaaclab.utils.math as math_utils


class PerArmVelocityController:
    """
    Per-arm task-space velocity controller.

    각 arm은 자기 grasp point의 desired velocity를 계산하고,
    damped pseudoinverse로 joint velocity로 변환 → velocity actuator에 명령.
    """

    TCP_OFFSET = 0.1034
    ROD_HALF_LENGTH = 0.4   # rod 0.8m / 2

    def __init__(
        self,
        env,
        damped_inv_lambda: float = 0.05,   # damped pseudoinverse damping factor
        velocity_limit: float = 2.0,        # joint velocity clamp (rad/s)
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

        self.damped_inv_lambda = damped_inv_lambda
        self.velocity_limit = velocity_limit

        # Grasp offset (rod local frame): +X 방향 (rod의 절반 길이 + TCP)
        # ee1 local = -grasp_offset (왼쪽 grasp from rod center)
        # ee2 local = +grasp_offset (오른쪽 grasp)
        # NOTE: TCP_OFFSET이 z 방향에 추가됨 — panda_hand가 rod로부터 TCP만큼 떨어진 점에서 잡음
        self._grasp_offset_local_1 = torch.tensor(
            [-self.ROD_HALF_LENGTH, 0.0, 0.0], device=self.device
        ).unsqueeze(0).expand(self.num_envs, 3).contiguous()
        self._grasp_offset_local_2 = torch.tensor(
            [+self.ROD_HALF_LENGTH, 0.0, 0.0], device=self.device
        ).unsqueeze(0).expand(self.num_envs, 3).contiguous()

        # Identity matrix (preallocated for damped inverse)
        self._eye6 = torch.eye(6, device=self.device).unsqueeze(0)  # (1, 6, 6)

        self._last_info: dict = {}

    # ──────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _quat_apply(q, v):
        """Rotate vector v (B,3) by quat q (B,4)."""
        qv = q[:, 1:4]
        qw = q[:, :1]
        t = 2.0 * torch.cross(qv, v, dim=-1)
        return v + qw * t + torch.cross(qv, t, dim=-1)

    def _get_jacobians(self):
        J1_full = self.robot_1.root_physx_view.get_jacobians()
        J1 = J1_full[:, self.ee_idx_1, :, :][:, :, self.joint_ids_1]  # (B, 6, 7)
        J2_full = self.robot_2.root_physx_view.get_jacobians()
        J2 = J2_full[:, self.ee_idx_2, :, :][:, :, self.joint_ids_2]
        return J1, J2

    def _damped_pseudoinverse(self, J):
        """
        Damped least squares pseudoinverse: J_pinv = J^T (J J^T + λ²I)⁻¹.

        Singularity 근처에서 발산 방지.

        Args:
            J: (B, 6, 7) geometric Jacobian

        Returns:
            J_pinv: (B, 7, 6)
        """
        B = J.shape[0]
        JJT = torch.bmm(J, J.transpose(-1, -2))                # (B, 6, 6)
        eye = self._eye6.expand(B, 6, 6)
        JJT_damped = JJT + (self.damped_inv_lambda ** 2) * eye  # (B, 6, 6)
        # Solve (JJT_damped × X = eye) for X = JJT_damped⁻¹
        JJT_inv = torch.linalg.solve(JJT_damped, eye)            # (B, 6, 6)
        J_pinv = torch.bmm(J.transpose(-1, -2), JJT_inv)        # (B, 7, 6)
        return J_pinv

    # ──────────────────────────────────────────────────────────────────
    # Main: rod velocity → joint velocities (양 arm)
    # ──────────────────────────────────────────────────────────────────
    def compute_joint_velocities(
        self,
        desired_rod_lin_vel: torch.Tensor,    # (B, 3) world frame
        desired_rod_ang_vel: torch.Tensor,    # (B, 3) world frame
    ):
        """
        Rod의 task-space velocity → 양 arm의 joint velocity target.

        Args:
            desired_rod_lin_vel: (B, 3) m/s
            desired_rod_ang_vel: (B, 3) rad/s

        Returns:
            dq1, dq2: (B, 7) joint velocities for arm 1, arm 2
            info: diagnostic dict
        """
        # ── 1. Grasp point offset (rod body frame → world frame) ──
        rod_quat = self.rod.data.root_quat_w
        r1_world = self._quat_apply(rod_quat, self._grasp_offset_local_1)  # (B, 3) 왼쪽
        r2_world = self._quat_apply(rod_quat, self._grasp_offset_local_2)  # (B, 3) 오른쪽

        # ── 2. Rod velocity → EE velocity ──
        # Rigid body 관계: v_ee = v_rod + ω_rod × r_ee
        ee1_lin_vel = desired_rod_lin_vel + torch.cross(desired_rod_ang_vel, r1_world, dim=-1)
        ee2_lin_vel = desired_rod_lin_vel + torch.cross(desired_rod_ang_vel, r2_world, dim=-1)
        # Angular velocity는 강체 → 동일
        ee1_ang_vel = desired_rod_ang_vel
        ee2_ang_vel = desired_rod_ang_vel
        # 6D EE velocity
        ee1_vel = torch.cat([ee1_lin_vel, ee1_ang_vel], dim=-1)  # (B, 6)
        ee2_vel = torch.cat([ee2_lin_vel, ee2_ang_vel], dim=-1)

        # ── 3. Jacobian pseudoinverse ──
        J1, J2 = self._get_jacobians()
        J1_pinv = self._damped_pseudoinverse(J1)  # (B, 7, 6)
        J2_pinv = self._damped_pseudoinverse(J2)

        # ── 4. Joint velocity = J_pinv × ee_vel ──
        dq1 = torch.bmm(J1_pinv, ee1_vel.unsqueeze(-1)).squeeze(-1)  # (B, 7)
        dq2 = torch.bmm(J2_pinv, ee2_vel.unsqueeze(-1)).squeeze(-1)

        # ── 5. Joint velocity clamp ──
        dq1 = torch.clamp(dq1, min=-self.velocity_limit, max=self.velocity_limit)
        dq2 = torch.clamp(dq2, min=-self.velocity_limit, max=self.velocity_limit)

        # ── 6. Diagnostics ──
        self._last_info = {
            "ee1_vel_norm": torch.norm(ee1_vel, dim=-1).mean().item(),
            "ee2_vel_norm": torch.norm(ee2_vel, dim=-1).mean().item(),
            "dq1_max": dq1.abs().max().item(),
            "dq2_max": dq2.abs().max().item(),
            "J1_condition": torch.linalg.cond(J1).mean().item() if J1.numel() > 0 else 0.0,
            "J2_condition": torch.linalg.cond(J2).mean().item() if J2.numel() > 0 else 0.0,
        }

        return dq1, dq2, self._last_info

    # ──────────────────────────────────────────────────────────────────
    # Compat wrapper — 기존 controller interface 호환 (target_obj_pos/quat 입력 받음)
    # 사용 안 함. env3가 직접 compute_joint_velocities 호출.
    # ──────────────────────────────────────────────────────────────────
    def compute_torques(self, *args, **kwargs):
        raise NotImplementedError(
            "PerArmVelocityController는 torque 인터페이스 미지원. "
            "env3._apply_action에서 compute_joint_velocities()를 직접 호출하고 "
            "set_joint_velocity_target()을 사용해야 합니다."
        )
