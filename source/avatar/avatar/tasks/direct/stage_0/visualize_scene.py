"""
Scene visualization script.

Phase 1 (current): kinematic rod 시각화 검증.
- 두 팔이 IK로 풀린 시작 자세에 위치하는지
- rod (갈색)가 두 그리퍼 사이에 정확히 끼어 있는지
- goal_rod (녹색)가 양 팔의 도달 가능 영역 내에 있는지
- 빨강/파랑 큐브(EE goal)가 녹색 rod의 양 끝과 일치하는지

학습 코드는 건드리지 않으며, 환경에 zero-velocity 명령만 보내고
주기적으로 reset 하여 다양한 샘플을 확인할 수 있다.

사용법:
    python visualize_scene.py --num_envs 4 --reset_period 200
    python visualize_scene.py --num_envs 1 --reset_period 100  # 단일 환경 정밀 확인
"""

import argparse
import math
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize Dual-Arm Scene Configuration")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel envs to render")
parser.add_argument(
    "--reset_period",
    type=int,
    default=0,
    help="Force a reset every N control steps (0 = disabled, lock on single sample). "
    "Set e.g. 600 to cycle through samples.",
)
parser.add_argument(
    "--episode_length_s",
    type=float,
    default=600.0,
    help="Episode length seconds. Default 600s (=10min) effectively disables auto-truncation "
    "so the scene stays put for inspection.",
)
parser.add_argument(
    "--hold_action",
    type=str,
    default="zero",
    choices=["zero", "random"],
    help="zero: arms hold IK start pose; random: small random velocity to verify dynamics",
)
parser.add_argument(
    "--freeze_arms",
    action="store_true",
    default=True,
    help="Re-write IK joint state every step to prevent gravity drift (visualization only)",
)
parser.add_argument(
    "--no_freeze_arms",
    action="store_false",
    dest="freeze_arms",
    help="Disable arm freezing to observe gravity drift",
)
parser.add_argument(
    "--drift_period",
    type=int,
    default=60,
    help="Print constraint drift diagnostics every N steps (0 = off). Default 60 = ~1s sim time.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from dual_arm_transport_env3 import DualrobotEnv, DualrobotCfg


def _print_header():
    print("=" * 70)
    print(f"  Scene Visualization | num_envs={args_cli.num_envs} | reset_period={args_cli.reset_period}")
    print("=" * 70)
    print("Color guide:")
    print("  Brown rod    : current shared object (start_obj_pose from sampler)")
    print("  Green rod    : goal pose for the shared object")
    print("  Red cube     : panda_hand_1 (wrist) goal pose ─ 10.7cm above green rod LEFT end")
    print("  Blue cube    : panda_hand_2 (wrist) goal pose ─ 10.7cm above green rod RIGHT end")
    print("-" * 70)
    print("Visual sanity checks (Phase 2 TCP-aligned):")
    print("  1. Brown rod horizontal, two grippers above it pointing DOWN (fingers around rod)")
    print("  2. Wrist (panda_hand body) sits ~10.7cm above each rod end")
    print("  3. Fingers tips coincide with rod ends (TCP at rod end)")
    print("  4. Green rod inside both arms' reachable workspace")
    print("  5. Red/Blue cubes ABOVE green rod ends (at wrist positions, not rod surface)")
    print("=" * 70)
    print("Tip: rotate camera in viewport, ESC or Ctrl+C in terminal to quit.")
    print("=" * 70)


def _capture_locked_state(env):
    """현재 IK 풀린 자세 + rod root pose를 freeze 용으로 캡처"""
    q1 = env.robot_1.data.joint_pos[:, env.robot_1_joint_ids].clone()
    q2 = env.robot_2.data.joint_pos[:, env.robot_2_joint_ids].clone()
    # Phase 2: rod도 함께 캡처 (fixed joint 일관성 유지)
    rod_pose = env.rod.data.root_state_w[:, :7].clone()
    return q1, q2, rod_pose


# ────────────────────────────────────────────────────────────────────
# Phase 2 진단: closed kinematic loop의 constraint 잔류 오차 측정
# ────────────────────────────────────────────────────────────────────
def _quat_conj(q):
    """quat (w, x, y, z) conjugate"""
    return torch.cat([q[:, :1], -q[:, 1:]], dim=-1)


def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def _quat_apply(q, v):
    """v (3D) rotated by quat q (wxyz)"""
    qv = q[:, 1:]  # vector part
    qw = q[:, :1]  # scalar part
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + qw * t + torch.cross(qv, t, dim=-1)


def _relative_pose(p_a, q_a, p_b, q_b):
    """B의 A 로컬 프레임 기준 상대 포즈"""
    q_a_inv = _quat_conj(q_a)
    rel_pos = _quat_apply(q_a_inv, p_b - p_a)
    rel_rot = _quat_mul(q_a_inv, q_b)
    return rel_pos, rel_rot


def _capture_initial_relative_pose(env):
    """reset 직후 EE-EE 상대 포즈를 reference로 저장"""
    ee1_state = env.robot_1.data.body_state_w[:, env.ee_body_idx_1, :7]
    ee2_state = env.robot_2.data.body_state_w[:, env.ee_body_idx_2, :7]
    rel_pos, rel_rot = _relative_pose(
        ee1_state[:, 0:3], ee1_state[:, 3:7],
        ee2_state[:, 0:3], ee2_state[:, 3:7],
    )
    return {"rel_pos": rel_pos.clone(), "rel_rot": rel_rot.clone()}


def _measure_drift(env, ref):
    """
    두 가지 측정:
    (A) EE-EE 상대 포즈 drift: reset 직후 reference 대비 변화
    (B) Joint anchor 위치 정합 오차: 두 fixed joint constraint 자체의 잔류 위반
    """
    # ── (A) EE-EE 상대 포즈 drift ──
    ee1_state = env.robot_1.data.body_state_w[:, env.ee_body_idx_1, :7]
    ee2_state = env.robot_2.data.body_state_w[:, env.ee_body_idx_2, :7]
    rel_pos, rel_rot = _relative_pose(
        ee1_state[:, 0:3], ee1_state[:, 3:7],
        ee2_state[:, 0:3], ee2_state[:, 3:7],
    )
    pos_drift = torch.norm(rel_pos - ref["rel_pos"], dim=-1)  # [num_envs]

    # 회전 drift = angle(R_now × R_ref^-1)
    diff_quat = _quat_mul(rel_rot, _quat_conj(ref["rel_rot"]))
    diff_v = diff_quat[:, 1:4]
    diff_w = diff_quat[:, 0].abs()
    rot_drift = 2.0 * torch.atan2(torch.norm(diff_v, dim=-1), diff_w)

    # ── (B) Joint anchor 위치 정합 오차 ──
    # joint_1: panda_hand_1.local(0,0,TCP) ↔ rod.local(-0.4,0,0)
    # joint_2: panda_hand_2.local(0,0,TCP) ↔ rod.local(+0.4,0,0)
    TCP_OFFSET = 0.1034
    HALF_W = 0.4

    rod_state = env.rod.data.root_state_w[:, :7]
    rod_p, rod_q = rod_state[:, 0:3], rod_state[:, 3:7]

    # body0_anchor world (panda_hand 기준)
    p_local_hand = torch.tensor([0.0, 0.0, TCP_OFFSET], device=env.device).expand(env.num_envs, 3)
    anchor0_1 = ee1_state[:, 0:3] + _quat_apply(ee1_state[:, 3:7], p_local_hand)
    anchor0_2 = ee2_state[:, 0:3] + _quat_apply(ee2_state[:, 3:7], p_local_hand)

    # body1_anchor world (rod 기준)
    p_local_rod_left = torch.tensor([-HALF_W, 0.0, 0.0], device=env.device).expand(env.num_envs, 3)
    p_local_rod_right = torch.tensor([+HALF_W, 0.0, 0.0], device=env.device).expand(env.num_envs, 3)
    anchor1_1 = rod_p + _quat_apply(rod_q, p_local_rod_left)
    anchor1_2 = rod_p + _quat_apply(rod_q, p_local_rod_right)

    j1_violation = torch.norm(anchor0_1 - anchor1_1, dim=-1)  # [num_envs]
    j2_violation = torch.norm(anchor0_2 - anchor1_2, dim=-1)

    return {
        "pos_drift_max_mm": pos_drift.max().item() * 1000,
        "pos_drift_mean_mm": pos_drift.mean().item() * 1000,
        "rot_drift_max_deg": rot_drift.max().item() * 180.0 / math.pi,
        "rot_drift_mean_deg": rot_drift.mean().item() * 180.0 / math.pi,
        "j1_viol_max_mm": j1_violation.max().item() * 1000,
        "j2_viol_max_mm": j2_violation.max().item() * 1000,
    }


def _freeze_arms(env, q1, q2, rod_pose):
    """매 스텝 IK 자세 + rod pose를 강제 write — 중력 드리프트 무력화 + fixed joint 잔류 오차 0
    (시각화 전용)"""
    zero_v1 = torch.zeros_like(q1)
    zero_v2 = torch.zeros_like(q2)
    env.robot_1.write_joint_state_to_sim(q1, zero_v1, env.robot_1_joint_ids)
    env.robot_2.write_joint_state_to_sim(q2, zero_v2, env.robot_2_joint_ids)
    # Phase 2: rod state도 함께 write (joint constraint 잔류 motion 제거)
    zero_vel_rod = torch.zeros((env.num_envs, 6), device=env.device)
    env.rod.write_root_pose_to_sim(rod_pose)
    env.rod.write_root_velocity_to_sim(zero_vel_rod)


def main():
    env_cfg = DualrobotCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # 환경 시각적 분리를 위해 spacing 유지
    env_cfg.scene.env_spacing = 4.0
    # Visualization 전용: 자동 truncation 무력화 (한 샘플을 충분히 관찰)
    env_cfg.episode_length_s = args_cli.episode_length_s

    env = DualrobotEnv(cfg=env_cfg, render_mode="human")

    _print_header()
    if args_cli.freeze_arms:
        print(">> freeze_arms=ON : arms are locked to IK pose every step (no gravity drift)")
    else:
        print(">> freeze_arms=OFF: arms will drift under gravity (Phase 1 limitation visualized)")
    if args_cli.reset_period > 0:
        print(f">> reset_period   : every {args_cli.reset_period} steps (cycling through samples)")
    else:
        print(">> reset_period   : DISABLED (locked on single sample; rerun script for new sample)")
    print(f">> episode_length : {args_cli.episode_length_s}s (auto-truncation horizon)")
    print("=" * 70)

    obs, _ = env.reset()
    q_lock_1, q_lock_2, rod_lock = _capture_locked_state(env)
    drift_ref = _capture_initial_relative_pose(env)
    step = 0

    print(f"[step 0] Initial reset done. Visualizing {args_cli.num_envs} sampled scenes.")
    if args_cli.drift_period > 0:
        d0 = _measure_drift(env, drift_ref)
        print(f"[step 0] Drift baseline | "
              f"EE-EE pos {d0['pos_drift_mean_mm']:.3f}/{d0['pos_drift_max_mm']:.3f} mm "
              f"rot {d0['rot_drift_mean_deg']:.4f}/{d0['rot_drift_max_deg']:.4f} deg | "
              f"joint viol max j1={d0['j1_viol_max_mm']:.3f} mm, j2={d0['j2_viol_max_mm']:.3f} mm")

    while simulation_app.is_running():
        if args_cli.hold_action == "zero":
            actions = torch.zeros(env.num_envs, env.cfg.action_space, device=env.device)
        else:
            actions = 0.1 * (2 * torch.rand(env.num_envs, env.cfg.action_space, device=env.device) - 1)

        obs, rew, terminated, truncated, info = env.step(actions)
        step += 1

        if args_cli.freeze_arms:
            _freeze_arms(env, q_lock_1, q_lock_2, rod_lock)

        # Drift 진단 출력
        if args_cli.drift_period > 0 and step % args_cli.drift_period == 0:
            d = _measure_drift(env, drift_ref)
            print(f"[step {step:5d}] "
                  f"EE-EE drift  pos {d['pos_drift_mean_mm']:6.3f}/{d['pos_drift_max_mm']:6.3f} mm  "
                  f"rot {d['rot_drift_mean_deg']:6.4f}/{d['rot_drift_max_deg']:6.4f} deg | "
                  f"joint viol  j1={d['j1_viol_max_mm']:5.3f}  j2={d['j2_viol_max_mm']:5.3f} mm")

        # 자동 reset (truncated/terminated)된 환경이 있으면 알림 + lock state 갱신
        dones = terminated | truncated
        if dones.any():
            done_idx = dones.nonzero(as_tuple=False).flatten().tolist()
            print(f"[step {step}] Auto-reset triggered for envs {done_idx}")
            q_lock_1, q_lock_2, rod_lock = _capture_locked_state(env)
            drift_ref = _capture_initial_relative_pose(env)

        # 주기적 강제 reset은 옵션 — 0이면 비활성화
        if args_cli.reset_period > 0 and step % args_cli.reset_period == 0:
            print(f"[step {step}] Manual reset: sampling {args_cli.num_envs} new scenes")
            obs, _ = env.reset()
            q_lock_1, q_lock_2, rod_lock = _capture_locked_state(env)
            drift_ref = _capture_initial_relative_pose(env)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
