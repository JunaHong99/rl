"""
Unstable env 진단 — drift test에서 폭주하는 env의 초기 조건 추적.

가설:
  - 일부 env가 zero-command drift test에서도 100mm+ 폭주
  - 원인: Jacobian near-singularity (특정 자세) 또는 joint limit 근접

방법:
  1. reset 직후 초기 상태 캡처 (joint angles, rod pose)
  2. Jacobian SVD → condition number, smallest singular value
  3. Joint margin (각 joint의 limit까지 거리 중 최소)
  4. settle_steps + num_steps drift 실행 → final pos_err
  5. unstable mask (final_err > threshold)로 stable vs unstable 통계 비교

사용:
    python -u diagnose_unstable.py --num_envs 128 --num_steps 20 --headless
"""
import argparse
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--num_steps", type=int, default=20, help="drift 측정 steps (5Hz). 20 = 4초")
parser.add_argument("--settle_steps", type=int, default=10)
parser.add_argument("--unstable_threshold_mm", type=float, default=50.0)
parser.add_argument("--show_top", type=int, default=5)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from dual_arm_transport_env3 import DualrobotEnv, DualrobotCfg


def jacobian_metrics(J):
    """SVD-based metrics. J: (B, 6, 7) → (cond, sigma_min)."""
    S = torch.linalg.svdvals(J)  # (B, 6)
    sigma_max = S.max(dim=-1).values
    sigma_min = S.min(dim=-1).values
    cond = sigma_max / sigma_min.clamp_min(1e-8)
    return cond, sigma_min


def main():
    device = torch.device("cuda")
    cfg = DualrobotCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.episode_length_s = (args.num_steps + args.settle_steps + 5) * cfg.sim.dt * cfg.decimation + 2.0  # bypass auto-reset
    env = DualrobotEnv(cfg=cfg, render_mode=None)

    print(f"\n== Unstable env diagnosis ==")
    print(f"  num_envs={args.num_envs}  settle={args.settle_steps}  drift_steps={args.num_steps}")
    print(f"  unstable threshold: {args.unstable_threshold_mm} mm final drift\n")

    obs, _ = env.reset()

    # ── 초기 상태 캡처 (reset 직후) ──
    q_init_1 = env.robot_1.data.joint_pos[:, env.robot_1_joint_ids].clone()  # (B, 7)
    q_init_2 = env.robot_2.data.joint_pos[:, env.robot_2_joint_ids].clone()
    env_origins = env.scene.env_origins
    rod_init_local = env.rod.data.root_pos_w.clone() - env_origins  # env-local

    # ── Jacobian metrics ──
    J1, J2 = env.controller._get_jacobians()  # (B, 6, 7)
    cond_1, sigma_min_1 = jacobian_metrics(J1)
    cond_2, sigma_min_2 = jacobian_metrics(J2)

    # ── Joint limit margin (Franka panda) ──
    # Standard Franka URDF joint limits
    q_lower = torch.tensor([-2.897, -1.7628, -2.897, -3.0718, -2.897, -0.0175, -2.897], device=device)
    q_upper = torch.tensor([ 2.897,  1.7628,  2.897, -0.0698,  2.897,  3.7525,  2.897], device=device)
    margin_per_joint_1 = torch.minimum(q_init_1 - q_lower, q_upper - q_init_1)  # (B, 7)
    margin_per_joint_2 = torch.minimum(q_init_2 - q_lower, q_upper - q_init_2)
    margin_1 = margin_per_joint_1.min(dim=-1).values  # (B,) most-constrained joint
    margin_2 = margin_per_joint_2.min(dim=-1).values

    # ── Settle: target = rod pose AT RESET (FIXED, not chasing current) ──
    # ★ chase 방식은 rod가 움직이면 target도 따라가 정지 신호 안 됨. snapshot 한 번만.
    a_zero = torch.zeros(args.num_envs, 6, device=device)
    target_pos_at_reset = env.rod.data.root_pos_w.clone()
    target_quat_at_reset = env.rod.data.root_quat_w.clone()
    env.target_obj_pos.copy_(target_pos_at_reset)
    env.target_obj_quat.copy_(target_quat_at_reset)
    for _ in range(args.settle_steps):
        env.step(a_zero)

    # Hold target = post-settle rod pose
    target_pos_hold = env.rod.data.root_pos_w.clone()
    target_quat_hold = env.rod.data.root_quat_w.clone()
    env.target_obj_pos.copy_(target_pos_hold)
    env.target_obj_quat.copy_(target_quat_hold)

    omega_after_settle = env.rod.data.root_ang_vel_w.norm(dim=-1)
    vel_after_settle = env.rod.data.root_lin_vel_w.norm(dim=-1)
    print(f"  After settle: rod |ω| mean={omega_after_settle.mean():.3f} max={omega_after_settle.max():.3f} rad/s")
    print(f"               rod |v| mean={vel_after_settle.mean()*1000:.2f} max={vel_after_settle.max()*1000:.2f} mm/s")

    # ── Drift loop with per-step trajectory recording ──
    drift_pos_err_steps = []  # list of (B,) per step in mm
    drift_rod_pos_steps = []  # list of (B, 3) per step in m, env-local
    drift_omega_steps = []    # list of (B,) per step in rad/s
    drift_vel_steps = []      # list of (B,) per step in mm/s
    env_origins = env.scene.env_origins
    for _ in range(args.num_steps):
        env.step(a_zero)
        env.target_obj_pos.copy_(target_pos_hold)
        env.target_obj_quat.copy_(target_quat_hold)
        # snapshot per-step
        rod_pos_now = env.rod.data.root_pos_w
        drift_pos_err_steps.append(torch.norm(rod_pos_now - target_pos_hold, dim=-1) * 1000)
        drift_rod_pos_steps.append((rod_pos_now - env_origins).clone())
        drift_omega_steps.append(env.rod.data.root_ang_vel_w.norm(dim=-1).clone())
        drift_vel_steps.append(env.rod.data.root_lin_vel_w.norm(dim=-1).clone() * 1000)
    drift_pos_err = torch.stack(drift_pos_err_steps)  # (T, B)
    drift_omega = torch.stack(drift_omega_steps)
    drift_vel = torch.stack(drift_vel_steps)

    rod_pos_final = env.rod.data.root_pos_w
    final_err_mm = torch.norm(rod_pos_final - target_pos_hold, dim=-1) * 1000  # (B,)

    # ── Classify ──
    unstable_mask = final_err_mm > args.unstable_threshold_mm
    stable_mask = ~unstable_mask
    n_unstable = int(unstable_mask.sum())
    n_stable = int(stable_mask.sum())

    print(f"\n=== Result ===")
    print(f"  Unstable: {n_unstable}/{args.num_envs} ({100*n_unstable/args.num_envs:.1f}%)")
    print(f"  Final err — stable group:   mean={final_err_mm[stable_mask].mean():.2f} max={final_err_mm[stable_mask].max():.2f} mm")
    if n_unstable > 0:
        print(f"  Final err — unstable group: mean={final_err_mm[unstable_mask].mean():.2f} max={final_err_mm[unstable_mask].max():.2f} mm")

    # ── Group comparison ──
    print(f"\n=== Group comparison (stable vs unstable) ===")
    def summarize(name, vec):
        s = vec[stable_mask]; u = vec[unstable_mask] if n_unstable > 0 else None
        line = f"  {name:25s}  stable: mean={s.mean().item():.4f} p50={s.median().item():.4f} max={s.max().item():.4f}"
        if u is not None and u.numel() > 0:
            line += f"  | unstable: mean={u.mean().item():.4f} p50={u.median().item():.4f} max={u.max().item():.4f}"
        print(line)

    summarize("Jac arm1 cond",         cond_1)
    summarize("Jac arm2 cond",         cond_2)
    summarize("Jac arm1 sigma_min",    sigma_min_1)
    summarize("Jac arm2 sigma_min",    sigma_min_2)
    summarize("Joint margin arm1 [rad]", margin_1)
    summarize("Joint margin arm2 [rad]", margin_2)
    summarize("Post-settle rod |ω|",    omega_after_settle)
    summarize("Post-settle rod |v| [mm/s]", vel_after_settle * 1000)

    # ── Top unstable envs detail with trajectory ──
    if n_unstable > 0:
        top_idx = torch.argsort(final_err_mm, descending=True)[:min(args.show_top, n_unstable)]
        print(f"\n=== Top {len(top_idx)} most unstable envs ===")
        for i in top_idx.tolist():
            print(f"\n  env {i}:  final_err={final_err_mm[i].item():.1f} mm")
            print(f"    arm1: cond={cond_1[i].item():.2f}  sigma_min={sigma_min_1[i].item():.4f}  joint_margin={margin_1[i].item():.3f} rad")
            print(f"    arm2: cond={cond_2[i].item():.2f}  sigma_min={sigma_min_2[i].item():.4f}  joint_margin={margin_2[i].item():.3f} rad")
            print(f"    rod_init (env-local): {rod_init_local[i].cpu().numpy().round(4)}")
            print(f"    post-settle: |ω|={omega_after_settle[i].item():.3f} rad/s  |v|={vel_after_settle[i].item()*1000:.2f} mm/s")
            # ★ Per-step drift trajectory
            print(f"    drift trajectory (pos_err mm / |ω| rad/s / |v| mm/s):")
            for s in range(args.num_steps):
                pe = drift_pos_err[s, i].item()
                w = drift_omega[s, i].item()
                v = drift_vel[s, i].item()
                marker = ""
                if s == 0: marker = "(start)"
                if s == args.num_steps - 1: marker = "(end)"
                print(f"      t={s*0.2:.1f}s (step{s:>2d}): pos_err={pe:>7.1f}mm  |ω|={w:>6.2f}  |v|={v:>7.1f}mm/s {marker}")
            # Which joint is closest to limit
            tight_j1 = int(margin_per_joint_1[i].argmin().item())
            tight_j2 = int(margin_per_joint_2[i].argmin().item())
            print(f"    tight joint arm1: panda_joint{tight_j1+1} (margin {margin_per_joint_1[i, tight_j1].item():.3f} rad)")
            print(f"    tight joint arm2: panda_joint{tight_j2+1} (margin {margin_per_joint_2[i, tight_j2].item():.3f} rad)")

    simulation_app.close()


if __name__ == "__main__":
    main()
