"""
Velocity Controller 검증 스크립트.

Stage 0~5 통과 시 RL 학습 진행.

Stages:
  0: velocity = 0          → 정지 유지 (drift < 1mm/sec)
  1: lin_vel = 0.02 m/s    → 직진 (실속도 ≥ 80% 명령)
  2: lin_vel = 0.10 m/s    → 빠른 직진 (≥ 70% 명령)
  3: ang_vel_z = 0.1 rad/s → 순수 회전 (≥ 80% 명령)
  4: ang_vel_z = 0.5 rad/s → 빠른 회전 (≥ 60%, 발산 없음)
  5: 병진 + 회전 동시       → 두 채널 모두 추종

사용:
  python test_velocity_tracking.py --stage 0 --num_envs 16 --num_steps 200
"""
import argparse
import math
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--stage", type=int, default=0, choices=[0, 1, 2, 3, 4, 5])
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--num_steps", type=int, default=200,
                    help="RL step (10Hz, decimation=24)")
parser.add_argument("--settle_steps", type=int, default=20,
                    help="reset transient 흡수용 정지 step")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.utils.math as math_utils
from dual_arm_transport_env3 import DualrobotEnv, DualrobotCfg


def percentile(t, q):
    return torch.quantile(t, q).item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = DualrobotCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.episode_length_s = max(cfg.episode_length_s, args.num_steps * cfg.sim.dt * cfg.decimation + 5.0)
    env = DualrobotEnv(cfg=cfg, render_mode=None)

    dt_rl = cfg.sim.dt * cfg.decimation
    print(f"\n== Velocity Controller Test — Stage {args.stage} ==")
    print(f"  num_envs={args.num_envs}  num_steps={args.num_steps}  RL dt={dt_rl*1000:.1f}ms ({1/dt_rl:.1f}Hz)")

    # Stage definitions
    if args.stage == 0:
        desired_lin = torch.zeros(args.num_envs, 3, device=device)
        desired_ang = torch.zeros(args.num_envs, 3, device=device)
        desc = "정지 유지"; pass_kind = "drift_max < 5mm"
    elif args.stage == 1:
        desired_lin = torch.tensor([[0.02, 0, 0]], device=device).expand(args.num_envs, -1).contiguous()
        desired_ang = torch.zeros(args.num_envs, 3, device=device)
        desc = "0.02 m/s +x"; pass_kind = "actual ≥ 80% command"
    elif args.stage == 2:
        desired_lin = torch.tensor([[0.10, 0, 0]], device=device).expand(args.num_envs, -1).contiguous()
        desired_ang = torch.zeros(args.num_envs, 3, device=device)
        desc = "0.10 m/s +x"; pass_kind = "actual ≥ 70% command"
    elif args.stage == 3:
        desired_lin = torch.zeros(args.num_envs, 3, device=device)
        desired_ang = torch.tensor([[0, 0, 0.1]], device=device).expand(args.num_envs, -1).contiguous()
        desc = "0.1 rad/s +z 회전"; pass_kind = "actual ≥ 80% command"
    elif args.stage == 4:
        desired_lin = torch.zeros(args.num_envs, 3, device=device)
        desired_ang = torch.tensor([[0, 0, 0.5]], device=device).expand(args.num_envs, -1).contiguous()
        desc = "0.5 rad/s +z 회전 (빠름)"; pass_kind = "actual ≥ 60%, no divergence"
    elif args.stage == 5:
        desired_lin = torch.tensor([[0.05, 0, 0]], device=device).expand(args.num_envs, -1).contiguous()
        desired_ang = torch.tensor([[0, 0, 0.2]], device=device).expand(args.num_envs, -1).contiguous()
        desc = "0.05m/s + 0.2rad/s 동시"; pass_kind = "두 채널 모두 추종"

    print(f"  Command: {desc}")
    print(f"  통과 기준: {pass_kind}\n")

    obs, _ = env.reset()

    # ── Settle phase: action=0으로 reset transient 흡수 ──
    a_zero = torch.zeros(args.num_envs, 6, device=device)
    for s in range(args.settle_steps):
        env.step(a_zero)
    rod_init_pos = env.rod.data.root_pos_w.clone()
    rod_init_quat = env.rod.data.root_quat_w.clone()
    init_w = env.rod.data.root_ang_vel_w.norm(dim=-1).max().item()
    print(f"  After {args.settle_steps} settle: max ω = {init_w:.4f} rad/s")

    # action = (desired_lin, desired_ang) 양식으로 전달
    action = torch.cat([desired_lin, desired_ang], dim=-1)

    rod_lin_hist = []      # (T, B, 3)
    rod_ang_hist = []      # (T, B, 3)
    rod_pos_hist = []
    rod_quat_hist = []
    for step in range(args.num_steps):
        env.step(action)
        rod_lin_hist.append(env.rod.data.root_lin_vel_w.clone().cpu())
        rod_ang_hist.append(env.rod.data.root_ang_vel_w.clone().cpu())
        rod_pos_hist.append(env.rod.data.root_pos_w.clone().cpu())
        rod_quat_hist.append(env.rod.data.root_quat_w.clone().cpu())

        if step % 20 == 0 or step == args.num_steps - 1:
            v = env.rod.data.root_lin_vel_w
            w = env.rod.data.root_ang_vel_w
            v_norm = torch.norm(v, dim=-1)
            w_norm = torch.norm(w, dim=-1)
            print(f"    step {step:>4d} ({step*dt_rl*1000:.0f} ms): "
                  f"|v_rod|={v_norm.mean().item()*1000:>6.2f} mm/s  "
                  f"|ω_rod|={w_norm.mean().item():>6.4f} rad/s  "
                  f"max|v|={v_norm.max().item()*1000:.2f}mm/s  max|ω|={w_norm.max().item():.3f}")

    # Analysis
    L = torch.stack(rod_lin_hist)  # (T, B, 3) m/s
    W = torch.stack(rod_ang_hist)
    P = torch.stack(rod_pos_hist)
    half = args.num_steps // 2
    L_ss = L[half:]   # steady state
    W_ss = W[half:]

    cmd_lin_norm = torch.norm(desired_lin, dim=-1).cpu()
    cmd_ang_norm = torch.norm(desired_ang, dim=-1).cpu()

    print(f"\n== 통계 (last {args.num_steps - half} steps, steady state) ==")
    print(f"  Commanded |lin_vel|: {cmd_lin_norm[0].item()*1000:.2f} mm/s")
    print(f"  Commanded |ang_vel|: {cmd_ang_norm[0].item():.4f} rad/s")
    print(f"  Actual   |lin_vel|: mean={torch.norm(L_ss, dim=-1).mean().item()*1000:.2f} mm/s  "
          f"max={torch.norm(L_ss, dim=-1).max().item()*1000:.2f} mm/s")
    print(f"  Actual   |ang_vel|: mean={torch.norm(W_ss, dim=-1).mean().item():.4f} rad/s  "
          f"max={torch.norm(W_ss, dim=-1).max().item():.4f} rad/s")

    if args.stage == 0:
        # Drift = ||rod_pos - rod_init_pos||
        drift = torch.norm(P[-10:].mean(dim=0) - rod_init_pos.cpu(), dim=-1) * 1000  # mm
        print(f"  Drift (last 10 step mean) per env [mm]:")
        for i in range(min(args.num_envs, 8)):
            print(f"    env {i}: {drift[i].item():.3f}")
        max_drift = drift.max().item()
        ok = max_drift < 5.0
        print(f"  Stage 0 ({'PASS' if ok else 'FAIL'}): max drift {max_drift:.3f}mm")
    elif args.stage in (1, 2):
        actual_speed = torch.norm(L_ss, dim=-1).mean().item()
        cmd_speed = cmd_lin_norm[0].item()
        ratio = actual_speed / max(cmd_speed, 1e-8)
        threshold = 0.8 if args.stage == 1 else 0.7
        ok = ratio >= threshold
        print(f"  Stage {args.stage} ({'PASS' if ok else 'FAIL'}): "
              f"actual/cmd = {ratio*100:.1f}% (>= {threshold*100:.0f}%)")
    elif args.stage in (3, 4):
        actual_w = torch.norm(W_ss, dim=-1).mean().item()
        cmd_w = cmd_ang_norm[0].item()
        ratio = actual_w / max(cmd_w, 1e-8)
        threshold = 0.8 if args.stage == 3 else 0.6
        # Check divergence (max ω should not vastly exceed cmd)
        max_w = torch.norm(W, dim=-1).max().item()
        diverge_factor = max_w / max(cmd_w, 1e-8)
        no_diverge = diverge_factor < 3.0
        ok = (ratio >= threshold) and no_diverge
        print(f"  Stage {args.stage} ({'PASS' if ok else 'FAIL'}): "
              f"actual/cmd = {ratio*100:.1f}% (>= {threshold*100:.0f}%), "
              f"max_ω/cmd = {diverge_factor:.2f}× (< 3.0×)")
    elif args.stage == 5:
        actual_v = torch.norm(L_ss, dim=-1).mean().item()
        actual_w = torch.norm(W_ss, dim=-1).mean().item()
        v_ratio = actual_v / max(cmd_lin_norm[0].item(), 1e-8)
        w_ratio = actual_w / max(cmd_ang_norm[0].item(), 1e-8)
        ok = (v_ratio >= 0.6) and (w_ratio >= 0.6)
        print(f"  Stage 5 ({'PASS' if ok else 'FAIL'}): "
              f"v: {v_ratio*100:.1f}%  ω: {w_ratio*100:.1f}%")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
