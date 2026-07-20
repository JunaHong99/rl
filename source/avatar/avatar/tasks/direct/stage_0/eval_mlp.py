"""
Deterministic eval of the trained MLP policy.

Loads checkpoint and runs the same env config used during training
(본 task: 2 Franka on z=0, rod via fixed joint to both EEs, dynamic rod,
 per-arm impedance controller — RL outputs 6-D object pose delta).

Reports: success_rate, mean episode reward / length, action norm,
         min_pos_err_mm, min_rot_err_deg.
"""

import argparse
import math
import os
import time
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="MLP deterministic eval")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--num_steps", type=int, default=300,
                    help="env steps to run. 30-step episodes → ~10 episodes per env.")
parser.add_argument("--model_path", type=str,
                    default="logs/phase3_sac_20260530-010944/model_final.pt")
parser.add_argument("--action_scale_pos", type=float, default=0.02)
parser.add_argument("--action_scale_rot", type=float, default=0.025)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from dual_arm_transport_env3 import DualrobotEnv, DualrobotCfg
import mlp_policy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 MLP eval on {device}")

    env_cfg = DualrobotCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = DualrobotEnv(cfg=env_cfg, render_mode=None)

    action_scale_vec = [args.action_scale_pos] * 3 + [args.action_scale_rot] * 3
    agent = mlp_policy.MLPSACAgent(
        action_dim=env.cfg.action_space,
        num_rounds=2,
        action_scale=action_scale_vec,
        hidden_dim=256,
        num_hidden_layers=2,
    ).to(device)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = args.model_path if os.path.isabs(args.model_path) else os.path.join(script_dir, args.model_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    agent.load_state_dict(ckpt["model"])
    agent.eval()
    print(f"📂 Loaded {ckpt_path}  (env_steps={ckpt.get('env_steps', '?')})")

    obs, _ = env.reset()
    current_batch = env._build_policy_batch()

    running_reward = torch.zeros(args.num_envs, device=device)
    running_len = torch.zeros(args.num_envs, device=device, dtype=torch.long)
    # ★ Per-env episode min err — success 판정용 (env3는 auto-reset 시 episode_min_pos_err 초기화).
    running_min_pos = torch.full((args.num_envs,), float("inf"), device=device)
    running_min_rot = torch.full((args.num_envs,), float("inf"), device=device)
    running_max_pos = torch.zeros((args.num_envs,), device=device)  # 에피소드 최대 거리 ≈ 운반 거리(버킷 키)
    ep_rewards, ep_lengths, ep_successes = [], [], []
    ep_min_pos_log, ep_min_rot_log, ep_dist_log = [], [], []
    action_norm_sum, action_norm_n = 0.0, 0

    import isaaclab.utils.math as math_utils
    POS_THRESH_M = 0.02
    ROT_THRESH_RAD = math.radians(10)

    t0 = time.time()
    print(f"⚙️  num_envs={args.num_envs}  num_steps={args.num_steps}  expected envs episodes ~{args.num_steps // 15:.0f} per env")
    print("=" * 80)

    for step in range(args.num_steps):
        with torch.no_grad():
            action, _, _ = agent.actor.get_action_and_log_prob(current_batch, deterministic=True)
        action_norm_sum += action.norm(dim=-1).mean().item()
        action_norm_n += 1

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated

        running_reward = running_reward + reward
        running_len = running_len + 1

        # ★ Per-env pos/rot err 매 step 업데이트 (env3 auto-reset 전에 capture)
        rod_pos = env.rod.data.root_pos_w
        goal_pos = env.goal_rod_marker.data.root_pos_w
        rod_quat = env.rod.data.root_quat_w
        goal_quat = env.goal_rod_marker.data.root_quat_w
        pos_err = torch.norm(goal_pos - rod_pos, dim=-1)
        rod_inv = math_utils.quat_conjugate(rod_quat)
        q_diff = math_utils.quat_mul(goal_quat, rod_inv)
        rot_err = 2.0 * torch.atan2(torch.norm(q_diff[:, 1:4], dim=-1), torch.abs(q_diff[:, 0]))
        running_min_pos = torch.min(running_min_pos, pos_err)
        running_min_rot = torch.min(running_min_rot, rot_err)
        running_max_pos = torch.max(running_max_pos, pos_err)

        if done.any():
            done_idx = done.nonzero(as_tuple=True)[0]
            for i in done_idx.tolist():
                ep_rewards.append(running_reward[i].item())
                ep_lengths.append(running_len[i].item())
                # ★ Success = 에피소드 중 한 번이라도 (pos<20mm AND rot<10°) 만족
                # env3 reward의 is_reached 정의와 일치 (line 481). terminated은 pos만 보고
                # 또 final step에 drift되면 False라 부정확.
                is_succ = (
                    running_min_pos[i].item() < POS_THRESH_M
                    and running_min_rot[i].item() < ROT_THRESH_RAD
                )
                ep_successes.append(is_succ)
                ep_min_pos_log.append(running_min_pos[i].item() * 1000)  # mm
                ep_min_rot_log.append(running_min_rot[i].item() * 180.0 / math.pi)  # deg
                ep_dist_log.append(running_max_pos[i].item() * 1000)  # mm ≈ 운반 거리
            running_reward[done_idx] = 0.0
            running_len[done_idx] = 0
            running_min_pos[done_idx] = float("inf")
            running_min_rot[done_idx] = float("inf")
            running_max_pos[done_idx] = 0.0

        current_batch = env._build_policy_batch()

    elapsed = time.time() - t0
    n_eps = len(ep_rewards)
    succ_rate = sum(ep_successes) / max(1, n_eps)
    rew_mean = sum(ep_rewards) / max(1, n_eps)
    len_mean = sum(ep_lengths) / max(1, n_eps)
    act_norm_mean = action_norm_sum / max(1, action_norm_n)
    mp_mean = sum(ep_min_pos_log) / max(1, len(ep_min_pos_log))
    mr_mean = sum(ep_min_rot_log) / max(1, len(ep_min_rot_log))

    print("=" * 80)
    print(f"✅ Eval complete in {elapsed:.1f}s")
    print(f"  Episodes:        {n_eps:,}")
    print(f"  Success rate:    {succ_rate*100:.1f}%   ({sum(ep_successes)}/{n_eps})")
    print(f"  Reward (mean):   {rew_mean:.2f}")
    print(f"  Length (mean):   {len_mean:.2f}  steps  (max 30)")
    print(f"  Action norm:     {act_norm_mean:.4f}  (uniform-random would be ~0.032 for current scale)")
    print(f"  Min pos err:     {mp_mean:.2f} mm  (threshold typically 20 mm)")
    print(f"  Min rot err:     {mr_mean:.2f} deg")

    # ── Phantom episode 진단 (2026-06-11) ──
    # success-termination 직후 RL 0-step done(length<=settle)이 실패로 카운트되어 metric 왜곡.
    # 진짜 정책 성공률 = phantom 제외 후 성공률.
    import numpy as np
    settle = getattr(env, "SETTLE_STEPS_AT_RESET", 0)
    L = np.array(ep_lengths); S = np.array(ep_successes, dtype=bool); MP = np.array(ep_min_pos_log)
    phantom = L <= settle
    genuine = ~phantom
    n_ph, n_gen = int(phantom.sum()), int(genuine.sum())
    gen_succ = float(S[genuine].mean()) if n_gen else 0.0
    print("-" * 80)
    print(f"  [phantom 진단] settle={settle}")
    print(f"    phantom (length<=settle, RL=0): {n_ph:,} ({n_ph/max(1,n_eps)*100:.1f}%)  "
          f"min_pos~{(MP[phantom].mean() if n_ph else 0):.0f}mm")
    print(f"    genuine (length>settle):        {n_gen:,}")
    print(f"    ▶ phantom 제외 success rate:    {gen_succ*100:.1f}%   ({int(S[genuine].sum())}/{n_gen})")
    gf = genuine & (~S)
    if int(gf.sum()):
        print(f"    genuine 실패 {int(gf.sum())}건 min_pos: mean {MP[gf].mean():.0f}mm  "
              f"median {np.median(MP[gf]):.0f}mm  min {MP[gf].min():.0f}mm  max {MP[gf].max():.0f}mm")

    # ── 거리 구간별 success 분해 (2026-06-11, maxreach 확장 진단) ──
    # 버킷 키 = 에피소드 max pos_err ≈ 운반 거리. genuine(phantom 제외) 에피소드만.
    D = np.array(ep_dist_log)
    buckets = [("가까움 100-250mm", 100, 250), ("중간 250-450mm", 250, 450), ("먼 450-650mm", 450, 650)]
    print("-" * 80)
    print("  [거리 구간별 success] (genuine만)")
    for name, lo, hi in buckets:
        m = genuine & (D >= lo) & (D < hi)
        n = int(m.sum())
        sr = float(S[m].mean()) * 100 if n else 0.0
        print(f"    {name:<18} n={n:>4}  success {sr:>5.1f}%")

    # Isaac simulation_app.close()가 hang하는 버그 → close() 호출 않고 즉시 강제 종료.
    # (-u 실행이라 print는 이미 flush됨; 안전을 위해 한 번 더 flush 후 os._exit.)
    import sys as _sys, os as _os
    _sys.stdout.flush()
    _os._exit(0)


if __name__ == "__main__":
    main()
