"""
저장된 실패 episode들을 GUI 시각화로 재현.

eval_collect_failures.py가 저장한 /tmp/failed_episodes.pt를 로드하여
해당 cache idx를 external_samples로 env에 주입 → 같은 episode 그대로 재생.

사용:
  python -u visualize_failures.py \
      --model_path logs/phase3_sac_20260609-154653/model_final.pt \
      --failures /tmp/failed_episodes.pt --num_envs 1
  python -u visualize_failures.py --model_path ... --failures /tmp/failed_episodes.pt --num_envs 4
"""
import argparse
import math
import os
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--failures", type=str, default="/tmp/failed_episodes.pt")
parser.add_argument("--num_envs", type=int, default=1,
                    help="동시 시각화할 env 수. 적을수록 천천히 한 케이스씩 관찰.")
parser.add_argument("--max_failures", type=int, default=-1,
                    help="-1이면 전부, 양수면 그 수만큼만 (예: 처음 10개)")
parser.add_argument("--action_scale_pos", type=float, default=0.02)
parser.add_argument("--action_scale_rot", type=float, default=0.05,
                    help="★ 학습 시점과 일치 필수.")
parser.add_argument("--episode_length_s", type=float, default=12.5)
parser.add_argument("--camera_eye", type=float, nargs=3, default=[1.5, 1.5, 1.0])
parser.add_argument("--camera_target", type=float, nargs=3, default=[0.0, 0.0, 0.5])
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from dual_arm_transport_env3 import DualrobotEnv, DualrobotCfg
import mlp_policy
import isaaclab.utils.math as math_utils


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎬 Failure replay on {device}")

    # ── 실패 episode 데이터 로드 ──
    data = torch.load(args.failures, weights_only=False)
    failed_idxs = data["cache_idxs"]
    failures = data["failures"]
    n_failures = int(failed_idxs.numel())
    if args.max_failures > 0:
        failed_idxs = failed_idxs[:args.max_failures]
        failures = failures[:args.max_failures]
        n_failures = int(failed_idxs.numel())
    print(f"📂 Loaded {n_failures} failed episodes from {args.failures}")

    # ── Env ──
    env_cfg = DualrobotCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.scene.env_spacing = 4.0
    env_cfg.episode_length_s = args.episode_length_s
    env_cfg.sim.render_interval = 8  # 30 fps
    env = DualrobotEnv(cfg=env_cfg, render_mode="human")

    try:
        from isaacsim.core.utils.viewports import set_camera_view
        set_camera_view(eye=args.camera_eye, target=args.camera_target)
    except Exception as e:
        print(f"⚠️ Camera setup failed: {e}")

    # Cache 직접 사용 (external_samples용)
    cache = env.pose_sampler.cache
    print(f"📦 Cache: {next(iter(cache.values())).shape[0]:,} samples")

    # ── Agent ──
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
    print(f"✅ Model: {ckpt_path}  (env_steps={ckpt.get('env_steps', '?')})")

    POS_THRESH_M = 0.02
    ROT_THRESH_RAD = math.radians(10)
    B = args.num_envs
    n_batches = (n_failures + B - 1) // B
    print(f"\nReplay {n_failures} failures in {n_batches} batches (batch_size={B})")
    print("=" * 80)

    succ_count = 0
    for batch_i in range(n_batches):
        s = batch_i * B
        e = min(s + B, n_failures)
        batch_idxs = failed_idxs[s:e]
        # B 못 채우면 마지막 idx 반복 (env 수 맞추기)
        while batch_idxs.numel() < B:
            batch_idxs = torch.cat([batch_idxs, batch_idxs[-1:]])

        # External samples
        external = {k: v[batch_idxs].clone().to(device) for k, v in cache.items()}
        env.external_samples = external
        env.reset()
        current_batch = env._build_policy_batch()

        ep_min_pos = torch.full((B,), float("inf"), device=device)
        ep_min_rot = torch.full((B,), float("inf"), device=device)
        step_in_ep = 0
        done_seen = torch.zeros(B, dtype=torch.bool, device=device)

        print(f"\n--- Batch {batch_i+1}/{n_batches}: failures {s+1}-{e} of {n_failures} ---")
        for i in range(min(B, e - s)):
            r = failures[s + i]
            print(f"  env {i}: ep#{s+i+1}  cache_idx={r['cache_idx']}  "
                  f"saved min_pos={r['min_pos_mm']:.1f}mm  min_rot={r['min_rot_deg']:.2f}°")

        while not done_seen.all() and simulation_app.is_running():
            with torch.no_grad():
                action, _, _ = agent.actor.get_action_and_log_prob(current_batch, deterministic=True)
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            step_in_ep += 1

            rod_pos = env.rod.data.root_pos_w
            goal_pos = env.goal_rod_marker.data.root_pos_w
            rod_quat = env.rod.data.root_quat_w
            goal_quat = env.goal_rod_marker.data.root_quat_w
            pos_err = torch.norm(goal_pos - rod_pos, dim=-1)
            rod_inv = math_utils.quat_conjugate(rod_quat)
            q_diff = math_utils.quat_mul(goal_quat, rod_inv)
            rot_err = 2.0 * torch.atan2(torch.norm(q_diff[:, 1:4], dim=-1), torch.abs(q_diff[:, 0]))
            ep_min_pos = torch.where(done_seen, ep_min_pos, torch.min(ep_min_pos, pos_err))
            ep_min_rot = torch.where(done_seen, ep_min_rot, torch.min(ep_min_rot, rot_err))

            new_done = done & ~done_seen
            if new_done.any():
                for i in new_done.nonzero(as_tuple=True)[0].tolist():
                    if i >= (e - s):
                        continue
                    is_succ = (
                        ep_min_pos[i].item() < POS_THRESH_M
                        and ep_min_rot[i].item() < ROT_THRESH_RAD
                    )
                    if is_succ:
                        succ_count += 1
                    tag = "✅ NOW SUCCESS" if is_succ else "❌ STILL FAIL"
                    print(f"    [Env {i}] {tag}  step={step_in_ep}  "
                          f"min_pos={ep_min_pos[i].item()*1000:.1f}mm  "
                          f"min_rot={ep_min_rot[i].item()*180/math.pi:.2f}°")
                done_seen = done_seen | done

            current_batch = env._build_policy_batch()

        env.external_samples = None

    print()
    print("=" * 80)
    print(f"✅ Replay done. {n_failures} failures replayed.")
    if n_failures > 0:
        print(f"  재실행 시 success로 잡힌 비율: {succ_count}/{n_failures} "
              f"({100*succ_count/n_failures:.1f}%) — variance 또는 비결정성 지표")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
