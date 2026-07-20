"""
Trained Policy 시각화 (Phase 3 디버깅용)

학습된 SAC 정책을 로드해서 한두 env에서 실제 움직임 관찰.

사용법:
    # 최신 학습 결과 시각화
    python -u visualize_trained_policy.py --model_path logs/phase3_sac_YYYYMMDD-XXXXXX/model_final.pt

    # 또는 특정 step 모델
    python -u visualize_trained_policy.py --model_path logs/.../model_step_002000000.pt --num_envs 4

Notes:
    - render_mode="human" — Isaac Sim GUI 띄움 (--headless 안 줌)
    - Deterministic 모드: action_mean 사용 (sampling 안 함)
    - 매 step pos_err / rot_err 콘솔 출력
    - Episode 종료 (success/timeout)되면 새로 reset
"""

import argparse
import time
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize trained Phase 3 policy")
parser.add_argument("--model_path", type=str, default=None,
                    help="Checkpoint .pt path. 없으면 scratch (untrained, random init) 정책 사용.")
parser.add_argument("--random_action", action="store_true",
                    help="정책 무시하고 매 step uniform random action 발행 (warmup과 동일).")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=10,
                    help="시각화할 에피소드 수")
parser.add_argument("--deterministic", action="store_true", default=True,
                    help="action mean 사용 (sampling X). Default True")
parser.add_argument("--stochastic", action="store_false", dest="deterministic",
                    help="Stochastic sampling 사용 (학습 때와 동일)")
parser.add_argument("--log_every", type=int, default=30,
                    help="N step마다 진행 출력")
parser.add_argument("--action_scale_pos", type=float, default=0.02,
                    help="★ train_phase3_sac.py default와 동기 필수. mismatch면 행동 축소 재생됨.")
parser.add_argument("--action_scale_rot", type=float, default=0.05)
parser.add_argument("--num_rounds", type=int, default=2)
parser.add_argument("--episode_length_s", type=float, default=12.5)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from dual_arm_transport_env3 import DualrobotEnv, DualrobotCfg
import gnn_policy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎬 Visualizing trained policy on {device}")
    print(f"   model: {args.model_path}")
    print(f"   num_envs: {args.num_envs}")
    print(f"   mode: {'deterministic (mean)' if args.deterministic else 'stochastic (sample)'}")

    # ── Env ──
    env_cfg = DualrobotCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.scene.env_spacing = 4.0
    env_cfg.episode_length_s = args.episode_length_s
    # Visualization smoothness: physics 240Hz × render_interval=8 → 30 fps rendering.
    # 학습 default (decimation=48 → render_interval=48 → 5 fps)는 끊겨 보여서 override.
    # Physics/RL step은 영향 없음.
    env_cfg.sim.render_interval = 8
    env = DualrobotEnv(cfg=env_cfg, render_mode="human")

    # ── Camera setup (rod 가까이 zoom in) ──
    try:
        from isaacsim.core.utils.viewports import set_camera_view
        set_camera_view(eye=[1.5, 1.5, 1.0], target=[0.0, 0.0, 0.5])
        print(f"📷 Camera: eye=(1.5, 1.5, 1.0), target=(0, 0, 0.5)")
    except Exception as e:
        print(f"⚠️ Camera setup failed: {e}")

    # ── Agent ──
    action_scale_vec = [args.action_scale_pos] * 3 + [args.action_scale_rot] * 3
    agent = gnn_policy.GNNSACAgent(
        action_dim=env.cfg.action_space,
        num_rounds=args.num_rounds,
        action_scale=action_scale_vec,
    ).to(device)
    agent.eval()

    if args.model_path is not None:
        ckpt = torch.load(args.model_path, map_location=device)
        agent.load_state_dict(ckpt["model"])
        env_steps_trained = ckpt.get("env_steps", "unknown")
        print(f"✅ Loaded model at env_steps = {env_steps_trained:,}" if isinstance(env_steps_trained, int)
              else f"✅ Loaded model (env_steps unknown)")
    else:
        print(f"🧪 SCRATCH mode: untrained agent (random init). "
              f"{'random_action override ON' if args.random_action else 'using agent.actor (μ≈0)'}")

    # ── Run ──
    obs, _ = env.reset()
    current_batch = env._build_policy_batch()

    print("=" * 80)
    print(f"Watching {args.num_episodes} episodes")
    print("=" * 80)

    ep_count = 0
    step_in_ep = 0
    ep_total_reward = torch.zeros(args.num_envs, device=device)
    ep_min_pos_err = torch.full((args.num_envs,), float("inf"), device=device)
    ep_min_rot_err = torch.full((args.num_envs,), float("inf"), device=device)

    scale_t = torch.tensor(action_scale_vec, device=device)
    while ep_count < args.num_episodes and simulation_app.is_running():
        # Action
        if args.random_action:
            action = scale_t * (2 * torch.rand(args.num_envs, env.cfg.action_space, device=device) - 1)
        else:
            with torch.no_grad():
                action, _, _ = agent.actor.get_action_and_log_prob(
                    current_batch, deterministic=args.deterministic
                )

        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated
        ep_total_reward += reward
        step_in_ep += 1

        # 진행 출력
        rod_pos = env.rod.data.root_pos_w
        goal_pos = env.goal_rod_marker.data.root_pos_w
        rod_quat = env.rod.data.root_quat_w
        goal_quat = env.goal_rod_marker.data.root_quat_w

        import isaaclab.utils.math as math_utils
        import math
        pos_err = torch.norm(goal_pos - rod_pos, dim=-1)        # (B,) m
        rod_inv = math_utils.quat_conjugate(rod_quat)
        q_diff = math_utils.quat_mul(goal_quat, rod_inv)
        rot_err = 2.0 * torch.atan2(
            torch.norm(q_diff[:, 1:4], dim=-1), torch.abs(q_diff[:, 0])
        )

        ep_min_pos_err = torch.min(ep_min_pos_err, pos_err)
        ep_min_rot_err = torch.min(ep_min_rot_err, rot_err)

        if step_in_ep % args.log_every == 0:
            print(f"  step {step_in_ep:>4d} | pos_err {pos_err[0].item()*1000:>7.1f} mm | "
                  f"rot_err {rot_err[0].item()*180/math.pi:>6.1f}° | "
                  f"action |a|={action[0].abs().max().item():.4f}",
                  flush=True)

        # Episode reset
        if done.any():
            for i in range(args.num_envs):
                if done[i]:
                    reason = "✅ SUCCESS" if terminated[i].item() else "⏱ TIMEOUT"
                    print(f"  [Env {i}] {reason} at step {step_in_ep}  "
                          f"min_pos_err {ep_min_pos_err[i].item()*1000:.1f}mm  "
                          f"min_rot_err {ep_min_rot_err[i].item()*180/math.pi:.1f}°  "
                          f"ep_reward {ep_total_reward[i].item():.1f}", flush=True)
                    ep_total_reward[i] = 0.0
                    ep_min_pos_err[i] = float("inf")
                    ep_min_rot_err[i] = float("inf")
            ep_count += done.sum().item()
            step_in_ep = 0
            print("-" * 60)

        current_batch = env._build_policy_batch()

    print("=" * 80)
    print(f"✅ Visualization done. {ep_count} episodes shown.")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
