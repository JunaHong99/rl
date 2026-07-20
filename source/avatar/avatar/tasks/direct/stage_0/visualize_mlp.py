"""
MLP 정책 시각화 (Phase 3 best model 관찰용)

eval_mlp.py + visualize_trained_policy.py 합친 MLP 전용 시각화.
Isaac Sim GUI 띄우고 1-N envs에서 정책 실행, pos/rot err 콘솔 출력.

사용법:
    # 본 task 18M 모델 (84% success)
    python -u visualize_mlp.py \
        --model_path logs/phase3_sac_20260531-122705/model_step_018432000.pt \
        --num_envs 1 --num_episodes 10

    # 4개 env 동시에 비교 보기
    python -u visualize_mlp.py \
        --model_path logs/phase3_sac_20260531-122705/model_step_018432000.pt \
        --num_envs 4 --num_episodes 20

Notes:
    - --headless 안 주면 자동 GUI mode (render_mode="human")
    - render_interval=8 → 30fps (학습 default 5fps는 끊겨 보임)
    - Deterministic (action mean) — sampling 안 함
    - action_scale_pos/rot 은 train과 동기 필수 (0.02 / 0.025 default)
"""

import argparse
import math
import os
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize trained MLP policy")
parser.add_argument("--model_path", type=str,
                    default="logs/phase3_sac_20260531-122705/model_step_018432000.pt",
                    help="Checkpoint .pt path.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--deterministic", action="store_true", default=True,
                    help="action mean 사용 (default).")
parser.add_argument("--stochastic", action="store_false", dest="deterministic",
                    help="Stochastic sampling (학습 때와 동일).")
parser.add_argument("--log_every", type=int, default=15,
                    help="N step마다 진행 출력.")
parser.add_argument("--action_scale_pos", type=float, default=0.02,
                    help="★ train과 동기 필수.")
parser.add_argument("--action_scale_rot", type=float, default=0.025,
                    help="★ train과 동기 필수.")
parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--num_hidden_layers", type=int, default=2)
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
    print(f"🎬 Visualizing MLP policy on {device}")
    print(f"   model: {args.model_path}")
    print(f"   num_envs: {args.num_envs}, episodes: {args.num_episodes}")
    print(f"   mode: {'deterministic (mean)' if args.deterministic else 'stochastic (sample)'}")

    # ── Env ──
    env_cfg = DualrobotCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.scene.env_spacing = 4.0
    env_cfg.episode_length_s = args.episode_length_s
    # 30 fps rendering (physics 240Hz / render_interval=8)
    env_cfg.sim.render_interval = 8
    env = DualrobotEnv(cfg=env_cfg, render_mode="human")

    # ── Camera ──
    try:
        from isaacsim.core.utils.viewports import set_camera_view
        set_camera_view(eye=args.camera_eye, target=args.camera_target)
        print(f"📷 Camera: eye={tuple(args.camera_eye)}, target={tuple(args.camera_target)}")
    except Exception as e:
        print(f"⚠️ Camera setup failed: {e}")

    # ── Agent ──
    action_scale_vec = [args.action_scale_pos] * 3 + [args.action_scale_rot] * 3
    agent = mlp_policy.MLPSACAgent(
        action_dim=env.cfg.action_space,
        num_rounds=2,
        action_scale=action_scale_vec,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
    ).to(device)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = args.model_path if os.path.isabs(args.model_path) \
        else os.path.join(script_dir, args.model_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    agent.load_state_dict(ckpt["model"])
    agent.eval()
    env_steps_trained = ckpt.get("env_steps", "?")
    print(f"✅ Loaded {ckpt_path}")
    print(f"   env_steps trained: {env_steps_trained:,}" if isinstance(env_steps_trained, int)
          else f"   env_steps: {env_steps_trained}")

    # ── Run ──
    obs, _ = env.reset()
    current_batch = env._build_policy_batch()

    print("=" * 80)
    print(f"Watching {args.num_episodes} episodes (Ctrl+C to stop)")
    print("=" * 80)

    ep_count = 0
    step_in_ep = 0
    ep_total_reward = torch.zeros(args.num_envs, device=device)
    ep_min_pos_err = torch.full((args.num_envs,), float("inf"), device=device)
    ep_min_rot_err = torch.full((args.num_envs,), float("inf"), device=device)
    succ_count = 0
    prev_settle = False  # settle/RL phase 전환 감지 (Env 0 기준)
    print(f"ℹ️  각 episode 시작 시 {env.SETTLE_STEPS_AT_RESET} step 동안 'SETTLE' 구간이 있음 "
          f"— reset 직후 PhysX 안정화용. 이 구간엔 모델이 출력은 해도 env가 action=0으로 "
          f"강제하므로 rod가 '의도적으로' 멈춰 있음 (모델 미작동/충돌 아님).")

    while ep_count < args.num_episodes and simulation_app.is_running():
        with torch.no_grad():
            action, _, _ = agent.actor.get_action_and_log_prob(
                current_batch, deterministic=args.deterministic
            )

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated
        ep_total_reward += reward
        step_in_ep += 1

        # ── Settle / RL phase 표시 (Env 0 기준) ──
        # env3가 settle 중엔 _pre_physics_step에서 action을 0으로 강제 (PhysX 안정화).
        # 모델 raw 출력(action)은 존재하지만 적용 안 됨 → "rod 안 움직임 = 모델 미작동/충돌" 오해 방지.
        settle_now = bool(env._is_settle_step[0]) if hasattr(env, "_is_settle_step") else False
        settle_left = int(env._settle_remaining[0]) if hasattr(env, "_settle_remaining") else 0
        raw_a0 = action[0].abs().max().item()
        if settle_now and not prev_settle:
            print(f"  ⏸️  SETTLE 시작 ({env.SETTLE_STEPS_AT_RESET} step): reset 직후 안정화 구간. "
                  f"모델은 출력하지만 env가 action=0으로 강제 → rod 의도적 정지 (모델 미작동/충돌 아님).",
                  flush=True)
        elif (not settle_now) and prev_settle:
            print(f"  ▶️  RL 제어 시작: 이제부터 모델 출력이 controller 통해 rod를 움직임.", flush=True)
        prev_settle = settle_now

        # err metrics
        rod_pos = env.rod.data.root_pos_w
        goal_pos = env.goal_rod_marker.data.root_pos_w
        rod_quat = env.rod.data.root_quat_w
        goal_quat = env.goal_rod_marker.data.root_quat_w

        pos_err = torch.norm(goal_pos - rod_pos, dim=-1)             # (B,) m
        rod_inv = math_utils.quat_conjugate(rod_quat)
        q_diff = math_utils.quat_mul(goal_quat, rod_inv)
        rot_err = 2.0 * torch.atan2(
            torch.norm(q_diff[:, 1:4], dim=-1), torch.abs(q_diff[:, 0])
        )

        ep_min_pos_err = torch.min(ep_min_pos_err, pos_err)
        ep_min_rot_err = torch.min(ep_min_rot_err, rot_err)

        if step_in_ep % args.log_every == 0:
            phase = f"⏸️SETTLE[{settle_left:>2d} left]" if settle_now else "▶️RL          "
            note = "  ← raw |a| 무시됨(action=0 강제)" if settle_now else ""
            print(f"  {phase} step {step_in_ep:>4d} | "
                  f"pos {pos_err[0].item()*1000:>7.1f}mm | "
                  f"rot {rot_err[0].item()*180/math.pi:>6.1f}° | "
                  f"|a|={action[0].abs().max().item():.4f}{note}",
                  flush=True)

        if done.any():
            for i in range(args.num_envs):
                if done[i]:
                    # ★ FIX (2026-06-09 v2): terminated 도 안 맞음. env3 dones는 pos<20mm만 보고 rot 무시,
                    # 또 episode 끝 step pos>20mm로 drift되면 terminated=False가 되지만 ep_rew=100인 케이스
                    # ("도달했다가 떠남")이 다수. 진짜 success 의도 = 에피소드 중 한 번이라도
                    # (min_pos<20mm AND min_rot<10°) 만족 → ep_min_*로 판정.
                    POS_THRESH_M = 0.02
                    ROT_THRESH_RAD = math.radians(10)
                    is_success = (
                        ep_min_pos_err[i].item() < POS_THRESH_M
                        and ep_min_rot_err[i].item() < ROT_THRESH_RAD
                    )
                    if is_success:
                        succ_count += 1
                    tag = "✅ SUCCESS" if is_success else "⏱ TIMEOUT"
                    # env3 settle 30 step + RL step 분리해서 표시 (settle 동안엔 action 0 강제)
                    settle = env.SETTLE_STEPS_AT_RESET if hasattr(env, "SETTLE_STEPS_AT_RESET") else 0
                    rl_step = max(0, step_in_ep - settle)
                    print(f"  [Env {i}] {tag} total_step={step_in_ep} (settle={settle}+RL={rl_step})  "
                          f"min_pos={ep_min_pos_err[i].item()*1000:.1f}mm  "
                          f"min_rot={ep_min_rot_err[i].item()*180/math.pi:.1f}°  "
                          f"ep_rew={ep_total_reward[i].item():.1f}", flush=True)
                    ep_total_reward[i] = 0.0
                    ep_min_pos_err[i] = float("inf")
                    ep_min_rot_err[i] = float("inf")
            ep_count += done.sum().item()
            step_in_ep = 0
            print("-" * 60)

        current_batch = env._build_policy_batch()

    print("=" * 80)
    print(f"✅ Done. {ep_count} episodes shown, "
          f"success {succ_count}/{ep_count} ({100*succ_count/max(1,ep_count):.1f}%)")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
