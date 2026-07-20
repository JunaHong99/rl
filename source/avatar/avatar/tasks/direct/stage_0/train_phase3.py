"""
Phase 3.3 Training — PPO + GNN policy + Cooperative Impedance Controller

Curriculum stage 1:
    - action = 6-dim (object pose delta only)
    - K_abs, K_rel are controller defaults (fixed)
    - Scenario 1: 기본 운반 (장애물 X)

사용:
    python train_phase3.py --num_envs 64 --total_steps 1_000_000 --headless
"""

import argparse
import os
import time
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from isaaclab.app import AppLauncher

# ── argparse before AppLauncher ──
parser = argparse.ArgumentParser(description="Phase 3.3 PPO + GNN training")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--total_steps", type=int, default=1_000_000,
                    help="총 env step (not gradient steps)")
parser.add_argument("--rollout_steps", type=int, default=128)
parser.add_argument("--minibatch_size", type=int, default=512)
parser.add_argument("--update_epochs", type=int, default=4)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--clip_eps", type=float, default=0.2)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--vf_coef", type=float, default=0.5)
parser.add_argument("--entropy_coef", type=float, default=0.01)
parser.add_argument("--max_grad_norm", type=float, default=0.5)
parser.add_argument("--action_scale_pos", type=float, default=0.001,
                    help="positional action delta scale (m per step). 0.001=0.06 m/s max")
parser.add_argument("--action_scale_rot", type=float, default=0.0005,
                    help="rotational action delta scale (rad per step). 0.0005=~1.7°/s max")
parser.add_argument("--num_rounds", type=int, default=2, help="GNN message passing rounds")
parser.add_argument("--save_every", type=int, default=50,
                    help="rollout iteration마다 모델 저장")
parser.add_argument("--resume_path", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── Module imports (after AppLauncher) ──
from dual_arm_transport_env3 import DualrobotEnv, DualrobotCfg
import graph_converter as gc
import gnn_policy
import ppo_trainer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Phase 3.3 PPO training on {device}")

    # ── Run name and log dir ──
    if args.resume_path:
        log_dir = os.path.dirname(args.resume_path)
        print(f"📂 Resuming from {args.resume_path}, log_dir={log_dir}")
    else:
        run_name = f"phase3_ppo_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(script_dir, "logs", run_name)
        print(f"📂 New log dir: {log_dir}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # ── Env ──
    env_cfg = DualrobotCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = DualrobotEnv(cfg=env_cfg, render_mode=None)

    # ── Policy ──
    # Per-dim action scale: position vs rotation 분리 (회전이 cold start에 민감)
    action_scale_vec = [args.action_scale_pos] * 3 + [args.action_scale_rot] * 3
    actor_critic = gnn_policy.GNNActorCritic(
        action_dim=env.cfg.action_space,
        num_rounds=args.num_rounds,
        action_scale=action_scale_vec,
    ).to(device)
    n_params = sum(p.numel() for p in actor_critic.parameters())
    print(f"🧠 GNN policy: {n_params:,} params  action_dim={env.cfg.action_space}  "
          f"scale=pos:{args.action_scale_pos} rot:{args.action_scale_rot}")

    if args.resume_path:
        ckpt = torch.load(args.resume_path, map_location=device)
        actor_critic.load_state_dict(ckpt["model"])
        start_iter = ckpt.get("iteration", 0) + 1
        print(f"⏩ Resumed from iter {start_iter - 1}")
    else:
        start_iter = 0

    # ── PPO trainer ──
    ppo_cfg = ppo_trainer.PPOConfig(
        rollout_steps=args.rollout_steps,
        num_envs=args.num_envs,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        lr=args.lr,
        clip_eps=args.clip_eps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        vf_coef=args.vf_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
    )
    trainer = ppo_trainer.PPOTrainer(actor_critic, ppo_cfg, device)

    if args.resume_path and "optimizer" in ckpt:
        trainer.optimizer.load_state_dict(ckpt["optimizer"])

    # ── Initial state ──
    obs, _ = env.reset()
    current_batch = env._build_policy_batch()

    # ── Main training loop ──
    steps_per_iter = args.rollout_steps * args.num_envs
    total_iters = max(1, args.total_steps // steps_per_iter)
    total_env_steps = start_iter * steps_per_iter

    print(f"⚙️  rollout/iter: {steps_per_iter:,}  total_iters: {total_iters:,}")
    print(f"⚙️  starting at iter {start_iter}, env_steps {total_env_steps:,}")
    print("=" * 80)

    t0 = time.time()
    for it in range(start_iter, total_iters):
        # Rollout
        rollout_start = time.time()
        current_batch, last_value, rollout_stats = trainer.collect_rollout(env, current_batch)
        rollout_time = time.time() - rollout_start

        # PPO update
        update_start = time.time()
        update_stats = trainer.update(last_value)
        update_time = time.time() - update_start

        total_env_steps += steps_per_iter
        elapsed = time.time() - t0
        fps = total_env_steps / elapsed if elapsed > 0 else 0

        # Logging
        print(
            f"[iter {it:5d}/{total_iters}] "
            f"steps {total_env_steps:>9,}  "
            f"rew {rollout_stats['ep_reward_mean']:>7.2f}  "
            f"ep_len {rollout_stats['ep_length_mean']:>5.1f}  "
            f"n_ep {rollout_stats['n_episodes']:>4d}  "
            f"loss_p {update_stats['policy_loss']:>6.3f}  "
            f"loss_v {update_stats['value_loss']:>6.3f}  "
            f"kl {update_stats['approx_kl']:>5.3f}  "
            f"clip {update_stats['clip_fraction']:>4.2f}  "
            f"fps {fps:>6.0f}  "
            f"rollout {rollout_time:.1f}s upd {update_time:.1f}s"
        )

        writer.add_scalar("Rollout/ep_reward_mean", rollout_stats["ep_reward_mean"], total_env_steps)
        writer.add_scalar("Rollout/ep_length_mean", rollout_stats["ep_length_mean"], total_env_steps)
        writer.add_scalar("Rollout/n_episodes", rollout_stats["n_episodes"], total_env_steps)
        writer.add_scalar("PPO/policy_loss", update_stats["policy_loss"], total_env_steps)
        writer.add_scalar("PPO/value_loss", update_stats["value_loss"], total_env_steps)
        writer.add_scalar("PPO/entropy", update_stats["entropy"], total_env_steps)
        writer.add_scalar("PPO/approx_kl", update_stats["approx_kl"], total_env_steps)
        writer.add_scalar("PPO/clip_fraction", update_stats["clip_fraction"], total_env_steps)
        writer.add_scalar("Perf/fps", fps, total_env_steps)

        # env extras (mean over rollout)
        for k, v in env.extras.items():
            if isinstance(v, torch.Tensor) and v.ndim == 0:
                writer.add_scalar(f"Env/{k}", v.item(), total_env_steps)

        # Save model
        if it > 0 and it % args.save_every == 0:
            ckpt_path = os.path.join(log_dir, f"model_iter_{it:06d}.pt")
            torch.save({
                "model": actor_critic.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "iteration": it,
                "env_steps": total_env_steps,
            }, ckpt_path)
            print(f"💾 Saved checkpoint: {ckpt_path}")

    # Final save
    final_path = os.path.join(log_dir, "model_final.pt")
    torch.save({
        "model": actor_critic.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "iteration": total_iters - 1,
        "env_steps": total_env_steps,
    }, final_path)
    print(f"💾 Final save: {final_path}")
    print(f"✅ Training complete. {total_env_steps:,} env steps in {time.time()-t0:.0f}s")

    writer.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
