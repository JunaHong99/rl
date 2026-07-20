"""
Phase 3.3-SAC training script.

Off-policy SAC + GNN policy + Cooperative Impedance Controller.

Curriculum stage 1:
    - action = 6-dim (object pose delta only)
    - K_abs, K_rel = controller defaults (fixed)
    - Scenario 1: 기본 운반

사용:
    python -u train_phase3_sac.py --num_envs 64 --total_steps 1000000 --headless
"""

import argparse
import os
import time
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Phase 3.3 SAC + GNN training")
parser.add_argument("--num_envs", type=int, default=1024,
                    help="★ 5-node로 model 작아져 1024 가능 (이전 17-node에서 OOM). diversity 2×.")
parser.add_argument("--total_steps", type=int, default=1_000_000)
parser.add_argument("--buffer_size", type=int, default=1_000_000,
                    help="500k → 1M: catastrophic forgetting 감소")
parser.add_argument("--batch_size", type=int, default=512,
                    help="256 → 512: gradient 더 안정")
parser.add_argument("--warmup_steps", type=int, default=250_000,
                    help="25k → 250k: HER buffer는 done 시점에만 push하므로 충분히 길어야 첫 update가 안정.")
parser.add_argument("--warmup_scale_factor", type=float, default=2.0,
                    help="Warmup random action amplitude × factor. 초기 exploration 폭 확장. 학습 phase에는 미적용.")
parser.add_argument("--explore_boost", type=float, default=1.0,
                    help="(선택) policy action × factor (학습 phase). 1.0 = off. 1.5~2.0 시도 가능하지만 controller stability 주의.")
parser.add_argument("--explore_boost_decay_steps", type=int, default=2_000_000,
                    help="explore_boost가 1.0으로 선형 decay하는 env_step 길이.")
parser.add_argument("--updates_per_step", type=int, default=4,
                    help="1 → 4: vectorized SAC sample efficiency 회복 (env step당 4 gradient update)")
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--target_entropy", type=float, default=-6.0)
parser.add_argument("--auto_alpha", action=argparse.BooleanOptionalAction, default=True,
                    help="SAC alpha auto-tune. Squashed Gaussian과 함께 정상 작동. (default ON)")
parser.add_argument("--fixed_alpha", type=float, default=0.05,
                    help="--no-auto_alpha일 때 사용할 고정 alpha.")
parser.add_argument("--action_scale_pos", type=float, default=0.02,
                    help="Accumulating mode: per-step target displacement 2cm. 30 step random walk std ~63mm (3-5cm goal cover). Controller 부담 작음.")
parser.add_argument("--action_scale_rot", type=float, default=0.05,
                    help="Accumulating mode: per-step rotation ~2.9°/step. 30 step random walk std ~16°.")
parser.add_argument("--init_log_std", type=float, default=0.0,
                    help="Squashed Gaussian: std=1 in unbounded space (tanh로 [-1,1] 전체 cover).")
parser.add_argument("--num_rounds", type=int, default=2)
parser.add_argument("--log_every", type=int, default=100, help="env step마다 한 번 로깅")
parser.add_argument("--save_every", type=int, default=2000,
                    help="vector_step 기준. 2000 × num_envs(=1024) = 2M env_steps마다 저장.")
parser.add_argument("--use_mlp", action="store_true",
                    help="GNN 대신 MLP 사용 (진단용). rod node + global feature만 input.")
parser.add_argument("--mlp_full_state", action="store_true",
                    help="MLP에 5개 노드 (robot1/EE1/rod/EE2/robot2) 전체 + global 입력. 기본은 rod+global only.")
parser.add_argument("--mlp_lean_obstacle", action="store_true",
                    help="MLP에 rod node(32) + 장애물 compact 요약[rel_pos(3)+dist(1)+radius(1)]*N + global 입력. "
                         "Stage 0의 깨끗한 rod+global 입력 유지(희석 X) + rod 라우팅용 장애물 정보만. full_state와 배타적.")
parser.add_argument("--use_her", action="store_true",
                    help="HER (Hindsight Experience Replay) 활성화.")
parser.add_argument("--her_strategy", choices=["future", "random_task"], default="random_task",
                    help="future: vg = rod 미래 위치 (정지정책에서 trivial-reach exploit). "
                         "random_task: vg = curriculum offset 풀에서 sampling (trajectory와 무관, exploit 없음).")
parser.add_argument("--k_future", type=int, default=4,
                    help="HER: 각 transition당 virtual goal 개수 (strategy 무관).")
parser.add_argument("--resume_path", type=str, default=None)
parser.add_argument("--init_weights_path", type=str, default=None,
                    help="네트워크 가중치만 로드(optimizer/buffer/step은 fresh, new log_dir). "
                         "2-phase warm-start용 (convert_phaseA_to_phaseB.py 출력). resume_path와 배타적.")
parser.add_argument("--no_rod_filter", action="store_true",
                    help="rod safety filter OFF로 학습 — RL이 회피를 스스로 학습하는지 검증 (Hong2025 노선).")
parser.add_argument("--curriculum", action="store_true",
                    help="Goal 거리 curriculum: 가까운 거리부터 시작해 점진 확대 (Stage 2 0% 위험 처치).")
parser.add_argument("--curriculum_start_frac", type=float, default=0.3,
                    help="Curriculum 시작 시 사용할 cache 비율 (거리 오름차순 하위 N%%). 0.3 ≈ 가까운 30%%.")
parser.add_argument("--curriculum_end_steps", type=int, default=15_000_000,
                    help="이 env_step에서 frac이 1.0(전체 분포)에 도달. ramp 시작점(=max(warmup, resume지점)) 이후 선형.")
parser.add_argument("--obstacle_curriculum", action="store_true",
                    help="장애물 curriculum: 운반 먼저 학습(0개) 후 점진 추가. cluttered transport용.")
parser.add_argument("--obstacle_curr_start", type=int, default=2_000_000,
                    help="이 env_step까지 장애물 0개(운반만 학습). 이후 ramp 시작.")
parser.add_argument("--obstacle_curr_end", type=int, default=6_000_000,
                    help="이 env_step에서 장애물 frac이 1.0(최대 n_obstacles) 도달.")
parser.add_argument("--resume_refill_steps", type=int, default=0,
                    help="resume 시 첫 N env_step은 정책 action으로 buffer만 채우고 gradient update 스킵 "
                         "(빈 buffer 콜드스타트 방지). 0=off. resume 권장 100000.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from dual_arm_transport_env3 import DualrobotEnv, DualrobotCfg
import gnn_policy
import mlp_policy
import sac_trainer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Phase 3.3-SAC training on {device}")

    # Log dir
    if args.resume_path:
        log_dir = os.path.dirname(args.resume_path)
        print(f"📂 Resuming from {args.resume_path}")
    else:
        run_name = f"phase3_sac_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(script_dir, "logs", run_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"📂 log_dir: {log_dir}")

    # Env
    env_cfg = DualrobotCfg()
    env_cfg.scene.num_envs = args.num_envs
    if args.no_rod_filter:
        env_cfg.use_rod_safety_filter = False
        print("⚠️  rod safety filter OFF — RL이 회피를 스스로 학습 (Hong2025 노선)")
    env = DualrobotEnv(cfg=env_cfg, render_mode=None)

    # Agent
    # pos(3)+rot(3) + per-arm K dim(env.cfg.action_space-6, scale 1.0 = raw tanh ∈ [-1,1])
    n_k_dims = max(0, env.cfg.action_space - 6)
    action_scale_vec = [args.action_scale_pos] * 3 + [args.action_scale_rot] * 3 + [1.0] * n_k_dims
    if args.use_mlp:
        agent = mlp_policy.MLPSACAgent(
            action_dim=env.cfg.action_space,
            num_rounds=args.num_rounds,
            action_scale=action_scale_vec,
            hidden_dim=256,
            num_hidden_layers=2,
            use_full_state=args.mlp_full_state,
            use_lean_obstacle=args.mlp_lean_obstacle,
        ).to(device)
        print(f"🧠 MLP SAC agent (replaces GNN for diagnostic). full_state={args.mlp_full_state} "
              f"lean_obstacle={args.mlp_lean_obstacle}")
    else:
        agent = gnn_policy.GNNSACAgent(
            action_dim=env.cfg.action_space,
            num_rounds=args.num_rounds,
            action_scale=action_scale_vec,
        ).to(device)
    # SAC actor needs init_log_std override (default -6 from gnn_policy, -4 better for SAC)
    with torch.no_grad():
        agent.actor.log_std.data.fill_(args.init_log_std)

    n_params = sum(p.numel() for p in agent.parameters())
    print(f"🧠 GNN SAC agent: {n_params:,} params  action_dim={env.cfg.action_space}  "
          f"scale=pos:{args.action_scale_pos} rot:{args.action_scale_rot}  init_log_std={args.init_log_std}")

    # Trainer
    sac_cfg = sac_trainer.SACConfig(
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        lr_actor=args.lr,
        lr_q=args.lr,
        lr_alpha=args.lr,
        target_entropy=args.target_entropy,
        auto_alpha=args.auto_alpha,
        fixed_alpha=args.fixed_alpha,
        warmup_steps=args.warmup_steps,
        updates_per_step=args.updates_per_step,
    )
    trainer = sac_trainer.SACTrainer(agent, sac_cfg, device)
    trainer.buffer.num_envs = args.num_envs  # set lazily

    # ★ HER buffer 교체 (옵션)
    if args.use_her:
        import her_buffer
        max_ep_len = int(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation)) + 5

        # random_task strategy용: cache에서 (start, goal) offset 풀 추출 (env-local frame)
        cache = env.pose_sampler.cache
        start_pose = cache["start_obj_pose"]  # (P, 7)
        goal_pose = cache["goal_obj_pose"]    # (P, 7)
        offset_pos_pool = (goal_pose[:, :3] - start_pose[:, :3]).clone()  # (P, 3)
        # q_offset such that goal_q = q_offset × start_q  →  q_offset = goal_q × inv(start_q)
        from her_buffer import _quat_mul as _qmul, _quat_conj as _qconj
        offset_quat_pool = _qmul(goal_pose[:, 3:7], _qconj(start_pose[:, 3:7])).clone()
        print(f"  HER random_task offset pool: {offset_pos_pool.shape[0]:,} entries "
              f"(pos range [{offset_pos_pool.norm(dim=-1).min():.3f}, {offset_pos_pool.norm(dim=-1).max():.3f}] m)")

        trainer.buffer = her_buffer.HERReplayBuffer(
            capacity=args.buffer_size,
            num_envs=args.num_envs,
            action_dim=env.cfg.action_space,
            device=device,
            k_future=args.k_future,
            max_episode_len=max_ep_len,
            strategy=args.her_strategy,
            goal_offset_pos_pool=offset_pos_pool,
            goal_offset_quat_pool=offset_quat_pool,
        )
        print(f"🎯 HER enabled: strategy={args.her_strategy}, k_future={args.k_future}, max_ep_len={max_ep_len}")

    if args.resume_path:
        ckpt = torch.load(args.resume_path, map_location=device)
        agent.load_state_dict(ckpt["model"])
        trainer.actor_opt.load_state_dict(ckpt["actor_opt"])
        trainer.q_opt.load_state_dict(ckpt["q_opt"])
        if trainer.alpha_opt is not None and "alpha_opt" in ckpt and ckpt["alpha_opt"] is not None:
            trainer.alpha_opt.load_state_dict(ckpt["alpha_opt"])
        # log_alpha: auto_alpha=False면 무시하고 fixed_alpha 그대로 유지
        if args.auto_alpha and "log_alpha" in ckpt:
            trainer.log_alpha.data = ckpt["log_alpha"].to(device)
        start_step = ckpt.get("env_steps", 0)
        print(f"⏩ Resumed at env_steps {start_step:,}")
    elif args.init_weights_path:
        ckpt = torch.load(args.init_weights_path, map_location=device)
        agent.load_state_dict(ckpt["model"])
        start_step = 0   # fresh optimizer/buffer/step (warm net weights만)
        print(f"🌱 Init weights from {args.init_weights_path} (fresh optimizer/buffer/step)")
    else:
        start_step = 0

    # ── Curriculum frac 스케줄 ──
    # ramp 시작점(=max(warmup, resume지점))부터 curriculum_end_steps까지 start_frac→1.0 선형.
    # resume이면 start_step부터 ramp (절대 step 기준이라 즉시 1.0 되는 것 방지).
    ramp_start = max(args.warmup_steps, start_step)

    def curriculum_frac_at(step):
        if not args.curriculum:
            return 1.0
        sf = args.curriculum_start_frac
        denom = max(1, args.curriculum_end_steps - ramp_start)
        prog = (step - ramp_start) / denom
        prog = min(1.0, max(0.0, prog))
        return sf + (1.0 - sf) * prog

    # resume buffer 콜드스타트 방지: 첫 refill 구간은 정책 action으로 buffer만 채우고 update 스킵.
    refill_until = (start_step + args.resume_refill_steps) if args.resume_path else 0

    if args.curriculum:
        env.pose_sampler.curriculum_frac = curriculum_frac_at(start_step)
        print(f"📚 Curriculum ON: start_frac={args.curriculum_start_frac} "
              f"→ 1.0 by {args.curriculum_end_steps:,} steps (ramp 시작 {ramp_start:,})")
    if refill_until > 0:
        print(f"🪣 Resume refill: {start_step:,} → {refill_until:,} env_steps 동안 update 스킵 (buffer 채우기)")

    # Initial state
    obs, _ = env.reset()
    current_batch = env._build_policy_batch()

    # ── Main loop ──
    print(f"⚙️  warmup_steps={args.warmup_steps:,}  total_steps={args.total_steps:,}  num_envs={args.num_envs}")
    print("=" * 80)

    env_steps = start_step
    t0 = time.time()
    episode_rewards = []
    episode_lengths = []
    running_reward = torch.zeros(args.num_envs, device=device)

    last_train_stats = {"q_loss": 0.0, "actor_loss": 0.0, "alpha_loss": 0.0,
                        "alpha": 0.0, "q1_mean": 0.0, "log_pi_mean": 0.0}

    # Action magnitude tracking (정책이 μ→0으로 collapse하는지 진단)
    action_norm_sum = 0.0
    action_abs_sum = 0.0
    action_norm_n = 0

    just_finished_warmup = False
    while env_steps < args.total_steps:
        # ── Curriculum: 이번 step의 frac을 sampler에 반영 (env.step 내부 reset이 사용) ──
        if args.curriculum:
            env.pose_sampler.curriculum_frac = curriculum_frac_at(env_steps)
        # ── 장애물 curriculum: 운반 먼저(0개) → 점진 추가 ──
        if args.obstacle_curriculum:
            if env_steps < args.obstacle_curr_start:
                env._obstacle_curr_frac = 0.0
            else:
                denom = max(1, args.obstacle_curr_end - args.obstacle_curr_start)
                env._obstacle_curr_frac = min(1.0, (env_steps - args.obstacle_curr_start) / denom)

        # ── Action selection ──
        if env_steps < args.warmup_steps:
            # Warmup: random action scaled to (action_scale × warmup_scale_factor)
            scale_t = torch.tensor(action_scale_vec, device=device) * args.warmup_scale_factor
            action = scale_t * (2 * torch.rand(args.num_envs, env.cfg.action_space, device=device) - 1)
        else:
            with torch.no_grad():
                action, _, _ = agent.actor.get_action_and_log_prob(current_batch, deterministic=False)
            # Optional policy-action explore boost (1.0 = off). Linear decay to 1.0.
            if args.explore_boost > 1.0:
                frac = min(1.0, env_steps / max(1, args.explore_boost_decay_steps))
                boost = args.explore_boost + (1.0 - args.explore_boost) * frac
                action = action * boost
            # First step after warmup: flush pending HER episodes to fill main buffer
            if not just_finished_warmup:
                if args.use_her and hasattr(trainer.buffer, "flush_pending_episodes"):
                    flushed = trainer.buffer.flush_pending_episodes()
                    print(f"🚿 HER flush at warmup end: pushed {flushed:,} transitions (buf now {trainer.buffer.size:,})")
                just_finished_warmup = True

        # Action magnitude 누적 (학습 phase 전용 — warmup은 random이라 의미 없음)
        if env_steps >= args.warmup_steps:
            with torch.no_grad():
                action_norm_sum += action.norm(dim=-1).mean().item()
                action_abs_sum += action.abs().mean().item()
                action_norm_n += 1

        # ── Env step ──
        ep_len_before = env.episode_length_buf.clone()
        # HER용: env.step **이전** rod/goal 상태를 먼저 캡처 (= 진짜 pre-step state).
        # env.step 후에 읽으면 post-step (done env는 post-reset)이라 잘못된 transition 라벨링됨.
        if args.use_her:
            env_origins = env.scene.env_origins
            rod_pos_pre = (env.rod.data.root_pos_w - env_origins).clone()
            rod_quat_pre = env.rod.data.root_quat_w.clone()
            goal_pos = (env.goal_rod_marker.data.root_pos_w - env_origins).clone()
            goal_quat = env.goal_rod_marker.data.root_quat_w.clone()

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated

        # ── Settle mask: env3 settle 중인 transition은 buffer에서 제외 ──
        # 정책에 'action=0이 valid' 학습 안 시키려는 목적. HER도 settle state로 virtual goal 합성 X
        valid_mask = ~env._is_settle_step if hasattr(env, "_is_settle_step") else None

        # ── Add to replay buffer ──
        next_batch = env._build_policy_batch()
        if args.use_her:
            # post-step rod state: env3._get_dones에서 reset 직전에 스냅샷한 _last_rod_pos_w 사용.
            # (env.rod.data는 done env에 대해 이미 post-reset이라 그대로 쓸 수 없음.)
            env_origins = env.scene.env_origins
            rod_pos_post = (env._last_rod_pos_w - env_origins).clone()
            rod_quat_post = env._last_rod_quat_w.clone()
            trainer.buffer.add_step(
                current_batch, action, reward, next_batch, done,
                rod_pos_pre, rod_quat_pre,
                rod_pos_post, rod_quat_post,
                goal_pos, goal_quat,
                valid_mask=valid_mask,
                goal_indep_reward=getattr(env, "_goal_indep_reward", None),
            )
        else:
            # add_batch는 valid_mask 미지원 — 전체 settle 중일 때만 skip (보수적)
            if valid_mask is None or valid_mask.all():
                trainer.buffer.add_batch(current_batch, action, reward, next_batch, done)
            elif valid_mask.any():
                # 일부 envs만 valid → partial push. add_batch가 환경별 mask 미지원이라 일단 skip.
                # TODO: 비-HER 경로에서도 partial push 필요시 add_batch에 mask 추가.
                pass

        # ── Episode bookkeeping ──
        running_reward += reward
        if done.any():
            done_mask = done.bool()
            actual_lengths = (ep_len_before[done_mask] + 1).float()
            episode_rewards.extend(running_reward[done_mask].cpu().tolist())
            episode_lengths.extend(actual_lengths.cpu().tolist())
            running_reward[done_mask] = 0.0

        current_batch = next_batch
        env_steps += args.num_envs

        # ── Gradient updates ── (warmup 이후 + resume refill 이후)
        if env_steps >= args.warmup_steps and env_steps >= refill_until:
            for _ in range(args.updates_per_step):
                stats = trainer.update(args.batch_size)
                if stats is not None:
                    last_train_stats = stats

        # ── Logging ──
        if env_steps // args.num_envs % args.log_every == 0:
            elapsed = time.time() - t0
            fps = env_steps / elapsed if elapsed > 0 else 0
            ep_rew = sum(episode_rewards[-100:]) / max(1, len(episode_rewards[-100:]))
            ep_len = sum(episode_lengths[-100:]) / max(1, len(episode_lengths[-100:]))

            phase = "WARM" if env_steps < args.warmup_steps else "TRAIN"
            print(
                f"[{phase} step {env_steps:>9,}] "
                f"ep_rew {ep_rew:>7.2f}  ep_len {ep_len:>6.1f}  buf {trainer.buffer.size:>7,}  "
                f"q_loss {last_train_stats['q_loss']:>7.3f}  "
                f"act_loss {last_train_stats['actor_loss']:>7.3f}  "
                f"α {last_train_stats['alpha']:>5.3f}  "
                f"logπ {last_train_stats['log_pi_mean']:>7.2f}  "
                f"fps {fps:>5.0f}  "
                f"elapsed {elapsed/60:.1f}m",
                flush=True
            )

            # Reward (외부 추적)
            writer.add_scalar("reward/ep_reward_mean", ep_rew, env_steps)
            # SAC internal — 핵심만 (q_loss, alpha, log_pi_mean). actor_loss/q1_mean은 q와 alpha로 충분.
            sac_keep = {"q_loss", "alpha", "log_pi_mean"}
            for k, v in last_train_stats.items():
                if k not in sac_keep:
                    continue
                writer.add_scalar(f"SAC/{k}", v, env_steps)
            writer.add_scalar("perf/fps", fps, env_steps)
            if args.curriculum:
                writer.add_scalar("curriculum/frac", env.pose_sampler.curriculum_frac, env_steps)

            # Action magnitude — 정책 μ가 0으로 collapse하는지 확인용.
            if action_norm_n > 0:
                writer.add_scalar("diag/action_norm_mean", action_norm_sum / action_norm_n, env_steps)
                writer.add_scalar("diag/action_abs_mean", action_abs_sum / action_norm_n, env_steps)
                action_norm_sum = 0.0
                action_abs_sum = 0.0
                action_norm_n = 0

            # env extras: env3에서 task/, reward/, diag/ prefix로 그룹화. 그대로 전달.
            for k, v in env.extras.items():
                if isinstance(v, torch.Tensor) and v.ndim == 0:
                    writer.add_scalar(k, v.item(), env_steps)

        # ── Save ──
        if env_steps > 0 and env_steps // args.num_envs % args.save_every == 0:
            ckpt_path = os.path.join(log_dir, f"model_step_{env_steps:09d}.pt")
            torch.save({
                "model": agent.state_dict(),
                "actor_opt": trainer.actor_opt.state_dict(),
                "q_opt": trainer.q_opt.state_dict(),
                "alpha_opt": trainer.alpha_opt.state_dict() if trainer.alpha_opt is not None else None,
                "log_alpha": trainer.log_alpha.detach().cpu(),
                "env_steps": env_steps,
            }, ckpt_path)
            print(f"💾 Saved {ckpt_path}", flush=True)

    # Final save
    final_path = os.path.join(log_dir, "model_final.pt")
    torch.save({
        "model": agent.state_dict(),
        "actor_opt": trainer.actor_opt.state_dict(),
        "q_opt": trainer.q_opt.state_dict(),
        "alpha_opt": trainer.alpha_opt.state_dict() if trainer.alpha_opt is not None else None,
        "log_alpha": trainer.log_alpha.detach().cpu(),
        "env_steps": env_steps,
    }, final_path)
    print(f"💾 Final save: {final_path}")
    print(f"✅ Training complete. {env_steps:,} env steps in {(time.time()-t0)/60:.1f}m")

    writer.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
