"""
100개 episode 평가 + 실패한 episode의 cache idx 저장.

목적:
  - model_final.pt를 100 ep deterministic eval
  - 실패 episode (min_pos≥20mm OR min_rot≥10°) 식별
  - 실패한 episode의 cache idx 저장 → visualize_failures.py로 재현 가능

사용:
  python -u eval_collect_failures.py \
      --model_path logs/phase3_sac_20260609-154653/model_final.pt \
      --num_episodes 100 --num_envs 100 --headless
"""
import argparse
import math
import os
import time
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=100)
parser.add_argument("--num_episodes", type=int, default=100,
                    help="목표 episode 수. 이만큼 done 발생할 때까지 계속.")
parser.add_argument("--max_steps", type=int, default=200,
                    help="안전 cap. 보통 settle 30 + RL 30 = 60 step 안에 다 done.")
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--action_scale_pos", type=float, default=0.02)
parser.add_argument("--action_scale_rot", type=float, default=0.05,
                    help="★ 학습 시점과 일치 필수. 이번 30M 학습은 0.05로 진행됨.")
parser.add_argument("--output", type=str,
                    default="/tmp/failed_episodes.pt")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from dual_arm_transport_env3 import DualrobotEnv, DualrobotCfg
import mlp_policy
import isaaclab.utils.math as math_utils


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Eval + failure collection on {device}")

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

    # Success threshold (reward function 의도)
    POS_THRESH_M = 0.02
    ROT_THRESH_RAD = math.radians(10)

    obs, _ = env.reset()
    current_batch = env._build_policy_batch()

    # Per-env tracking
    running_min_pos = torch.full((args.num_envs,), float("inf"), device=device)
    running_min_rot = torch.full((args.num_envs,), float("inf"), device=device)
    running_len = torch.zeros(args.num_envs, device=device, dtype=torch.long)

    # Collected episode results (마지막 args.num_episodes 까지 채울 때까지)
    ep_records = []  # list of dict(cache_idx, success, min_pos_mm, min_rot_deg, length)

    t0 = time.time()
    print(f"⚙️  num_envs={args.num_envs}  target_episodes={args.num_episodes}")
    print("=" * 80)

    step = 0
    while len(ep_records) < args.num_episodes and step < args.max_steps:
        with torch.no_grad():
            action, _, _ = agent.actor.get_action_and_log_prob(current_batch, deterministic=True)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated

        # Per-env err 업데이트
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
        running_len = running_len + 1

        if done.any():
            done_idx = done.nonzero(as_tuple=True)[0]
            for i in done_idx.tolist():
                if len(ep_records) >= args.num_episodes:
                    break
                cache_idx = env.current_sample_idxs[i].item()  # ★ 이 env의 현재 episode cache idx
                is_succ = (
                    running_min_pos[i].item() < POS_THRESH_M
                    and running_min_rot[i].item() < ROT_THRESH_RAD
                )
                ep_records.append({
                    "env_id": i,
                    "cache_idx": cache_idx,
                    "success": is_succ,
                    "min_pos_mm": running_min_pos[i].item() * 1000,
                    "min_rot_deg": running_min_rot[i].item() * 180.0 / math.pi,
                    "length": int(running_len[i].item()),
                })
            running_min_pos[done_idx] = float("inf")
            running_min_rot[done_idx] = float("inf")
            running_len[done_idx] = 0

        current_batch = env._build_policy_batch()
        step += 1

    elapsed = time.time() - t0

    # 결과 분석
    n_total = len(ep_records)
    successes = [r for r in ep_records if r["success"]]
    failures = [r for r in ep_records if not r["success"]]
    n_succ = len(successes)
    n_fail = len(failures)
    succ_rate = n_succ / max(1, n_total) * 100

    print("=" * 80)
    print(f"✅ Eval complete in {elapsed:.1f}s ({step} env-steps)")
    print(f"  Total episodes:  {n_total}")
    print(f"  Success rate:    {succ_rate:.1f}%   ({n_succ}/{n_total})")
    print(f"  Failures:        {n_fail}")
    print()

    if successes:
        sp = [r["min_pos_mm"] for r in successes]
        sr = [r["min_rot_deg"] for r in successes]
        sl = [r["length"] for r in successes]
        print(f"  Successes: min_pos {sum(sp)/len(sp):.1f}mm avg ({min(sp):.1f}-{max(sp):.1f})")
        print(f"             min_rot {sum(sr)/len(sr):.2f}° avg ({min(sr):.1f}-{max(sr):.1f})")
        print(f"             length  {sum(sl)/len(sl):.1f} avg ({min(sl)}-{max(sl)})")
    if failures:
        fp = [r["min_pos_mm"] for r in failures]
        fr = [r["min_rot_deg"] for r in failures]
        print(f"  Failures:  min_pos {sum(fp)/len(fp):.1f}mm avg ({min(fp):.1f}-{max(fp):.1f})")
        print(f"             min_rot {sum(fr)/len(fr):.2f}° avg ({min(fr):.1f}-{max(fr):.1f})")
        print()
        print(f"  실패 분류:")
        only_pos = sum(1 for r in failures if r["min_rot_deg"] < 10 and r["min_pos_mm"] >= 20)
        only_rot = sum(1 for r in failures if r["min_pos_mm"] < 20 and r["min_rot_deg"] >= 10)
        both = sum(1 for r in failures if r["min_pos_mm"] >= 20 and r["min_rot_deg"] >= 10)
        print(f"    pos만 미달 (rot OK):  {only_pos}")
        print(f"    rot만 미달 (pos OK):  {only_rot}")
        print(f"    둘 다 미달:           {both}")

    # 실패 episode의 cache idx 저장
    failed_cache_idxs = [r["cache_idx"] for r in failures if r["cache_idx"] >= 0]
    if failed_cache_idxs:
        save_data = {
            "cache_idxs": torch.tensor(failed_cache_idxs, dtype=torch.long),
            "failures": failures,  # 상세 메타데이터
            "model_path": ckpt_path,
            "action_scale_pos": args.action_scale_pos,
            "action_scale_rot": args.action_scale_rot,
        }
        torch.save(save_data, args.output)
        print()
        print(f"💾 Failed episodes saved: {args.output}")
        print(f"   {len(failed_cache_idxs)} cache idxs → visualize_failures.py로 재현 가능")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
