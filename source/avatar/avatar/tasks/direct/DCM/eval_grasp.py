"""파지 일반화 eval (obstacle-free) — 학습 파지 vs held-out 파지 success 비교.

eval_cluttered.py의 검증된 deterministic rollout(min_pos<2cm & min_rot<10°, genuine ep)을 그대로 따오되:
  - 장애물 제거(obstacle-free)
  - **파지 버킷별 success 분해** (env._grasp_bucket_idx / _grasp_bucket_d / _grasp_bucket_theta)
  - --grasp_preset: 없으면 **학습 8버킷(seed 12345 재현)** = 학습 파지 success,
                    있으면 그 preset의 파지들 = held-out success (gen_grasp.py로 생성).
  - --same_side: 학습과 일치시킬 것(이 S2 런은 same_side ON이므로 반드시 붙일 것).

사용:
  export LD_LIBRARY_PATH=/home/hjh/anaconda3/envs/env-isaaclab/lib:$LD_LIBRARY_PATH
  # 1) 학습 파지 success (8버킷 재현)
  python -u eval_grasp.py --model_path logs/.../model_final.pt --same_side --num_envs 1024 --headless
  # 2) held-out 파지 success (안 준 랜덤 파지)
  python gen_grasp.py --n 32 --seed 777 --out heldout.pt
  python -u eval_grasp.py --model_path logs/.../model_final.pt --same_side --grasp_preset heldout.pt --num_envs 1024 --headless
generalization gap = (1)success − (2)success. per-버킷/θ구간 표로 어디서 무너지는지 확인.
"""
import argparse, math, time
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--num_steps", type=int, default=300)
parser.add_argument("--action_scale_pos", type=float, default=0.05)
parser.add_argument("--action_scale_rot", type=float, default=0.05)
parser.add_argument("--num_rounds", type=int, default=2, help="GNN message-passing rounds.")
parser.add_argument("--grasp_preset", type=str, default=None, help="held-out 파지 세트(.pt). 없으면 학습 8버킷 재현.")
parser.add_argument("--same_side", action="store_true", help="학습과 일치(이 S2 런은 ON이므로 필수).")
parser.add_argument("--cache_size", type=int, default=100_000, help="pose 캐시 크기(학습과 같게 두면 재사용).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args); sim_app = app.app

import torch, os, numpy as np
from dual_arm_transport_env3 import DualrobotEnv, DualrobotCfg
import mlp_policy
import isaaclab.utils.math as math_utils

dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = DualrobotCfg()
cfg.scene.num_envs = args.num_envs
cfg.vary_grasp = True                        # 파지 변이 ON (학습과 동일)
cfg.grasp_same_side = args.same_side
cfg.pose_cache_size = args.cache_size
cfg.n_obstacles = 0                           # obstacle-free
if args.grasp_preset:
    cfg.grasp_preset_path = args.grasp_preset
env = DualrobotEnv(cfg, render_mode=None)
A = env.cfg.action_space
B = args.num_envs
POS_T, ROT_T = 0.02, math.radians(10)
settle = getattr(env, "SETTLE_STEPS_AT_RESET", 0)
if hasattr(env, "_obstacle_curr_frac"):
    env._obstacle_curr_frac = 0.0

# 버킷 (d,θ) 표 (per-env는 round-robin)
nb = int(cfg.grasp_n_buckets)
b_d = env._grasp_bucket_d.detach().cpu()
b_th = env._grasp_bucket_theta.detach().cpu()
env_bucket = env._grasp_bucket_idx.detach().to(dev)   # (B,) 고정

# --- 모델 로드 (eval_cluttered와 동일 판별) ---
sd = torch.load(os.path.abspath(args.model_path), map_location=dev, weights_only=False)["model"]
scale = [args.action_scale_pos]*3 + [args.action_scale_rot]*3 + [1.0]*(A-6)
if "actor.mean_head.0.weight" in sd:
    in_dim = sd["actor.mean_head.0.weight"].shape[1]
    use_full = (in_dim == mlp_policy._state_dim(True))
    use_lean = (in_dim == mlp_policy._state_dim(False, True))
    agent = mlp_policy.MLPSACAgent(action_dim=A, action_scale=scale, hidden_dim=256,
                                   num_hidden_layers=2, use_full_state=use_full,
                                   use_lean_obstacle=use_lean).to(dev)
    mode = "full_state" if use_full else ("lean_obstacle" if use_lean else "rod+global")
else:
    import gnn_policy
    agent = gnn_policy.GNNSACAgent(action_dim=A, num_rounds=args.num_rounds, action_scale=scale).to(dev)
    mode, in_dim = "GNN", "graph"
agent.load_state_dict(sd); agent.eval()

src = os.path.basename(args.grasp_preset) if args.grasp_preset else "학습 8버킷(seed 12345)"
print("=" * 74)
print(f"✅ {os.path.basename(args.model_path)}  input={mode}({in_dim})  파지={src}  "
      f"same_side={args.same_side}  buckets={nb}")

env.reset(); batch = env._build_policy_batch()
rr_min_pos = torch.full((B,), float("inf"), device=dev)
rr_min_rot = torch.full((B,), float("inf"), device=dev)
rr_len = torch.zeros(B, dtype=torch.long, device=dev)
ep_succ, ep_bkt = [], []   # genuine 에피소드별 (성공, 버킷)
t0 = time.time()
for step in range(args.num_steps):
    with torch.no_grad():
        action, _, _ = agent.actor.get_action_and_log_prob(batch, deterministic=True)
        _, _, term, trunc, _ = env.step(action)
    rod_pos = env.rod.data.root_pos_w; goal_pos = env.goal_rod_marker.data.root_pos_w
    pos_err = torch.norm(goal_pos - rod_pos, dim=-1)
    q_diff = math_utils.quat_mul(env.goal_rod_marker.data.root_quat_w,
                                 math_utils.quat_conjugate(env.rod.data.root_quat_w))
    rot_err = 2.0 * torch.atan2(torch.norm(q_diff[:, 1:4], dim=-1), torch.abs(q_diff[:, 0]))
    rr_min_pos = torch.min(rr_min_pos, pos_err); rr_min_rot = torch.min(rr_min_rot, rot_err)
    rr_len += 1
    done = term | trunc
    if done.any():
        for i in done.nonzero(as_tuple=True)[0].tolist():
            if rr_len[i].item() > settle:
                succ = (rr_min_pos[i].item() < POS_T) and (rr_min_rot[i].item() < ROT_T)
                ep_succ.append(succ); ep_bkt.append(int(env_bucket[i].item()))
            rr_min_pos[i] = float("inf"); rr_min_rot[i] = float("inf"); rr_len[i] = 0
    batch = env._build_policy_batch()

S = np.array(ep_succ); Bk = np.array(ep_bkt); n = len(S)
print(f"  Eval {time.time()-t0:.0f}s  genuine episodes: {n}")
print(f"  ▶ 전체 success: {100*S.mean() if n else 0:.1f}%")
print(f"  {'bkt':>3} {'d':>6} {'θ(deg)':>7} {'n':>5} {'success':>8}")
for b in range(nb):
    m = Bk == b
    if m.sum() == 0:
        continue
    print(f"  {b:>3} {float(b_d[b]):>6.3f} {float(b_th[b])*57.3:>7.1f} {int(m.sum()):>5} "
          f"{100*S[m].mean():>7.1f}%")
# |θ| 구간 분해 (어디서 무너지나)
th_abs = np.abs(b_th.numpy()[Bk]) * 57.3
print("  ── |θ| 구간별 ──")
for lo, hi in [(0, 15), (15, 30), (30, 46)]:
    m = (th_abs >= lo) & (th_abs < hi)
    if m.sum() > 0:
        print(f"   |θ|∈[{lo:>2},{hi:>2})°: n={int(m.sum()):>5}  success {100*S[m].mean():>5.1f}%")
os._exit(0)
