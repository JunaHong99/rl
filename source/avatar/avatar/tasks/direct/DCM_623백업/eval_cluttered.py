"""
Cluttered transport deterministic eval.
운반 success(min_pos<2cm & min_rot<10°) + 충돌(에피소드 중 min_clearance<0 발생) 측정,
장애물 개수별 버킷. full 장애물(frac=1.0)에서 평가.

사용:
  export LD_LIBRARY_PATH=/home/hjh/anaconda3/envs/env-isaaclab/lib:$LD_LIBRARY_PATH
  # 단일
  python -u eval_cluttered.py --model_path logs/.../model_final.pt --num_envs 1024 --headless
  # 여러 체크포인트 한 프로세스에서 연속(부팅 1회): 엔트리별 @on/@off로 필터 지정 가능
  python -u eval_cluttered.py --model_paths "a.pt@off,b.pt@off,b.pt@on" --num_envs 1024 --headless
"""
import argparse, math, time
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default=None)
parser.add_argument("--model_paths", type=str, default=None,
                    help="comma-separated 'path' 또는 'path@on'/'path@off'. 한 프로세스에서 연속 평가(부팅 1회).")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--num_steps", type=int, default=300)
parser.add_argument("--action_scale_pos", type=float, default=0.05)
parser.add_argument("--action_scale_rot", type=float, default=0.05)
parser.add_argument("--obstacle_frac", type=float, default=1.0, help="평가 시 장애물 curriculum frac")
parser.add_argument("--zero_nullspace", action="store_true", help="action[6:8](nullspace)를 0으로 강제 — 진단용")
parser.add_argument("--no_filter", action="store_true", help="rod safety filter OFF (model_paths의 per-entry @가 우선).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args); sim_app = app.app

import torch, os, numpy as np
from dual_arm_transport_env3 import DualrobotEnv, DualrobotCfg
import mlp_policy
import isaaclab.utils.math as math_utils

dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = DualrobotCfg(); cfg.scene.num_envs = args.num_envs
env = DualrobotEnv(cfg, render_mode=None)            # ★ 부팅/씬 로딩 1회
env._obstacle_curr_frac = args.obstacle_frac
A = env.cfg.action_space
B = args.num_envs
POS_T, ROT_T = 0.02, math.radians(10)
settle = getattr(env, "SETTLE_STEPS_AT_RESET", 0)


def run_eval(model_path, no_filter):
    """체크포인트 1개 평가. 부팅된 env 재사용. no_filter로 rod safety filter on/off."""
    env.cfg.use_rod_safety_filter = (not no_filter)
    sd = torch.load(os.path.abspath(model_path), map_location=dev, weights_only=False)["model"]
    scale = [args.action_scale_pos]*3 + [args.action_scale_rot]*3 + [1.0]*(A-6)
    in_dim = sd["actor.mean_head.0.weight"].shape[1]
    use_full = (in_dim == mlp_policy._state_dim(True))
    use_lean = (in_dim == mlp_policy._state_dim(False, True))
    agent = mlp_policy.MLPSACAgent(action_dim=A, action_scale=scale, hidden_dim=256,
                                   num_hidden_layers=2, use_full_state=use_full,
                                   use_lean_obstacle=use_lean).to(dev)
    agent.load_state_dict(sd); agent.eval()
    mode = "full_state" if use_full else ("lean_obstacle" if use_lean else "rod+global")
    fmode = "OFF" if no_filter else "ON"
    print("=" * 70)
    print(f"✅ {os.path.basename(model_path)}  input={mode}(dim={in_dim})  filter={fmode}  frac={args.obstacle_frac}")

    env.reset(); batch = env._build_policy_batch()
    rr_min_pos = torch.full((B,), float("inf"), device=dev)
    rr_min_rot = torch.full((B,), float("inf"), device=dev)
    rr_collided = torch.zeros(B, dtype=torch.bool, device=dev)
    rr_rod_coll = torch.zeros(B, dtype=torch.bool, device=dev)
    rr_arm_coll = torch.zeros(B, dtype=torch.bool, device=dev)
    rr_nobs = torch.zeros(B, dtype=torch.long, device=dev)
    rr_len = torch.zeros(B, dtype=torch.long, device=dev)
    ep = []
    t0 = time.time()
    for step in range(args.num_steps):
        with torch.no_grad():
            action, _, _ = agent.actor.get_action_and_log_prob(batch, deterministic=True)
            if args.zero_nullspace and action.shape[1] >= 8:
                action = action.clone(); action[:, 6:8] = 0.0
            _, _, term, trunc, _ = env.step(action)
        rod_pos = env.rod.data.root_pos_w; goal_pos = env.goal_rod_marker.data.root_pos_w
        pos_err = torch.norm(goal_pos - rod_pos, dim=-1)
        q_diff = math_utils.quat_mul(env.goal_rod_marker.data.root_quat_w,
                                     math_utils.quat_conjugate(env.rod.data.root_quat_w))
        rot_err = 2.0 * torch.atan2(torch.norm(q_diff[:, 1:4], dim=-1), torch.abs(q_diff[:, 0]))
        rr_min_pos = torch.min(rr_min_pos, pos_err); rr_min_rot = torch.min(rr_min_rot, rot_err)
        rr_len += 1
        ost = env._get_obstacle_state()
        nonsettle = ~env._is_settle_step if hasattr(env, "_is_settle_step") else torch.ones(B, dtype=torch.bool, device=dev)
        rr_rod_coll |= (ost["min_clearance"] < 0.0) & nonsettle
        rr_arm_coll |= (ost["arm_min_clearance"] < 0.0) & nonsettle
        rr_collided |= (torch.minimum(ost["min_clearance"], ost["arm_min_clearance"]) < 0.0) & nonsettle
        rr_nobs = torch.maximum(rr_nobs, env.obstacle_active.sum(1))
        done = term | trunc
        if done.any():
            for i in done.nonzero(as_tuple=True)[0].tolist():
                if rr_len[i].item() > settle:
                    succ = (rr_min_pos[i].item() < POS_T) and (rr_min_rot[i].item() < ROT_T)
                    ep.append((succ, bool(rr_collided[i].item()), int(rr_nobs[i].item()),
                               bool(rr_rod_coll[i].item()), bool(rr_arm_coll[i].item())))
                rr_min_pos[i] = float("inf"); rr_min_rot[i] = float("inf")
                rr_collided[i] = False; rr_rod_coll[i] = False; rr_arm_coll[i] = False
                rr_nobs[i] = 0; rr_len[i] = 0
        batch = env._build_policy_batch()

    S = np.array([e[0] for e in ep]); C = np.array([e[1] for e in ep]); N = np.array([e[2] for e in ep])
    RC = np.array([e[3] for e in ep]); AC = np.array([e[4] for e in ep])
    n = len(S)
    print(f"  Eval {time.time()-t0:.0f}s  genuine episodes: {n}")
    print(f"  Success (도달):            {100*S.mean() if n else 0:.1f}%")
    print(f"  Collision-free success:    {100*(S & ~C).mean() if n else 0:.1f}%")
    print(f"  Collision rate (ep중 충돌): {100*C.mean() if n else 0:.1f}%")
    print(f"    ├ rod-장애물 충돌:       {100*RC.mean() if n else 0:.1f}%")
    print(f"    └ 팔-장애물 충돌:        {100*AC.mean() if n else 0:.1f}%")
    for k in range(0, cfg.n_obstacles + 1):
        m = N == k
        if m.sum() > 0:
            print(f"  장애물 {k}개: n={int(m.sum()):>4}  success {100*S[m].mean():>5.1f}%  "
                  f"충돌없는success {100*(S[m] & ~C[m]).mean():>5.1f}%  collision {100*C[m].mean():>5.1f}%")


# 평가 스펙 파싱: model_paths 우선, 없으면 단일 model_path
specs = []
if args.model_paths:
    for entry in args.model_paths.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "@" in entry:
            p, m = entry.rsplit("@", 1); specs.append((p, m.strip().lower() == "off"))
        else:
            specs.append((entry, args.no_filter))
elif args.model_path:
    specs.append((args.model_path, args.no_filter))
else:
    raise SystemExit("--model_path 또는 --model_paths 필요")

for p, nf in specs:
    run_eval(p, nf)
os._exit(0)
