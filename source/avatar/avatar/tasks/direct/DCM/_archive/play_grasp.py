"""학습된 정책을 held-out 파지에서 **실제 구동**해 운반을 눈으로 재생 (Isaac GUI, --headless 금지).

eval_grasp.py의 모델 로드/deterministic rollout을 그대로 쓰되, 렌더하면서 정책이 rod를 목표로
운반하는 걸 본다. env 여러 개(각자 다른 파지)를 띄우고 카메라가 --seconds마다 다음 env로 이동.
에피소드 끝날 때마다 [env b=버킷 d/θ] success/step 콘솔 출력 → 보는 것과 숫자 대조.

사용 (⚠️ 이 모델은 action_scale pos=0.02):
  export LD_LIBRARY_PATH=/home/hjh/anaconda3/envs/env-isaaclab/lib:$LD_LIBRARY_PATH
  python gen_grasp.py --n 6 --seed 777 --out heldout_view.pt     # 볼 held-out 파지 (평가와 같은 seed)
  python -u play_grasp.py --model_path logs/<run>/model_final.pt --grasp_preset heldout_view.pt --same_side
옵션: --n(env수=파지수), --seconds(env당 카메라 초), --num_steps. 종료: 뷰포트 닫기/Ctrl-C.
"""
import argparse, math, time
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--grasp_preset", type=str, default=None, help="볼 파지 세트(.pt). 없으면 학습 8버킷.")
parser.add_argument("--n", type=int, default=6, help="띄울 env 수(=파지 수). 카메라가 순차로 비춤.")
parser.add_argument("--seconds", type=float, default=10.0, help="env당 카메라 표시 시간[s] (한 에피소드 여유).")
parser.add_argument("--env_spacing", type=float, default=4.0)
parser.add_argument("--num_steps", type=int, default=100000, help="총 스텝(크게=계속 재생).")
parser.add_argument("--action_scale_pos", type=float, default=0.02, help="★ 이 모델은 0.02.")
parser.add_argument("--action_scale_rot", type=float, default=0.05)
parser.add_argument("--num_rounds", type=int, default=2)
parser.add_argument("--same_side", action="store_true", help="학습과 일치(이 런은 ON).")
parser.add_argument("--cache_size", type=int, default=4000, help="pose 캐시(뷰어는 작게).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.headless:
    print("⚠️  --headless면 화면이 안 뜹니다. 재생이 목적이면 --headless 빼세요.")
app = AppLauncher(args); sim_app = app.app

import torch, os, math as _m
from dual_arm_transport_env3 import DualrobotEnv, DualrobotCfg
import mlp_policy
import isaaclab.utils.math as math_utils

dev = "cuda" if torch.cuda.is_available() else "cpu"

# 볼 파지 수 = env 수. preset 있으면 그 버킷 수를 n으로 맞춤(1:1).
if args.grasp_preset:
    n_preset = int(torch.load(args.grasp_preset, map_location="cpu")["d"].numel())
    N = min(args.n, n_preset) if args.n else n_preset
    N = n_preset  # preset 전체를 라운드로빈 1:1로
else:
    N = args.n

cfg = DualrobotCfg()
cfg.scene.num_envs = N
cfg.scene.env_spacing = args.env_spacing
cfg.vary_grasp = True
cfg.grasp_same_side = args.same_side
cfg.pose_cache_size = args.cache_size
cfg.n_obstacles = 0
if args.grasp_preset:
    cfg.grasp_preset_path = args.grasp_preset
env = DualrobotEnv(cfg, render_mode=None)
A = env.cfg.action_space
POS_T, ROT_T = 0.02, math.radians(10)
settle = getattr(env, "SETTLE_STEPS_AT_RESET", 0)
if hasattr(env, "_obstacle_curr_frac"):
    env._obstacle_curr_frac = 0.0

b_d = env._grasp_bucket_d.detach().cpu(); b_th = env._grasp_bucket_theta.detach().cpu()
env_bucket = env._grasp_bucket_idx.detach().to(dev)
origins = env.scene.env_origins

# --- 모델 로드 (eval_grasp와 동일) ---
sd = torch.load(os.path.abspath(args.model_path), map_location=dev, weights_only=False)["model"]
scale = [args.action_scale_pos]*3 + [args.action_scale_rot]*3 + [1.0]*(A-6)
if "actor.mean_head.0.weight" in sd:
    in_dim = sd["actor.mean_head.0.weight"].shape[1]
    use_full = (in_dim == mlp_policy._state_dim(True))
    use_lean = (in_dim == mlp_policy._state_dim(False, True))
    agent = mlp_policy.MLPSACAgent(action_dim=A, action_scale=scale, hidden_dim=256,
                                   num_hidden_layers=2, use_full_state=use_full,
                                   use_lean_obstacle=use_lean).to(dev)
else:
    import gnn_policy
    agent = gnn_policy.GNNSACAgent(action_dim=A, num_rounds=args.num_rounds, action_scale=scale).to(dev)
agent.load_state_dict(sd); agent.eval()

src = os.path.basename(args.grasp_preset) if args.grasp_preset else "학습 8버킷"
print("=" * 74)
print(f"▶ 재생: {os.path.basename(args.model_path)}  파지={src}  same_side={args.same_side}  "
      f"envs={N}  scale=pos{args.action_scale_pos}/rot{args.action_scale_rot}")
print(f"  카메라가 {args.seconds:.0f}초마다 env를 순차로 비춥니다. 에피소드 종료 시 결과 출력.")


def look_at(i):
    o = origins[i].tolist()
    env.sim.set_camera_view(eye=(o[0]+1.8, o[1]+1.8, o[2]+1.4), target=(o[0], o[1], o[2]+0.4))


env.reset(); batch = env._build_policy_batch()
rr_min_pos = torch.full((N,), float("inf"), device=dev)
rr_min_rot = torch.full((N,), float("inf"), device=dev)
rr_len = torch.zeros(N, dtype=torch.long, device=dev)
cam = 0; look_at(cam); t_switch = time.time()
for step in range(args.num_steps):
    if not sim_app.is_running():
        break
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
                if i == cam:  # 지금 보는 env만 출력(스팸 방지)
                    b = int(env_bucket[i].item())
                    mark = "✅성공" if succ else "❌실패"
                    print(f"  [env {i} bkt{b} d={float(b_d[b]):.3f} θ={float(b_th[b])*57.3:+.0f}°] "
                          f"{mark}  min_pos={rr_min_pos[i].item()*100:.1f}cm min_rot={math.degrees(rr_min_rot[i].item()):.0f}° ({rr_len[i].item()}s)")
            rr_min_pos[i] = float("inf"); rr_min_rot[i] = float("inf"); rr_len[i] = 0
    # 카메라 순차 이동
    if time.time() - t_switch >= args.seconds:
        cam = (cam + 1) % N; look_at(cam); t_switch = time.time()
        b = int(env_bucket[cam].item())
        print(f"  → 카메라 env {cam} (bkt{b} d={float(b_d[b]):.3f} θ={float(b_th[b])*57.3:+.0f}°)")
    batch = env._build_policy_batch()

env.close(); sim_app.close()
