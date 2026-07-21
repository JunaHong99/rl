"""학습 정책을 held-out 파지에서 구동해 운반을 **mp4로 녹화** (headless 서버용, --headless + --enable_cameras).

play_grasp.py의 policy rollout를 그대로 쓰되 GUI 대신 DirectRLEnv.render()(rgb_array)로 프레임을 잡아
cv2로 mp4 저장. 파지마다 카메라를 그 env로 옮겨 한 세그먼트씩 녹화 → 한 영상에 여러 파지 운반 수록.
프레임에 [버킷 d/θ, min_pos/min_rot, 성공] 오버레이 → 눈으로 본 것과 숫자 대조.

사용 (⚠️ headless라도 --enable_cameras 필수, 이 모델 scale pos=0.02):
  export LD_LIBRARY_PATH=/home/hjh/anaconda3/envs/env-isaaclab/lib:$LD_LIBRARY_PATH
  python gen_grasp.py --n 6 --seed 777 --out heldout_view.pt
  python -u record_grasp.py --model_path logs/<run>/model_final.pt --grasp_preset heldout_view.pt \
      --same_side --headless --enable_cameras --out grasp_replay.mp4
결과: grasp_replay.mp4 (파지 6종 × 세그먼트). scp로 받아서 재생.
"""
import argparse, math, os, time
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--grasp_preset", type=str, default=None, help="볼 파지 세트(.pt). 없으면 학습 8버킷.")
parser.add_argument("--out", type=str, default="grasp_replay.mp4")
parser.add_argument("--steps_per_env", type=int, default=45, help="파지당 녹화 스텝(에피소드 1~2개 = ~9s sim).")
parser.add_argument("--fps", type=int, default=10, help="출력 영상 fps(콘텐츠는 5Hz제어라 10이면 약 2x).")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--env_spacing", type=float, default=4.0)
parser.add_argument("--action_scale_pos", type=float, default=0.02, help="★ 이 모델은 0.02.")
parser.add_argument("--action_scale_rot", type=float, default=0.05)
parser.add_argument("--num_rounds", type=int, default=2)
parser.add_argument("--same_side", action="store_true")
parser.add_argument("--cache_size", type=int, default=4000)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True                 # 녹화(rgb_array)엔 필수 (headless여도)
app = AppLauncher(args); sim_app = app.app

import torch, numpy as np, cv2
from dual_arm_transport_env3 import DualrobotEnv, DualrobotCfg
import mlp_policy
import isaaclab.utils.math as math_utils

dev = "cuda" if torch.cuda.is_available() else "cpu"

if args.grasp_preset:
    N = int(torch.load(args.grasp_preset, map_location="cpu")["d"].numel())
else:
    N = 8

cfg = DualrobotCfg()
cfg.scene.num_envs = N
cfg.scene.env_spacing = args.env_spacing
cfg.viewer.resolution = (args.width, args.height)
cfg.vary_grasp = True
cfg.grasp_same_side = args.same_side
cfg.pose_cache_size = args.cache_size
cfg.n_obstacles = 0
if args.grasp_preset:
    cfg.grasp_preset_path = args.grasp_preset
env = DualrobotEnv(cfg, render_mode="rgb_array")     # ★ rgb_array 녹화 모드
A = env.cfg.action_space
POS_T, ROT_T = 0.02, math.radians(10)
settle = getattr(env, "SETTLE_STEPS_AT_RESET", 0)
if hasattr(env, "_obstacle_curr_frac"):
    env._obstacle_curr_frac = 0.0

b_d = env._grasp_bucket_d.detach().cpu(); b_th = env._grasp_bucket_theta.detach().cpu()
env_bucket = env._grasp_bucket_idx.detach().to(dev)
origins = env.scene.env_origins

# 모델 로드
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
print(f"▶ 녹화: {os.path.basename(args.model_path)}  파지={src}  envs={N}  → {args.out}")

writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))


def look_at(i):
    o = origins[i].tolist()
    env.sim.set_camera_view(eye=(o[0]+1.8, o[1]+1.8, o[2]+1.4), target=(o[0], o[1], o[2]+0.4))


env.reset(); batch = env._build_policy_batch()
rr_min_pos = torch.full((N,), float("inf"), device=dev)
rr_min_rot = torch.full((N,), float("inf"), device=dev)
rr_len = torch.zeros(N, dtype=torch.long, device=dev)
t0 = time.time()
for i in range(N):                                   # 파지(=env)마다 한 세그먼트
    look_at(i)
    b = int(env_bucket[i].item()); dd = float(b_d[b]); th = float(b_th[b]) * 57.3
    seg_best_pos, seg_best_rot, seg_success = 9.9, 999.0, False
    for s in range(args.steps_per_env):
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
        # 녹화 대상 env i 통계
        cp = float(rr_min_pos[i].item()) * 100; cr = math.degrees(float(rr_min_rot[i].item()))
        seg_best_pos = min(seg_best_pos, cp); seg_best_rot = min(seg_best_rot, cr)
        if cp < POS_T*100 and cr < math.degrees(ROT_T):
            seg_success = True
        done = term | trunc
        if done.any():
            for j in done.nonzero(as_tuple=True)[0].tolist():
                rr_min_pos[j] = float("inf"); rr_min_rot[j] = float("inf"); rr_len[j] = 0
        # 프레임 캡처 + 오버레이
        frame = env.render()
        if frame is None or frame.size == 0:
            batch = env._build_policy_batch(); continue
        img = np.ascontiguousarray(frame[:, :, :3])
        col = (0, 220, 0) if seg_success else (255, 210, 0)
        cv2.putText(img, f"grasp {i+1}/{N}  d={dd:.3f}m  theta={th:+.0f}deg", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"min_pos={seg_best_pos:.1f}cm  min_rot={seg_best_rot:.0f}deg  "
                         f"{'REACHED' if seg_success else '...'}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2, cv2.LINE_AA)
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        batch = env._build_policy_batch()
    print(f"  [{i+1}/{N}] bkt{b} d={dd:.3f} θ={th:+.0f}°  "
          f"{'✅REACHED' if seg_success else '❌미도달'}  min_pos={seg_best_pos:.1f}cm min_rot={seg_best_rot:.0f}°")

writer.release()
print(f"✅ 저장: {args.out}  ({time.time()-t0:.0f}s)")
env.close(); sim_app.close()
