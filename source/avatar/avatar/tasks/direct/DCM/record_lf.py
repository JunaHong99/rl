"""리더-팔로워 정책 운반을 mp4 녹화 (headless 서버, --headless + --enable_cameras).

eval_lf의 정책 구동을 그대로 쓰되 DirectRLEnv.render()(rgb_array)로 프레임 잡아 cv2 mp4 저장.
env N개(멀리 배치) 중 카메라를 순차로 옮겨 한 에피소드씩 녹화 → 여러 운반 사례 수록.
프레임에 [rod-goal 오차, 도달여부] 오버레이.

사용 (⚠️ headless라도 --enable_cameras 필수):
  export LD_LIBRARY_PATH=/home/hjh/anaconda3/envs/env-isaaclab/lib:$LD_LIBRARY_PATH
  python -u record_lf.py --model_path logs/<run>/model_final.pt --headless --enable_cameras --out lf_replay.mp4
결과: lf_replay.mp4. scp로 받아서 재생.
"""
import argparse, math, os, time
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--out", type=str, default="lf_replay.mp4")
parser.add_argument("--n_clips", type=int, default=6, help="녹화할 env(=운반 사례) 수. 카메라가 순차로 비춤.")
parser.add_argument("--steps_per_env", type=int, default=40, help="env당 녹화 스텝(에피소드 ~30 + 여유).")
parser.add_argument("--fps", type=int, default=5, help="제어 5Hz라 fps=5면 실시간(1x), 10이면 2x배속.")
parser.add_argument("--res_w", type=int, default=1280)
parser.add_argument("--res_h", type=int, default=720)
parser.add_argument("--env_spacing", type=float, default=4.0)
parser.add_argument("--use_kin_graph", action="store_true", help="kinematic 그래프 + KinGNN 로드.")
parser.add_argument("--num_rounds", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args); sim_app = app.app

import torch, numpy as np, cv2
from dual_arm_transport_env3 import DualrobotEnv, DualrobotCfg
import mlp_policy, graph_converter as gc
import isaaclab.utils.math as math_utils

dev = "cuda" if torch.cuda.is_available() else "cpu"
N = args.n_clips
cfg = DualrobotCfg()
cfg.scene.num_envs = N
cfg.scene.env_spacing = args.env_spacing
cfg.viewer.resolution = (args.res_w, args.res_h)
cfg.leader_follower = True
cfg.action_space = 7
cfg.n_obstacles = 0
if args.use_kin_graph:
    cfg.use_kin_graph = True
else:
    gc.GLOBAL_FEATURE_DIM = 1 + 14 + 1 + 3 + 6
env = DualrobotEnv(cfg, render_mode="rgb_array")
A = env.cfg.action_space
POS_T, ROT_T = 0.02, math.radians(10)
origins = env.scene.env_origins

sd = torch.load(os.path.abspath(args.model_path), map_location=dev, weights_only=False)["model"]
scale = [float(cfg.joint_dq_scale)] * A
if args.use_kin_graph:
    import gnn_policy_kin
    agent = gnn_policy_kin.KinSACAgent(num_rounds=args.num_rounds, action_scale=scale).to(dev)
    print(f"🕸️ KinGNN 로드 (rounds={args.num_rounds})")
else:
    agent = mlp_policy.MLPSACAgent(action_dim=A, action_scale=scale, hidden_dim=256,
                                   num_hidden_layers=2, use_full_state=False, use_lean_obstacle=False).to(dev)
agent.load_state_dict(sd); agent.eval()
print(f"▶ 녹화: {os.path.basename(args.model_path)}  clips={N}  → {args.out}")

writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.res_w, args.res_h))


def look_at(i):
    o = origins[i].tolist()
    env.sim.set_camera_view(eye=(o[0]+1.8, o[1]+1.8, o[2]+1.4), target=(o[0], o[1], o[2]+0.4))


env.reset(); batch = env._build_policy_batch()
t0 = time.time()
for i in range(N):
    look_at(i)
    best_p = 9.9; best_r = 999.0; reached = False
    for s in range(args.steps_per_env):
        if not sim_app.is_running():
            break
        with torch.no_grad():
            action, _, _ = agent.actor.get_action_and_log_prob(batch, deterministic=True)
            _, _, term, trunc, _ = env.step(action)
        rod = env.rod.data.root_pos_w; goal = env.goal_rod_marker.data.root_pos_w
        perr = (goal[i] - rod[i]).norm().item()
        qd = math_utils.quat_mul(env.goal_rod_marker.data.root_quat_w[i:i+1],
                                 math_utils.quat_conjugate(env.rod.data.root_quat_w[i:i+1]))
        rerr = math.degrees(2.0 * math.atan2(qd[:, 1:4].norm().item(), abs(qd[0, 0].item())))
        best_p = min(best_p, perr); best_r = min(best_r, rerr)
        # 성공 판정 = eval_lf와 동일: 위치·자세 각각 독립 min이 임계 미만(동시 아님).
        reached = (best_p < POS_T) and (best_r < math.degrees(ROT_T))
        frame = env.render()
        if frame is None or frame.size == 0:
            batch = env._build_policy_batch(); continue
        img = np.ascontiguousarray(frame[:, :, :3])
        col = (0, 220, 0) if reached else (255, 210, 0)
        cv2.putText(img, f"clip {i+1}/{N}  kin leader-follower", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"pos={best_p*100:.1f}cm(<2)  rot={best_r:.0f}deg(<10)  {'REACHED' if reached else '...'}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        batch = env._build_policy_batch()
    # 실패 원인 진단: 위치는 됐는데 자세 때문인지
    cause = ""
    if not reached:
        cause = " (자세 미달)" if best_p < POS_T else (" (위치 미달)" if best_r < math.degrees(ROT_T) else " (둘다)")
    print(f"  [{i+1}/{N}] {'✅REACHED' if reached else '❌미도달'+cause}  "
          f"min pos={best_p*100:.1f}cm  min rot={best_r:.0f}deg")

writer.release()
print(f"✅ 저장: {args.out}  ({time.time()-t0:.0f}s)")
env.close(); sim_app.close()
