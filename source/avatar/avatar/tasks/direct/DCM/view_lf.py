"""리더-팔로워 학습 환경 육안 확인 (Isaac GUI, --headless 금지). env 1개.

--model_path 있으면 학습된 정책 구동(운반 재생), 없으면 랜덤 리더 Δq(팔로워 추종 확인).
매 에피소드 reset되며 rod가 목표(녹색 마커)로 가는지, 팔로워가 리더를 잘 추종하는지 관찰.

사용:
  export LD_LIBRARY_PATH=/home/hjh/anaconda3/envs/env-isaaclab/lib:$LD_LIBRARY_PATH
  python -u view_lf.py                                   # 랜덤 액션(팔로워 추종만 확인)
  python -u view_lf.py --model_path logs/<run>/model_final.pt   # 학습 정책 구동
종료: 뷰포트 닫기/Ctrl-C.
"""
import argparse, math, os
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default=None, help="있으면 학습 정책 구동, 없으면 랜덤 리더 Δq.")
parser.add_argument("--cache_size", type=int, default=2000)
parser.add_argument("--random_scale", type=float, default=0.3, help="랜덤 모드 리더 Δq 진폭(모델 없을 때).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.headless:
    print("⚠️  --headless면 화면이 안 뜹니다. 육안 확인이면 --headless 빼세요.")
app = AppLauncher(args); sim_app = app.app

import torch
from dual_arm_transport_env3 import DualrobotEnv, DualrobotCfg
import graph_converter as gc

dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = DualrobotCfg()
cfg.scene.num_envs = 1
cfg.leader_follower = True
cfg.action_space = 7
cfg.n_obstacles = 0
cfg.pose_cache_size = args.cache_size
gc.GLOBAL_FEATURE_DIM = 1 + 14 + 1 + 3 + 6      # train과 동일 (leader_follower)
env = DualrobotEnv(cfg, render_mode=None)
A = env.cfg.action_space

agent = None
if args.model_path:
    import mlp_policy
    sd = torch.load(os.path.abspath(args.model_path), map_location=dev, weights_only=False)["model"]
    scale = [float(cfg.joint_dq_scale)] * A
    agent = mlp_policy.MLPSACAgent(action_dim=A, action_scale=scale, hidden_dim=256,
                                   num_hidden_layers=2, use_full_state=False, use_lean_obstacle=False).to(dev)
    agent.load_state_dict(sd); agent.eval()
    print(f"▶ 정책 구동: {os.path.basename(args.model_path)}")
else:
    print(f"▶ 랜덤 리더 Δq (scale={args.random_scale}) — 팔로워 추종 확인용")

# env 0 원점 기준 카메라
o = env.scene.env_origins[0].tolist()
env.sim.set_camera_view(eye=(o[0]+2.0, o[1]+2.0, o[2]+1.6), target=(o[0], o[1], o[2]+0.4))

env.reset(); batch = env._build_policy_batch()
gen = torch.Generator(device=dev).manual_seed(0)
step = 0
while sim_app.is_running():
    with torch.no_grad():
        if agent is not None:
            action, _, _ = agent.actor.get_action_and_log_prob(batch, deterministic=True)
        else:
            action = (torch.rand(1, A, generator=gen, device=dev) * 2 - 1) * args.random_scale
        _, _, term, trunc, _ = env.step(action)
    batch = env._build_policy_batch()
    step += 1
    if (term | trunc).any():
        rod = env.rod.data.root_pos_w[0]; goal = env.goal_rod_marker.data.root_pos_w[0]
        err = (goal - rod).norm().item()
        print(f"  ep 종료(step {step}): rod-goal 오차 {err*100:.1f}cm  {'✅도달' if err<0.02 else ''}")

env.close(); sim_app.close()
