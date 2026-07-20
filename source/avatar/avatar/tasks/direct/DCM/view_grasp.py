"""파지 변이(vary_grasp=True) 육안 확인용 뷰어 (Isaac GUI 필요, --headless 붙이지 말 것).

per-env (d,θ) 서로 다른 파지로 env를 띄우고, target을 rod 초기 자세에 동결(zero action)한 채
계속 렌더. Isaac 뷰포트에서 각 env(그리드 배치)의 파지 위치/각도를 직접 눈으로 비교.

사용 (서버, GUI/x11 또는 로컬):
  export LD_LIBRARY_PATH=/home/hjh/anaconda3/envs/env-isaaclab/lib:$LD_LIBRARY_PATH
  python -u view_grasp.py --num_envs 8 --env_spacing 3.0
  # 버킷별 1개만 보고 싶으면 num_envs = grasp_n_buckets (기본 8)
  # 매 env 무작위 파지를 더 많이 보려면 --num_envs 16/32 (버킷 재사용)
종료: 뷰포트 닫기 또는 Ctrl-C.
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8, help="띄울 env 수(=파지 샘플 수). 기본 8=버킷당 1개.")
parser.add_argument("--env_spacing", type=float, default=3.0, help="env 간 간격[m] (넓힐수록 안 겹침).")
parser.add_argument("--max_steps", type=int, default=0, help="0이면 무한 렌더(수동 종료).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.headless:
    print("⚠️  --headless면 화면이 안 뜹니다. 육안 확인이면 --headless 빼세요.")
app = AppLauncher(args); sim_app = app.app

import torch
from dual_arm_transport_env3 import DualrobotEnv, DualrobotCfg

cfg = DualrobotCfg()
cfg.scene.num_envs = args.num_envs
cfg.scene.env_spacing = args.env_spacing
cfg.vary_grasp = True                 # ★ 파지 변이 ON
cfg.n_obstacles = 0                   # 장애물 없이 파지만
env = DualrobotEnv(cfg, render_mode=None)
B = args.num_envs
A = env.cfg.action_space

env.reset()
zero_act = torch.zeros(B, A, device=env.device)

# 각 env의 파지 파라미터 출력 (뷰포트 env 순서 = 그리드 순서)
print("\n===== 띄운 파지 (env별 d,θ) =====")
d = env._grasp_d; th = env._grasp_theta; bidx = env._grasp_bucket_idx
for i in range(B):
    print(f"  env {i:2d}: bucket={int(bidx[i])}  d={float(d[i]):.3f}m  θ={float(th[i]):+.3f}rad ({float(th[i])*57.3:+.1f}°)")
print("  (뷰포트에서 각 env는 그리드로 배치됨. zero-action으로 자세 동결 — 카메라 돌려서 비교하세요.)")

t = 0
while sim_app.is_running():
    env.step(zero_act)
    t += 1
    if args.max_steps and t >= args.max_steps:
        break

env.close()
sim_app.close()
