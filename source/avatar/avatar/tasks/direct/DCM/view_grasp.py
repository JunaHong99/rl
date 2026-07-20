"""파지 변이(vary_grasp=True) 육안 확인용 뷰어 (Isaac GUI 필요, --headless 붙이지 말 것).

per-env (d,θ) 서로 다른 파지로 env를 띄우고, target을 rod 초기 자세에 동결(zero action)한 채
계속 렌더. Isaac 뷰포트에서 각 env(그리드 배치)의 파지 위치/각도를 직접 눈으로 비교.

권장 흐름 (저장 → 확인):
  python gen_grasp.py --n 8 --out grasps.pt            # 파지 세트 생성·저장 (Isaac 불필요)
  python -u view_grasp.py --load grasps.pt             # 저장된 세트 시각화 (GUI)

즉석 랜덤(저장 없이):
  export LD_LIBRARY_PATH=/home/hjh/anaconda3/envs/env-isaaclab/lib:$LD_LIBRARY_PATH
  python -u view_grasp.py --num_envs 8 --env_spacing 3.0
종료: 뷰포트 닫기 또는 Ctrl-C.
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--load", type=str, default=None, help="gen_grasp.py로 저장한 파지 세트(.pt). 지정 시 env 수=파일 파지 수.")
parser.add_argument("--num_envs", type=int, default=8, help="띄울 env 수(=파지 샘플 수). --load 없을 때만 사용. 기본 8=버킷당 1개.")
parser.add_argument("--env_spacing", type=float, default=3.0, help="env 간 간격[m] (넓힐수록 안 겹침).")
parser.add_argument("--cache_size", type=int, default=2000, help="초기포즈 캐시 크기(뷰어는 작게=빠름). 학습은 100k.")
parser.add_argument("--max_steps", type=int, default=0, help="0이면 무한 렌더(수동 종료).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.headless:
    print("⚠️  --headless면 화면이 안 뜹니다. 육안 확인이면 --headless 빼세요.")
app = AppLauncher(args); sim_app = app.app

import torch
from dual_arm_transport_env3 import DualrobotEnv, DualrobotCfg

# --load면 파일 파지 수만큼 env를 띄워 세트 1:1 시각화 (라운드로빈이라 env=버킷수여야 각자 유일)
if args.load:
    n_preset = int(torch.load(args.load, map_location="cpu")["d"].numel())
    num_envs = n_preset
else:
    num_envs = args.num_envs

cfg = DualrobotCfg()
cfg.scene.num_envs = num_envs
cfg.scene.env_spacing = args.env_spacing
cfg.vary_grasp = True                 # ★ 파지 변이 ON
cfg.n_obstacles = 0                   # 장애물 없이 파지만
cfg.pose_cache_size = args.cache_size # 뷰어는 작게 → 캐시 생성 몇 초
if args.load:
    cfg.grasp_preset_path = args.load  # 랜덤 대신 저장된 (d,θ) 로드
env = DualrobotEnv(cfg, render_mode=None)
B = num_envs
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
