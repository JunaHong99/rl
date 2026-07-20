"""파지 변이 육안 확인용 뷰어 — env **1개**만 띄워 파지 하나를 자세히 봄 (Isaac GUI, --headless 금지).

8개를 한 번에 띄우면 렉 → 한 번에 1개. --index로 어떤 파지를 볼지 선택하고, 바꿔가며 재실행.
zero-action으로 자세 동결 → 카메라 돌려 관찰.

권장 흐름 (저장 → 확인):
  python gen_grasp.py --n 8 --out grasps.pt         # 파지 세트 생성·저장 (Isaac 불필요)
  python -u view_grasp.py --load grasps.pt --index 0    # 0번 파지 1개만 렌더
  python -u view_grasp.py --load grasps.pt --index 4    # 4번으로 바꿔 재실행

저장 없이 즉석(랜덤): --index를 seed로 사용 → index 바꾸면 다른 랜덤 파지 1개.
  export LD_LIBRARY_PATH=/home/hjh/anaconda3/envs/env-isaaclab/lib:$LD_LIBRARY_PATH
  python -u view_grasp.py --index 0
종료: 뷰포트 닫기 또는 Ctrl-C.
"""
import argparse, math, os
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--load", type=str, default=None, help="gen_grasp.py로 저장한 파지 세트(.pt). 그 중 --index 하나만 렌더.")
parser.add_argument("--index", type=int, default=0, help="볼 파지 번호(--load면 파일 내 인덱스, 랜덤이면 seed).")
parser.add_argument("--cache_size", type=int, default=1000, help="초기포즈 캐시 크기(뷰어 1개는 작게=빠름).")
parser.add_argument("--same_side", action="store_true", help="두 파지점이 베이스축 같은 편에 오도록(straddle 배제).")
parser.add_argument("--max_steps", type=int, default=0, help="0이면 무한 렌더(수동 종료).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.headless:
    print("⚠️  --headless면 화면이 안 뜹니다. 육안 확인이면 --headless 빼세요.")
app = AppLauncher(args); sim_app = app.app

import torch
from dual_arm_transport_env3 import DualrobotEnv, DualrobotCfg

cfg = DualrobotCfg()

# 볼 파지 하나(d,θ) 결정 → 1엔트리 임시 preset으로 저장(파일명에 태그 포함 → 캐시 충돌 방지).
th_cap = min(float(cfg.grasp_theta_max), 0.999 * (math.pi / 4.0))
if args.load:
    blob = torch.load(args.load, map_location="cpu")
    n = int(blob["d"].numel())
    i = args.index % n
    d_i = blob["d"][i:i+1].clone().float()
    th_i = blob["theta"][i:i+1].clone().float().clamp(-th_cap, th_cap)
    tag = f"{os.path.basename(args.load).replace('.','_')}_{i}"
    src = f"{args.load} #{i}/{n}"
else:
    d_lo, d_hi = float(cfg.grasp_d_range[0]), float(cfg.grasp_d_range[1])
    g = torch.Generator().manual_seed(args.index)                 # index=seed → 바꾸면 다른 파지
    d_i = d_lo + torch.rand(1, generator=g) * (d_hi - d_lo)
    th_i = (torch.rand(1, generator=g) * 2.0 - 1.0) * th_cap
    tag = f"rand{args.index}"
    src = f"random seed={args.index}"

tmp = f"/tmp/_view_grasp_{tag}{'_ss' if args.same_side else ''}.pt"
torch.save({"d": d_i, "theta": th_i}, tmp)

cfg.scene.num_envs = 1                 # ★ 한 번에 1개만 (렉 방지)
cfg.vary_grasp = True                  # 파지 변이 ON
cfg.n_obstacles = 0                    # 장애물 없이 파지만
cfg.pose_cache_size = args.cache_size  # 1개라 작게 → 캐시 생성 빠름
cfg.grasp_same_side = args.same_side   # 두 파지점 베이스축 같은 편(straddle 배제)
cfg.grasp_preset_path = tmp            # 고른 파지 1개 로드
env = DualrobotEnv(cfg, render_mode=None)
A = env.cfg.action_space

env.reset()
zero_act = torch.zeros(1, A, device=env.device)

print("\n===== 렌더 중인 파지 =====")
print(f"  src={src}  same_side={args.same_side}")
print(f"  d={float(env._grasp_d[0]):.3f}m  θ={float(env._grasp_theta[0]):+.3f}rad ({float(env._grasp_theta[0])*57.3:+.1f}°)")
print("  (env 1개, zero-action 동결. 다른 파지 보려면 --index 바꿔 재실행.)")

t = 0
while sim_app.is_running():
    env.step(zero_act)
    t += 1
    if args.max_steps and t >= args.max_steps:
        break

env.close()
sim_app.close()
