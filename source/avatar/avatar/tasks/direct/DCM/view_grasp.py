"""파지 변이 육안 확인용 뷰어 — 파지를 **하나씩 순차로** 보여줌 (Isaac GUI, --headless 금지).

8개를 한 번에 띄우면 렉 → 매번 env 1개만 만들고 3초 보여준 뒤 다음 파지로 교체(연속).
파지는 scene 빌드 때 용접에 baking되므로 파지마다 env를 새로(close→재생성) 만든다.
zero-action으로 자세 동결 → 각 파지를 3초간 관찰.

권장 흐름 (저장 → 확인):
  python gen_grasp.py --n 8 --out grasps.pt         # 파지 세트 생성·저장 (Isaac 불필요)
  python -u view_grasp.py --load grasps.pt           # 저장된 8개를 3초씩 순차 렌더

저장 없이 즉석(랜덤 --n개):
  export LD_LIBRARY_PATH=/home/hjh/anaconda3/envs/env-isaaclab/lib:$LD_LIBRARY_PATH
  python -u view_grasp.py --n 8
옵션: --seconds 3(파지당 초), --loop(끝나면 처음부터 반복), --same_side. 종료: 뷰포트 닫기/Ctrl-C.
"""
import argparse, math, os, time
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--load", type=str, default=None, help="gen_grasp.py로 저장한 파지 세트(.pt). 전부 순차 렌더.")
parser.add_argument("--n", type=int, default=8, help="--load 없을 때 랜덤 파지 개수(seed 0..n-1).")
parser.add_argument("--seconds", type=float, default=3.0, help="파지당 표시 시간[s].")
parser.add_argument("--loop", action="store_true", help="세트 끝나면 처음부터 반복.")
parser.add_argument("--cache_size", type=int, default=800, help="초기포즈 캐시 크기(1개라 작게=빠름).")
parser.add_argument("--same_side", action="store_true", help="두 파지점이 베이스축 같은 편에 오도록(straddle 배제).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.headless:
    print("⚠️  --headless면 화면이 안 뜹니다. 육안 확인이면 --headless 빼세요.")
app = AppLauncher(args); sim_app = app.app

import torch
from dual_arm_transport_env3 import DualrobotEnv, DualrobotCfg

# 볼 파지 목록 (d, θ) 구성
_cfg0 = DualrobotCfg()
th_cap = min(float(_cfg0.grasp_theta_max), 0.999 * (math.pi / 4.0))
if args.load:
    blob = torch.load(args.load, map_location="cpu")
    ds = blob["d"].float()
    ths = blob["theta"].float().clamp(-th_cap, th_cap)
    tags = [f"{os.path.basename(args.load).replace('.','_')}_{i}" for i in range(ds.numel())]
    srcs = [f"{os.path.basename(args.load)} #{i}/{ds.numel()}" for i in range(ds.numel())]
else:
    d_lo, d_hi = float(_cfg0.grasp_d_range[0]), float(_cfg0.grasp_d_range[1])
    ds, ths, tags, srcs = [], [], [], []
    for i in range(args.n):
        g = torch.Generator().manual_seed(i)
        ds.append(d_lo + torch.rand(1, generator=g).item() * (d_hi - d_lo))
        ths.append((torch.rand(1, generator=g).item() * 2.0 - 1.0) * th_cap)
        tags.append(f"rand{i}"); srcs.append(f"random seed={i}")
    ds = torch.tensor(ds); ths = torch.tensor(ths)

N = int(ds.numel())
print(f"\n===== 파지 {N}개를 {args.seconds:.0f}초씩 순차 렌더 (same_side={args.same_side}) =====")


def show_one(i):
    """파지 i를 위한 env를 만들어 args.seconds 동안 동결 렌더 후 close."""
    d_i = ds[i:i+1].clone(); th_i = ths[i:i+1].clone()
    tmp = f"/tmp/_view_grasp_{tags[i]}{'_ss' if args.same_side else ''}.pt"
    torch.save({"d": d_i, "theta": th_i}, tmp)

    cfg = DualrobotCfg()
    cfg.scene.num_envs = 1                 # ★ 항상 1개 (렉 방지)
    cfg.vary_grasp = True
    cfg.n_obstacles = 0
    cfg.pose_cache_size = args.cache_size
    cfg.grasp_same_side = args.same_side
    cfg.grasp_preset_path = tmp
    env = DualrobotEnv(cfg, render_mode=None)
    A = env.cfg.action_space
    env.reset()
    zero_act = torch.zeros(1, A, device=env.device)
    # env 0 원점 기준 카메라 배치
    o = env.scene.env_origins[0].tolist()
    env.sim.set_camera_view(eye=(o[0]+2.0, o[1]+2.0, o[2]+1.6), target=(o[0], o[1], o[2]+0.3))
    print(f"  [{i+1}/{N}] {srcs[i]}  d={float(d_i):.3f}m  θ={float(th_i):+.3f}rad ({float(th_i)*57.3:+.1f}°)")

    t0 = time.time()
    while sim_app.is_running() and (time.time() - t0) < args.seconds:
        env.step(zero_act)
    env.close()


first = True
while sim_app.is_running() and (first or args.loop):
    first = False
    for i in range(N):
        if not sim_app.is_running():
            break
        show_one(i)

sim_app.close()
