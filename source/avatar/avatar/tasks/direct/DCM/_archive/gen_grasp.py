"""파지 세트 생성 + 저장 (Isaac 불필요). view_grasp.py로 로드해 육안 확인.

N개의 (d, θ) 파지를 샘플해 .pt로 저장. 저장 포맷: {'d': (N,), 'theta': (N,), 'meta': {...}}.
제약: a1·a2>0 (=cos(2θ)>0 → |θ|<45°). th_max로 clamp해 항상 보장.
env는 grasp_preset_path로 이 파일을 로드 → 파일의 (d,θ)를 버킷으로 사용(랜덤 샘플링 대신).

사용:
  python gen_grasp.py --n 8 --seed 0 --out grasps.pt         # 8개 랜덤 파지 저장
  python gen_grasp.py --n 12 --d_range 0.25 0.40 --theta_max 0.70 --out grasps.pt
그다음 확인:
  python -u view_grasp.py --load grasps.pt
"""
import argparse, math, torch

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=8, help="생성할 파지 수")
p.add_argument("--seed", type=int, default=0, help="샘플 재현 seed")
p.add_argument("--d_range", type=float, nargs=2, default=[0.25, 0.40], help="파지 거리 d ∈ [lo,hi] [m]")
p.add_argument("--theta_max", type=float, default=0.70, help="tilt |θ| 최대 [rad] (<π/4로 clamp됨)")
p.add_argument("--out", type=str, default="grasps.pt", help="저장 경로 (.pt)")
args = p.parse_args()

d_lo, d_hi = float(args.d_range[0]), float(args.d_range[1])
th_cap = min(float(args.theta_max), 0.999 * (math.pi / 4.0))   # a1·a2>0 보장
gen = torch.Generator().manual_seed(args.seed)

du = torch.rand(args.n, generator=gen)
tu = torch.rand(args.n, generator=gen)
d = d_lo + du * (d_hi - d_lo)                    # (N,) ∈ [lo,hi]
theta = (tu * 2.0 - 1.0) * th_cap                # (N,) ∈ [-cap,+cap]

# a1·a2 = cos(2θ) 검증 (approach축 내적)
dots = torch.cos(2.0 * theta)
assert bool((dots > 0).all()), "a1·a2>0 위반 (theta_max 확인)"

torch.save({
    "d": d, "theta": theta,
    "meta": {"n": args.n, "seed": args.seed, "d_range": [d_lo, d_hi], "theta_cap": th_cap},
}, args.out)

print(f"\n===== {args.n}개 파지 저장 → {args.out} =====")
print(f"  d∈[{d_lo:.2f},{d_hi:.2f}]  |θ|≤{th_cap:.3f}rad  seed={args.seed}  a1·a2 min={float(dots.min()):.3f}")
for i in range(args.n):
    print(f"  #{i:2d}: d={float(d[i]):.3f}m  θ={float(theta[i]):+.3f}rad ({float(theta[i])*57.3:+.1f}°)  a1·a2={float(dots[i]):.3f}")
print(f"\n확인:  python -u view_grasp.py --load {args.out}")
