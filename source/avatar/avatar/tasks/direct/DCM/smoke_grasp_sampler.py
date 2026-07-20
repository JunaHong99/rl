"""Isaac-free 스모크: 파지 변이 (d, θ) 기하 + HARD 제약 a1·a2>0 검증.

env3.py의 grasp_quat_ry_rxpi / grasp_approach_axis 로직을 그대로 재현(Isaac import 없이)해
버킷 (d,θ) 샘플 다수에 대해:
  - a1·a2 > 0 (모든 env, HARD 제약)
  - off_pos_1 x = -d, off_pos_2 x = +d (오프셋 일관성)
  - d ∈ grasp_d_range, |θ| ≤ theta_cap
을 assert. env의 _compute_grasp_specs / _grasp_bucket_grasp_offs 와 동일한 수식.

실행: /home/hjh/anaconda3/envs/env-isaaclab_copy/bin/python smoke_grasp_sampler.py
"""
import math
import torch

TCP_OFFSET = 0.1034   # VectorizedPoseSampler.TCP_OFFSET


def grasp_quat_ry_rxpi(theta: torch.Tensor) -> torch.Tensor:
    """(N,) θ → (N,4) Ry(θ)·Rx(π) = (0, cos(θ/2), 0, -sin(θ/2)) (wxyz)."""
    h = 0.5 * theta
    z = torch.zeros_like(theta)
    return torch.stack([z, torch.cos(h), z, -torch.sin(h)], dim=-1).contiguous()


def grasp_approach_axis(theta: torch.Tensor) -> torch.Tensor:
    """(N,) θ → (N,3) approach축 = Ry(θ)·Rx(π)·(0,0,1) = (-sinθ, 0, -cosθ)."""
    z = torch.zeros_like(theta)
    return torch.stack([-torch.sin(theta), z, -torch.cos(theta)], dim=-1).contiguous()


def _quat_apply(q, v):
    """rotate v (N,3) by quat q (N,4) wxyz — approach축 재계산 교차검증용(수식 독립)."""
    w = q[:, :1]; u = q[:, 1:]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def main():
    # cfg 기본값 (dual_arm_transport_cfg.py 와 일치)
    d_lo, d_hi = 0.25, 0.40
    theta_max = 0.70
    n_buckets = 8
    num_envs = 4096

    th_cap = min(theta_max, 0.999 * (math.pi / 4.0))

    # ── 버킷 (d,θ) (env._compute_grasp_specs 와 동일: seed 12345, 균등) ──
    gen = torch.Generator(device="cpu").manual_seed(12345)
    du = torch.rand(n_buckets, generator=gen)
    tu = torch.rand(n_buckets, generator=gen)
    bucket_d = d_lo + du * (d_hi - d_lo)
    bucket_theta = (tu * 2.0 - 1.0) * th_cap

    # env → 버킷 라운드로빈
    bucket_idx = torch.arange(num_envs) % n_buckets
    d = bucket_d[bucket_idx]
    theta = bucket_theta[bucket_idx]

    q1 = grasp_quat_ry_rxpi(theta)
    q2 = grasp_quat_ry_rxpi(-theta)

    # ── 1. HARD 제약 a1·a2 > 0 (수식 + quat rotate 두 방식 교차검증) ──
    a1 = grasp_approach_axis(theta)
    a2 = grasp_approach_axis(-theta)
    dots = (a1 * a2).sum(dim=-1)
    assert bool((dots > 0).all()), f"a1·a2>0 위반: min={float(dots.min()):.5f}"
    # 교차검증: quat로 (0,0,1) 회전한 approach축이 수식과 일치
    zaxis = torch.tensor([0.0, 0.0, 1.0]).expand(num_envs, 3).contiguous()
    a1_q = _quat_apply(q1, zaxis)
    a2_q = _quat_apply(q2, zaxis)
    assert torch.allclose(a1_q, a1, atol=1e-5), "approach축 수식≠quat"
    assert torch.allclose(a2_q, a2, atol=1e-5), "approach축 수식≠quat"
    dots_q = (a1_q * a2_q).sum(dim=-1)
    assert bool((dots_q > 0).all()), f"quat a1·a2>0 위반: min={float(dots_q.min()):.5f}"
    # dot == cos(2θ) 이론값 확인
    assert torch.allclose(dots, torch.cos(2.0 * theta), atol=1e-5), "dot≠cos(2θ)"

    # ── 2. 오프셋 일관성: off_pos_1 x=-d, off_pos_2 x=+d, z=TCP ──
    off_pos_1 = torch.stack([-d, torch.zeros_like(d), torch.full_like(d, TCP_OFFSET)], dim=1)
    off_pos_2 = torch.stack([+d, torch.zeros_like(d), torch.full_like(d, TCP_OFFSET)], dim=1)
    assert torch.allclose(off_pos_1[:, 0], -d), "off_pos_1 x != -d"
    assert torch.allclose(off_pos_2[:, 0], +d), "off_pos_2 x != +d"
    assert torch.allclose(off_pos_1[:, 1], torch.zeros_like(d)), "off_pos_1 y != 0"
    assert torch.allclose(off_pos_1[:, 2], torch.full_like(d, TCP_OFFSET)), "off_pos_1 z != TCP"

    # ── 3. 범위: d ∈ [lo,hi], |θ| ≤ cap ──
    assert bool((d >= d_lo - 1e-6).all() and (d <= d_hi + 1e-6).all()), "d 범위 벗어남"
    assert bool((theta.abs() <= th_cap + 1e-6).all()), "|θ| > cap"

    # ── 4. 미러 대칭: q2 = grasp(-θ) 이고 off_pos_2 = -off_pos_1 (x부호) ──
    assert torch.allclose(off_pos_2[:, 0], -off_pos_1[:, 0]), "미러 대칭 위반"
    assert torch.allclose(q2, grasp_quat_ry_rxpi(-theta)), "q2 미러 위반"

    # ── 5. θ=0 특수 케이스 = Rx(π) top-grasp (기본과 동일) ──
    z0 = torch.zeros(1)
    assert torch.allclose(grasp_quat_ry_rxpi(z0), torch.tensor([[0.0, 1.0, 0.0, 0.0]])), "θ=0 ≠ Rx(π)"

    print("[smoke_grasp_sampler] ALL PASS")
    print(f"  n_buckets={n_buckets} num_envs={num_envs}")
    print(f"  d range sampled: [{float(d.min()):.3f}, {float(d.max()):.3f}] (cfg [{d_lo},{d_hi}])")
    print(f"  |θ| max: {float(theta.abs().max()):.3f} (cap {th_cap:.3f})")
    print(f"  a1·a2 (=cos2θ): min={float(dots.min()):.4f}, max={float(dots.max()):.4f} (>0 ✓)")
    print(f"  bucket d: {[round(float(x),3) for x in bucket_d]}")
    print(f"  bucket θ: {[round(float(x),3) for x in bucket_theta]}")


if __name__ == "__main__":
    main()
