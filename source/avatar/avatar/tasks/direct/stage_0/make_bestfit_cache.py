"""
Reset 충격(fixed-joint snap) 감소용 cache 후처리.

진단 결과: reset 시 rod를 '샘플된 start_obj_pose'에 놓는데, IK로 푼 실제 손 위치와
~6mm 어긋나 fixed joint가 snap을 일으킴 (settle이 흡수하던 transient의 원천).
프레임은 정상(상수 bias 없음), IK iter↑도 무효(solver 바닥 6mm).

해법(①a, best-fit): rod를 '샘플 pose'가 아니라 'IK로 푼 실제 두 손 TCP'에 best-fit으로 배치.
  - 위치: 두 TCP의 중점
  - 회전: rod x축을 TCP-TCP 선에 정렬(+ sampled 회전의 roll 유지)
→ 잔여 snap 6.2mm → 2.5mm (~60% 감소). 잔여는 두 손 간 거리 오차(IK 부정확)라 best-fit 한계.

원본 cache는 보존하고 *.BESTFIT.pt로 저장. 채택하려면 학습/eval 전에 이 파일을 사용.
"""
import os, math, torch
from vectorized_pose_sampler import CachedPoseSampler, VectorizedPoseSampler

DEV = "cuda:0"
TCP = VectorizedPoseSampler.TCP_OFFSET
HW = 0.4


def mat_to_quat(R):
    """배치 회전행렬 (N,3,3) → 쿼터니언 wxyz (N,4). Shepperd 안정 버전."""
    N = R.shape[0]
    m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]
    t = m00 + m11 + m22
    q = torch.zeros(N, 4, device=R.device, dtype=R.dtype)
    c0 = t > 0
    c1 = (~c0) & (m00 >= m11) & (m00 >= m22)
    c2 = (~c0) & (~c1) & (m11 >= m22)
    c3 = (~c0) & (~c1) & (~c2)

    def fill(mask, w, x, y, z):
        s = torch.stack([w, x, y, z], dim=-1)
        s = s / s.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        q[mask] = s[mask]

    s0 = torch.sqrt((t + 1.0).clamp_min(1e-9)) * 2
    fill(c0, 0.25 * s0, (m21 - m12) / s0, (m02 - m20) / s0, (m10 - m01) / s0)
    s1 = torch.sqrt((1.0 + m00 - m11 - m22).clamp_min(1e-9)) * 2
    fill(c1, (m21 - m12) / s1, 0.25 * s1, (m01 + m10) / s1, (m02 + m20) / s1)
    s2 = torch.sqrt((1.0 - m00 + m11 - m22).clamp_min(1e-9)) * 2
    fill(c2, (m02 - m20) / s2, (m01 + m10) / s2, 0.25 * s2, (m12 + m21) / s2)
    s3 = torch.sqrt((1.0 - m00 - m11 + m22).clamp_min(1e-9)) * 2
    fill(c3, (m10 - m01) / s3, (m02 + m20) / s3, (m12 + m21) / s3, 0.25 * s3)
    return q


def bestfit_start_pose(base, cache, idx):
    """idx 슬라이스에 대해 best-fit start_obj_pose (pos+quat) 계산."""
    sp = cache["start_obj_pose"][idx]
    b1 = cache["base_pose_1"][idx]; b2 = cache["base_pose_2"][idx]
    q1 = cache["q_start_1"][idx]; q2 = cache["q_start_2"][idx]
    n = idx.numel()
    # FK로 실제 손 origin/rot (base-local) → 실제 TCP world
    zoff = torch.tensor([0, 0, TCP], device=DEV).repeat(n, 1).unsqueeze(2)
    p1, R1 = base.ik_solver.forward_kinematics(q1)
    p2, R2 = base.ik_solver.forward_kinematics(q2)
    tcp1l = p1 + torch.bmm(R1, zoff).squeeze(2)
    tcp2l = p2 + torch.bmm(R2, zoff).squeeze(2)
    Rb1 = base._quat_to_matrix(b1[:, 3:7]); Rb2 = base._quat_to_matrix(b2[:, 3:7])
    tcp1 = b1[:, :3] + torch.bmm(Rb1, tcp1l.unsqueeze(2)).squeeze(2)
    tcp2 = b2[:, :3] + torch.bmm(Rb2, tcp2l.unsqueeze(2)).squeeze(2)
    # 위치 = 중점
    mid = (tcp1 + tcp2) / 2
    # 회전: x축 = TCP-TCP 선, roll = sampled 회전 유지(Gram-Schmidt)
    nx = (tcp2 - tcp1)
    nx = nx / nx.norm(dim=1, keepdim=True).clamp_min(1e-9)
    Rs = base._quat_to_matrix(sp[:, 3:7])
    ys = Rs[:, :, 1]
    ys = ys - (ys * nx).sum(1, keepdim=True) * nx
    ny = ys / ys.norm(dim=1, keepdim=True).clamp_min(1e-9)
    nz = torch.cross(nx, ny, dim=1)
    Rn = torch.stack([nx, ny, nz], dim=2)  # columns
    quat = mat_to_quat(Rn)
    new_sp = torch.cat([mid, quat], dim=1)
    # 검증용 잔여 snap
    snap1 = torch.norm((mid - HW * nx) - tcp1, dim=1)
    snap2 = torch.norm((mid + HW * nx) - tcp2, dim=1)
    return new_sp, torch.cat([snap1, snap2]) * 1000


def main():
    s = CachedPoseSampler(device=DEV, cache_size=100_000, fixed_grasp_roll=True)
    base, cache = s._base, s.cache
    P = next(iter(cache.values())).shape[0]
    print(f"cache entries: {P:,}  TCP={TCP}  half_width={HW}")

    new_starts = torch.empty(P, 7, device=DEV)
    snaps = []
    B = 20000
    for i in range(0, P, B):
        idx = torch.arange(i, min(i + B, P), device=DEV)
        ns, snap = bestfit_start_pose(base, cache, idx)
        new_starts[idx] = ns
        snaps.append(snap)
    snap = torch.cat(snaps)
    # 기존 snap (비교)
    print(f"\n■ best-fit 적용 후 잔여 snap: mean {snap.mean():.2f}mm  median {snap.median():.2f}mm  "
          f"p90 {snap.quantile(0.9):.2f}mm  max {snap.max():.2f}mm  (기존 ~6.2mm)")

    # 저장 (원본 보존, 별도 파일)
    out = {k: v.clone() for k, v in cache.items()}
    out["start_obj_pose"] = new_starts
    out_cpu = {k: v.cpu() for k, v in out.items()}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, "pose_cache_100000.BESTFIT.pt")
    torch.save(out_cpu, path)
    print(f"✓ saved {path}  (원본 pose_cache_100000.pt 보존됨)")


if __name__ == "__main__":
    main()
