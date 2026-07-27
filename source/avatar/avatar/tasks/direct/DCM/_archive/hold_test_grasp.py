"""서버 실행용 weld-hold 스모크 (Isaac 필요) — 파지 변이(vary_grasp=True) 용접 정합 검증.

vary_grasp=True로 env를 띄우고, target을 rod 초기 자세에 동결(zero action)한 채 N step 유지.
per-env (d,θ) 다양한 파지에서 rod가 두 panda_hand에 계속 물려 있는지(weld gap 작음) 확인.

측정 gap (양팔): |TCP_hand - grasppoint_rod|
  TCP_hand      = hand_pos_w + R_hand · (0,0,TCP)          (용접 body0 앵커의 world)
  grasppoint_rod= rod_pos_w  + R_rod  · (±d, 0, 0)          (용접 body1 앵커의 world)
용접이 성립하면 두 world 점이 일치 → gap ≈ 0. 파지 tilt θ가 커도 유지돼야 함.

사용:
  export LD_LIBRARY_PATH=/home/hjh/anaconda3/envs/env-isaaclab/lib:$LD_LIBRARY_PATH
  python -u hold_test_grasp.py --num_envs 256 --num_steps 120 --headless
합격 기준: 정착(settle) 후 평균 gap < 2cm, 최대 gap < 5cm (버킷별 통계 출력).
"""
import argparse, math
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--num_steps", type=int, default=120)
parser.add_argument("--gap_mean_thr", type=float, default=0.02, help="합격 평균 gap [m]")
parser.add_argument("--gap_max_thr", type=float, default=0.05, help="합격 최대 gap [m]")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args); sim_app = app.app

import torch
from dual_arm_transport_env3 import DualrobotEnv, DualrobotCfg

dev = "cuda" if torch.cuda.is_available() else "cpu"
TCP = 0.1034   # VectorizedPoseSampler.TCP_OFFSET

cfg = DualrobotCfg()
cfg.scene.num_envs = args.num_envs
cfg.vary_grasp = True                 # ★ 파지 변이 ON
# 장애물 없는 순수 hold 검증(운반 무관). 활성 0이면 base transport 보존 경로.
cfg.n_obstacles = 0
env = DualrobotEnv(cfg, render_mode=None)
B = args.num_envs
A = env.cfg.action_space


def _quat_apply(q, v):
    """rotate v (B,3) by quat q (B,4) wxyz."""
    w = q[:, :1]; u = q[:, 1:]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def weld_gaps():
    """양팔 용접 gap (B,) 각각 반환: |TCP_hand - grasppoint_rod|."""
    rod_p = env.rod.data.root_pos_w
    rod_q = env.rod.data.root_quat_w
    h1_p = env.robot_1.data.body_pos_w[:, env.ee_body_idx_1, :]
    h1_q = env.robot_1.data.body_quat_w[:, env.ee_body_idx_1, :]
    h2_p = env.robot_2.data.body_pos_w[:, env.ee_body_idx_2, :]
    h2_q = env.robot_2.data.body_quat_w[:, env.ee_body_idx_2, :]
    d = env._grasp_d.unsqueeze(-1)                                   # (B,1)
    z = torch.zeros_like(d)
    tcpv = torch.cat([z, z, torch.full_like(d, TCP)], dim=-1)         # (B,3)
    gp1 = rod_p + _quat_apply(rod_q, torch.cat([-d, z, z], dim=-1))   # rod 파지점(왼)
    gp2 = rod_p + _quat_apply(rod_q, torch.cat([+d, z, z], dim=-1))   # rod 파지점(오)
    tcp1 = h1_p + _quat_apply(h1_q, tcpv)                             # hand1 TCP
    tcp2 = h2_p + _quat_apply(h2_q, tcpv)                             # hand2 TCP
    return (tcp1 - gp1).norm(dim=-1), (tcp2 - gp2).norm(dim=-1)


env.reset()
zero_act = torch.zeros(B, A, device=env.device)
settle = getattr(env, "SETTLE_STEPS_AT_RESET", 0)

g1_final = g2_final = None
for t in range(args.num_steps):
    env.step(zero_act)
    if t == args.num_steps - 1:
        g1_final, g2_final = weld_gaps()

allg = torch.cat([g1_final, g2_final])
mean_gap = float(allg.mean()); max_gap = float(allg.max())

print("\n===== hold_test_grasp 결과 (vary_grasp=True) =====")
print(f"  envs={B} steps={args.num_steps} settle={settle} buckets={cfg.grasp_n_buckets}")
print(f"  weld gap: mean={mean_gap*1000:.2f}mm  max={max_gap*1000:.2f}mm")
# 버킷별 gap
bidx = env._grasp_bucket_idx
for b in range(cfg.grasp_n_buckets):
    m = (bidx == b)
    if not bool(m.any()):
        continue
    gb = torch.cat([g1_final[m], g2_final[m]])
    d_b = float(env._grasp_bucket_d[b]); th_b = float(env._grasp_bucket_theta[b])
    print(f"    bucket {b}: d={d_b:.3f} θ={th_b:+.3f}  gap mean={float(gb.mean())*1000:.2f}mm max={float(gb.max())*1000:.2f}mm")

ok = (mean_gap < args.gap_mean_thr) and (max_gap < args.gap_max_thr)
print(f"  => {'PASS ✓' if ok else 'FAIL ✗'} (mean<{args.gap_mean_thr*1000:.0f}mm, max<{args.gap_max_thr*1000:.0f}mm)")
env.close()
sim_app.close()
