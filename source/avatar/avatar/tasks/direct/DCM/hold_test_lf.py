"""리더-팔로워 추종 스모크 (Isaac 필요, RL 없이) — 팔로워 IK가 리더를 잘 따라가나 검증.

leader_follower=True로 env를 띄우고:
  Phase A (hold): 리더 Δq=0 → 리더/팔로워 q_start 유지 → rod 들고 정지, f_int 낮음.
  Phase B (leader move): 리더 관절에 Δq 램프 → rod가 움직임 → **팔로워가 추종해 f_int 낮게 유지되나**.
    (리더-팔로워의 핵심: 움직여도 내력이 안 튀어야 = 협응 구조적 보장. joint 독립제어와 대비.)

사용:
  export LD_LIBRARY_PATH=/home/hjh/anaconda3/envs/env-isaaclab/lib:$LD_LIBRARY_PATH
  python -u hold_test_lf.py --num_envs 64 --headless
합격: HOLD f_int 낮음·rod 유지 / MOVE 시 rod 움직임 + f_int 낮게 유지(팔로워 추종) + NaN 없음.
"""
import argparse, math
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--hold_steps", type=int, default=30)
parser.add_argument("--move_steps", type=int, default=40)
parser.add_argument("--vary_grasp", action="store_true", help="파지 변이 켜고 팔로워 추종 검증(per-env d,θ).")
parser.add_argument("--same_side", action="store_true")
parser.add_argument("--kp", type=float, default=None, help="포지션 stiffness override(원인 격리용).")
parser.add_argument("--kd", type=float, default=None)
parser.add_argument("--ik_iters", type=int, default=None)
parser.add_argument("--grav_comp", action="store_true", help="중력보상 feedforward ON.")
parser.add_argument("--follower_hold", action="store_true", help="팔로워 IK 끄고 q_start 고정(처짐 원인 판별).")
parser.add_argument("--no_gravity", action="store_true", help="중력 OFF(로봇+rod).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args); sim_app = app.app

import torch
from dual_arm_transport_env3 import DualrobotEnv, DualrobotCfg

dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = DualrobotCfg()
cfg.scene.num_envs = args.num_envs
cfg.leader_follower = True
cfg.action_space = 7
cfg.n_obstacles = 0
if args.kp is not None: cfg.joint_kp = args.kp
if args.kd is not None: cfg.joint_kd = args.kd
if args.ik_iters is not None: cfg.lf_ik_iters = args.ik_iters
if args.grav_comp: cfg.lf_grav_comp = True
if args.follower_hold: cfg.lf_follower_hold = True
if args.no_gravity: cfg.disable_gravity_all = True
if args.vary_grasp:
    cfg.vary_grasp = True
    cfg.grasp_same_side = args.same_side
env = DualrobotEnv(cfg, render_mode=None)
B = args.num_envs
J1, J2 = env.robot_1_joint_ids, env.robot_2_joint_ids
print(f"  포지션 액추에이터 kp={cfg.joint_kp} kd={cfg.joint_kd}, IK iters={cfg.lf_ik_iters}, Δq/step={cfg.joint_dq_scale}")


def f_int():
    w1, w2 = env._get_grasp_wrenches()
    return (0.5 * (w1[:, :3] - w2[:, :3])).norm(dim=-1)


def rod_z():
    return env.rod.data.root_pos_w[:, 2]


def rod_xy():
    return env.rod.data.root_pos_w[:, :2]


# ── Phase A: HOLD (리더 Δq=0) ──
env.reset()
zero = torch.zeros(B, 7, device=dev)
# ★ 초기 snap(reset 시 rod-hand 배치 불일치를 PhysX가 강제정렬) 흡수 후 z0 기록 → 순수 '이후 처짐'만 측정.
z_reset = rod_z().clone()                         # reset 직후(snap 전)
for _ in range(35):                               # env 내부 settle(30)+여유 → snap 완전 흡수
    env.step(zero)
z0 = rod_z().clone()                              # snap 후 기준
snap_dz = (z0 - z_reset).abs().mean().item()
print(f"  [snap] reset→settle후 rod z 이동 mean={snap_dz*100:.1f}cm (초기 배치 불일치 흡수량)")
fi_hold_max = 0.0; fi_hold_means = []
for _ in range(args.hold_steps):
    env.step(zero)
    fi = f_int(); fi_hold_max = max(fi_hold_max, fi.max().item()); fi_hold_means.append(fi.mean().item())
dz_all = (rod_z() - z0).abs()                                # (B,) env별 처짐
dz = dz_all.max().item()
dz_mean = dz_all.mean().item(); dz_p90 = dz_all.quantile(0.9).item()
n_bad = int((dz_all > 0.05).sum())                          # 5cm 초과 env 수
# 진단: 팔로워/리더 IK 목표 vs 실제 hand 추종오차
q1c = env.robot_1.data.joint_pos[:, J1]; q2c = env.robot_2.data.joint_pos[:, J2]
d_lead = (q1c - env._lf_q1_des).abs().max().item()
d_foll = (q2c - env._lf_q2_des).abs().max().item() if env._lf_q2_des is not None else -1
print(f"  [진단] 리더 관절 실제-명령 오차 max={d_lead:.3f}rad  팔로워 실제-IK목표 오차 max={d_foll:.3f}rad")
fi_hold_mean = sum(fi_hold_means[-10:]) / min(10, len(fi_hold_means))
print("\n===== Phase A: HOLD (리더 Δq=0) =====")
print(f"  rodΔz: mean={dz_mean*100:.1f}cm  p90={dz_p90*100:.1f}cm  max={dz*100:.1f}cm  (>5cm인 env: {n_bad}/{B})")
print(f"  f_int mean={fi_hold_mean:.0f}N (max spike={fi_hold_max:.0f}N)")
# 판정 = 평균 처짐(대부분 env) 기준. max는 극단 파지 outlier라 참고만.
hold_ok = (dz_mean < 0.05) and (fi_hold_mean < 100) and not torch.isnan(rod_z()).any()
print(f"  => HOLD {'PASS ✓' if hold_ok else 'FAIL ✗'} (mean rodΔz<5cm, mean f_int<100N)")

# ── Phase B: LEADER MOVE (리더 관절1,2에 Δq 램프) ──
env.reset()
xy0 = rod_xy().clone()
act = torch.zeros(B, 7, device=dev); act[:, 1] = 0.5; act[:, 3] = -0.5   # 리더 어깨/팔꿈치
fi_move_max = 0.0; fi_move_means = []; nan = False
for _ in range(args.move_steps):
    env.step(act)
    fi = f_int(); fi_move_max = max(fi_move_max, fi.max().item()); fi_move_means.append(fi.mean().item())
    if torch.isnan(rod_z()).any():
        nan = True; break
rod_moved = (rod_xy() - xy0).norm(dim=-1).mean().item()
fi_move_mean = sum(fi_move_means) / max(1, len(fi_move_means))
print("\n===== Phase B: LEADER MOVE (리더 관절 Δq 램프) =====")
print(f"  rod 이동={rod_moved*100:.1f}cm  f_int mean={fi_move_mean:.0f}N (max spike={fi_move_max:.0f}N)  NaN={nan}")
# 핵심: rod 이동(운반 성립) + 평균 내력 관리 + NaN 없음(폭발X). max spike는 transient.
move_ok = (rod_moved > 0.02) and (fi_move_mean < 300) and (not nan)
print(f"  => MOVE {'PASS ✓' if move_ok else 'FAIL ✗'} (rod 이동>2cm + mean f_int<300N + NaN없음)")

print(f"\n판정: 리더-팔로워 {'추종 정상 → RL 배선 진행 ✓' if (hold_ok and move_ok) else 'IK iters/Δq/게인 조정 필요 (출력 공유) ✗'}")
if fi_move_mean >= 300:
    print("  (mean f_int 높으면 → 팔로워 추종: lf_ik_iters↑ 또는 joint_dq_scale↓)")
env.close(); sim_app.close()
