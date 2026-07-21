"""Phase1 A 제어 substrate 스모크 (Isaac 필요) — joint-velocity servo 안정성 검증(RL 없이).

joint_action=True로 env를 띄우고:
  Phase A (hold): dq_des=0 → 팔이 중력 버티고 정지 유지하나 (중력보상 sign/servo 안정).
  Phase B (track): 작은 dq_des 명령 → servo가 추종하고 시스템이 *발산 안 하나*(bounded).
                   (양팔 독립 dq는 용접이 저항해 내력↑ = 예상 — 여기선 '터지지 않음'이 핵심.)
목적: 그래프/RL 대공사 전에 관절속도 제어가 닫힌사슬에서 안정한지 값싸게 확인.

사용:
  export LD_LIBRARY_PATH=/home/hjh/anaconda3/envs/env-isaaclab/lib:$LD_LIBRARY_PATH
  python -u hold_test_joint.py --num_envs 64 --headless
합격: hold 시 관절 drift<0.1rad·|q̇|<0.1·rod z 유지, track 시 |q̇|<10(발산X)·NaN 없음.
"""
import argparse, math
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--hold_steps", type=int, default=40)
parser.add_argument("--track_steps", type=int, default=40)
parser.add_argument("--dq_test", type=float, default=0.15, help="track 단계 명령 dq [rad/s] (관절1 both arm).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args); sim_app = app.app

import torch
from dual_arm_transport_env3 import DualrobotEnv, DualrobotCfg

dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = DualrobotCfg()
cfg.scene.num_envs = args.num_envs
cfg.joint_action = True              # ★ joint-velocity 액션
cfg.action_space = 14                # 팔당 7
cfg.n_obstacles = 0
env = DualrobotEnv(cfg, render_mode=None)
B = args.num_envs
J1, J2 = env.robot_1_joint_ids, env.robot_2_joint_ids


def jstate():
    q1 = env.robot_1.data.joint_pos[:, J1]; q2 = env.robot_2.data.joint_pos[:, J2]
    qd1 = env.robot_1.data.joint_vel[:, J1]; qd2 = env.robot_2.data.joint_vel[:, J2]
    return torch.cat([q1, q2], 1), torch.cat([qd1, qd2], 1)


def f_int():
    w1, w2 = env._get_grasp_wrenches()
    return (0.5 * (w1[:, :3] - w2[:, :3])).norm(dim=-1)   # (B,) N


def rod_z():
    return env.rod.data.root_pos_w[:, 2]


env.reset()
q0, _ = jstate(); z0 = rod_z().clone()

# ── Phase A: hold (dq=0) ──
zero = torch.zeros(B, 14, device=dev)
for _ in range(args.hold_steps):
    env.step(zero)
qh, qdh = jstate(); fih = f_int(); zh = rod_z()
drift = (qh - q0).abs().max().item()
jvel = qdh.abs().max().item()
dz = (zh - z0).abs().max().item()
print("\n===== Phase A: HOLD (dq_des=0) =====")
print(f"  관절 drift max={drift:.3f} rad   |q̇| max={jvel:.3f} rad/s")
print(f"  rod z 변화 max={dz*100:.1f} cm   f_int mean={fih.mean():.1f}N max={fih.max():.1f}N")
hold_ok = (drift < 0.1) and (jvel < 0.1) and (dz < 0.05) and not torch.isnan(qh).any()
print(f"  => HOLD {'PASS ✓' if hold_ok else 'FAIL ✗'} (drift<0.1, q̇<0.1, dz<5cm)")

# ── Phase B: track (관절1 both arm에 +dq_test) ──
act = torch.zeros(B, 14, device=dev)
act[:, 0] = args.dq_test / cfg.joint_vel_scale     # dq_des = scale*action = dq_test
act[:, 7] = args.dq_test / cfg.joint_vel_scale
fi_max = 0.0; jv_max = 0.0; nan = False
for _ in range(args.track_steps):
    env.step(act)
    _, qd = jstate()
    jv_max = max(jv_max, qd.abs().max().item())
    fi_max = max(fi_max, f_int().max().item())
    if torch.isnan(qd).any():
        nan = True; break
# 관절1 실제 속도 vs 명령
_, qd = jstate()
track_j1 = qd[:, 0].mean().item()
print("\n===== Phase B: TRACK (관절1 dq_des=%.2f) =====" % args.dq_test)
print(f"  관절1 실제 q̇ mean={track_j1:.3f} (명령 {args.dq_test:.3f})")
print(f"  |q̇| max={jv_max:.2f} rad/s   f_int max={fi_max:.1f}N   NaN={nan}")
track_ok = (jv_max < 10.0) and (not nan)
print(f"  => TRACK {'PASS ✓' if track_ok else 'FAIL ✗'} (발산X: |q̇|<10, NaN 없음)")

print(f"\n판정: 제어 substrate {'안정 — 조각2(RL) 진행 가능 ✓' if (hold_ok and track_ok) else '불안정 — kd/scale/중력sign 조정 필요 ✗'}")
env.close(); sim_app.close()
