"""Phase1 A 제어 substrate 스모크 — joint 포지션(ImplicitActuator) 제어 안정성 (RL 없이).

action=Δq → q_des 누적 → set_joint_position_target. PhysX가 PD(kp/kd)+용접제약을 암시적 co-solve.
  Phase A (hold): dq=0 → q_des=q_start 고정 → 팔이 중력 버티고 rod 들고 정지 유지하나.
  Phase B (track): 작은 Δq 램프 → 관절이 부드럽게 이동 + 발산 없나.

사용:
  export LD_LIBRARY_PATH=/home/hjh/anaconda3/envs/env-isaaclab/lib:$LD_LIBRARY_PATH
  python -u hold_test_joint.py --num_envs 64 --headless
합격: HOLD drift<0.1·rod z 유지·f_int 낮음 / TRACK 이동 있음+발산X.
"""
import argparse, math
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--hold_steps", type=int, default=40)
parser.add_argument("--track_steps", type=int, default=40)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args); sim_app = app.app

import torch
from dual_arm_transport_env3 import DualrobotEnv, DualrobotCfg

dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = DualrobotCfg()
cfg.scene.num_envs = args.num_envs
cfg.joint_action = True
cfg.action_space = 14
cfg.n_obstacles = 0
env = DualrobotEnv(cfg, render_mode=None)
B = args.num_envs
J1, J2 = env.robot_1_joint_ids, env.robot_2_joint_ids
zero = torch.zeros(B, 14, device=dev)
print(f"  액추에이터 포지션 모드: kp={cfg.joint_kp} kd={cfg.joint_kd}  Δq/step={cfg.joint_dq_scale}")


def jstate():
    q = torch.cat([env.robot_1.data.joint_pos[:, J1], env.robot_2.data.joint_pos[:, J2]], 1)
    qd = torch.cat([env.robot_1.data.joint_vel[:, J1], env.robot_2.data.joint_vel[:, J2]], 1)
    return q, qd


def f_int():
    w1, w2 = env._get_grasp_wrenches()
    return (0.5 * (w1[:, :3] - w2[:, :3])).norm(dim=-1)


def rod_z():
    return env.rod.data.root_pos_w[:, 2]


# ── Phase A: HOLD ──
env.reset()
q0, _ = jstate(); z0 = rod_z().clone()
for _ in range(args.hold_steps):
    env.step(zero)
qh, qdh = jstate(); fih = f_int(); zh = rod_z()
drift = (qh - q0).abs().max().item(); jvel = qdh.abs().max().item()
dz = (zh - z0).abs().max().item()
print("\n===== Phase A: HOLD (dq=0) =====")
print(f"  drift={drift:.3f}rad  |q̇|={jvel:.3f}  rodΔz={dz*100:.1f}cm  f_int mean={fih.mean():.0f}N max={fih.max():.0f}N")
hold_ok = (drift < 0.1) and (jvel < 0.2) and (dz < 0.05) and not torch.isnan(qh).any()
print(f"  => HOLD {'PASS ✓' if hold_ok else 'FAIL ✗'}")

# ── Phase B: TRACK (관절1 Δq 램프) ──
env.reset()
q_before, _ = jstate()
act = torch.zeros(B, 14, device=dev); act[:, 0] = 1.0; act[:, 7] = 1.0
jv_max = 0.0; fi_max = 0.0; nan = False
for _ in range(args.track_steps):
    env.step(act)
    _, qd = jstate()
    jv_max = max(jv_max, qd.abs().max().item()); fi_max = max(fi_max, f_int().max().item())
    if torch.isnan(qd).any():
        nan = True; break
q_after, _ = jstate()
moved = (q_after[:, 0] - q_before[:, 0]).mean().item()
print("\n===== Phase B: TRACK (관절1 Δq 램프 %.3f/step) =====" % cfg.joint_dq_scale)
print(f"  관절1 이동={moved:.2f}rad (목표~{cfg.joint_dq_scale*args.track_steps:.2f})  "
      f"|q̇|max={jv_max:.2f}  f_int max={fi_max:.0f}N  NaN={nan}")
track_ok = (jv_max < 10.0) and (not nan) and (abs(moved) > 0.1)
print(f"  => TRACK {'PASS ✓' if track_ok else 'FAIL ✗'}")

print(f"\n판정: 제어 substrate {'안정 → 조각2(RL) 진행 ✓' if (hold_ok and track_ok) else 'kp/kd 조정 필요 (출력 공유) ✗'}")
env.close(); sim_app.close()
