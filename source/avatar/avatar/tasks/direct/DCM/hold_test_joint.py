"""Phase1 A 제어 substrate 스모크 — joint 위치-PD servo 안정성 + 중력보상 부호 확정(RL 없이).

HOLD 실패의 원인은 대개 중력보상 G. q_des=q_start면 위치오차=0 → τ=G뿐이라, HOLD가 곧
'G가 중력을 상쇄하나' 테스트. 그래서 **grav_sign ±1 둘 다 자동 비교** → 무너지지 않는 쪽 확정.

사용:
  export LD_LIBRARY_PATH=/home/hjh/anaconda3/envs/env-isaaclab/lib:$LD_LIBRARY_PATH
  python -u hold_test_joint.py --num_envs 64 --headless
합격: 한 부호에서 HOLD drift<0.1·rod z 유지 → 그 부호로 joint_grav_sign 확정. TRACK 발산X.
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


def jstate():
    q = torch.cat([env.robot_1.data.joint_pos[:, J1], env.robot_2.data.joint_pos[:, J2]], 1)
    qd = torch.cat([env.robot_1.data.joint_vel[:, J1], env.robot_2.data.joint_vel[:, J2]], 1)
    return q, qd


def f_int():
    w1, w2 = env._get_grasp_wrenches()
    return (0.5 * (w1[:, :3] - w2[:, :3])).norm(dim=-1)


def rod_z():
    return env.rod.data.root_pos_w[:, 2]


def grav_mag():
    g1 = env._gravity_comp(env.robot_1, J1); g2 = env._gravity_comp(env.robot_2, J2)
    return torch.cat([g1, g2], 1).abs().mean().item()


def run_hold(sign):
    cfg.joint_grav_sign = float(sign)
    env.reset()
    q0, _ = jstate(); z0 = rod_z().clone()
    gm = grav_mag()
    for _ in range(args.hold_steps):
        env.step(zero)
    qh, qdh = jstate(); fih = f_int(); zh = rod_z()
    return {
        "drift": (qh - q0).abs().max().item(), "jvel": qdh.abs().max().item(),
        "dz": (zh - z0).abs().max().item(), "fint": fih.mean().item(),
        "gmag": gm, "nan": bool(torch.isnan(qh).any()),
    }


print("\n===== 중력보상 부호 자동 비교 (HOLD) =====")
res = {}
for s in (+1.0, -1.0):
    r = run_hold(s); res[s] = r
    ok = (r["drift"] < 0.1) and (r["dz"] < 0.05) and not r["nan"]
    print(f"  grav_sign={s:+.0f}: |G|평균={r['gmag']:.1f}Nm  drift={r['drift']:.3f}rad  "
          f"rodΔz={r['dz']*100:.1f}cm  |q̇|={r['jvel']:.2f}  f_int={r['fint']:.0f}N  "
          f"=> {'HOLD ✓' if ok else 'FAIL ✗'}")

best = min(res, key=lambda s: res[s]["drift"])
best_ok = (res[best]["drift"] < 0.1) and (res[best]["dz"] < 0.05)
print(f"\n  → 더 나은 부호: grav_sign={best:+.0f}  ({'안정 ✓' if best_ok else '둘 다 불안정 — kp/clamp/method 재검토 ✗'})")

# ── TRACK: 좋은 부호로 관절1 Δq 램프 ──
cfg.joint_grav_sign = float(best)
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
print(f"\n===== TRACK (grav_sign={best:+.0f}, 관절1 Δq 램프 {cfg.joint_dq_scale}/step) =====")
print(f"  관절1 이동={moved:.2f}rad (목표~{cfg.joint_dq_scale*args.track_steps:.2f})  "
      f"|q̇|max={jv_max:.2f}  f_int max={fi_max:.0f}N  NaN={nan}")
track_ok = (jv_max < 10.0) and (not nan) and (abs(moved) > 0.1)
print(f"  => TRACK {'PASS ✓' if track_ok else 'FAIL ✗'}")

print(f"\n판정: {'grav_sign=%+.0f로 substrate 안정 → cfg 반영 후 조각2(RL) ✓' % best if (best_ok and track_ok) else '추가 조정 필요 (아래 출력 공유) ✗'}")
env.close(); sim_app.close()
