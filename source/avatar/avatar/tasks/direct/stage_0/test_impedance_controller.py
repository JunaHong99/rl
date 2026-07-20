"""
Phase 3.1 단독 검증: Cooperative Impedance Controller

목적: RL 통합 전에 controller 자체가 정상 동작하는지 확인.

테스트 시나리오 (script-controlled goal sequence):
    Phase 0:  500 step  | delta=(0,0,0)         | 시작 자세 유지 (0 명령에 reaction 없는지)
    Phase 1: 1000 step  | delta=(0, 0, +1e-4)   | 천천히 위로 (총 +10cm 상승)
    Phase 2: 1000 step  | delta=(+1e-4, 0, 0)   | 오른쪽으로 (+10cm)
    Phase 3: 1000 step  | delta=(0, +1e-4, 0)   | 앞으로 (+10cm)
    Phase 4:  500 step  | delta=(0,0,0)         | 정지 — 잔여 진동 확인

매 50 step마다 콘솔에 진단 출력:
    - 위치 오차 (target − rod) 크기
    - 회전 오차 (deg)
    - controller wrench 크기
    - joint torque 최댓값
    - fixed joint anchor 위반 (Phase 2 검증과 동일)

사용법:
    python test_impedance_controller.py --num_envs 1
"""

import argparse
import json
import math
import os
import time
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Standalone test for Cooperative Impedance Controller")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--episode_length_s", type=float, default=3600.0,
                    help="Long enough to run full sequence without auto-reset")
parser.add_argument("--diag_period", type=int, default=50, help="Print/log diagnostics every N steps")
parser.add_argument("--log_file", type=str, default="impedance_test_log.json",
                    help="Output JSON path (relative to script dir)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from dual_arm_transport_env3 import DualrobotEnv, DualrobotCfg


# ─────────────────────────────────────────────────────────
# Goal sequence — 각 phase마다 (n_steps, 6-dim delta_per_step) 정의
#   delta[0:3] = positional delta (m/step)
#   delta[3:6] = rotation delta as axis-angle (rad/step)
# ─────────────────────────────────────────────────────────
SEQUENCE = [
    # phase name,      steps, (dx, dy, dz, drx, dry, drz)
    ("hold_initial",     500, (0.0,    0.0,    0.0,   0.0,  0.0,  0.0)),
    ("move_up",         1000, (0.0,    0.0,    +1e-4, 0.0,  0.0,  0.0)),     # +10cm Z
    ("move_right",      1000, (+1e-4,  0.0,    0.0,   0.0,  0.0,  0.0)),     # +10cm X
    ("move_forward",    1000, (0.0,    +1e-4,  0.0,   0.0,  0.0,  0.0)),     # +10cm Y
    ("rotate_yaw",      1000, (0.0,    0.0,    0.0,   0.0,  0.0,  +5e-4)),   # +0.5rad ≈ 28.6° yaw
    ("rotate_pitch",     500, (0.0,    0.0,    0.0,   0.0,  +5e-4, 0.0)),    # +0.25rad ≈ 14.3° pitch
    ("settle",           500, (0.0,    0.0,    0.0,   0.0,  0.0,  0.0)),
]


def _measure_joint_anchor_violation(env):
    """Phase 2 검증과 동일한 fixed-joint anchor 위반 측정"""
    TCP_OFFSET = 0.1034
    HALF_W = 0.4

    ee1 = env.robot_1.data.body_state_w[:, env.ee_body_idx_1, :7]
    ee2 = env.robot_2.data.body_state_w[:, env.ee_body_idx_2, :7]
    rod = env.rod.data.root_state_w[:, :7]

    # body0 anchor (panda_hand local TCP) world 위치
    p_local = torch.tensor([0.0, 0.0, TCP_OFFSET], device=env.device).expand(env.num_envs, 3)

    def quat_apply(q, v):
        qv = q[:, 1:]
        qw = q[:, :1]
        t = 2.0 * torch.cross(qv, v, dim=-1)
        return v + qw * t + torch.cross(qv, t, dim=-1)

    a0_1 = ee1[:, 0:3] + quat_apply(ee1[:, 3:7], p_local)
    a0_2 = ee2[:, 0:3] + quat_apply(ee2[:, 3:7], p_local)

    p_left = torch.tensor([-HALF_W, 0.0, 0.0], device=env.device).expand(env.num_envs, 3)
    p_right = torch.tensor([+HALF_W, 0.0, 0.0], device=env.device).expand(env.num_envs, 3)
    a1_1 = rod[:, 0:3] + quat_apply(rod[:, 3:7], p_left)
    a1_2 = rod[:, 0:3] + quat_apply(rod[:, 3:7], p_right)

    return (
        torch.norm(a0_1 - a1_1, dim=-1).max().item() * 1000,  # mm
        torch.norm(a0_2 - a1_2, dim=-1).max().item() * 1000,
    )


def main():
    env_cfg = DualrobotCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = 4.0
    env_cfg.episode_length_s = args_cli.episode_length_s

    env = DualrobotEnv(cfg=env_cfg, render_mode="human")

    log = {
        "config": {
            "num_envs": args_cli.num_envs,
            "K_abs_pos": env.controller.K_abs_pos,
            "D_abs_pos": env.controller.D_abs_pos,
            "K_abs_rot": env.controller.K_abs_rot,
            "D_abs_rot": env.controller.D_abs_rot,
            "K_rel": env.controller.K_rel,
            "D_rel": env.controller.D_rel,
            "TCP_OFFSET": env.controller.TCP_OFFSET,
            "decimation": env.cfg.decimation,
            "diag_period": args_cli.diag_period,
            "sequence": [
                {"name": n, "n_steps": s, "delta_per_step": list(d),
                 "cumulative_pos": [d[0] * s, d[1] * s, d[2] * s],
                 "cumulative_rot_aa": [d[3] * s, d[4] * s, d[5] * s]}
                for (n, s, d) in SEQUENCE
            ],
        },
        "initial_rod_pos": None,
        "samples": [],
        "final": None,
        "completed": False,
        "ended_early": False,
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, args_cli.log_file)

    def _save(verbose: bool = True):
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        if verbose:
            print(f"📝 Saved {len(log['samples'])} samples to {log_path}")

    print("=" * 80)
    print(f"  Cooperative Impedance Controller — Standalone Test")
    print(f"  num_envs={args_cli.num_envs}  K_abs_pos={env.controller.K_abs_pos}  D_abs_pos={env.controller.D_abs_pos}")
    print(f"                                 K_abs_rot={env.controller.K_abs_rot}  D_abs_rot={env.controller.D_abs_rot}")
    print(f"                                 K_rel={env.controller.K_rel}  D_rel={env.controller.D_rel}")
    print(f"  log file: {log_path}")
    print("=" * 80)
    import math as _math
    for name, n, d in SEQUENCE:
        cum_pos = (d[0] * n, d[1] * n, d[2] * n)
        cum_rot_rad = (d[3] * n, d[4] * n, d[5] * n)
        cum_rot_deg = tuple(round(r * 180.0 / _math.pi, 1) for r in cum_rot_rad)
        print(f"  {name:14s}: {n:5d} steps  pos={cum_pos}m  rot={cum_rot_deg}°")
    print("=" * 80)

    obs, _ = env.reset()

    # 초기 state 기록
    initial_rod_pos = env.rod.data.root_pos_w.clone()
    log["initial_rod_pos"] = initial_rod_pos[0].cpu().tolist()
    print(f"\n[step 0] initial rod_pos (env 0): {initial_rod_pos[0].cpu().numpy()}")

    step = 0
    t0 = time.time()
    try:
        for phase_name, n_steps, delta_t in SEQUENCE:
            print(f"\n────────────  Phase: {phase_name}  ({n_steps} steps, delta/step={delta_t})  ────────────")
            delta_tensor = torch.tensor(delta_t, device=env.device).expand(env.num_envs, 6).clone()

            for _ in range(n_steps):
                if not simulation_app.is_running():
                    log["ended_early"] = True
                    break

                obs, rew, terminated, truncated, info = env.step(delta_tensor)
                step += 1

                if step % args_cli.diag_period == 0:
                    ci = env._last_ctrl_info
                    rod_now = env.rod.data.root_pos_w[0].cpu().tolist()
                    rod_quat_now = env.rod.data.root_quat_w[0].cpu().tolist()
                    target_now = env.target_obj_pos[0].cpu().tolist()
                    drift_from_init = (env.rod.data.root_pos_w - initial_rod_pos).norm(dim=-1).max().item()
                    j1, j2 = _measure_joint_anchor_violation(env)

                    # Phase 3.5: internal wrench 측정값 추출
                    f_int_lin_mean = env.extras.get("log/f_int_lin_mean", torch.tensor(0.0)).item()
                    f_int_lin_max = env.extras.get("log/f_int_lin_max", torch.tensor(0.0)).item()
                    f_int_ang_mean = env.extras.get("log/f_int_ang_mean", torch.tensor(0.0)).item()
                    f_int_ang_max = env.extras.get("log/f_int_ang_max", torch.tensor(0.0)).item()

                    # console
                    print(
                        f"[step {step:5d}] "
                        f"pos_err {ci['pos_err_norm']*1000:6.2f}mm  "
                        f"rot_err {ci['rot_err_deg']:5.2f}deg  "
                        f"|f| {ci['f_ext_lin_norm']:5.1f}N  "
                        f"|τ| {ci['f_ext_ang_norm']:5.2f}Nm  "
                        f"τ_max {max(ci['tau_1_max'], ci['tau_2_max']):5.1f}  "
                        f"j_viol {j1:.2f}/{j2:.2f}mm  "
                        f"f_int_lin {f_int_lin_max:5.2f}N  "
                        f"f_int_ang {f_int_ang_max:5.3f}Nm  "
                        f"|rod−rod₀| {drift_from_init*1000:5.1f}mm"
                    )

                    # log entry
                    log["samples"].append({
                        "step": step,
                        "phase": phase_name,
                        "pos_err_mm": ci["pos_err_norm"] * 1000,
                        "rot_err_deg": ci["rot_err_deg"],
                        "f_lin_norm_N": ci["f_ext_lin_norm"],
                        "f_ang_norm_Nm": ci["f_ext_ang_norm"],
                        "tau_1_max": ci["tau_1_max"],
                        "tau_2_max": ci["tau_2_max"],
                        "j1_viol_mm": j1,
                        "j2_viol_mm": j2,
                        "drift_from_init_mm": drift_from_init * 1000,
                        "rod_pos": rod_now,
                        "rod_quat": rod_quat_now,
                        "target_pos": target_now,
                        # Phase 3.5: internal wrench measurement
                        "f_int_lin_mean_N": f_int_lin_mean,
                        "f_int_lin_max_N": f_int_lin_max,
                        "f_int_ang_mean_Nm": f_int_ang_mean,
                        "f_int_ang_max_Nm": f_int_ang_max,
                    })
                    # 매 diag마다 incremental save (어떤 종료 방식에도 데이터 보존)
                    _save(verbose=False)

            if not simulation_app.is_running():
                log["ended_early"] = True
                break

        log["completed"] = True

        print("\n" + "=" * 80)
        print(f"  Test sequence finished. Total {step} steps in {time.time()-t0:.1f}s.")
        if env.num_envs == 1:
            final_rod = env.rod.data.root_pos_w[0].cpu().tolist()
            final_target = env.target_obj_pos[0].cpu().tolist()
            final_err = torch.norm(env.rod.data.root_pos_w - env.target_obj_pos, dim=-1)[0].item() * 1000
            print(f"  Final rod_pos    : {final_rod}")
            print(f"  Final target     : {final_target}")
            print(f"  Final pos error  : {final_err:.2f} mm")
            log["final"] = {
                "rod_pos": final_rod,
                "target_pos": final_target,
                "pos_err_mm": final_err,
                "total_steps": step,
                "elapsed_s": time.time() - t0,
            }
        print("=" * 80)

        # ★ 시퀀스 끝나자마자 즉시 저장 (idle 루프에서 죽어도 데이터 보존)
        _save()

        print("  Test data saved. Idle loop active — close window or Ctrl+C to exit.")
        print("=" * 80)

        # idle: 끝나도 시뮬은 계속 돌려서 최종 상태 관찰 가능
        idle_step = 0
        while simulation_app.is_running():
            zero_delta = torch.zeros(env.num_envs, 6, device=env.device)
            env.step(zero_delta)
            idle_step += 1

    except KeyboardInterrupt:
        print("\n[!] KeyboardInterrupt — saving partial log…")
    except Exception as e:
        print(f"\n[!] Exception: {e!r} — saving partial log…")
        log["error"] = repr(e)
    finally:
        _save()
        try:
            env.close()
        except Exception:
            pass
        try:
            simulation_app.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
