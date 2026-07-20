"""
Controller tracking validation — RL 학습 없이 reference 주입 후 tracking error 측정.

테스트 종류:
1) RANDOM:  각 step random action (RL warmup과 동일). RL 학습 시 controller가 받는 부하 시뮬레이션.
2) STEP:    일정 offset target 유지. Step response — settling 시간 직접 측정.
3) RAMP:    constant velocity target. Steady-state lag 측정.

사용:
    python test_tracking.py --mode random --num_envs 64 --num_steps 200
    python test_tracking.py --mode step   --num_envs 16 --num_steps 100 --offset 0.05
    python test_tracking.py --mode ramp   --num_envs 16 --num_steps 100
"""
import argparse
import math
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["random", "step", "ramp", "goal", "hold"], default="random")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_steps", type=int, default=200)
parser.add_argument("--action_scale_pos", type=float, default=0.05)
parser.add_argument("--action_scale_rot", type=float, default=0.1)
parser.add_argument("--offset", type=float, default=0.05,
                    help="STEP mode: fixed target offset (m) in +x direction")
parser.add_argument("--ramp_v", type=float, default=0.025,
                    help="RAMP mode: target velocity (m per RL step) in +x")
parser.add_argument("--hold_offset", type=float, default=0.0,
                    help="HOLD mode: target = rod_init + (hold_offset, 0, 0) m.")
parser.add_argument("--hold_yaw_deg", type=float, default=0.0,
                    help="HOLD mode: target yaw = rod_init_yaw + hold_yaw_deg (degrees). Stage 4 rotation test.")
parser.add_argument("--settle_steps", type=int, default=10,
                    help="HOLD mode: 설정 전 settle phase. target=rod_pos로 N step. reset transient 제거용.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from dual_arm_transport_env3 import DualrobotEnv, DualrobotCfg


def percentile(t: torch.Tensor, q: float) -> float:
    return torch.quantile(t, q).item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = DualrobotCfg()
    cfg.scene.num_envs = args.num_envs
    # goal/hold mode: 한 episode 안에 test 끝나도록 episode_length 늘리기 (auto-reset 회피)
    # ★ FIX (2026-06-05): hold mode도 추가. 이전엔 step 30(6s)마다 reset되면서 pos_err 점프가
    #   controller 불안정으로 오인됨. step 0~16 (3.2s)만 의미 있는 데이터였음.
    if args.mode in ("goal", "hold"):
        cfg.episode_length_s = max(cfg.episode_length_s, args.num_steps * cfg.sim.dt * cfg.decimation + 5.0)
    env = DualrobotEnv(cfg=cfg, render_mode=None)

    dt_rl = cfg.sim.dt * cfg.decimation
    print(f"\n== Controller Tracking Test ==")
    print(f"  mode={args.mode}  num_envs={args.num_envs}  num_steps={args.num_steps}")
    print(f"  decimation={cfg.decimation}  RL dt={dt_rl*1000:.1f} ms ({1/dt_rl:.1f} Hz)")
    print(f"  action_scale_pos={args.action_scale_pos}  action_scale_rot={args.action_scale_rot}\n")

    obs, _ = env.reset()
    scale_pos = torch.tensor([args.action_scale_pos] * 3, device=device)
    scale_rot = torch.tensor([args.action_scale_rot] * 3, device=device)
    scale = torch.cat([scale_pos, scale_rot])

    pos_errs = []  # list of (num_envs,) per step (rod ↔ target)
    rot_errs = []
    rod_xs = []     # rod_pos_x per step (for STEP mode 진단)
    rod_x0 = None
    goal_dists = []  # rod ↔ goal distance (goal mode 전용)

    # hold mode (Stage 0): target = rod_init. 정지 유지 검증.
    # → controller에 0 외력 명령. drift가 발생하면 controller에 버그.
    if args.mode == "hold":
        # ── Settle phase: target = rod pose AT RESET (FIXED, not chasing current) ──
        # ★ FIX (2026-06-05): 이전엔 매 step target=current rod pose로 set ("chase") →
        #   rod 움직이면 target도 움직여서 "여기 멈춰" 신호 없었음. unstable env에서 settle 무력.
        #   이제 reset 직후 rod pose 한 번 snapshot → controller에 명확한 정지 명령.
        a_zero = torch.zeros(args.num_envs, 6, device=device)
        with torch.no_grad():
            env.target_obj_pos.copy_(env.rod.data.root_pos_w)
            env.target_obj_quat.copy_(env.rod.data.root_quat_w)
        for s in range(args.settle_steps):
            env.step(a_zero)
        # Settle 후의 rod 상태를 init으로
        rod_init_pos = env.rod.data.root_pos_w.clone()
        rod_init_quat = env.rod.data.root_quat_w.clone()
        rod_init_w = env.rod.data.root_ang_vel_w.clone()
        print(f"  After {args.settle_steps} settle steps: max ω = {rod_init_w.norm(dim=-1).max().item():.4f} rad/s")
        # target = rod_init + hold_offset in +x
        hold_target_pos = rod_init_pos.clone()
        hold_target_pos[:, 0] += args.hold_offset
        # target rotation = rod_init + yaw (z-axis rotation in world)
        hold_target_quat = rod_init_quat.clone()
        if args.hold_yaw_deg != 0.0:
            import math as _m
            yaw_rad = args.hold_yaw_deg * _m.pi / 180.0
            half = yaw_rad * 0.5
            # q_delta = (cos(half), 0, 0, sin(half)) for yaw around z
            cw = _m.cos(half); sw = _m.sin(half)
            q_delta = torch.tensor([cw, 0.0, 0.0, sw], device=device).unsqueeze(0).expand(args.num_envs, -1)
            # apply: target_quat = q_delta * rod_init_quat (world frame yaw)
            import isaaclab.utils.math as _math_utils
            hold_target_quat = _math_utils.quat_mul(q_delta, rod_init_quat)
            hold_target_quat = hold_target_quat / torch.norm(hold_target_quat, dim=-1, keepdim=True)
        with torch.no_grad():
            env.target_obj_pos.copy_(hold_target_pos)
            env.target_obj_quat.copy_(hold_target_quat)
        offset_mm = args.hold_offset * 1000
        if args.hold_offset == 0.0:
            print(f"  [hold mode, Stage 0] target = rod_init (zero offset). drift 측정.")
            print(f"  통과 기준: drift < 1mm/sec.")
        else:
            print(f"  [hold mode, Step Response] target = rod_init + {offset_mm:.1f}mm in +x. step response 측정.")
            if offset_mm <= 1.5:
                print(f"  통과 기준 (Stage 1, 1mm): monotonic, no overshoot.")
            elif offset_mm <= 15:
                print(f"  통과 기준 (Stage 2, ~1cm): < 20% overshoot, 5초 내 settling.")
            else:
                print(f"  통과 기준 (Stage 3, ~5cm): settling 가능, overshoot 허용.")

    # goal mode: action 무시하고 매 step env.target_obj_pos를 goal pose로 직접 override
    # → controller가 goal을 직접 reference로 받는 best-case 시나리오
    goal_pos_world = None
    goal_quat_world = None
    if args.mode == "goal":
        goal_pos_world = env.goal_rod_marker.data.root_pos_w.clone()  # absolute world
        goal_quat_world = env.goal_rod_marker.data.root_quat_w.clone()
        # Step 0부터 target = goal로 시작 (loop 시작 전에 override)
        with torch.no_grad():
            env.target_obj_pos.copy_(goal_pos_world)
            env.target_obj_quat.copy_(goal_quat_world)
        print(f"  [goal mode] initial rod-goal distance per env [mm]:")
        rod_p0 = env.rod.data.root_pos_w
        d0 = torch.norm(rod_p0 - goal_pos_world, dim=-1) * 1000
        for i in range(min(args.num_envs, 8)):
            print(f"    env {i}: {d0[i].item():.2f}")

    for step in range(args.num_steps):
        if args.mode == "random":
            a = scale * (2 * torch.rand(args.num_envs, 6, device=device) - 1)
        elif args.mode == "step":
            a = torch.zeros(args.num_envs, 6, device=device)
            a[:, 0] = args.offset
        elif args.mode == "ramp":
            a = torch.zeros(args.num_envs, 6, device=device)
            a[:, 0] = args.ramp_v
        elif args.mode == "goal":
            # action = 0 → accumulating mode에서 target 변화 없음.
            a = torch.zeros(args.num_envs, 6, device=device)
        elif args.mode == "hold":
            # action = 0 → accumulating mode에서 target 그대로 유지 (= rod_init).
            a = torch.zeros(args.num_envs, 6, device=device)

        obs, r, term, trunc, info = env.step(a)

        # GOAL mode: 매 step env.step 이후 target을 goal로 강제 override
        # (다음 step의 _apply_action이 이 target을 사용)
        if args.mode == "goal":
            with torch.no_grad():
                env.target_obj_pos.copy_(goal_pos_world)
                env.target_obj_quat.copy_(goal_quat_world)
        elif args.mode == "hold":
            # hold mode: target을 매 step 고정.
            with torch.no_grad():
                env.target_obj_pos.copy_(hold_target_pos)
                env.target_obj_quat.copy_(hold_target_quat)

        # Tracking error — env3가 매 step extras로 publish
        pe_mm = env.extras.get("diag/track_err_pos_mm", None)
        re_deg = env.extras.get("diag/track_err_rot_deg", None)
        # scalar (mean over envs) ── 우리는 per-env vector도 원함
        # env3 코드: track_err_pos = torch.norm(rod_pos - target, dim=-1) → (B,) 그 후 mean
        # 직접 계산
        with torch.no_grad():
            rod_pos = env.rod.data.root_pos_w
            target_pos = env.target_obj_pos
            pe_vec = torch.norm(rod_pos - target_pos, dim=-1)  # (B,) m
            pos_errs.append(pe_vec.cpu())
            if rod_x0 is None:
                rod_x0 = rod_pos[:, 0].clone()  # baseline x at step 0
            rod_xs.append((rod_pos[:, 0] - rod_x0).cpu())  # x displacement from start

            rod_quat = env.rod.data.root_quat_w
            target_quat = env.target_obj_quat
            # Quaternion error → axis-angle norm
            import isaaclab.utils.math as math_utils
            rod_inv = math_utils.quat_conjugate(rod_quat)
            q_diff = math_utils.quat_mul(target_quat, rod_inv)
            re_rad = 2.0 * torch.atan2(torch.norm(q_diff[:, 1:4], dim=-1), torch.abs(q_diff[:, 0]))
            rot_errs.append(re_rad.cpu())

            if args.mode == "goal":
                d_goal = torch.norm(rod_pos - goal_pos_world, dim=-1)
                goal_dists.append(d_goal.cpu())

        if step % 20 == 0 or step == args.num_steps - 1:
            pe = pe_vec * 1000  # mm
            re = re_rad * 180 / math.pi  # deg
            print(f"  step {step:>4d}  pos_err [mm] mean={pe.mean():.2f} p50={percentile(pe, 0.5):.2f} "
                  f"p95={percentile(pe, 0.95):.2f}  rot_err [deg] mean={re.mean():.2f} p95={percentile(re, 0.95):.2f}")

    # 통계: 마지막 절반 (transient 제외)
    P = torch.stack(pos_errs)  # (T, B)
    R = torch.stack(rot_errs)
    half = args.num_steps // 2
    P_ss = P[half:].flatten() * 1000  # mm
    R_ss = R[half:].flatten() * 180 / math.pi  # deg

    print(f"\n== 통계 (steady-state, last {args.num_steps - half} steps) ==")
    print(f"  pos_err [mm]: mean={P_ss.mean():.2f}  p50={percentile(P_ss, 0.5):.2f}  "
          f"p95={percentile(P_ss, 0.95):.2f}  max={P_ss.max():.2f}")
    print(f"  rot_err [deg]: mean={R_ss.mean():.2f}  p50={percentile(R_ss, 0.5):.2f}  "
          f"p95={percentile(R_ss, 0.95):.2f}  max={R_ss.max():.2f}")

    # Hold mode 결과 분석
    if args.mode == "hold":
        # ★ FIX (2026-06-05): D를 mm로 통일. 이전엔 D=P(meters)인데 "mm" 라벨로 출력 → 1000× 작게 보임,
        #   PASS verdict도 단위 mix로 잘못 찍힘 (e.g. 51mm를 0.051"mm"로 보고 PASS).
        D = P * 1000  # pos_err (rod ↔ target) in mm. offset=0이면 drift, offset>0이면 step response gap.
        D_mean = D.mean(dim=1)
        offset_mm = args.hold_offset * 1000
        # Rotation analysis (if yaw_deg != 0)
        if args.hold_yaw_deg != 0.0:
            R_arr = R  # already in deg in earlier processing? Let me check — R is from rot_errs in rad, then *180/pi later
            # We need to print rotation error per step
            R_deg_per_step = (torch.stack(rot_errs) * 180 / math.pi)  # (T, B) deg
            R_mean_step = R_deg_per_step.mean(dim=1)
            print(f"\n  [hold mode] rotation tracking (yaw target {args.hold_yaw_deg:.1f}°):")
            sample_idx = list(range(0, args.num_steps, max(1, args.num_steps // 10)))
            if args.num_steps - 1 not in sample_idx: sample_idx.append(args.num_steps - 1)
            for i in sample_idx:
                print(f"    step {i:>3d} ({i*dt_rl*1000:.0f} ms): rot_err mean={R_mean_step[i].item():.3f}°  max={R_deg_per_step[i].max().item():.3f}°")
            R_final = R_deg_per_step[-10:].mean(dim=0)
            print(f"  Final rot_err (last 10 step mean) per env:")
            for i in range(min(args.num_envs, 8)):
                print(f"    env {i}: {R_final[i].item():.3f}°")
            R_peak = R_deg_per_step.max().item()
            R_final_mean = R_final.mean().item()
            target_yaw = abs(args.hold_yaw_deg)
            print(f"\n  Rotation summary:")
            print(f"    Initial target yaw:  {target_yaw:.1f}°")
            print(f"    Peak rot_err:        {R_peak:.3f}°")
            print(f"    Final rot_err mean:  {R_final_mean:.3f}°  ({(R_final_mean/target_yaw)*100:.1f}% of target)")
            ok = R_final_mean < target_yaw * 0.1
            print(f"  Stage 4 rotation: final < 10% → {'✓ PASS' if ok else '✗ FAIL'}")

        label = "drift" if args.hold_offset == 0.0 else "pos_err (rod↔target)"
        print(f"\n  [hold mode] {label} over time [mm]:")
        sample_idx = list(range(0, args.num_steps, max(1, args.num_steps // 12)))
        if args.num_steps - 1 not in sample_idx: sample_idx.append(args.num_steps - 1)
        for i in sample_idx:
            print(f"    step {i:>3d} ({i*dt_rl*1000:.0f} ms): mean={D_mean[i].item():.3f}mm  max={D[i].max().item():.3f}mm")
        final_err = D[-10:].mean(dim=0)
        total_time_s = args.num_steps * dt_rl

        if args.hold_offset == 0.0:
            # Stage 0 — drift 측정
            print(f"\n  Final drift (last 10 step mean) per env [mm]:")
            for i in range(min(args.num_envs, 8)):
                print(f"    env {i}: {final_err[i].item():.3f} mm")
            max_final = final_err.max().item()
            print(f"\n  PASS criterion (Stage 0): drift < {total_time_s * 1.0:.1f}mm (1mm/sec)")
            print(f"  {'✓ PASS' if max_final < total_time_s * 1.0 else '✗ FAIL'} — max drift {max_final:.3f}mm")
        else:
            # Step response — pos_err 분석
            print(f"\n  Final pos_err (last 10 step mean) per env [mm]:")
            for i in range(min(args.num_envs, 8)):
                print(f"    env {i}: {final_err[i].item():.3f} mm")
            max_pos_err = D.max(dim=1).values  # max over envs per step
            peak_over_time = max_pos_err.max().item()
            overshoot_pct = ((peak_over_time - offset_mm) / offset_mm) * 100 if offset_mm > 0 else 0
            final_mean = final_err.mean().item()
            print(f"\n  Step response summary:")
            print(f"    Initial gap:    {offset_mm:.3f} mm")
            print(f"    Peak gap:       {peak_over_time:.3f} mm  (overshoot vs initial: {overshoot_pct:.1f}%)")
            print(f"    Final gap mean: {final_mean:.3f} mm  ({(final_mean/offset_mm)*100:.1f}% of initial)")
            # PASS criteria
            if offset_mm <= 1.5:
                pass_overshoot = overshoot_pct < 5
                pass_final = final_mean < offset_mm * 0.3
                ok = pass_overshoot and pass_final
                print(f"  Stage 1 (~1mm): no overshoot 거의 0, settling 70% 이상 → {'✓ PASS' if ok else '✗ FAIL'}")
            elif offset_mm <= 15:
                pass_overshoot = overshoot_pct < 20
                pass_final = final_mean < offset_mm * 0.1
                ok = pass_overshoot and pass_final
                print(f"  Stage 2 (~1cm): overshoot < 20%, final < 10% → {'✓ PASS' if ok else '✗ FAIL'}")
            else:
                pass_final = final_mean < offset_mm * 0.1
                print(f"  Stage 3 (~5cm): final < 10% → {'✓ PASS' if pass_final else '✗ FAIL'}")

    if args.mode == "goal" and goal_dists:
        G = torch.stack(goal_dists) * 1000  # (T, B) mm
        G_mean_per_step = G.mean(dim=1)
        print(f"\n  [goal mode] rod ↔ goal distance over time (mean across envs) [mm]:")
        sample_idx = list(range(0, args.num_steps, max(1, args.num_steps // 12)))
        if args.num_steps - 1 not in sample_idx: sample_idx.append(args.num_steps - 1)
        for i in sample_idx:
            print(f"    step {i:>3d} ({i*dt_rl*1000:.0f} ms): {G_mean_per_step[i].item():.2f} mm")
        G_final = G[-10:].mean(dim=0)  # last 10 step mean per env
        print(f"\n  Final rod-goal distance per env (last 10 step avg) [mm]:")
        for i in range(min(args.num_envs, 8)):
            print(f"    env {i}: {G_final[i].item():.2f}")
        # success at 2cm threshold
        success_mask = G_final < 20
        print(f"\n  Reached 2cm threshold: {int(success_mask.sum())}/{args.num_envs} envs")

    # Rod 실제 변위 — controller 실측 진단 (STEP / RAMP 시)
    if args.mode in ("step", "ramp"):
        X = torch.stack(rod_xs) * 1000  # (T, B) mm
        x_mean = X.mean(dim=1)
        print(f"\n  rod x displacement (mean across envs) [mm]:")
        sample_idx = list(range(0, args.num_steps, max(1, args.num_steps // 10)))
        if args.num_steps - 1 not in sample_idx: sample_idx.append(args.num_steps - 1)
        for i in sample_idx:
            print(f"    step {i:>3d} ({i*dt_rl*1000:.0f} ms): {x_mean[i].item():.2f} mm")
        # rod velocity 추정 (last half average)
        x_last = x_mean[-10:].mean().item()
        x_mid = x_mean[args.num_steps // 2].mean().item()
        dx = x_last - x_mid
        dt_half = (args.num_steps - args.num_steps // 2 - 5) * dt_rl
        v_est = dx / dt_half if dt_half > 0 else 0
        print(f"  rod velocity est (last half): {v_est*1000:.1f} mm/s")

    # STEP / RAMP 의 경우 settling 시간 추가 출력
    if args.mode in ("step", "ramp"):
        P_mean_per_step = P.mean(dim=1) * 1000  # mm
        target_mm = args.offset * 1000 if args.mode == "step" else args.ramp_v * 1000
        # 5% settling
        thresh = target_mm * 0.05
        below = (P_mean_per_step < thresh)
        if below.any():
            first = int(below.float().argmax().item())
            print(f"\n  5% settling: step {first} ({first * dt_rl * 1000:.0f} ms)")
        else:
            print(f"\n  5% settling: NOT reached within {args.num_steps} steps")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
