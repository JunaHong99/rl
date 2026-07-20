"""
Freeze 원인 진단 — 모델 vs 컨트롤러.

rollout에서 rod가 멈춰있는(rod_speed≈0) step을 정책/컨트롤러 신호로 분류한다.
모든 신호는 env에 이미 존재 (target_obj_pos, rod.root_lin_vel_w, goal_rod_marker).
기존 로직은 건드리지 않는 순수 관찰 스크립트.

분류 (rod_speed < V_THR 인 RL step 대상):
  MODEL_HOLD_OK : 정책 명령 작음 & pos_err<thr  → goal 도달 후 정지 (정상)
  MODEL_GIVEUP  : 정책 명령 작음 & pos_err>thr  → 미도달 정지 (모델/time penalty 문제)
  CTRL_STUCK    : 정책 명령 큼 & track_err 큼    → 명령 있는데 rod 안 따라옴 (컨트롤러 문제)
  CTRL_FINE     : 정책 명령 큼 & track_err 작음  → 추종은 되나 미세조정/저속 (애매)

사용:
  python -u diagnose_freeze.py --num_envs 64 --num_steps 360 \
      --action_scale_pos 0.05 --action_scale_rot 0.05 \
      --model_path logs/.../model_final.pt --headless
"""

import argparse
import math
import os
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Freeze cause diagnostic (model vs controller)")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_steps", type=int, default=360)
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--action_scale_pos", type=float, default=0.05)
parser.add_argument("--action_scale_rot", type=float, default=0.05)
# 진단 임계값
parser.add_argument("--v_thr", type=float, default=0.02, help="rod_speed < 이 값(m/s) = frozen")
parser.add_argument("--a_small", type=float, default=None,
                    help="정책 positional 명령 norm < 이 값(m) = '안 미는 중'. 기본 0.1*action_scale_pos")
parser.add_argument("--track_large", type=float, default=0.02, help="track_err > 이 값(m) = 컨트롤러 미추종")
parser.add_argument("--pos_thr", type=float, default=0.02, help="pos_err < 이 값(m) = goal 도달")
parser.add_argument("--trace_envs", type=int, default=2, help="per-step trace 덤프할 env 수")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from dual_arm_transport_env3 import DualrobotEnv, DualrobotCfg
import mlp_policy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A_SMALL = args.a_small if args.a_small is not None else 0.1 * args.action_scale_pos
    print(f"🔍 Freeze diagnostic on {device}  "
          f"(V_THR={args.v_thr} m/s, A_SMALL={A_SMALL:.4f} m, TRACK_LARGE={args.track_large} m, POS_THR={args.pos_thr} m)")

    env_cfg = DualrobotCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = DualrobotEnv(cfg=env_cfg, render_mode=None)

    action_scale_vec = [args.action_scale_pos] * 3 + [args.action_scale_rot] * 3
    agent = mlp_policy.MLPSACAgent(
        action_dim=env.cfg.action_space,
        num_rounds=2,
        action_scale=action_scale_vec,
        hidden_dim=256,
        num_hidden_layers=2,
    ).to(device)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = args.model_path if os.path.isabs(args.model_path) else os.path.join(script_dir, args.model_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    agent.load_state_dict(ckpt["model"])
    agent.eval()
    print(f"📂 Loaded {ckpt_path}  (env_steps={ckpt.get('env_steps', '?')})")

    obs, _ = env.reset()
    current_batch = env._build_policy_batch()
    prev_target = env.target_obj_pos.clone()

    # 누적 카운터 (slow step 분류)
    cats = ["MODEL_HOLD_OK", "MODEL_GIVEUP", "CTRL_STUCK", "CTRL_FINE"]
    cnt = {c: 0 for c in cats}
    cnt_sum_pos = {c: 0.0 for c in cats}     # 분류별 pos_err 합 (mm)
    cnt_sum_track = {c: 0.0 for c in cats}   # 분류별 track_err 합 (mm)
    n_rl_steps = 0       # settle 제외 RL step 총수 (env*step)
    n_slow = 0           # 그 중 rod_speed<V_THR
    n_moving = 0
    # ── 명령 park 분석 (rod 속도 무관): 정책이 명령(a_pos, dtarget)을 멈췄는가 ──
    # reset-boundary step(track_err≈0 & dtarget 큼)은 제외해야 정확.
    n_park_reached = 0       # 명령 멈춤 & 도달 (정상 hold)
    n_park_short = 0         # 명령 멈춤 & 미도달 (정책 조기 포기 = time penalty 의심)
    park_short_sum_pos = 0.0
    park_short_sum_track = 0.0
    ep_had_park_short = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    n_ep_park_short = 0
    # episode가 한 번이라도 frozen-미도달(GIVEUP/STUCK) 겪었는지 per-env 플래그
    ep_had_problem_freeze = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    n_ep = 0
    n_ep_problem = 0

    # trace 덤프용
    trace = {i: [] for i in range(min(args.trace_envs, args.num_envs))}

    for step in range(args.num_steps):
        with torch.no_grad():
            action, _, _ = agent.actor.get_action_and_log_prob(current_batch, deterministic=True)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated

        is_settle = env._is_settle_step if hasattr(env, "_is_settle_step") else torch.zeros(args.num_envs, dtype=torch.bool, device=device)

        # ── 신호 수집 (post-step) ──
        rod_pos = env.rod.data.root_pos_w
        rod_speed = torch.norm(env.rod.data.root_lin_vel_w, dim=-1)            # (B,) m/s
        target = env.target_obj_pos
        track_err = torch.norm(rod_pos - target, dim=-1)                       # (B,) m
        goal_pos = env.goal_rod_marker.data.root_pos_w
        pos_err = torch.norm(goal_pos - rod_pos, dim=-1)                       # (B,) m
        a_pos = torch.norm(action[:, :3], dim=-1)                              # (B,) 정책 positional 명령 norm
        dtarget = torch.norm(target - prev_target, dim=-1)                     # (B,) target 전진량

        rl_mask = ~is_settle                                                   # RL step만 분류
        pushing = (a_pos > A_SMALL) | (dtarget > A_SMALL)
        reached = pos_err < args.pos_thr
        slow = rod_speed < args.v_thr

        # trace 덤프 (RL step만)
        for i in trace:
            if rl_mask[i]:
                trace[i].append((step, a_pos[i].item(), dtarget[i].item(),
                                 track_err[i].item()*1000, rod_speed[i].item(),
                                 pos_err[i].item()*1000, bool(slow[i])))

        # 분류 (RL & slow 인 env만)
        sel = rl_mask & slow
        n_rl_steps += int(rl_mask.sum())
        n_slow += int(sel.sum())
        n_moving += int((rl_mask & ~slow).sum())
        if sel.any():
            idx = sel.nonzero(as_tuple=True)[0]
            for i in idx.tolist():
                if reached[i]:
                    c = "MODEL_HOLD_OK"
                elif not pushing[i]:
                    c = "MODEL_GIVEUP"
                elif track_err[i] > args.track_large:
                    c = "CTRL_STUCK"
                else:
                    c = "CTRL_FINE"
                cnt[c] += 1
                cnt_sum_pos[c] += pos_err[i].item() * 1000
                cnt_sum_track[c] += track_err[i].item() * 1000
                if c in ("MODEL_GIVEUP", "CTRL_STUCK"):
                    ep_had_problem_freeze[i] = True

        # ── 명령 park 분석 (rod 속도 무관, reset-boundary 제외) ──
        # reset 직후 step: track_err≈0 & dtarget 큼 → 제외.
        is_reset_boundary = (track_err < 0.005) & (dtarget > A_SMALL)
        parked = rl_mask & ~is_reset_boundary & (a_pos < A_SMALL) & (dtarget < A_SMALL)
        if parked.any():
            pr = parked & reached
            ps = parked & ~reached
            n_park_reached += int(pr.sum())
            n_park_short += int(ps.sum())
            if ps.any():
                psi = ps.nonzero(as_tuple=True)[0]
                park_short_sum_pos += pos_err[ps].sum().item() * 1000
                park_short_sum_track += track_err[ps].sum().item() * 1000
                ep_had_park_short[psi] = True

        # episode 종료 집계
        if done.any():
            di = done.nonzero(as_tuple=True)[0]
            n_ep += int(done.sum())
            n_ep_problem += int(ep_had_problem_freeze[di].sum())
            n_ep_park_short += int(ep_had_park_short[di].sum())
            ep_had_problem_freeze[di] = False
            ep_had_park_short[di] = False

        prev_target = env.target_obj_pos.clone()
        # done env는 reset돼 target 점프 → 다음 step dtarget 왜곡 방지
        if done.any():
            prev_target[done] = env.target_obj_pos[done]

        current_batch = env._build_policy_batch()

    # ── 결과 ──
    print("=" * 80)
    print(f"RL steps (settle 제외): {n_rl_steps:,}   moving: {n_moving:,}   slow(<{args.v_thr}m/s): {n_slow:,} "
          f"({100*n_slow/max(1,n_rl_steps):.1f}%)")
    print(f"episodes: {n_ep:,}   그중 '미도달 freeze(GIVEUP/STUCK)' 겪은 episode: {n_ep_problem:,} "
          f"({100*n_ep_problem/max(1,n_ep):.1f}%)")
    print("-" * 80)
    print(f"{'분류':<16}{'count':>8}{'slow중%':>9}{'평균pos_err':>13}{'평균track_err':>15}")
    for c in cats:
        n = cnt[c]
        pct = 100 * n / max(1, n_slow)
        mp = cnt_sum_pos[c] / max(1, n)
        mt = cnt_sum_track[c] / max(1, n)
        print(f"{c:<16}{n:>8,}{pct:>8.1f}%{mp:>11.0f}mm{mt:>13.0f}mm")
    print("-" * 80)
    print("해석: MODEL_GIVEUP 우세→time penalty 약함/value부족 | CTRL_STUCK 우세→컨트롤러 | "
          "MODEL_HOLD_OK 우세→도달후정지(정상)")
    print("=" * 80)
    print("■ 명령 park 분석 (rod 속도 무관, reset-boundary 제외): 정책이 명령을 멈췄는가")
    n_park = n_park_reached + n_park_short
    print(f"  parked step(명령≈0): {n_park:,}  ({100*n_park/max(1,n_rl_steps):.1f}% of RL steps)")
    print(f"    ├ 도달 후 park(정상):   {n_park_reached:,}")
    print(f"    └ 미도달 park(조기포기): {n_park_short:,}  "
          f"평균pos_err {park_short_sum_pos/max(1,n_park_short):.0f}mm  "
          f"평균track_err {park_short_sum_track/max(1,n_park_short):.0f}mm")
    print(f"  '미도달 park' 겪은 episode: {n_ep_park_short:,}/{n_ep:,} ({100*n_ep_park_short/max(1,n_ep):.1f}%)")
    print("  → 미도달 park의 pos_err가 컨트롤러 floor(~29mm) 근처면: 정책이 '더 밀어도 안 되는' 한계 인지하고 멈춤")
    print("     (=컨트롤러 한계의 발현). pos_err가 그보다 크면: 정책 조기 포기(time penalty/value 문제).")

    # trace 덤프
    for i, rows in trace.items():
        print("=" * 80)
        print(f"[Env {i}] per-step trace (RL steps only, 처음 50개)")
        print(f"  {'step':>5}{'a_pos':>9}{'dtarget':>9}{'track_mm':>10}{'speed':>9}{'pos_mm':>9}  slow")
        for r in rows[:50]:
            st, ap, dt, tk, sp, pe, sl = r
            print(f"  {st:>5}{ap:>9.4f}{dt:>9.4f}{tk:>10.1f}{sp:>9.4f}{pe:>9.1f}  {'■' if sl else ''}")

    simulation_app.close()


if __name__ == "__main__":
    main()
