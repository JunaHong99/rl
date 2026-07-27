"""
Offline IK-feasibility diagnostic.

목적: 학습된 SAC 정책을 롤아웃하며 각 스텝에서 정책이 명령한 rod pose로부터 유도되는
per-arm EE target들이 IK 해가 존재하는지(도달가능) 판정하고, 그것이 에피소드 실패와
상관되는지 확정적으로 확인한다. 이는 "IK 없음/도달불가 명령"을 "특이점(singularity)"
(manipulability로 따로 로깅됨)과 분리하기 위한 진단이다.

★ 설계 원칙 (2026-07-20 재작성):
  롤아웃 루프는 KNOWN-GOOD eval_cluttered.py 의 롤아웃을 그대로 복제한다
  (동일 env setup / 동일 action 계산 / 동일 env.step / 동일 success 집계 / 동일 settle 처리).
  그 위에 "EE target 기록 + 배치 IK feasibility 판정 + success/fail split 리포트" 만 얹는다.
  → 이렇게 해야 IK 숫자가 학습/eval 과 동일한 정책 거동에서 나온 것임이 보장된다.
  (이전 버전은 success 라벨을 env.reached_once 로, genuine 판정을 무조건 True 로 잘못 처리해
   eval_cluttered 와 다른 집계를 냈다. 이제 eval_cluttered 와 동일하게
   min_pos<2cm & min_rot<10° 누적 + rr_len>settle genuine 게이트를 쓴다.)

파이프라인:
  1. eval_cluttered.py 와 동일하게 env + 학습 정책(GNN/MLP) 로드
     (action_scale 은 constructor 에 넘기지만 load_state_dict 가 ckpt 값으로 덮어씀 — eval 과 동일).
  2. eval_cluttered 와 동일한 deterministic 롤아웃 (_build_policy_batch → actor(det=True) → env.step).
  3. 매 스텝(settle step 제외), per-env 로 controller._compute_ee_targets(target_obj_pos, target_obj_quat)
     가 반환하는 world-frame EE target(arm1/arm2)과 각 arm base pose(root_pos_w/root_quat_w)
     를 기록. FrankaTensorIK 는 ARM BASE frame 에서 동작하므로 world→base 변환 후 저장.
  4. 에피소드 종료 시 eval_cluttered 와 동일하게 (min_pos<2cm & min_rot<10°) 로 success 라벨링,
     rr_len>settle 인 genuine 에피소드만 채택. 라벨은 running episode id 로 back-fill.
  5. 롤아웃 후, 기록된 모든 EE target(양 arm)에 대해 배치 IK(solve_ik_gradient) 실행,
     residual = ||FK(q_sol) - target_pos|| 계산. residual > threshold → IK 해 없음(infeasible).
  6. 먼저 롤아웃 success rate 를 출력(eval_cluttered ~93% 와 일치하는지 sanity gate),
     그다음 SUCCESS vs FAIL 에피소드로 나누어 infeasible 비율, mean/max reach 리포트.

사용:
  export LD_LIBRARY_PATH=/home/hjh/anaconda3/envs/env-isaaclab/lib:$LD_LIBRARY_PATH
  python -u ik_feasibility_diag.py --model_path logs/phase3_sac_XXXX/model_final.pt \
      --num_envs 256 --num_steps 300 --headless --max_active_obstacles 0

  # Isaac 없이 IK 판정기만 자체검증(reachable vs 2m-away):
  python -u ik_feasibility_diag.py --self_test
"""
import argparse
import math
import os


# ──────────────────────────────────────────────────────────────────────────
# 프레임 변환 + IK-feasibility 판정 (Isaac 독립 — self-test 가능)
# ──────────────────────────────────────────────────────────────────────────
def world_to_base(ee_pos_w, ee_quat_w, base_pos_w, base_quat_w):
    """World-frame EE target 을 arm base frame 으로 변환.

    p_base = R_base^{-1} (p_ee_world - p_base_world)
    q_base = quat_inv(q_base_world) * q_ee_world  (→ rotation matrix)

    Args (모두 (B,·) 텐서, quat 은 wxyz):
        ee_pos_w:   (B,3)   world EE target pos
        ee_quat_w:  (B,4)   world EE target quat
        base_pos_w: (B,3)   world arm base pos
        base_quat_w:(B,4)   world arm base quat
    Returns:
        p_base:   (B,3)
        R_base:   (B,3,3)  target rotation matrix in base frame
    """
    import torch
    import isaaclab.utils.math as mu

    p_rel = ee_pos_w - base_pos_w
    p_base = mu.quat_apply_inverse(base_quat_w, p_rel)               # R_base^{-1} p_rel
    q_base = mu.quat_mul(mu.quat_inv(base_quat_w), ee_quat_w)
    q_base = q_base / torch.norm(q_base, dim=-1, keepdim=True).clamp_min(1e-8)
    R_base = mu.matrix_from_quat(q_base)
    return p_base, R_base


def classify_ik_feasible(ik, target_pos_base, target_rot_base, threshold=0.03,
                         max_iter=100, tol=1e-4, chunk=8192):
    """배치 IK 를 돌려 각 target 의 residual 과 feasible 여부 반환.

    Args:
        ik: FrankaTensorIK
        target_pos_base: (N,3)   base-frame target pos
        target_rot_base: (N,3,3) base-frame target rot mat
        threshold: residual(m) 임계값. residual>threshold → infeasible(IK 해 없음)
    Returns:
        residual: (N,)   ||FK(q_sol) - target_pos||
        feasible: (N,) bool  residual < threshold
    """
    import torch

    N = target_pos_base.shape[0]
    res = torch.empty(N, device=target_pos_base.device)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        out = ik.solve_ik_gradient(
            target_pos_base[s:e], target_rot_base[s:e], max_iter=max_iter, tol=tol
        )
        q = out[0] if isinstance(out, tuple) else out          # tuple 반환도 허용
        fk_pos, _ = ik.forward_kinematics(q)
        res[s:e] = torch.norm(fk_pos - target_pos_base[s:e], dim=1)
    feasible = res < threshold
    return res, feasible


# ──────────────────────────────────────────────────────────────────────────
# Isaac 없이 판정기만 자체검증
# ──────────────────────────────────────────────────────────────────────────
def run_self_test():
    import torch
    from franka_tensor_ik import FrankaTensorIK

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ik = FrankaTensorIK(dev)
    print("=" * 70)
    print("DRY SELF-TEST: IK-feasibility classifier (Isaac 불필요)")

    # frame 변환 정합성: base=identity 이면 world target == base target
    ee_w = torch.tensor([[0.4, 0.0, 0.5]], device=dev)
    ee_q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev)
    b_p = torch.zeros(1, 3, device=dev)
    b_q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev)
    p_b, R_b = world_to_base(ee_w, ee_q, b_p, b_q)
    ok_frame = torch.allclose(p_b, ee_w, atol=1e-5) and torch.allclose(R_b, torch.eye(3, device=dev).unsqueeze(0), atol=1e-5)
    print(f"  frame identity check: {'PASS' if ok_frame else 'FAIL'} (p_base={p_b.tolist()})")

    # base 이동/회전 변환 검증: 알고있는 pose 를 base frame 으로 돌렸다가 되돌림
    b_p2 = torch.tensor([[1.0, 2.0, 0.3]], device=dev)
    b_q2 = torch.tensor([[0.9238795, 0.0, 0.0, 0.3826834]], device=dev)  # 45deg about z
    world_target = torch.tensor([[1.3, 2.1, 0.7]], device=dev)
    import isaaclab.utils.math as mu
    p_b2, _ = world_to_base(world_target, ee_q, b_p2, b_q2)
    back = mu.quat_apply(b_q2, p_b2) + b_p2
    ok_rt = torch.allclose(back, world_target, atol=1e-5)
    print(f"  world->base->world roundtrip: {'PASS' if ok_rt else 'FAIL'}")

    # feasible vs infeasible: 도달가능 target 과 2m-away target
    tgt = torch.tensor([[0.4, 0.0, 0.5], [2.0, 0.0, 0.5]], device=dev)
    R = torch.eye(3, device=dev).unsqueeze(0).repeat(2, 1, 1)
    res, feas = classify_ik_feasible(ik, tgt, R, threshold=0.03, max_iter=200, tol=1e-4)
    print(f"  residuals: reachable={res[0].item():.4f}m  unreachable={res[1].item():.4f}m")
    print(f"  labels:    reachable feasible={bool(feas[0])} (expect True), "
          f"unreachable feasible={bool(feas[1])} (expect False)")
    ok_cls = bool(feas[0]) and (not bool(feas[1]))
    print(f"  classification: {'PASS' if ok_cls else 'FAIL'}")
    print("=" * 70)
    print("SELF-TEST RESULT:", "ALL PASS" if (ok_frame and ok_rt and ok_cls) else "FAILURE")
    return ok_frame and ok_rt and ok_cls


# ──────────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--num_steps", type=int, default=300)
    # ★ eval_cluttered 와 동일한 기본값(0.05/0.05). ckpt 의 actor.action_scale 버퍼가
    #   load_state_dict 로 덮어쓰므로 실제로는 ckpt 값이 사용됨(= eval_cluttered 와 완전 동일).
    parser.add_argument("--action_scale_pos", type=float, default=0.05)
    parser.add_argument("--action_scale_rot", type=float, default=0.05)
    parser.add_argument("--obstacle_frac", type=float, default=1.0)
    parser.add_argument("--max_active_obstacles", type=int, default=0,
                        help="장애물 활성 상한. 기본 0 = obstacle-free(진단 기본). env 기본 쓰려면 4.")
    parser.add_argument("--num_rounds", type=int, default=2,
                        help="GNN message-passing rounds (GNN 체크포인트).")
    parser.add_argument("--ik_threshold", type=float, default=0.03,
                        help="IK residual(m) 임계값. residual>thr → infeasible(no IK).")
    parser.add_argument("--ik_max_iter", type=int, default=100)
    parser.add_argument("--self_test", action="store_true",
                        help="Isaac 없이 IK-feasibility 판정기만 자체검증.")
    # AppLauncher args 는 Isaac 경로에서만 붙임
    args, _unknown = parser.parse_known_args()

    if args.self_test:
        ok = run_self_test()
        raise SystemExit(0 if ok else 1)

    if not args.model_path:
        raise SystemExit("--model_path 필요 (또는 --self_test)")

    # ── Isaac 부팅 ──────────────────────────────────────────────────────
    from isaaclab.app import AppLauncher
    parser2 = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser2)
    app_args, _ = parser2.parse_known_args()
    # 위 파서에서 정의한 args 값을 app_args 로 병합
    for k, v in vars(args).items():
        setattr(app_args, k, v)
    app = AppLauncher(app_args)
    sim_app = app.app

    import torch
    import numpy as np
    from dual_arm_transport_env3 import DualrobotEnv, DualrobotCfg
    from franka_tensor_ik import FrankaTensorIK
    import isaaclab.utils.math as math_utils

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ── env 구성 (eval_cluttered 와 동일) ──────────────────────────────
    cfg = DualrobotCfg()
    cfg.scene.num_envs = args.num_envs
    env = DualrobotEnv(cfg, render_mode=None)             # ★ 부팅/씬 로딩 1회
    env._obstacle_curr_frac = args.obstacle_frac
    # obstacle-free 진단: 장애물 활성 상한. eval_cluttered 는 건드리지 않지만
    # 진단 목적상 --max_active_obstacles 0 으로 obstacle-free 롤아웃을 얻는다.
    if args.max_active_obstacles is not None:
        try:
            env.cfg.max_active_obstacles = args.max_active_obstacles
        except Exception:
            pass
        try:
            env.max_active_obstacles = args.max_active_obstacles
        except Exception:
            pass
    A = env.cfg.action_space
    B = args.num_envs
    POS_T, ROT_T = 0.02, math.radians(10)                 # eval_cluttered 와 동일한 success 임계
    settle = getattr(env, "SETTLE_STEPS_AT_RESET", 0)

    # ── 정책 로드 (eval_cluttered 와 동일 구조) ─────────────────────────
    sd = torch.load(os.path.abspath(args.model_path), map_location=dev,
                    weights_only=False)["model"]
    # eval_cluttered 와 동일: constructor 에는 args 기반 scale 을 주지만,
    # load_state_dict 가 ckpt 의 actor.action_scale 버퍼를 덮어쓰므로 실제 사용값은 ckpt 값.
    scale = [args.action_scale_pos] * 3 + [args.action_scale_rot] * 3 + [1.0] * (A - 6)
    if "actor.mean_head.0.weight" in sd:                  # MLP vs GNN 자동 판별 (eval_cluttered 와 동일)
        import mlp_policy
        in_dim = sd["actor.mean_head.0.weight"].shape[1]
        use_full = (in_dim == mlp_policy._state_dim(True))
        use_lean = (in_dim == mlp_policy._state_dim(False, True))
        agent = mlp_policy.MLPSACAgent(action_dim=A, action_scale=scale, hidden_dim=256,
                                       num_hidden_layers=2, use_full_state=use_full,
                                       use_lean_obstacle=use_lean).to(dev)
        mode = "full_state" if use_full else ("lean_obstacle" if use_lean else "rod+global")
        mode = f"MLP/{mode}"
    else:
        import gnn_policy
        agent = gnn_policy.GNNSACAgent(action_dim=A, num_rounds=args.num_rounds,
                                       action_scale=scale).to(dev)
        mode = "GNN"
    agent.load_state_dict(sd)                             # ★ action_scale 버퍼도 ckpt 값으로 덮어씀
    agent.eval()
    used_scale = agent.actor.action_scale.detach().cpu().tolist()
    print("=" * 70)
    print(f"IK-feasibility diag  |  {os.path.basename(args.model_path)}  "
          f"policy={mode}  num_envs={B}  steps={args.num_steps}  thr={args.ik_threshold}m")
    print(f"  action_scale (load_state_dict 후, 실사용): {[round(s, 3) for s in used_scale]}")
    print(f"  obstacle_frac={args.obstacle_frac}  max_active_obstacles={args.max_active_obstacles}")

    ik = FrankaTensorIK(dev)

    # ── 롤아웃 (eval_cluttered.run_eval 를 그대로 복제 + EE target 기록) ──
    env.reset()
    batch = env._build_policy_batch()

    # eval_cluttered 의 에피소드 집계 상태 (동일)
    rr_min_pos = torch.full((B,), float("inf"), device=dev)
    rr_min_rot = torch.full((B,), float("inf"), device=dev)
    rr_len = torch.zeros(B, dtype=torch.long, device=dev)

    # ── EE target 기록 (얹은 부분) ──
    # 각 (step, env, arm) 로 base-frame EE target(pos, rot) + reach + running episode id 기록.
    # 에피소드 라벨(success)은 done 시점에 eval_cluttered 와 동일 기준으로 확정 → back-fill.
    rec_pos = []      # list of (2B,3) base-frame pos  (arm1 concat arm2)
    rec_rot = []      # list of (2B,3,3)
    rec_reach = []    # list of (2B,) ||EE-base||
    rec_epid = []     # list of (2B,) long  running episode id per (env,arm)

    ep_label = {}     # global_epid(int) -> success(bool)   (genuine 에피소드만 등재)
    global_epid = torch.arange(B, dtype=torch.long, device=dev)  # env 별 현재 진행 episode id
    next_id = B

    import time
    t0 = time.time()
    for step in range(args.num_steps):
        # ── 정책 action (eval_cluttered 와 동일: deterministic) ──
        with torch.no_grad():
            action, _, _ = agent.actor.get_action_and_log_prob(batch, deterministic=True)
            _, _, term, trunc, _ = env.step(action)

        # settle step 마스크 (eval_cluttered 와 동일한 노출 사용)
        nonsettle = ~env._is_settle_step if hasattr(env, "_is_settle_step") \
            else torch.ones(B, dtype=torch.bool, device=dev)

        # ── success 집계 (eval_cluttered 와 동일: rod↔goal min pos/rot 누적) ──
        rod_pos = env.rod.data.root_pos_w
        goal_pos = env.goal_rod_marker.data.root_pos_w
        pos_err = torch.norm(goal_pos - rod_pos, dim=-1)
        q_diff = math_utils.quat_mul(env.goal_rod_marker.data.root_quat_w,
                                     math_utils.quat_conjugate(env.rod.data.root_quat_w))
        rot_err = 2.0 * torch.atan2(torch.norm(q_diff[:, 1:4], dim=-1), torch.abs(q_diff[:, 0]))
        rr_min_pos = torch.min(rr_min_pos, pos_err)
        rr_min_rot = torch.min(rr_min_rot, rot_err)
        rr_len += 1

        # ── EE target 기록 (settle step 은 target 동결이라 제외) ──
        ee1_p, ee1_q, ee2_p, ee2_q = env.controller._compute_ee_targets(
            env.target_obj_pos, env.target_obj_quat
        )
        b1_p = env.robot_1.data.root_pos_w
        b1_q = env.robot_1.data.root_quat_w
        b2_p = env.robot_2.data.root_pos_w
        b2_q = env.robot_2.data.root_quat_w
        p1_b, R1_b = world_to_base(ee1_p, ee1_q, b1_p, b1_q)
        p2_b, R2_b = world_to_base(ee2_p, ee2_q, b2_p, b2_q)
        reach1 = torch.norm(ee1_p - b1_p, dim=-1)
        reach2 = torch.norm(ee2_p - b2_p, dim=-1)

        keep = nonsettle                                  # settle step 제외 (genuine 만 기록)
        if keep.any():
            ki = keep.nonzero(as_tuple=True)[0]
            rec_pos.append(torch.cat([p1_b[ki], p2_b[ki]], dim=0))
            rec_rot.append(torch.cat([R1_b[ki], R2_b[ki]], dim=0))
            rec_reach.append(torch.cat([reach1[ki], reach2[ki]], dim=0))
            rec_epid.append(global_epid[ki].repeat(2))

        # ── 종료 처리 (eval_cluttered 와 동일: rr_len>settle 인 genuine 만 라벨 + 카운트) ──
        done = term | trunc
        if done.any():
            for i in done.nonzero(as_tuple=True)[0].tolist():
                if rr_len[i].item() > settle:
                    succ = (rr_min_pos[i].item() < POS_T) and (rr_min_rot[i].item() < ROT_T)
                    ep_label[int(global_epid[i].item())] = bool(succ)
                # eval_cluttered 와 동일: 상태 리셋
                rr_min_pos[i] = float("inf")
                rr_min_rot[i] = float("inf")
                rr_len[i] = 0
            # 종료된 env 에 새 unique episode id 발급 (다음 에피소드 기록용)
            di = done.nonzero(as_tuple=True)[0]
            n = di.numel()
            global_epid[di] = torch.arange(next_id, next_id + n, device=dev, dtype=torch.long)
            next_id += n

        batch = env._build_policy_batch()

    # ── 롤아웃 success rate (sanity gate: eval_cluttered ~93% obstacle-free 와 일치해야 함) ──
    genuine_ids = list(ep_label.keys())
    n_ep = len(genuine_ids)
    n_succ_ep = sum(1 for e in genuine_ids if ep_label[e])
    n_fail_ep = n_ep - n_succ_ep
    succ_rate = (100.0 * n_succ_ep / n_ep) if n_ep else 0.0
    print("=" * 70)
    print(f"[ROLLOUT SANITY GATE]  eval_cluttered 롤아웃 복제 결과 ({time.time()-t0:.0f}s)")
    print(f"  genuine finished episodes: {n_ep}  (success={n_succ_ep}, fail={n_fail_ep})")
    print(f"  ROLLOUT SUCCESS RATE (도달, min_pos<2cm & min_rot<10°): {succ_rate:.1f}%")
    print(f"  → 이 값이 eval_cluttered 의 success(~93% obstacle-free)와 일치해야 IK 숫자가 유효.")

    # ── 배치 IK ────────────────────────────────────────────────────────
    if not rec_pos:
        print("기록된 EE target 없음 (모든 step 이 settle?). 종료.")
        os._exit(0)
    all_pos = torch.cat(rec_pos, dim=0)     # (M,3)
    all_rot = torch.cat(rec_rot, dim=0)     # (M,3,3)
    all_reach = torch.cat(rec_reach, dim=0)  # (M,)
    all_epid = torch.cat(rec_epid, dim=0)   # (M,)
    M = all_pos.shape[0]
    print("-" * 70)
    print(f"Recorded {M} EE targets (settle 제외, 2 arms). Running batched IK...")

    res, feas = classify_ik_feasible(ik, all_pos, all_rot, threshold=args.ik_threshold,
                                     max_iter=args.ik_max_iter, tol=1e-4)
    infeasible = ~feas

    # 각 레코드에 대응하는 에피소드 라벨. genuine(=종료 라벨 확정)만 통계에 포함.
    labeled = torch.tensor([e in ep_label for e in all_epid.tolist()],
                           dtype=torch.bool, device=dev)
    succ_arr = torch.tensor([ep_label.get(int(e), False) for e in all_epid.tolist()],
                            dtype=torch.bool, device=dev)

    # ── 리포트 ─────────────────────────────────────────────────────────
    def _stats(mask):
        n = int(mask.sum().item())
        if n == 0:
            return 0, float("nan"), float("nan"), float("nan")
        inf_rate = 100.0 * infeasible[mask].float().mean().item()
        mean_reach = all_reach[mask].mean().item()
        max_reach = all_reach[mask].max().item()
        return n, inf_rate, mean_reach, max_reach

    succ_mask = labeled & succ_arr
    fail_mask = labeled & (~succ_arr)
    n_s, inf_s, mr_s, xr_s = _stats(succ_mask)
    n_f, inf_f, mr_f, xr_f = _stats(fail_mask)
    n_all, inf_all, mr_all, xr_all = _stats(labeled)

    print("=" * 70)
    print("IK-FEASIBILITY DIAGNOSTIC REPORT")
    print(f"  IK residual threshold: {args.ik_threshold} m  (residual>thr => NO IK / unreachable)")
    print(f"  genuine finished episodes: {n_ep}  (success={n_succ_ep}, fail={n_fail_ep})")
    print(f"  (통계는 종료 라벨이 확정된 genuine 에피소드의 EE target 만 포함)")
    print("-" * 70)
    print(f"  {'group':<10}{'#EEtargets':>12}{'infeasible%':>14}{'mean_reach':>12}{'max_reach':>12}")
    print(f"  {'SUCCESS':<10}{n_s:>12}{inf_s:>13.2f}%{mr_s:>11.3f}m{xr_s:>11.3f}m")
    print(f"  {'FAIL':<10}{n_f:>12}{inf_f:>13.2f}%{mr_f:>11.3f}m{xr_f:>11.3f}m")
    print(f"  {'OVERALL':<10}{n_all:>12}{inf_all:>13.2f}%{mr_all:>11.3f}m{xr_all:>11.3f}m")
    print("-" * 70)
    print(f"  SUMMARY: SUCCESS episodes: infeasible={inf_s:.1f}%, mean_reach={mr_s:.3f}m | "
          f"FAIL episodes: infeasible={inf_f:.1f}%, mean_reach={mr_f:.3f}m")
    if not math.isnan(inf_f) and not math.isnan(inf_s):
        if inf_f > 2.0 * max(inf_s, 1e-6):
            print("  INTERPRETATION: FAIL >> SUCCESS in infeasible rate "
                  "=> policy commands unreachable (no-IK) poses on failures.")
        else:
            print("  INTERPRETATION: infeasible rate similar between success/fail "
                  "=> failures NOT primarily driven by unreachable (no-IK) commands.")
    print("=" * 70)

    os._exit(0)


if __name__ == "__main__":
    main()
