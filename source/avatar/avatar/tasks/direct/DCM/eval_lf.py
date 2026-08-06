"""리더-팔로워(또는 joint_action) 모델 진단 eval — success + rod가 움직이나(collapse 판별).

train_phase3_sac.py의 leader_follower/joint 구성을 그대로 재현(GLOBAL_DIM 갱신, action_scale, MLP).
학습이 정체(ep_rew flat)한 게 (a) 정책이 rod를 못 움직여 붕괴인지 (b) 움직이나 목표 도달만 못하는지 판별.

사용:
  export LD_LIBRARY_PATH=/home/hjh/anaconda3/envs/env-isaaclab/lib:$LD_LIBRARY_PATH
  python -u eval_lf.py --model_path logs/<run>/model_step_016384000.pt --leader_follower --num_envs 256 --headless
"""
import argparse, math, time
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--num_steps", type=int, default=300)
parser.add_argument("--leader_follower", action="store_true")
parser.add_argument("--joint_action", action="store_true")
parser.add_argument("--use_kin_graph", action="store_true", help="kinematic 그래프 + KinGNN 정책 로드.")
parser.add_argument("--num_rounds", type=int, default=8, help="KinGNN message-passing rounds(학습과 일치).")
parser.add_argument("--vary_grasp", action="store_true", help="파지 변이(학습과 일치). 없으면 고정파지.")
parser.add_argument("--same_side", action="store_true")
parser.add_argument("--no_gravity", action="store_true", help="중력 OFF(학습과 일치).")
parser.add_argument("--grasp_preset", type=str, default=None, help="held-out 파지 세트(.pt, gen_grasp로 생성). 없으면 학습 8버킷.")
parser.add_argument("--cache_size", type=int, default=20000, help="reset용 pose 캐시(eval은 작게=빠름. 학습은 100k). preset 캐시 생성 가속.")
parser.add_argument("--stochastic", action="store_true", help="탐험 확인용(기본 deterministic).")
parser.add_argument("--lf_ik_iters", type=int, default=None, help="팔로워 IK 반복수 override(기본 cfg=12). IK잔차 원인 진단용(예: 50).")
parser.add_argument("--lf_ik_rate", type=int, default=1, help="팔로워 IK control-rate(학습과 일치시켜야 함). 1=현재, 2~4=서브스텝 재추종.")
parser.add_argument("--single_action_scale", action="store_true", help="스케일 이중적용 수정(학습과 일치). single_action_scale로 학습한 모델 eval 시 필수.")
parser.add_argument("--use_near_goal_fine", action="store_true", help="목표 근처 Δq 축소(근접 정밀 테스트). travel은 유지.")
parser.add_argument("--fine_gate", type=float, default=3.0)
parser.add_argument("--fine_min_scale", type=float, default=0.3)
parser.add_argument("--oracle", action="store_true", help="정책 대신 리더를 정답 q*(target_joint_pos)로 P-제어 → 컨트롤러/닫힌사슬 도달 상한 측정(정책 vs 컨트롤러 한계 판별).")
parser.add_argument("--oracle_cap", type=float, default=1.0, help="오라클 스텝당 액션 clamp(작을수록 완만=팔로워 추종 쉬움). 1.0=최대속도, 0.2≈정책속도.")
parser.add_argument("--episode_len_s", type=float, default=None, help="에피소드 길이[s] override(기본6=30step). 오라클 컨트롤러 천장 테스트용(예:20=100step).")
parser.add_argument("--save_failures", type=str, default=None, help="실패 에피소드 config를 .pt로 저장(external_samples 형식) → 나중에 record_lf --replay_cases로 재생/시각화.")
parser.add_argument("--randomize_base", action="store_true", help="베이스 간격+yaw 랜덤화(학습과 일치). 별도 _basevar 캐시.")
parser.add_argument("--base_spacing_min", type=float, default=0.8)
parser.add_argument("--base_spacing_max", type=float, default=1.4)
parser.add_argument("--base_yaw_range", type=float, default=0.2618)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args); sim_app = app.app

import torch, os, numpy as np
from dual_arm_transport_env3 import DualrobotEnv, DualrobotCfg
import mlp_policy, graph_converter as gc
import isaaclab.utils.math as math_utils

dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = DualrobotCfg()
cfg.scene.num_envs = args.num_envs
cfg.n_obstacles = 0
if args.leader_follower:
    cfg.leader_follower = True; cfg.action_space = 7
if args.joint_action:
    cfg.joint_action = True; cfg.action_space = 14
if args.use_kin_graph:
    cfg.use_kin_graph = True    # leader_follower와 함께
# ★ 학습 조건 일치 (안 맞추면 mismatch로 결과 무의미)
if args.vary_grasp:
    cfg.vary_grasp = True; cfg.grasp_same_side = args.same_side
if args.randomize_base:
    cfg.randomize_base = True
    cfg.base_spacing_range = (args.base_spacing_min, args.base_spacing_max)
    cfg.base_yaw_range = args.base_yaw_range
    print(f"🤖 베이스 랜덤화 ON (eval): 간격 {args.base_spacing_min}~{args.base_spacing_max}m, yaw ±{args.base_yaw_range:.3f}rad")
if args.no_gravity:
    cfg.disable_gravity_all = True
if args.grasp_preset:
    cfg.grasp_preset_path = args.grasp_preset   # held-out 파지 주입(랜덤 8버킷 대신)
cfg.pose_cache_size = args.cache_size           # eval은 작게 → preset 캐시 생성 빠름
if args.lf_ik_iters is not None:
    cfg.lf_ik_iters = args.lf_ik_iters          # 팔로워 IK 반복수 override (IK잔차 원인 진단)
    print(f"🔧 lf_ik_iters override → {cfg.lf_ik_iters}")
cfg.lf_ik_rate = args.lf_ik_rate
if args.lf_ik_rate > 1:
    print(f"⏱️ control-rate {args.lf_ik_rate} (eval): 서브스텝 재추종")
cfg.single_action_scale = args.single_action_scale
if args.single_action_scale:
    print(f"🔧 single_action_scale ON (eval): Δq=joint_dq_scale·tanh")
if args.use_near_goal_fine:
    cfg.use_near_goal_fine = True; cfg.fine_gate = args.fine_gate; cfg.fine_min_scale = args.fine_min_scale
    print(f"🔬 near-goal fine ON (eval): gate={args.fine_gate} min_scale={args.fine_min_scale}")
if args.episode_len_s is not None:
    cfg.episode_length_s = args.episode_len_s   # 에피소드 길이 override (오라클 컨트롤러 천장 테스트)
    print(f"🔧 episode_length_s override → {cfg.episode_length_s} ({int(args.episode_len_s/0.2)} step)")
# train과 동일: leader_follower=time+리더sin/cos(14)+f_int+base pos(3)+rot6d(6)=25, joint=time+28+1=30
if args.leader_follower and not args.use_kin_graph:
    gc.GLOBAL_FEATURE_DIM = 1 + 14 + 1 + 3 + 6
elif not args.use_kin_graph:
    gc.GLOBAL_FEATURE_DIM = 1 + 28 + 1
env = DualrobotEnv(cfg, render_mode=None)
A = env.cfg.action_space
B = args.num_envs
POS_T, ROT_T = 0.02, math.radians(10)
settle = getattr(env, "SETTLE_STEPS_AT_RESET", 0)
scale = [float(cfg.joint_dq_scale)] * A

sd = torch.load(os.path.abspath(args.model_path), map_location=dev, weights_only=False)["model"]
if args.use_kin_graph:
    import gnn_policy_kin
    # 체크포인트 키로 KinGNN(joint_head) vs KinMLP(mean_head) 자동 판별.
    is_kinmlp = any("actor.mean_head" in k for k in sd.keys())
    if is_kinmlp:
        agent = gnn_policy_kin.KinMLPSACAgent(action_scale=scale).to(dev)
        print("🧠 KinMLP 로드 (flatten ablation)")
    else:
        agent = gnn_policy_kin.KinSACAgent(num_rounds=args.num_rounds, action_scale=scale).to(dev)
        print(f"🕸️ KinGNN 로드 (rounds={args.num_rounds})")
    agent.load_state_dict(sd); agent.eval()
else:
    agent = mlp_policy.MLPSACAgent(action_dim=A, action_scale=scale, hidden_dim=256,
                                   num_hidden_layers=2, use_full_state=False, use_lean_obstacle=False).to(dev)
    agent.load_state_dict(sd); agent.eval()
print("=" * 70)
print(f"✅ {os.path.basename(args.model_path)}  action_dim={A}  global_dim={gc.GLOBAL_FEATURE_DIM}  "
      f"{'stochastic' if args.stochastic else 'deterministic'}")

env.reset(); batch = env._build_policy_batch()
rr_min_pos = torch.full((B,), float("inf"), device=dev)
rr_min_rot = torch.full((B,), float("inf"), device=dev)
rr_reached = torch.zeros(B, dtype=torch.bool, device=dev)   # ★ 동시 도달(pos&rot 같은 스텝) 여부
rr_len = torch.zeros(B, dtype=torch.long, device=dev)
rod_start = env.rod.data.root_pos_w.clone()
rod_disp_max = torch.zeros(B, device=dev)          # 에피소드 내 rod 최대 이동량
act_abs_sum = 0.0; n_act = 0
ep_succ = []
# ★ 버킷별 성공률 추적 (vary_grasp/preset일 때). env i → 고정 버킷.
bkt_idx = getattr(env, "_grasp_bucket_idx", None)
ep_bkt = []
# ★ 팔로워 IK 잔차·manipulability 진단 (성공 vs 실패 대비 → IK/특이점 가설 검증)
env._lf_diag = True
ep_diag = []   # (success, ikres_pos_mm, ikres_rot_deg, manip_min)
ikp_max = torch.zeros(B, device=dev); ikr_max = torch.zeros(B, device=dev)
manip_min = torch.full((B,), float("inf"), device=dev)
# ── ★ 진동 진단 (H1 네트워크출력 진동 vs H2 협응 진동) ──
#   마지막 OSC_K 스텝(목표 근처)에서 리더 액션 방향반전율(액션flip)·오차 진동(err flip) 측정.
#   실패가 액션flip↑ = H1(출력 진동). 액션flip 낮고 err flip↑ = H2(팔로워 협응이 진동).
OSC_K = 8
act_h = torch.zeros(B, OSC_K, A, device=dev)
pe_h = torch.zeros(B, OSC_K, device=dev)
re_h = torch.zeros(B, OSC_K, device=dev)
ep_osc = []   # (success, act_flip, act_mag, pe_flip, re_flip)
# ★ 스케일 버그 검증: 실제 per-step Δq(=_lf_q1_des 변화) 측정 (ON≈0.15, OFF≈0.045 기대)
q1des_prev = None; dq_sum = 0.0; dq_n = 0
fint_h = torch.zeros(B, OSC_K, device=dev)   # ★ 근접-구간 내력(f_int) 롤링 (제약위반/내력 가설)
ep_fint = []   # (success, near_goal_fint)
jscale = float(cfg.joint_dq_scale)
fail_idxs = []   # ★ 실패 에피소드의 cache 글로벌 인덱스 (--save_failures로 저장 → 나중에 재생)
ep_pr = []       # ★ (reached, min_pos, min_rot) — 실패 pos/rot/timing 분해
for step in range(args.num_steps):
    cur_idx = env.current_sample_idxs.clone()   # 이 스텝에 도는 에피소드의 cache idx (reset 전 스냅샷)
    with torch.no_grad():
        if args.oracle:
            # 오라클: 정책 대신 리더를 정답 관절 q*(=target_joint_pos 앞7, 골 rod pose의 IK해)로 P-제어.
            #   팔로워 IK가 추종 → rod가 실제 도달하나 측정 = 컨트롤러/닫힌사슬 achievability 상한.
            q1c = env.robot_1.data.joint_pos[:, env.robot_1_joint_ids]
            qstar = env.target_joint_pos[:, :7]
            action = ((qstar - q1c) / max(1e-6, jscale)).clamp(-args.oracle_cap, args.oracle_cap)
        else:
            action, _, _ = agent.actor.get_action_and_log_prob(batch, deterministic=not args.stochastic)
        act_abs_sum += action.abs().mean().item(); n_act += 1
        _, _, term, trunc, _ = env.step(action)
    # 실제 Δq 측정 (스케일 검증)
    if getattr(env, "_lf_q1_des", None) is not None:
        _q1d = env._lf_q1_des
        if q1des_prev is not None:
            dq_sum += (_q1d - q1des_prev).abs().mean().item(); dq_n += 1
        q1des_prev = _q1d.clone()
    rod_pos = env.rod.data.root_pos_w; goal_pos = env.goal_rod_marker.data.root_pos_w
    pos_err = torch.norm(goal_pos - rod_pos, dim=-1)
    q_diff = math_utils.quat_mul(env.goal_rod_marker.data.root_quat_w,
                                 math_utils.quat_conjugate(env.rod.data.root_quat_w))
    rot_err = 2.0 * torch.atan2(torch.norm(q_diff[:, 1:4], dim=-1), torch.abs(q_diff[:, 0]))
    # 진동 진단용 롤링 버퍼 (최근 OSC_K 스텝)
    act_h = torch.cat([act_h[:, 1:], action.detach().unsqueeze(1)], dim=1)
    pe_h = torch.cat([pe_h[:, 1:], pos_err.unsqueeze(1)], dim=1)
    re_h = torch.cat([re_h[:, 1:], rot_err.unsqueeze(1)], dim=1)
    _w1, _w2 = env._get_grasp_wrenches()                                  # 내력(f_int) per-env
    _fint = 0.5 * torch.norm(_w1[:, :3] - _w2[:, :3], dim=-1)
    fint_h = torch.cat([fint_h[:, 1:], _fint.unsqueeze(1)], dim=1)
    rr_min_pos = torch.min(rr_min_pos, pos_err); rr_min_rot = torch.min(rr_min_rot, rot_err)
    rr_reached |= (pos_err < POS_T) & (rot_err < ROT_T)     # ★ 이 스텝에 pos&rot 동시 만족
    rod_disp_max = torch.maximum(rod_disp_max, (rod_pos - rod_start).norm(dim=-1))
    if hasattr(env, "_lf_ik_res_pos"):     # 팔로워 IK 진단 누적 (에피소드 내 최악)
        ikp_max = torch.maximum(ikp_max, env._lf_ik_res_pos)
        ikr_max = torch.maximum(ikr_max, env._lf_ik_res_rot)
        manip_min = torch.minimum(manip_min, env._lf_manip)
    rr_len += 1
    done = term | trunc
    if done.any():
        for i in done.nonzero(as_tuple=True)[0].tolist():
            if rr_len[i].item() > settle:
                ep_succ.append(bool(rr_reached[i].item()))   # 동시 도달 기준
                if bkt_idx is not None:
                    ep_bkt.append(int(bkt_idx[i].item()))
                ep_diag.append((bool(rr_reached[i].item()), float(ikp_max[i]) * 1000.0,
                                float(ikr_max[i]) * 57.3, float(manip_min[i])))
                # 진동 지표: 액션 방향반전율(액션flip), 액션 크기(mag/scale), pos·rot 오차 진동율.
                _a = act_h[i]; _sg = torch.sign(_a)
                _af = ((_sg[1:] * _sg[:-1]) < 0).float().mean().item()          # 액션 부호반전 비율
                _am = _a.abs().mean().item() / max(1e-6, jscale)               # |액션|/scale ∈[0,1]
                _dpe = pe_h[i][1:] - pe_h[i][:-1]; _sp = torch.sign(_dpe)
                _pf = ((_sp[1:] * _sp[:-1]) < 0).float().mean().item()          # pos_err 방향반전
                _dre = re_h[i][1:] - re_h[i][:-1]; _sr = torch.sign(_dre)
                _rf = ((_sr[1:] * _sr[:-1]) < 0).float().mean().item()          # rot_err 방향반전
                ep_osc.append((bool(rr_reached[i].item()), _af, _am, _pf, _rf))
                ep_fint.append((bool(rr_reached[i].item()), float(fint_h[i].mean())))
                # ★ 실패 에피소드의 cache idx 저장 (external 주입/-1 제외)
                ep_pr.append((bool(rr_reached[i].item()), float(rr_min_pos[i]), float(rr_min_rot[i])))
                if args.save_failures and (not rr_reached[i].item()) and int(cur_idx[i]) >= 0:
                    # 정밀도(timing) 실패만 저장: pos·rot 각각 임계 도달했으나 동시 아님
                    _tim = (rr_min_pos[i] < POS_T) and (rr_min_rot[i] < ROT_T)
                    if _tim:
                        fail_idxs.append(int(cur_idx[i]))
            rr_min_pos[i] = float("inf"); rr_min_rot[i] = float("inf"); rr_reached[i] = False
            rr_len[i] = 0
            rod_start[i] = env.rod.data.root_pos_w[i]; rod_disp_max[i] = 0.0
            ikp_max[i] = 0.0; ikr_max[i] = 0.0; manip_min[i] = float("inf")
            act_h[i] = 0.0; pe_h[i] = 0.0; re_h[i] = 0.0; fint_h[i] = 0.0
    batch = env._build_policy_batch()

S = np.array(ep_succ)
print(f"  genuine ep: {len(S)}   success: {100*S.mean() if len(S) else 0:.1f}%")
# ── 실패 분해 (timing=정밀도 실패) ──
if ep_pr:
    PR = np.array(ep_pr); reached = PR[:,0] > 0.5; fail = ~reached
    pok = PR[:,1] < POS_T; rok = PR[:,2] < ROT_T
    timing = fail & pok & rok; posonly = fail & ~pok & rok
    rotonly = fail & pok & ~rok; both = fail & ~pok & ~rok
    nF = int(fail.sum())
    def _r(n): return f"{int(n.sum()):4d} ({100*n.sum()/max(1,nF):4.1f}%)"
    print(f"  ── 실패분해(총{nF}): 타이밍{_r(timing)} pos만{_r(posonly)} rot만{_r(rotonly)} 둘다{_r(both)}")
# ── 버킷별 성공률 (worst 먼저) ──
if bkt_idx is not None and len(ep_bkt):
    eb = np.array(ep_bkt); nb = int(env.cfg.grasp_n_buckets)
    bd = env._grasp_bucket_d.cpu().numpy(); bth = env._grasp_bucket_theta.cpu().numpy()
    rows = []
    for b in range(nb):
        m = eb == b; n = int(m.sum())
        if n: rows.append((100.0 * S[m].mean(), b, n, float(bd[b]), float(bth[b])))
    print("  ── 버킷별 성공률 (낮은 순) ──")
    for sr, b, n, d, th in sorted(rows):
        print(f"    버킷{b:2d}: {sr:5.1f}%  (n={n:4d})  d={d:.3f}m  θ={th:+.3f}rad ({th*57.3:+.0f}°)")
# ── ★팔로워 IK 잔차·특이점: 성공 vs 실패 (IK/특이점 가설 검증) ──
if ep_diag:
    D = np.array([[float(s), ip, ir, mp] for (s, ip, ir, mp) in ep_diag])
    sm = D[:, 0] > 0.5; fm = ~sm
    def _st(m):
        if m.sum() == 0: return "n=0"
        return (f"n={int(m.sum()):5d}  IK잔차 pos={np.median(D[m,1]):5.1f}mm rot={np.median(D[m,2]):4.1f}°  "
                f"manip_min(median)={np.median(D[m,3]):.3f}  (p10={np.percentile(D[m,3],10):.3f})")
    print("  ── 팔로워 IK 잔차·특이점 (성공 vs 실패, median) ──")
    print(f"    성공: {_st(sm)}")
    print(f"    실패: {_st(fm)}")
    print(f"    → 실패가 IK잔차↑ 또는 manip↓면 IK/특이점 원인(가설), 비슷하면 리워드/정밀도")
# ── ★ 진동 진단 (H1 네트워크출력 진동 vs H2 협응 진동) ──
if ep_osc:
    O = np.array([[float(s), af, am, pf, rf] for (s, af, am, pf, rf) in ep_osc])
    sm = O[:, 0] > 0.5; fm = ~sm
    def _o(m):
        if m.sum() == 0: return "n=0"
        return (f"n={int(m.sum()):5d}  액션flip={np.median(O[m,1]):.2f}  액션mag={np.median(O[m,2]):.2f}  "
                f"pos_err flip={np.median(O[m,3]):.2f}  rot_err flip={np.median(O[m,4]):.2f}")
    print(f"  ── 진동 진단 (마지막 {OSC_K}스텝, 성공 vs 실패, median) ──")
    print(f"    성공: {_o(sm)}")
    print(f"    실패: {_o(fm)}")
    print(f"    → 실패 '액션flip'↑ = H1(네트워크 출력 진동). '액션flip' 낮고 'err flip'↑ = H2(팔로워 협응 진동).")
# ── ★ 근접-구간 내력 (내력/제약위반이 동시수렴 방해하는가) ──
if ep_fint:
    F = np.array([[float(s), fv] for (s, fv) in ep_fint])
    sm = F[:, 0] > 0.5; fm = ~sm
    def _f(m):
        return (f"n={int(m.sum()):5d}  f_int(median)={np.median(F[m,1]):6.1f}N  p90={np.percentile(F[m,1],90):6.0f}N"
                if m.sum() else "n=0")
    print(f"  ── 근접 내력 (마지막 {OSC_K}스텝 평균 f_int, 성공 vs 실패) ──")
    print(f"    성공: {_f(sm)}")
    print(f"    실패: {_f(fm)}")
    print(f"    → 실패 내력↑ = 내력/제약위반이 동시수렴 방해(가설 지지). 비슷하면 내력은 주범 아님.")
print(f"  rod 최대이동(에피소드): mean={rod_disp_max.mean()*100:.1f}cm  max={rod_disp_max.max()*100:.1f}cm")
print(f"  액션 |mean|: {act_abs_sum/max(1,n_act):.3f}  (0에 가까우면 정책 붕괴=freeze)")
print(f"  실제 |Δq|/step: {dq_sum/max(1,dq_n):.4f} rad  (single_action_scale ON≈0.15, OFF≈0.045; 액션|mean|×{'1' if args.single_action_scale else 'joint_dq_scale'})")
print(f"  → 진단: rod가 {'거의 안 움직임 → 정책 붕괴/탐험실패' if rod_disp_max.mean()<0.03 else '움직임 → 도달만 못함(탐험/HER)'}")
# ── ★ 실패 케이스 저장 (external_samples 형식, record_lf --replay_cases로 재생) ──
if args.save_failures and fail_idxs:
    cache = env.pose_sampler.cache
    fi = torch.tensor(sorted(set(fail_idxs)), dtype=torch.long, device=dev)
    sub = {k: v[fi].cpu() for k, v in cache.items()
           if isinstance(v, torch.Tensor) and v.shape[0] == next(iter(cache.values())).shape[0]}
    # 버킷 정보도 저장(재생 시 env._grasp_bucket_idx 세팅용 — grasp offset 정합)
    torch.save({"samples": sub, "cache_idx": fi.cpu()}, args.save_failures)
    print(f"  💾 실패 {len(fi)}개 config 저장 → {args.save_failures} "
          f"(record_lf --replay_cases {args.save_failures} 로 재생)")
os._exit(0)
