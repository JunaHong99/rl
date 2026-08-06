"""리더-팔로워 정책 운반을 mp4 녹화 (headless 서버, --headless + --enable_cameras).

eval_lf의 정책 구동을 그대로 쓰되 DirectRLEnv.render()(rgb_array)로 프레임 잡아 cv2 mp4 저장.
env N개(멀리 배치) 중 카메라를 순차로 옮겨 한 에피소드씩 녹화 → 여러 운반 사례 수록.
프레임에 [rod-goal 오차, 도달여부] 오버레이.

사용 (⚠️ headless라도 --enable_cameras 필수):
  export LD_LIBRARY_PATH=/home/hjh/anaconda3/envs/env-isaaclab/lib:$LD_LIBRARY_PATH
  python -u record_lf.py --model_path logs/<run>/model_final.pt --headless --enable_cameras --out lf_replay.mp4
결과: lf_replay.mp4. scp로 받아서 재생.
"""
import argparse, math, os, time
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--out", type=str, default="lf_replay.mp4")
parser.add_argument("--n_clips", type=int, default=6, help="녹화할 env(=운반 사례) 수. 카메라가 순차로 비춤.")
parser.add_argument("--steps_per_env", type=int, default=40, help="env당 녹화 스텝(에피소드 ~30 + 여유).")
parser.add_argument("--fps", type=int, default=5, help="제어 5Hz라 fps=5면 실시간(1x), 10이면 2x배속.")
parser.add_argument("--res_w", type=int, default=1280)
parser.add_argument("--res_h", type=int, default=720)
parser.add_argument("--env_spacing", type=float, default=4.0)
parser.add_argument("--use_kin_graph", action="store_true", help="kinematic 그래프 + KinGNN 로드.")
parser.add_argument("--num_rounds", type=int, default=8)
parser.add_argument("--vary_grasp", action="store_true", help="파지 변이(학습과 일치). 없으면 고정파지.")
parser.add_argument("--same_side", action="store_true")
parser.add_argument("--no_gravity", action="store_true", help="중력 OFF(학습과 일치).")
parser.add_argument("--replay_cases", type=str, default=None, help="eval --save_failures로 저장한 실패 config(.pt) 재생 → 실패 에피소드만 녹화.")
parser.add_argument("--randomize_base", action="store_true", help="베이스 간격+yaw 랜덤화(학습과 일치).")
parser.add_argument("--base_spacing_min", type=float, default=0.8)
parser.add_argument("--base_spacing_max", type=float, default=1.4)
parser.add_argument("--base_yaw_range", type=float, default=0.2618)
parser.add_argument("--grasp_preset", type=str, default=None, help="held-out 파지 세트(.pt).")
parser.add_argument("--cache_size", type=int, default=20000, help="reset용 pose 캐시.")
parser.add_argument("--seed", type=int, default=0, help="matched 비교용: reset 재현 seed. GNN·MLP 같은 seed면 동일 에피소드.")
parser.add_argument("--min_start_dist", type=float, default=0.15, help="녹화 에피소드 최소 시작-골 거리[m] (trivial 배제).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args); sim_app = app.app

import torch, numpy as np, cv2
from dual_arm_transport_env3 import DualrobotEnv, DualrobotCfg
import mlp_policy, graph_converter as gc
import isaaclab.utils.math as math_utils

# ★ 렌더링엔 CUDA_VISIBLE_DEVICES 금지(GLX 깨짐) → --device cuda:N 사용. torch도 이 device로 통일.
dev = getattr(args, "device", "cuda") if torch.cuda.is_available() else "cpu"
N = args.n_clips
cfg = DualrobotCfg()
# ★ 실패 재생: 저장된 실패 config를 로드, num_envs=재생 사례 수로 맞춤.
_replay = None
if args.replay_cases:
    _rd = torch.load(args.replay_cases, map_location="cpu")
    _rs = _rd["samples"]
    _ncase = min(int(next(iter(_rs.values())).shape[0]), args.n_clips)
    _replay = {k: v[:_ncase] for k, v in _rs.items()}
    N = _ncase
    print(f"🎬 실패 재생: {args.replay_cases}에서 {_ncase}개 (총 {int(next(iter(_rs.values())).shape[0])}개 중)")
# vary_grasp면 env마다 다른 버킷(=다른 grasp) → 클립마다 다른 env 녹화 위해 num_envs 확보.
cfg.scene.num_envs = N if _replay is not None else (max(8, args.n_clips) if args.vary_grasp else 4)
cfg.scene.env_spacing = args.env_spacing
cfg.viewer.resolution = (args.res_w, args.res_h)
cfg.leader_follower = True
cfg.action_space = 7
cfg.n_obstacles = 0
if args.use_kin_graph:
    cfg.use_kin_graph = True
else:
    gc.GLOBAL_FEATURE_DIM = 1 + 14 + 1 + 3 + 6
# ★ 학습 조건 일치 (안 맞추면 mismatch로 이상하게 나옴)
if args.vary_grasp:
    cfg.vary_grasp = True; cfg.grasp_same_side = args.same_side
if args.randomize_base:
    cfg.randomize_base = True
    cfg.base_spacing_range = (args.base_spacing_min, args.base_spacing_max)
    cfg.base_yaw_range = args.base_yaw_range
    print(f"🤖 베이스 랜덤화 ON (record): 간격 {args.base_spacing_min}~{args.base_spacing_max}m, yaw ±{args.base_yaw_range:.3f}")
if args.no_gravity:
    cfg.disable_gravity_all = True
if args.grasp_preset:
    cfg.grasp_preset_path = args.grasp_preset
cfg.pose_cache_size = args.cache_size
env = DualrobotEnv(cfg, render_mode="rgb_array")
# ★ 실패 재생 주입: external_samples(base/q_start/q_goal/obj pose) + grasp 버킷 정합.
if _replay is not None:
    bk = _replay.get("bucket", None)
    env.external_samples = {k: v.to(dev) for k, v in _replay.items() if k != "bucket"}
    if bk is not None and getattr(env, "_grasp_bucket_d", None) is not None:
        bk = bk.to(dev).long()
        env._grasp_bucket_idx = bk
        env._grasp_d = env._grasp_bucket_d[bk]
        env._grasp_theta = env._grasp_bucket_theta[bk]
    print(f"🎬 external_samples 주입 완료 ({env.num_envs} env = 실패 사례)")
A = env.cfg.action_space
POS_T, ROT_T = 0.02, math.radians(10)
origins = env.scene.env_origins

sd = torch.load(os.path.abspath(args.model_path), map_location=dev, weights_only=False)["model"]
scale = [float(cfg.joint_dq_scale)] * A
if args.use_kin_graph:
    import gnn_policy_kin
    is_kinmlp = any("actor.mean_head" in k for k in sd.keys())   # 체크포인트 키로 자동판별
    if is_kinmlp:
        agent = gnn_policy_kin.KinMLPSACAgent(action_scale=scale).to(dev)
        print("🧠 KinMLP 로드 (flatten ablation)")
    else:
        agent = gnn_policy_kin.KinSACAgent(num_rounds=args.num_rounds, action_scale=scale).to(dev)
        print(f"🕸️ KinGNN 로드 (rounds={args.num_rounds})")
else:
    agent = mlp_policy.MLPSACAgent(action_dim=A, action_scale=scale, hidden_dim=256,
                                   num_hidden_layers=2, use_full_state=False, use_lean_obstacle=False).to(dev)
agent.load_state_dict(sd); agent.eval()
print(f"▶ 녹화: {os.path.basename(args.model_path)}  clips={N}  → {args.out}")

writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.res_w, args.res_h))


def look_at(i):
    o = origins[i].tolist()
    env.sim.set_camera_view(eye=(o[0]+1.8, o[1]+1.8, o[2]+1.4), target=(o[0], o[1], o[2]+0.4))


ROT_T_DEG = math.degrees(ROT_T)
settle = getattr(env, "SETTLE_STEPS_AT_RESET", 0)
total_steps = settle + args.steps_per_env
NP = env.num_envs                             # 병렬 env 수(=4)
look_at(0)
writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.res_w, args.res_h))
t0 = time.time()


def rollout_record(record_env, min_start_dist=0.0):
    """한 batch rollout. record_env가 None이면 렌더 없이 성공판정만(빠름).
       int면 그 env의 프레임을 오버레이해 buf 반환. 반환: (buf, best_p(NP,), best_r(NP,))."""
    # ★ 녹화 대상 env의 시작-골 거리 >= min_start_dist 될 때까지 reset (seed 고정 시 재현 → 모델 무관 동일 에피소드).
    while True:
        env.reset()
        if record_env is None or min_start_dist <= 0:
            break
        rp = env.rod.data.root_pos_w; gp = env.goal_rod_marker.data.root_pos_w
        if (gp[record_env] - rp[record_env]).norm().item() >= min_start_dist:
            break
    batch = env._build_policy_batch()
    bp = np.full(NP, 9.9); br = np.full(NP, 999.0)
    reached_once = np.zeros(NP, dtype=bool); buf = []
    start_perr = None
    ci = 0 if record_env is None else record_env
    for s in range(total_steps):
        if not sim_app.is_running():
            break
        with torch.no_grad():
            action, _, _ = agent.actor.get_action_and_log_prob(batch, deterministic=True)
            _, _, term, trunc, _ = env.step(action)
        # ★ done이면 env가 이미 자동 리셋 → 지금 rod/goal은 새 에피소드(텔레포트). 렌더/판정 말고 즉시 종료.
        #   ★ 녹화 대상 env(ci) 기준으로 break해야 함 (env0 하드코딩 시 tenv≠0 클립에 다른 에피소드 프레임 섞임).
        if s >= settle and (bool(term[ci].item()) or bool(trunc[ci].item())):
            break
        rod = env.rod.data.root_pos_w; goal = env.goal_rod_marker.data.root_pos_w
        perr = (goal - rod).norm(dim=-1).cpu().numpy()
        qd = math_utils.quat_mul(env.goal_rod_marker.data.root_quat_w,
                                 math_utils.quat_conjugate(env.rod.data.root_quat_w))
        rerr = np.degrees(2.0 * np.arctan2(qd[:, 1:4].norm(dim=-1).cpu().numpy(), qd[:, 0].abs().cpu().numpy()))
        if s >= settle:                        # settle 스킵
            if start_perr is None:             # 첫 유효스텝 시작 위치오차 기록
                start_perr = perr.copy()
            bp = np.minimum(bp, perr); br = np.minimum(br, rerr)
            reached_once |= (perr < POS_T) & (rerr < ROT_T_DEG)   # ★ 이 스텝에 동시 만족
            if record_env is not None:
                frame = env.render()
                if frame is not None and frame.size > 0:
                    img = np.ascontiguousarray(frame[:, :, :3])
                    now = (perr[ci] < POS_T) and (rerr[ci] < ROT_T_DEG)   # 지금 순간 도달중?
                    col = (0, 220, 0) if now else (255, 210, 0)
                    cv2.putText(img, f"kin leader-follower (matched eval)", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(img, f"pos={perr[ci]*100:.1f}cm(<2)  rot={rerr[ci]:.0f}deg(<10)  {'REACHED' if now else '...'}",
                                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)
                    buf.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        batch = env._build_policy_batch()
    sp = float(start_perr[ci]) if start_perr is not None else -1.0
    return buf, bp, br, reached_once, sp


# ★ matched 녹화: seed 고정으로 GNN·MLP가 정확히 같은 에피소드(start/goal/grasp)를 보게 함.
#   성공/실패 무관 녹화(cherry-pick 안 함, 정직) + min_start_dist로 trivial 배제.
#   start/goal은 policy 이전(reset)에 정해지므로 seed 같으면 모델과 무관하게 동일 → 공정 비교.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
collected = 0
for c in range(N):
    if not sim_app.is_running():
        break
    tenv = c % NP                         # 클립마다 다른 env = 다른 grasp 버킷
    look_at(tenv)                         # 카메라를 그 env로
    buf, bp, br, reached, sp = rollout_record(record_env=tenv, min_start_dist=args.min_start_dist)
    for f in buf:
        writer.write(f)
    collected += 1
    tag = "REACHED" if bool(reached[tenv]) else "MISS"
    print(f"  clip {c+1}/{N} (env{tenv}, {tag})  시작{sp*100:.0f}cm  min pos={bp[tenv]*100:.1f}cm rot={br[tenv]:.0f}deg  ({len(buf)}프레임)")

writer.release()
print(f"✅ 저장: {args.out}  {collected}/{N} 클립 (matched seed={args.seed}, min_dist={args.min_start_dist}m, {time.time()-t0:.0f}s)")
env.close(); sim_app.close()
