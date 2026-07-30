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
parser.add_argument("--grasp_preset", type=str, default=None, help="held-out 파지 세트(.pt).")
parser.add_argument("--cache_size", type=int, default=20000, help="reset용 pose 캐시.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args); sim_app = app.app

import torch, numpy as np, cv2
from dual_arm_transport_env3 import DualrobotEnv, DualrobotCfg
import mlp_policy, graph_converter as gc
import isaaclab.utils.math as math_utils

dev = "cuda" if torch.cuda.is_available() else "cpu"
N = args.n_clips
cfg = DualrobotCfg()
cfg.scene.num_envs = 4              # env 0만 녹화(카메라 고정), 나머지는 무시. 시도마다 reset=새 에피소드.
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
if args.no_gravity:
    cfg.disable_gravity_all = True
if args.grasp_preset:
    cfg.grasp_preset_path = args.grasp_preset
cfg.pose_cache_size = args.cache_size
env = DualrobotEnv(cfg, render_mode="rgb_array")
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


def rollout_record(record_env):
    """한 batch rollout. record_env가 None이면 렌더 없이 성공판정만(빠름).
       int면 그 env의 프레임을 오버레이해 buf 반환. 반환: (buf, best_p(NP,), best_r(NP,))."""
    env.reset(); batch = env._build_policy_batch()
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
        if s >= settle and (bool(term[0].item()) or bool(trunc[0].item())):
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
                    cv2.putText(img, f"success clip  kin leader-follower", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(img, f"pos={perr[ci]*100:.1f}cm(<2)  rot={rerr[ci]:.0f}deg(<10)  {'REACHED' if now else '...'}",
                                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)
                    buf.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        batch = env._build_policy_batch()
    sp = float(start_perr[ci]) if start_perr is not None else -1.0
    return buf, bp, br, reached_once, sp


# 렌더 없이 여러 batch 판정 → 성공 env 좌표(batch, env_idx) 수집.
# 근데 batch마다 pose가 새로 랜덤이라 재현 불가 → 판정과 녹화를 한 rollout에서:
# 각 batch를 "env 0 카메라로 녹화"하되, env 0이 성공한 batch만 keep. (렌더는 매 batch 있지만
# 실패해도 렌더 비용은 동일 — 대신 num_envs 병렬로 시도수 자체를 줄임: 한 batch에 4개 판정,
# env 0 성공률 93.7%라 대부분 첫 시도 성공 → 시도수 급감.)
collected = 0; attempt = 0; max_attempts = N * 4
while collected < N and attempt < max_attempts and sim_app.is_running():
    attempt += 1
    buf, bp, br, reached, sp = rollout_record(record_env=0)
    # genuine 성공 = 동시도달 + 시작 시 목표에서 충분히 멀었음(운반) + 프레임 있음
    genuine = bool(reached[0]) and (sp > 0.08) and (len(buf) >= 3)
    if genuine:
        for f in buf:
            writer.write(f)
        collected += 1
        print(f"  ✅ clip {collected}/{N} (시도 {attempt})  시작{sp*100:.0f}cm→도달  min pos={bp[0]*100:.1f}cm rot={br[0]:.0f}deg  ({len(buf)}프레임)")
    else:
        why = "미도달" if not bool(reached[0]) else ("trivial(시작가까움)" if sp <= 0.08 else "짧음")
        print(f"  ✗ env0 제외:{why} (시도 {attempt})  시작{sp*100:.0f}cm  min pos={bp[0]*100:.1f}cm rot={br[0]:.0f}deg")

writer.release()
print(f"✅ 저장: {args.out}  성공 {collected}/{N} ({attempt}회, {time.time()-t0:.0f}s)")
env.close(); sim_app.close()
