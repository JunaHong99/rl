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
parser.add_argument("--stochastic", action="store_true", help="탐험 확인용(기본 deterministic).")
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
for step in range(args.num_steps):
    with torch.no_grad():
        action, _, _ = agent.actor.get_action_and_log_prob(batch, deterministic=not args.stochastic)
        act_abs_sum += action.abs().mean().item(); n_act += 1
        _, _, term, trunc, _ = env.step(action)
    rod_pos = env.rod.data.root_pos_w; goal_pos = env.goal_rod_marker.data.root_pos_w
    pos_err = torch.norm(goal_pos - rod_pos, dim=-1)
    q_diff = math_utils.quat_mul(env.goal_rod_marker.data.root_quat_w,
                                 math_utils.quat_conjugate(env.rod.data.root_quat_w))
    rot_err = 2.0 * torch.atan2(torch.norm(q_diff[:, 1:4], dim=-1), torch.abs(q_diff[:, 0]))
    rr_min_pos = torch.min(rr_min_pos, pos_err); rr_min_rot = torch.min(rr_min_rot, rot_err)
    rr_reached |= (pos_err < POS_T) & (rot_err < ROT_T)     # ★ 이 스텝에 pos&rot 동시 만족
    rod_disp_max = torch.maximum(rod_disp_max, (rod_pos - rod_start).norm(dim=-1))
    rr_len += 1
    done = term | trunc
    if done.any():
        for i in done.nonzero(as_tuple=True)[0].tolist():
            if rr_len[i].item() > settle:
                ep_succ.append(bool(rr_reached[i].item()))   # 동시 도달 기준
            rr_min_pos[i] = float("inf"); rr_min_rot[i] = float("inf"); rr_reached[i] = False
            rr_len[i] = 0
            rod_start[i] = env.rod.data.root_pos_w[i]; rod_disp_max[i] = 0.0
    batch = env._build_policy_batch()

S = np.array(ep_succ)
print(f"  genuine ep: {len(S)}   success: {100*S.mean() if len(S) else 0:.1f}%")
print(f"  rod 최대이동(에피소드): mean={rod_disp_max.mean()*100:.1f}cm  max={rod_disp_max.max()*100:.1f}cm")
print(f"  액션 |mean|: {act_abs_sum/max(1,n_act):.3f}  (0에 가까우면 정책 붕괴=freeze)")
print(f"  → 진단: rod가 {'거의 안 움직임 → 정책 붕괴/탐험실패' if rod_disp_max.mean()<0.03 else '움직임 → 도달만 못함(탐험/HER)'}")
os._exit(0)
