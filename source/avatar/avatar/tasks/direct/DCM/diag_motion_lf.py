"""리더-팔로워 정책의 '과격함' 정량 진단 — 관절 속도/저크/액션 변화/내력 측정.

과격함 후보: (1) 관절 각속도 큼(dq_scale), (2) 액션 부호 급변(저크), (3) 내력 스파이크, (4) 도달 후에도 계속 움직임.
success와 함께 이 지표들을 측정해 어디가 과격한지 진단.

사용:
  python -u diag_motion_lf.py --model_path logs/<run>/model_final.pt --num_envs 256 --headless
"""
import argparse, math, os
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--num_steps", type=int, default=300)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args); sim_app = app.app

import torch, numpy as np
from dual_arm_transport_env3 import DualrobotEnv, DualrobotCfg
import mlp_policy, graph_converter as gc

dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = DualrobotCfg()
cfg.scene.num_envs = args.num_envs
cfg.leader_follower = True; cfg.action_space = 7; cfg.n_obstacles = 0
gc.GLOBAL_FEATURE_DIM = 1 + 14 + 1 + 3 + 6
env = DualrobotEnv(cfg, render_mode=None)
A = env.cfg.action_space; B = args.num_envs
J1, J2 = env.robot_1_joint_ids, env.robot_2_joint_ids

sd = torch.load(os.path.abspath(args.model_path), map_location=dev, weights_only=False)["model"]
scale = [float(cfg.joint_dq_scale)] * A
agent = mlp_policy.MLPSACAgent(action_dim=A, action_scale=scale, hidden_dim=256,
                               num_hidden_layers=2, use_full_state=False, use_lean_obstacle=False).to(dev)
agent.load_state_dict(sd); agent.eval()
print(f"✅ {os.path.basename(args.model_path)}  dq_scale={cfg.joint_dq_scale} kp={cfg.joint_kp} kd={cfg.joint_kd}")

env.reset(); batch = env._build_policy_batch()
prev_act = None
jvel_all, jerk_all, dact_all, fint_all = [], [], [], []
def f_int():
    w1, w2 = env._get_grasp_wrenches()
    return (0.5 * (w1[:, :3] - w2[:, :3])).norm(dim=-1)
for step in range(args.num_steps):
    with torch.no_grad():
        action, _, _ = agent.actor.get_action_and_log_prob(batch, deterministic=True)
        _, _, term, trunc, _ = env.step(action)
    # 관절 속도(리더+팔로워)
    qd1 = env.robot_1.data.joint_vel[:, J1]; qd2 = env.robot_2.data.joint_vel[:, J2]
    jvel_all.append(torch.cat([qd1, qd2], 1).abs().max(dim=1).values.cpu())    # env별 최대 관절속도
    fint_all.append(f_int().cpu())
    # 액션 변화(저크 proxy): |a_t - a_{t-1}|
    if prev_act is not None:
        dact_all.append((action - prev_act).abs().mean(dim=1).cpu())
    prev_act = action.clone()
    done = term | trunc
    if done.any():
        prev_act[done] = 0.0    # reset env는 액션변화 계산 리셋
    batch = env._build_policy_batch()

jvel = torch.stack(jvel_all)      # (T,B)
dact = torch.stack(dact_all)      # (T-1,B)
fint = torch.stack(fint_all)      # (T,B)
print("\n===== 모션 과격함 진단 =====")
print(f"  관절 각속도 |q̇|:  mean={jvel.mean():.2f}  p90={jvel.flatten().quantile(0.9):.2f}  max={jvel.max():.2f} rad/s")
print(f"    (dq_scale/dt = {cfg.joint_dq_scale}/0.2 = {cfg.joint_dq_scale/0.2:.2f} rad/s 이 명령 상한)")
print(f"  액션 스텝변화 |Δa|:  mean={dact.mean():.3f}  p90={dact.flatten().quantile(0.9):.3f}  max={dact.max():.3f}")
print(f"    (0에 가까울수록 부드러움. 1에 가까우면 매 스텝 반전=과격/저크↑)")
print(f"  내력 f_int:  mean={fint.mean():.0f}  p90={fint.flatten().quantile(0.9):.0f}  max={fint.max():.0f} N")
# 저크 지표: 액션 부호 반전 빈도
print(f"\n  판정 힌트:")
print(f"   - |q̇| p90이 명령상한({cfg.joint_dq_scale/0.2:.1f})에 가까우면 → 항상 최대속도로 움직임(dq_scale 과다)")
print(f"   - |Δa| p90>0.5면 → 액션 급변(smoothness 페널티 부족, w_smooth↑ 필요)")
print(f"   - f_int p90 높으면 → 급가속으로 내력 스파이크")
os._exit(0)
