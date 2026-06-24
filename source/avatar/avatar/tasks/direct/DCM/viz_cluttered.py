"""
Cluttered transport 환경/정책 시각화.

GUI로 장애물(빨간 구)·rod(갈색 막대)·목표(녹색 막대)·로봇을 보여준다.
- --model_path 없으면: zero(또는 --random_action) action으로 환경만 (임피던스가 rod 유지).
- --model_path 주면: 학습 정책(MLP/GNN 자동인식)을 deterministic으로 돌려 운반+회피 장면.

실행 (GUI; --headless 빼기):
  export LD_LIBRARY_PATH=/home/hjh/anaconda3/envs/env-isaaclab/lib:$LD_LIBRARY_PATH
  python -u viz_cluttered.py --num_envs 2 --model_path logs/phase3_sac_XX/model_final.pt
옵션: --no_filter (rod safety filter 끄고 정책 raw 동작), --obstacle_frac (기본 1.0).
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1, help="viz는 1~2개 권장 (렉 방지)")
parser.add_argument("--random_action", action="store_true", help="zero 대신 랜덤 action (정책 미사용시)")
parser.add_argument("--model_path", type=str, default=None, help="학습 정책 (MLP/GNN 자동인식). 주면 정책 구동.")
parser.add_argument("--obstacle_frac", type=float, default=1.0, help="장애물 curriculum frac (1.0=full)")
parser.add_argument("--no_filter", action="store_true", help="rod safety filter 끄기")
parser.add_argument("--num_rounds", type=int, default=2, help="GNN message-passing rounds")
parser.add_argument("--action_scale_pos", type=float, default=0.02)
parser.add_argument("--action_scale_rot", type=float, default=0.05)
parser.add_argument("--light", action=argparse.BooleanOptionalAction, default=True,
                    help="viz용 경량 물리(solver iter↓, 렌더 자주). 기본 ON.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch, os
from dual_arm_transport_env3 import DualrobotEnv, DualrobotCfg

cfg = DualrobotCfg()
cfg.scene.num_envs = args.num_envs
if args.no_filter:
    cfg.use_rod_safety_filter = False
if args.light:
    # viz 전용 경량화 (학습 정확도 불필요): 렌더 더 자주 + solver iter 축소 → 렉 감소.
    cfg.decimation = 12
    cfg.sim.render_interval = cfg.decimation
    for r in (cfg.robot_1, cfg.robot_2):
        r.spawn.articulation_props.solver_position_iteration_count = 16
        r.spawn.articulation_props.solver_velocity_iteration_count = 2
env = DualrobotEnv(cfg, render_mode=None)
env._obstacle_curr_frac = args.obstacle_frac
A = cfg.action_space
dev = env.device

# ── 정책 로드 (옵션) ──
agent = None
if args.model_path:
    import mlp_policy
    sd = torch.load(os.path.abspath(args.model_path), map_location=dev, weights_only=False)["model"]
    scale = [args.action_scale_pos]*3 + [args.action_scale_rot]*3 + [1.0]*(A-6)
    if "actor.mean_head.0.weight" in sd:   # MLP
        in_dim = sd["actor.mean_head.0.weight"].shape[1]
        use_full = (in_dim == mlp_policy._state_dim(True))
        use_lean = (in_dim == mlp_policy._state_dim(False, True))
        agent = mlp_policy.MLPSACAgent(action_dim=A, action_scale=scale, hidden_dim=256,
                                       num_hidden_layers=2, use_full_state=use_full,
                                       use_lean_obstacle=use_lean).to(dev)
        mode = "full_state" if use_full else ("lean_obstacle" if use_lean else "rod+global")
    else:   # GNN
        import gnn_policy
        agent = gnn_policy.GNNSACAgent(action_dim=A, num_rounds=args.num_rounds,
                                       action_scale=scale).to(dev)
        mode = "GNN"
    agent.load_state_dict(sd); agent.eval()
    print(f"🎬 정책 구동: {os.path.basename(args.model_path)}  input={mode}  filter={'OFF' if args.no_filter else 'ON'}")

print(f"🎬 Cluttered viz: {args.num_envs} envs, n_obstacles={cfg.n_obstacles}, frac={args.obstacle_frac}")
print("   빨간 구=장애물, 갈색 막대=rod, 녹색 막대=목표. 창 닫으면 종료.")

env.reset()
batch = env._build_policy_batch() if agent is not None else None

while simulation_app.is_running():
    if agent is not None:
        with torch.no_grad():
            action, _, _ = agent.actor.get_action_and_log_prob(batch, deterministic=True)
        env.step(action)
        batch = env._build_policy_batch()
    elif args.random_action:
        env.step(0.05 * (2 * torch.rand(args.num_envs, A, device=dev) - 1))
    else:
        env.step(torch.zeros(args.num_envs, A, device=dev))

env.close()
simulation_app.close()
