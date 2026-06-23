"""
Cluttered transport 환경 시각화 (학습 정책 불필요).

GUI로 장애물(빨간 구)·rod(갈색 막대)·목표(녹색 막대)·로봇을 보여준다.
zero action으로 step → 임피던스가 rod를 시작 자세에 유지, 에피소드 timeout(~6s)마다
자동 reset되어 새로운 랜덤 장애물 배치가 보인다.

실행 (GUI; --headless 빼기):
  export LD_LIBRARY_PATH=/home/hjh/anaconda3/envs/env-isaaclab/lib:$LD_LIBRARY_PATH
  python -u viz_cluttered.py --num_envs 4
옵션: --random_action 으로 정책 대신 랜덤 움직임 보기.
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1, help="viz는 1~2개 권장 (렉 방지)")
parser.add_argument("--random_action", action="store_true", help="zero 대신 랜덤 action")
parser.add_argument("--light", action=argparse.BooleanOptionalAction, default=True,
                    help="viz용 경량 물리(solver iter↓, 렌더 자주). 기본 ON.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from dual_arm_transport_env3 import DualrobotEnv, DualrobotCfg

cfg = DualrobotCfg()
cfg.scene.num_envs = args.num_envs
if args.light:
    # viz 전용 경량화 (학습 정확도 불필요): 렌더 더 자주 + solver iter 축소 → 렉 감소.
    cfg.decimation = 12                       # 48→12: env.step당 물리 substep↓, 렌더 4× 자주
    cfg.sim.render_interval = cfg.decimation
    for r in (cfg.robot_1, cfg.robot_2):
        r.spawn.articulation_props.solver_position_iteration_count = 16   # 64→16
        r.spawn.articulation_props.solver_velocity_iteration_count = 2    # 8→2
env = DualrobotEnv(cfg, render_mode=None)
A = cfg.action_space

print(f"🎬 Cluttered env viz: {args.num_envs} envs, n_obstacles={cfg.n_obstacles}")
print("   빨간 구=장애물, 갈색 막대=rod, 녹색 막대=목표. 에피소드마다 장애물 랜덤.")
print("   창을 닫으면 종료.")

env.reset()

while simulation_app.is_running():
    if args.random_action:
        action = 0.05 * (2 * torch.rand(args.num_envs, A, device=env.device) - 1)
    else:
        action = torch.zeros(args.num_envs, A, device=env.device)
    env.step(action)

env.close()
simulation_app.close()
