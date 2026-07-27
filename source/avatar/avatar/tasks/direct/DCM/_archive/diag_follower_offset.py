"""팔로워 IK offset convention 검증: FK(q_start)이 rod frame에서 어느 offset에 있나 직접 측정.
reset 후 q_start_2·rod pose로 FK를 계산해 rod frame 상대 pose를 뽑아, 두 후보 offset과 대조.
사용: python -u diag_follower_offset.py --num_envs 16 --vary_grasp --same_side --headless
"""
import argparse, math
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--vary_grasp", action="store_true")
parser.add_argument("--same_side", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args); sim_app = app.app

import torch
from dual_arm_transport_env3 import DualrobotEnv, DualrobotCfg
from vectorized_pose_sampler import VectorizedPoseSampler as VPS

dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = DualrobotCfg(); cfg.scene.num_envs = args.num_envs
cfg.leader_follower = True; cfg.action_space = 7; cfg.n_obstacles = 0
if args.vary_grasp:
    cfg.vary_grasp = True; cfg.grasp_same_side = args.same_side
env = DualrobotEnv(cfg, render_mode=None)
env.reset()
# settle 몇 스텝 (rod-hand 정합)
for _ in range(10):
    env.step(torch.zeros(args.num_envs, 7, device=dev))

sampler = env.pose_sampler._base; ik = sampler.ik_solver
J2 = env.robot_2_joint_ids
q2 = env.robot_2.data.joint_pos[:, J2]
# FK(q2) → world (팔로워 base frame → world)
fk_p, fk_R = ik.forward_kinematics(q2)
b2_p = env.robot_2.data.root_pos_w; b2_q = env.robot_2.data.root_quat_w
R_b2 = sampler._quat_to_matrix(b2_q)
ee_p = torch.bmm(R_b2, fk_p.unsqueeze(2)).squeeze(2) + b2_p       # world FK-EE 위치
ee_R = torch.bmm(R_b2, fk_R)
# rod pose (world)
rod_p = env.rod.data.root_pos_w; rod_q = env.rod.data.root_quat_w
R_rod = sampler._quat_to_matrix(rod_q)
# FK-EE를 rod frame으로: off = R_rodᵀ·(ee_p − rod_p)
off_measured = torch.bmm(R_rod.transpose(1, 2), (ee_p - rod_p).unsqueeze(2)).squeeze(2)   # (B,3)
# 후보 offset
TCP = VPS.TCP_OFFSET
if args.vary_grasp:
    d = env._grasp_d
else:
    d = torch.full((args.num_envs,), 0.4, device=dev)
print("\n===== 팔로워 FK-EE의 rod frame offset 측정 (B개 평균/샘플) =====")
print(f"  측정 off (rod frame) 평균: x={off_measured[:,0].mean():.4f} y={off_measured[:,1].mean():.4f} z={off_measured[:,2].mean():.4f}")
print(f"  팔로워 d 평균: {d.mean():.4f}  (후보 x=+d≈{d.mean():.3f})")
print(f"  후보A (+d,0,TCP): x={d.mean():.4f} z={TCP:.4f}")
print("  → 측정 x가 +d에 가깝고 z가 TCP(0.103)에 가까우면 후보A((±d,0,TCP)) 맞음.")
print("  샘플 5개 (측정 x,y,z | d):")
for i in range(min(5, args.num_envs)):
    print(f"    env{i}: ({off_measured[i,0]:.3f},{off_measured[i,1]:.3f},{off_measured[i,2]:.3f}) | d={float(d[i]):.3f}")
env.close(); sim_app.close()
