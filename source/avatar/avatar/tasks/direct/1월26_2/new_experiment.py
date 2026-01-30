import torch
import argparse
import os
import numpy as np
import time
from isaaclab.app import AppLauncher
import isaaclab.utils.math as math_utils

# ---------------------------------------------------------
# 1. Argparse Setup
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description="Experiment with Pre-generated Dataset")
parser.add_argument("--model_path", type=str, required=True, help="Path to saved actor model")
parser.add_argument("--dataset_path", type=str, default="test_dataset_strict.pt", help="Path to test dataset")
parser.add_argument("--num_envs", type=int, default=100, help="Number of parallel environments")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------
# 2. Imports (After App Launch)
# ---------------------------------------------------------
from dual_arm_transport_env3 import DualrobotEnv, DualrobotCfg
from graph_converter import convert_batch_state_to_graph, NODE_FEATURE_DIM, EDGE_FEATURE_DIM, GLOBAL_FEATURE_DIM
from agent import TD3

def force_apply_dataset(env, batch_data, env_ids):
    device = env.device
    num_envs = len(env_ids)
    
    env_origins = env.scene.env_origins[env_ids]
    
    def to_d(k): return batch_data[k].to(device)
    
    b1 = to_d("base_pose_1"); b2 = to_d("base_pose_2")
    q1 = to_d("q_start_1"); q2 = to_d("q_start_2")
    g1 = to_d("goal_ee1_pose"); g2 = to_d("goal_ee2_pose")
    
    zeros_vel = torch.zeros(num_envs, 6, device=device)
    zeros_j_vel = torch.zeros_like(q1)
    
    # Apply Robot 1
    p1 = b1.clone(); p1[:, :3] += env_origins
    env.robot_1.write_root_pose_to_sim(p1, env_ids)
    env.robot_1.write_root_velocity_to_sim(zeros_vel, env_ids)
    env.robot_1.write_joint_state_to_sim(q1, zeros_j_vel, env.robot_1_joint_ids, env_ids)
    
    # Apply Robot 2
    p2 = b2.clone(); p2[:, :3] += env_origins
    env.robot_2.write_root_pose_to_sim(p2, env_ids)
    env.robot_2.write_root_velocity_to_sim(zeros_vel, env_ids)
    env.robot_2.write_joint_state_to_sim(q2, zeros_j_vel, env.robot_2_joint_ids, env_ids)
    
    # Markers
    m1 = g1.clone(); m1[:, :3] += env_origins
    env.goal_marker_ee1.write_root_pose_to_sim(m1, env_ids)
    m2 = g2.clone(); m2[:, :3] += env_origins
    env.goal_marker_ee2.write_root_pose_to_sim(m2, env_ids)
    
    # Update Internal State
    p_g1 = g1[:, 0:3]; q_g1 = g1[:, 3:7]
    p_g2 = g2[:, 0:3]; q_g2 = g2[:, 3:7]
    
    q1_inv = math_utils.quat_conjugate(q_g1)
    p_diff = p_g2 - p_g1 # Correct: Goal2 Pos - Goal1 Pos
    rel_pos = math_utils.quat_apply(q1_inv, p_diff)
    rel_rot = math_utils.quat_mul(q1_inv, q_g2)
    
    env.target_ee_rel_poses[env_ids] = torch.cat([rel_pos, rel_rot], dim=-1)
    env.violation_occurred[env_ids] = False
    
    if hasattr(env, 'episode_max_pos_error'):
        env.episode_max_pos_error[env_ids] = 0.0
        env.episode_max_rot_error[env_ids] = 0.0
    if hasattr(env, 'prev_dist'):
        env.prev_dist[env_ids] = float('inf')
        
    env.episode_length_buf[env_ids] = 0

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Optimized Experiment Start on {device}")
    
    full_dataset = torch.load(args_cli.dataset_path, map_location=device)
    total_episodes = full_dataset["base_pose_1"].shape[0]
    
    env_cfg = DualrobotCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = DualrobotEnv(cfg=env_cfg, render_mode=None)
    
    agent = TD3(gnn_params={'node_dim': 14, 'edge_dim': 9, 'global_dim': 19, 'action_dim': 7}, max_action=1.0)
    agent.actor.load_state_dict(torch.load(args_cli.model_path, map_location=device))
    agent.actor.eval()
    
    success_count = 0
    violation_count = 0
    reached_count = 0 # [Fix] Initialize reached_count
    total_evaluated = 0
    
    num_envs = args_cli.num_envs
    all_env_ids = torch.arange(num_envs, device=device)
    
    # First Reset
    env.reset()
    
    for start_idx in range(0, total_episodes, num_envs):
        end_idx = min(start_idx + num_envs, total_episodes)
        cur_batch_size = end_idx - start_idx
        
        # Prepare Batch Samples
        batch = {k: v[start_idx:end_idx] for k, v in full_dataset.items() if isinstance(v, torch.Tensor)}
        if cur_batch_size < num_envs:
            pad = num_envs - cur_batch_size
            for k in batch:
                batch[k] = torch.cat([batch[k], batch[k][-1:].repeat(pad, *([1]*(batch[k].ndim-1)))], dim=0)

        force_apply_dataset(env, batch, all_env_ids)
        obs_dict = env._get_observations()
        
        done_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
        success_batch = torch.zeros(num_envs, dtype=torch.bool, device=device)
        violation_batch = torch.zeros(num_envs, dtype=torch.bool, device=device)
        reached_batch = torch.zeros(num_envs, dtype=torch.bool, device=device) # [Fix]
        
        for step in range(100):
            current_batch_graph = convert_batch_state_to_graph(obs_dict['policy'], num_envs)
            
            with torch.no_grad():
                full_actions = agent.actor(current_batch_graph)
                robot_actions = full_actions.view(num_envs, -1, 7)[:, :2, :]
                env_actions = robot_actions.reshape(num_envs, -1)
            
            obs_dict, _, terminated, truncated, extras = env.step(env_actions)
            dones = terminated | truncated
            
            if dones.any():
                new_dones = dones & (~done_mask)
                if new_dones.any():
                    success_batch[new_dones] = extras["log/success"][new_dones] > 0.5
                    violation_batch[new_dones] = extras["log/violation"][new_dones] > 0.5
                    if "log/is_reached" in extras:
                        reached_batch[new_dones] = extras["log/is_reached"][new_dones] > 0.5
                    done_mask |= new_dones
            
            if done_mask.all(): break

        not_done = ~done_mask
        if not_done.any():
            success_batch[not_done] = extras["log/success"][not_done] > 0.5
            violation_batch[not_done] = extras["log/violation"][not_done] > 0.5
            if "log/is_reached" in extras:
                reached_batch[not_done] = extras["log/is_reached"][not_done] > 0.5

        success_count += success_batch[:cur_batch_size].sum().item()
        violation_count += violation_batch[:cur_batch_size].sum().item()
        reached_count += reached_batch[:cur_batch_size].sum().item() # [Fix]
        total_evaluated += cur_batch_size
        
        print(f"Batch {start_idx//num_envs + 1}: SR {success_count/total_evaluated:.1%} | RR {reached_count/total_evaluated:.1%}")

    print(f"\n✅ Final Results: SR {success_count/total_evaluated:.2%}, VR {violation_count/total_evaluated:.2%}, RR {reached_count/total_evaluated:.2%}")
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()