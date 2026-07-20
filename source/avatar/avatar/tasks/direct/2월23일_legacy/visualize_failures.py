import torch
import numpy as np
import os
import argparse
import time

# Isaac Lab Imports
from isaaclab.app import AppLauncher

# argparse
parser = argparse.ArgumentParser(description="Visualize Failed Episodes")
parser.add_argument("--model_path", type=str, required=True, help="Path to the checkpoint")
parser.add_argument("--file", type=str, default="failed_episodes.pt", help="Path to failed episodes file")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# App Launch
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports after app launch
import isaaclab.utils.math as math_utils
from dual_arm_transport_env3 import DualrobotEnv, DualrobotCfg
from graph_converter import convert_batch_state_to_graph, NODE_FEATURE_DIM, EDGE_FEATURE_DIM, GLOBAL_FEATURE_DIM
from agent import TD3

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(args_cli.file):
        print(f"❌ Error: Failed episodes file '{args_cli.file}' not found.")
        return

    print(f"Loading failed episodes from {args_cli.file}...")
    dataset = torch.load(args_cli.file, map_location=device)
    
    num_failures = dataset["base_pose_1"].shape[0]
    print(f"Found {num_failures} failed episodes.")
    
    reasons = dataset.get("failure_reason", ["unknown"] * num_failures)
    
    # [1] Sort indices: 'not_reached' first
    sorted_indices = []
    not_reached_idxs = [i for i, r in enumerate(reasons) if r == "not_reached"]
    other_idxs = [i for i, r in enumerate(reasons) if r != "not_reached"]
    sorted_indices = not_reached_idxs + other_idxs
    
    print(f"Sorted: {len(not_reached_idxs)} 'not_reached' cases first, then {len(other_idxs)} others.")

    # [2] Env Setup (Single Env)
    env_cfg = DualrobotCfg()
    env_cfg.scene.num_envs = 1  # Show one by one
    render_mode = "human" if not args_cli.headless else None
    env = DualrobotEnv(cfg=env_cfg, render_mode=render_mode)

    # [3] Agent Load
    gnn_params = {
        'node_dim': NODE_FEATURE_DIM,
        'edge_dim': EDGE_FEATURE_DIM,
        'global_dim': GLOBAL_FEATURE_DIM,
        'action_dim': 7
    }
    agent = TD3(gnn_params=gnn_params, max_action=1.0)
    
    print(f"Loading model from {args_cli.model_path}...")
    try:
        agent.actor.load_state_dict(torch.load(args_cli.model_path, map_location=device))
    except:
        agent.load(args_cli.model_path)
    
    agent.actor.eval()
    
    print("Starting Visualization Sequence... (Press Ctrl+C to stop)")
    
    try:
        for seq_idx, original_idx in enumerate(sorted_indices):
            reason = reasons[original_idx]
            print(f"\n▶️ Playing Case {seq_idx+1}/{num_failures} (Reason: {reason})")
            
            # Prepare single sample batch
            single_sample = {}
            for k, v in dataset.items():
                if k == "failure_reason": continue
                if isinstance(v, torch.Tensor):
                    # Extract the single item and keep dim (1, ...)
                    single_sample[k] = v[original_idx].unsqueeze(0).to(env.device)
            
            # Inject
            env.external_samples = single_sample
            
            obs_dict, _ = env.reset()
            current_batch_graph = convert_batch_state_to_graph(obs_dict['policy'], 1)
            
            max_steps = 1200
            
            violation_active = False
            for step in range(max_steps):
                with torch.no_grad():
                    full_actions = agent.actor(current_batch_graph)
                    
                    # Match experiment.py logic: handle graph having more nodes than just robots
                    total_nodes = full_actions.shape[0]
                    num_nodes_per_env = total_nodes // env.num_envs
                    
                    reshaped_actions = full_actions.view(env.num_envs, num_nodes_per_env, -1)
                    robot_actions = reshaped_actions[:, :2, :] 
                    env_actions = robot_actions.reshape(env.num_envs, -1)
                
                # Slower playback
                time.sleep(0.03)
                
                next_obs_dict, rewards, terminated, truncated, extras = env.step(env_actions)
                
                # --- [1] Joint Limit Check (Both Robots) ---
                margin = 0.05
                # R1
                q1 = env.robot_1.data.joint_pos[:, :7]
                limits_1 = env.robot_1.data.soft_joint_pos_limits[0, :7, :]
                viol_l1 = (q1 - limits_1[:, 0] < margin).any().item()
                viol_u1 = (limits_1[:, 1] - q1 < margin).any().item()
                # R2
                q2 = env.robot_2.data.joint_pos[:, :7]
                limits_2 = env.robot_2.data.soft_joint_pos_limits[0, :7, :]
                viol_l2 = (q2 - limits_2[:, 0] < margin).any().item()
                viol_u2 = (limits_2[:, 1] - q2 < margin).any().item()

                if (step % 20 == 0) and (viol_l1 or viol_u1 or viol_l2 or viol_u2):
                    print(f"  ⚠️ [Step {step}] Joint Limit Warning:", end="")
                    if viol_l1 or viol_u1: print(" R1", end="")
                    if viol_l2 or viol_u2: print(" R2", end="")
                    print()

                # --- [2] Relative Pose Constraint Check ---
                pos_thresh = 0.3
                rot_thresh = 0.3
                
                ee1_pos = env.robot_1.data.body_state_w[:, env.ee_body_idx_1, 0:3]
                ee2_pos = env.robot_2.data.body_state_w[:, env.ee_body_idx_2, 0:3]
                ee1_quat = env.robot_1.data.body_state_w[:, env.ee_body_idx_1, 3:7]
                ee2_quat = env.robot_2.data.body_state_w[:, env.ee_body_idx_2, 3:7]
                target_rel_pos = env.target_ee_rel_poses[:, 0:3]
                target_rel_rot = env.target_ee_rel_poses[:, 3:7]

                ee1_inv_quat = math_utils.quat_conjugate(ee1_quat)
                curr_rel_pos_local = math_utils.quat_apply(ee1_inv_quat, ee2_pos - ee1_pos)
                pos_error = torch.norm(curr_rel_pos_local - target_rel_pos, dim=-1).item()

                curr_rel_rot = math_utils.quat_mul(ee1_inv_quat, ee2_quat)
                target_rel_rot_inv = math_utils.quat_conjugate(target_rel_rot)
                q_diff = math_utils.quat_mul(curr_rel_rot, target_rel_rot_inv)
                q_diff_v = q_diff[:, 1:4]
                q_diff_w = q_diff[:, 0]
                rot_error = (2.0 * torch.atan2(torch.norm(q_diff_v, dim=-1), torch.abs(q_diff_w))).item()

                currently_violated = (pos_error > pos_thresh or rot_error > rot_thresh)
                
                if currently_violated and not violation_active:
                    print(f"  🛑 [Step {step}] CONSTRAINT VIOLATION STARTED!")
                    print(f"     Pos Err: {pos_error:.3f}, Rot Err: {rot_error:.3f}")
                    violation_active = True
                elif not currently_violated and violation_active:
                    print(f"  ✅ [Step {step}] Constraint Violation Resolved.")
                    violation_active = False
                elif currently_violated and (step % 50 == 0):
                    print(f"  🛑 [Step {step}] Violation Ongoing (Pos: {pos_error:.3f}, Rot: {rot_error:.3f})")

                current_batch_graph = convert_batch_state_to_graph(next_obs_dict['policy'], 1)
                
                if terminated.item() or truncated.item():
                    print(f"   Episode finished at step {step+1}")
                    break
            
            # Optional: Wait for user input to continue?
            # input("Press Enter for next case...") 

    except KeyboardInterrupt:
        print("\nVisualization stopped by user.")
    
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
