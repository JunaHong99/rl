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

# Force rendering
args_cli.headless = False

# App Launch
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports after app launch
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
    env = DualrobotEnv(cfg=env_cfg, render_mode="human")

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
            
            max_steps = 150 
            
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
