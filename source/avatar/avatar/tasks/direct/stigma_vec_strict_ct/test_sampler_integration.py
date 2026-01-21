
import argparse
from isaaclab.app import AppLauncher

# Set up parser
parser = argparse.ArgumentParser(description="Test Vectorized Pose Sampler Integration")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch App
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after app launch
import torch
from dual_arm_transport_env3 import DualrobotEnv, DualrobotCfg

def main():
    print(f"Testing with {args_cli.num_envs} environments...")
    
    # Configure Env
    env_cfg = DualrobotCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # Create Env
    # This will trigger __init__ -> VectorizedPoseSampler init
    print("Creating environment...")
    env = DualrobotEnv(cfg=env_cfg, render_mode="human" if args_cli.num_envs == 1 else None)
    
    # Reset
    # This will trigger _reset_idx -> VectorizedPoseSampler.sample_episodes
    print("Resetting environment (Testing Sampler)...")
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    obs, _ = env.reset()
    end_time.record()
    torch.cuda.synchronize()
    
    print(f"Reset finished in {start_time.elapsed_time(end_time):.2f} ms")
    
    # Check Observation Shapes
    print("Checking observations...")
    policy_obs = obs['policy']
    print(f"Robot Node Shape: {policy_obs['robot_nodes'].shape}")
    print(f"Goal Poses Shape: {policy_obs['goal_poses'].shape}")
    
    # Step Loop
    print("Running simulation steps...")
    for i in range(10):
        actions = torch.rand((args_cli.num_envs, 14), device=env.device) * 2 - 1
        obs, rew, terminated, truncated, extras = env.step(actions)
        
        if (i+1) % 10 == 0:
            print(f"Step {i+1} done.")
            
    print("Test Complete!")
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
