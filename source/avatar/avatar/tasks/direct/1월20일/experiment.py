import torch
import numpy as np
import os
import argparse
from datetime import datetime
import isaaclab.utils.math as math_utils

# Isaac Lab Imports
from isaaclab.app import AppLauncher

# argparse 설정
parser = argparse.ArgumentParser(description="Test Agent on Fixed Dataset")
parser.add_argument("--model_path", type=str, required=True, help="Path to the checkpoint (e.g., logs/.../model_step_50000)")
parser.add_argument("--num_envs", type=int, default=100, help="Fixed to 100 for consistent testing")
parser.add_argument("--dataset_file", type=str, default="test_dataset_100.pt", help="File to save/load fixed test episodes")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# App 실행
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 나머지 임포트
from dual_arm_transport_env3 import DualrobotEnv, DualrobotCfg
from graph_converter import convert_batch_state_to_graph, NODE_FEATURE_DIM, EDGE_FEATURE_DIM, GLOBAL_FEATURE_DIM
from agent import TD3
from vectorized_pose_sampler import VectorizedPoseSampler

def generate_and_save_dataset(filepath, num_samples, device="cpu"):
    """
    테스트용 초기 상태를 벡터화된 샘플러를 통해 생성하여 파일로 저장합니다.
    """
    print(f"Generating new test dataset with {num_samples} samples using VectorizedPoseSampler...")
    sampler = VectorizedPoseSampler(device=device)
    
    # 벡터화된 샘플링 실행
    samples = sampler.sample_episodes(num_samples)
    
    data = {
        "base_pose_1": samples["base_pose_1"],
        "base_pose_2": samples["base_pose_2"],
        "q_start_1": samples["q_start_1"],
        "q_start_2": samples["q_start_2"],
        "goal_ee1": samples["goal_ee1_pose"],
        "goal_ee2": samples["goal_ee2_pose"]
    }
        
    torch.save(data, filepath)
    print(f"Saved dataset to {filepath}")
    return data

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Dataset 준비
    if os.path.exists(args_cli.dataset_file):
        print(f"Loading existing dataset: {args_cli.dataset_file}")
        dataset = torch.load(args_cli.dataset_file, map_location=device)
        
        # [Validation] Check size compatibility
        loaded_size = dataset["base_pose_1"].shape[0]
        if loaded_size != args_cli.num_envs:
            print(f"⚠️ Dataset size mismatch! Loaded: {loaded_size}, Requested: {args_cli.num_envs}")
            
            if loaded_size > args_cli.num_envs:
                print(f"📉 Slicing dataset to match {args_cli.num_envs}...")
                for k in dataset:
                    dataset[k] = dataset[k][:args_cli.num_envs]
            else:
                print(f"📈 Requested more envs than dataset has. Regenerating {args_cli.num_envs} samples...")
                dataset = generate_and_save_dataset(args_cli.dataset_file, args_cli.num_envs, device=device)
    else:
        dataset = generate_and_save_dataset(args_cli.dataset_file, args_cli.num_envs, device=device)

    # 2. 환경 설정
    env_cfg = DualrobotCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = DualrobotEnv(cfg=env_cfg, render_mode=None) # 속도를 위해 렌더링 끔

    # 3. 에이전트 로드
    gnn_params = {
        'node_dim': NODE_FEATURE_DIM,
        'edge_dim': EDGE_FEATURE_DIM,
        'global_dim': GLOBAL_FEATURE_DIM,
        'action_dim': 7
    }
    agent = TD3(gnn_params=gnn_params, max_action=1.0)
    
    # 모델 로드 (actor만 있어도 테스트 가능하지만 구조상 load 함수 사용)
    # resume_path가 full path라고 가정
    print(f"Loading model from {args_cli.model_path}...")
    try:
        # [수정] Actor만 직접 로드 (Actor 파일 경로가 직접 주어졌을 경우 대응)
        agent.actor.load_state_dict(torch.load(args_cli.model_path, map_location=device))
        print("✅ Actor model loaded successfully.")
    except Exception as e:
        print(f"⚠️ Failed to load actor directly: {e}")
        print("Attempting agent.load() fallback...")
        try:
            agent.load(args_cli.model_path)
        except Exception as e2:
            print(f"❌ Error loading model: {e2}")
            return

    # 4. 테스트 루프
    print(f"Starting evaluation on {args_cli.num_envs} fixed episodes...")
    
    # [Optimized] Inject dataset directly into env to skip expensive IK sampling in reset()
    # Key mapping: dataset uses 'goal_ee1', env expects 'goal_ee1_pose'
    external_samples = {
        "base_pose_1": dataset["base_pose_1"],
        "base_pose_2": dataset["base_pose_2"],
        "q_start_1": dataset["q_start_1"],
        "q_start_2": dataset["q_start_2"],
        "goal_ee1_pose": dataset["goal_ee1"],
        "goal_ee2_pose": dataset["goal_ee2"]
    }
    env.external_samples = external_samples

    # Now reset() uses the external_samples efficiently
    obs_dict, _ = env.reset()
    
    # force_apply_dataset is no longer needed as reset() applied the external_samples
    
    # 물리 엔진에 적용된 상태를 반영하기 위해 관측 다시 가져오기
    obs_dict = env._get_observations() 
    
    current_batch_graph = convert_batch_state_to_graph(obs_dict['policy'], args_cli.num_envs)

    # 통계 변수 (Latches & Masks)
    total_rewards = torch.zeros(env.num_envs, device=device)
    any_success = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    any_violation = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    any_reached = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    active_mask = torch.ones(env.num_envs, dtype=torch.bool, device=device) # 첫 에피소드 진행 중이면 True
    
    # 에피소드 길이만큼 실행 (환경의 max_length 사용)
    max_steps = 100 # 혹시 모르니 하드코딩 혹은 env.max_episode_length
    
    agent.actor.eval()
    
    print(f"Running simulation for {max_steps} steps...")
    
    for step in range(max_steps):
        # Action Inference (No Grad for Policy)
        with torch.no_grad():
            full_actions = agent.actor(current_batch_graph)
            
            # Reshape & Slice (Train과 동일 로직)
            total_nodes = full_actions.shape[0]
            num_nodes_per_env = total_nodes // args_cli.num_envs
            reshaped_actions = full_actions.view(args_cli.num_envs, num_nodes_per_env, -1)
            robot_actions = reshaped_actions[:, :2, :] 
            env_actions = robot_actions.reshape(args_cli.num_envs, -1)
        
        # Step (Grad Enabled for Internal IK/Sim Ops)
        next_obs_dict, rewards, terminated, truncated, extras = env.step(env_actions)
        
        if (step + 1) % 10 == 0:
            print(f"Step {step+1}/{max_steps} completed.")
        
        dones = terminated | truncated
        
        # 1. Rewards Accumulation (활성 상태인 환경만)
        total_rewards += rewards * active_mask.float()
        
        # 2. Success Latch
        if "log/success" in extras:
            current_success = (extras["log/success"] > 0.5)
            any_success = torch.logical_or(any_success, current_success & active_mask)

        # 3. Violation Latch
        if "log/violation" in extras:
            current_violation = (extras["log/violation"] > 0.5)
            any_violation = torch.logical_or(any_violation, current_violation & active_mask)

        # [NEW] 4. Reached Latch
        if "log/is_reached" in extras:
            current_reached = (extras["log/is_reached"] > 0.5)
            any_reached = torch.logical_or(any_reached, current_reached & active_mask)
        
        # 5. Mask Update (끝난 환경은 비활성화)
        active_mask = active_mask & (~dones)
        
        # 모든 환경이 끝났으면 조기 종료
        if not active_mask.any():
            print(f"All episodes finished at step {step+1}.")
            break
        
        current_batch_graph = convert_batch_state_to_graph(next_obs_dict['policy'], args_cli.num_envs)

    # 5. 결과 집계
    print("\n" + "="*50)
    print(f"Evaluation Results ({args_cli.num_envs} Episodes)")
    print("="*50)
    print(f"Success Rate        : {torch.mean(any_success.float()).item()*100:.2f}%")
    print(f"Violation Rate      : {torch.mean(any_violation.float()).item()*100:.2f}%")
    print(f"Reached Rate        : {torch.mean(any_reached.float()).item()*100:.2f}%")
    print(f"Avg Total Reward    : {torch.mean(total_rewards).item():.4f}")
    print("="*50)

    # 결과 파일 저장 (선택)
    # with open("test_results.txt", "a") as f:
    #     f.write(f"{args_cli.model_path}, {torch.mean(final_success).item()}\n")

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()