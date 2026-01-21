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

def force_apply_dataset(env, dataset, env_ids):
    """
    환경의 초기 상태를 데이터셋 값으로 강제로 덮어씌웁니다.
    """
    # 1. 데이터 가져오기
    base_pose_1 = dataset["base_pose_1"][env_ids].to(env.device)
    base_pose_2 = dataset["base_pose_2"][env_ids].to(env.device)
    q_start_1 = dataset["q_start_1"][env_ids].to(env.device)
    q_start_2 = dataset["q_start_2"][env_ids].to(env.device)
    goal_ee1 = dataset["goal_ee1"][env_ids].to(env.device)
    goal_ee2 = dataset["goal_ee2"][env_ids].to(env.device)
    
    zeros_vel = torch.zeros_like(base_pose_1[:, :6])
    zeros_joint_vel = torch.zeros_like(q_start_1)
    
    # 2. Sim에 쓰기 (env_origins 보정 필수)
    env_origins = env.scene.env_origins[env_ids]
    
    # Robot 1
    r1_pose = base_pose_1.clone()
    r1_pose[:, :3] += env_origins
    env.robot_1.write_root_pose_to_sim(r1_pose, env_ids)
    env.robot_1.write_root_velocity_to_sim(zeros_vel, env_ids)
    env.robot_1.write_joint_state_to_sim(q_start_1, zeros_joint_vel, env.robot_1_joint_ids, env_ids)
    
    # Robot 2
    r2_pose = base_pose_2.clone()
    r2_pose[:, :3] += env_origins
    env.robot_2.write_root_pose_to_sim(r2_pose, env_ids)
    env.robot_2.write_root_velocity_to_sim(zeros_vel, env_ids)
    env.robot_2.write_joint_state_to_sim(q_start_2, zeros_joint_vel, env.robot_2_joint_ids, env_ids)
    
    # Markers
    m1_pose = goal_ee1.clone()
    m1_pose[:, :3] += env_origins
    env.goal_marker_ee1.write_root_pose_to_sim(m1_pose, env_ids)
    
    m2_pose = goal_ee2.clone()
    m2_pose[:, :3] += env_origins
    env.goal_marker_ee2.write_root_pose_to_sim(m2_pose, env_ids)
    
    # 3. 내부 변수 업데이트 (Reward/Obs 계산용) - 중요!
    # env._reset_idx 로직과 동일하게 target_ee_rel_poses를 다시 계산해줘야 함
    p1 = goal_ee1[:, 0:3]
    q1 = goal_ee1[:, 3:7]
    p2 = goal_ee2[:, 0:3]
    q2 = goal_ee2[:, 3:7]
    
    q1_inv = math_utils.quat_conjugate(q1)
    p_diff = p2 - p1
    rel_pos = math_utils.quat_apply(q1_inv, p_diff)
    rel_rot = math_utils.quat_mul(q1_inv, q2)
    
    env.target_ee_rel_poses[env_ids] = torch.cat([rel_pos, rel_rot], dim=-1)
    env.violation_occurred[env_ids] = False
    
    # Max Error 버퍼도 초기화
    if hasattr(env, 'episode_max_pos_error'):
        env.episode_max_pos_error[env_ids] = 0.0
        env.episode_max_rot_error[env_ids] = 0.0

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Dataset 준비
    if os.path.exists(args_cli.dataset_file):
        print(f"Loading existing dataset: {args_cli.dataset_file}")
        dataset = torch.load(args_cli.dataset_file, map_location=device)
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
        agent.load(args_cli.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 4. 테스트 루프
    print(f"Starting evaluation on {args_cli.num_envs} fixed episodes...")
    
    obs_dict, _ = env.reset()
    
    # [핵심] Reset 직후 데이터셋 강제 주입
    all_indices = torch.arange(env.num_envs, device=env.device)
    force_apply_dataset(env, dataset, all_indices)
    
    # 물리 엔진에 적용된 상태를 반영하기 위해 관측 다시 가져오기
    obs_dict = env._get_observations() 
    
    current_batch_graph = convert_batch_state_to_graph(obs_dict['policy'], args_cli.num_envs)

    # 통계 변수 (Latches & Masks)
    total_rewards = torch.zeros(env.num_envs, device=device)
    any_success = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    any_violation = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    active_mask = torch.ones(env.num_envs, dtype=torch.bool, device=device) # 첫 에피소드 진행 중이면 True
    
    # 에피소드 길이만큼 실행 (환경의 max_length 사용)
    max_steps = 100 # 혹시 모르니 하드코딩 혹은 env.max_episode_length
    
    agent.actor.eval()
    
    print(f"Running simulation for {max_steps} steps...")
    
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
            # extras["log/success"]는 done 시점에만 1.0 (성공 시)
            current_success = (extras["log/success"] > 0.5)
            # 현재 활성 상태이고 성공했다면 기록
            any_success = torch.logical_or(any_success, current_success & active_mask)

        # 3. Violation Latch
        if "log/violation" in extras:
            # extras["log/violation"]은 위반 기록이 있으면 1.0
            current_violation = (extras["log/violation"] > 0.5)
            any_violation = torch.logical_or(any_violation, current_violation & active_mask)
        
        # 4. Mask Update (끝난 환경은 비활성화)
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
    # Max Error는 정확한 집계가 어려우므로(reset 문제) 생략하거나 extras 로깅 의존
    # print(f"Avg Max Pos Error   : {torch.mean(final_max_pos_err).item():.4f} m") 
    print(f"Avg Total Reward    : {torch.mean(total_rewards).item():.4f}")
    print("="*50)

    # 결과 파일 저장 (선택)
    # with open("test_results.txt", "a") as f:
    #     f.write(f"{args_cli.model_path}, {torch.mean(final_success).item()}\n")

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()