import torch
import argparse
import os
import numpy as np
from isaaclab.app import AppLauncher

# ---------------------------------------------------------
# 1. Argparse 설정 (수정됨)
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description="Test RoboBallet Agent")

# [수정] 모델 경로 인자 추가 (필수)
parser.add_argument("--model_path", type=str, required=True, help="Path to saved actor model (e.g., logs/.../model_step_10000_actor)")
# 테스트는 기본적으로 1개 환경에서 수행
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments for testing")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 앱 실행
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------
# 2. 모듈 임포트 (앱 실행 후)
# ---------------------------------------------------------
from dual_arm_transport_env2 import DualrobotEnv, DualrobotCfg
from graph_converter import (
    convert_state_to_graph, 
    NODE_FEATURE_DIM, EDGE_FEATURE_DIM, GLOBAL_FEATURE_DIM
)
from agent import TD3

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Testing Start on {device}")

    # ---------------------------------------------------------
    # 3. 환경 및 에이전트 초기화
    # ---------------------------------------------------------
    env_cfg = DualrobotCfg()
    
    # [중요] 테스트 시에는 환경을 1개로 고정하는 것이 시각화에 유리함
    # (CLI에서 --num_envs를 따로 주지 않았다면 1로 설정)
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # 렌더링 켜기
    env = DualrobotEnv(cfg=env_cfg, render_mode="human")

    # GNN 파라미터 (train.py와 동일)
    gnn_params = {
        'node_dim': NODE_FEATURE_DIM,
        'edge_dim': EDGE_FEATURE_DIM,
        'global_dim': GLOBAL_FEATURE_DIM,
        'action_dim': 7
    }
    
    # 에이전트 생성
    agent = TD3(gnn_params=gnn_params, max_action=1.0)
    
    # ---------------------------------------------------------
    # 4. 모델 가중치 로드
    # ---------------------------------------------------------
    print(f"📂 Loading model from: {args_cli.model_path}")
    
    if not os.path.exists(args_cli.model_path):
        print(f"❌ Error: Model file not found at {args_cli.model_path}")
        return

    # Actor 모델 로드
    agent.actor.load_state_dict(torch.load(args_cli.model_path, map_location=device))
    agent.actor.eval() # 평가 모드 (Dropout 등 비활성화)

    print("✅ Model loaded. Starting simulation loop...")
    
    # ---------------------------------------------------------
    # 5. 테스트 루프
    # ---------------------------------------------------------
    while simulation_app.is_running():
        # --- 새 에피소드를 위한 리셋 ---
        obs_dict, _ = env.reset()
        
        # [NEW] 위반 추적을 위한 변수 초기화
        is_in_violation = [False] * env.num_envs
        violation_start_step = [-1] * env.num_envs
        step_counter = 0
        
        print("\n" + "="*60)
        print("Starting New Test Episode...")
        print("="*60)


        # --- 에피소드 스텝 루프 ---
        while True:
            # 스텝 카운터 증가
            step_counter += 1
            
            # (1) 그래프 변환 (기존 로직 유지)
            graph_list = []
            keys = list(obs_dict['policy'].keys())
            for i in range(env.num_envs):
                single_env_obs = {k: obs_dict['policy'][k][i] for k in keys}
                graph_list.append(convert_state_to_graph(single_env_obs))
            
            from torch_geometric.data import Batch
            batch_graph = Batch.from_data_list(graph_list).to(device)

            with torch.no_grad():
                # (2) 액션 추론 (기존 로직 유지)
                actions_tensor = agent.actor(batch_graph)
                num_nodes_per_graph = actions_tensor.shape[0] // env.num_envs
                actions_reshaped = actions_tensor.view(env.num_envs, num_nodes_per_graph, 7)
                robot_actions = actions_reshaped[:, :2, :]
                env_actions_tensor = robot_actions.reshape(env.num_envs, -1)

            # (3) 환경 스텝
            obs_dict, rewards, terminated, truncated, extras = env.step(env_actions_tensor)
            
            # [NEW] 위반 로깅 로직
            currently_violating_tensor = extras["log/is_currently_violated"]
            #raw_pos_errors = extras["log/raw_err_pos"]
            #raw_rot_errors = extras["log/raw_err_rot"]

            for i in range(env.num_envs):
                is_currently_violated = currently_violating_tensor[i].item()
                #current_pos_error = raw_pos_errors[i].item()
                #current_rot_error = raw_rot_errors[i].item()

                # Case 1: 위반 시작
                if is_currently_violated and not is_in_violation[i]:
                    is_in_violation[i] = True
                    violation_start_step[i] = step_counter
                    print(f"🔴 [Env {i}, Step {step_counter}] Constraint violation STARTED.")
                
                # Case 2: 위반 종료
                elif not is_currently_violated and is_in_violation[i]:
                    print(f"🟢 [Env {i}] Constraint violation ENDED. (Duration: {violation_start_step[i]} ~ {step_counter - 1})")
                    is_in_violation[i] = False
                    violation_start_step[i] = -1
                
                # Case 3: 위반 중 (지속적 로깅)
                if is_in_violation[i]:
                     print(f"   🔸 [Env {i}, Step {step_counter}] Violation Ongoing ") #| Pos Err: {current_pos_error:.4f} | Rot Err: {current_rot_error:.4f}
            
            # (4) 종료 확인
            dones = terminated | truncated
            if dones.any():
                # [NEW] 에피소드 종료 시, 진행 중이던 위반이 있었는지 확인
                for i in range(env.num_envs):
                    if dones[i] and is_in_violation[i]:
                         print(f"🟡 [Env {i}] Episode ended while in violation. (Started at step {violation_start_step[i]})")

                # 최종 결과 출력 (기존 로직 유지)
                env_idx = 0 # 1개 환경 기준
                is_success = extras["log/success"][env_idx].item() > 0.5
                is_reached = "log/is_reached" in extras and extras["log/is_reached"][env_idx].item() > 0.5
                is_violated_final = extras["log/violation"][env_idx].item() > 0.5
                final_reward = rewards[env_idx].item()

                if is_success:
                    status_icon, status_msg = "🏆", "Perfect Success (Reached & Safe)"
                elif is_reached and is_violated_final:
                    status_icon, status_msg = "⚠️", "Reached but Violated"
                elif not is_reached and is_violated_final:
                    status_icon, status_msg = "❌", "Failed (Violated & Not Reached)"
                else:
                    status_icon, status_msg = "⏳", "Time Out (Safe but Not Reached)"

                print("-" * 60)
                print(f"Episode Finished at step {step_counter}!")
                print(f"Total Reward : {final_reward:.4f}")
                print(f"Status       : {status_icon} {status_msg}")
                print(f"Details      : Reached={is_reached}, Final Violation Status={is_violated_final}")
                print("-" * 60)
                
                # [NEW] 안쪽 루프를 탈출하여 새 에피소드에서 다시 시작하도록 함
                break
    
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()