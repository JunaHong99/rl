# import torch
# import argparse
# import os
# import numpy as np
# from isaaclab.app import AppLauncher

# # ---------------------------------------------------------
# # 1. Argparse 설정 (수정됨)
# # ---------------------------------------------------------
# parser = argparse.ArgumentParser(description="Test RoboBallet Agent")

# # [수정] 모델 경로 인자 추가 (필수)
# parser.add_argument("--model_path", type=str, required=True, help="Path to saved actor model (e.g., logs/.../model_step_10000_actor)")
# # 테스트는 기본적으로 1개 환경에서 수행
# parser.add_argument("--num_envs", type=int, default=1, help="Number of environments for testing")

# AppLauncher.add_app_launcher_args(parser)
# args_cli = parser.parse_args()

# # 앱 실행
# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app

# # ---------------------------------------------------------
# # 2. 모듈 임포트 (앱 실행 후)
# # ---------------------------------------------------------
# from dual_arm_transport_env2 import DualrobotEnv, DualrobotCfg
# from graph_converter import (
#     convert_state_to_graph, 
#     NODE_FEATURE_DIM, EDGE_FEATURE_DIM, GLOBAL_FEATURE_DIM
# )
# from agent import TD3

# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"🚀 Testing Start on {device}")

#     # ---------------------------------------------------------
#     # 3. 환경 및 에이전트 초기화
#     # ---------------------------------------------------------
#     env_cfg = DualrobotCfg()
    
#     # [중요] 테스트 시에는 환경을 1개로 고정하는 것이 시각화에 유리함
#     # (CLI에서 --num_envs를 따로 주지 않았다면 1로 설정)
#     env_cfg.scene.num_envs = args_cli.num_envs
    
#     # 렌더링 켜기
#     env = DualrobotEnv(cfg=env_cfg, render_mode="human")

#     # GNN 파라미터 (train.py와 동일)
#     gnn_params = {
#         'node_dim': NODE_FEATURE_DIM,
#         'edge_dim': EDGE_FEATURE_DIM,
#         'global_dim': GLOBAL_FEATURE_DIM,
#         'action_dim': 7
#     }
    
#     # 에이전트 생성
#     agent = TD3(gnn_params=gnn_params, max_action=1.0)
    
#     # ---------------------------------------------------------
#     # 4. 모델 가중치 로드
#     # ---------------------------------------------------------
#     print(f"📂 Loading model from: {args_cli.model_path}")
    
#     if not os.path.exists(args_cli.model_path):
#         print(f"❌ Error: Model file not found at {args_cli.model_path}")
#         return

#     # Actor 모델 로드
#     agent.actor.load_state_dict(torch.load(args_cli.model_path, map_location=device))
#     agent.actor.eval() # 평가 모드 (Dropout 등 비활성화)

#     print("✅ Model loaded. Starting simulation loop...")
    
#     # ---------------------------------------------------------
#     # 5. 테스트 루프
#     # ---------------------------------------------------------
#     while simulation_app.is_running():
#         # 환경 리셋
#         obs_dict, _ = env.reset()
        
#         while True:
#             # (1) 그래프 변환
#             # Test 모드에서는 주로 1개 환경이므로 직접 변환
#             # obs_dict['policy']의 각 텐서는 [Num_Envs, ...] 형태임
#             # 0번 환경의 데이터만 가져와서 그래프로 변환
            
#             # 만약 num_envs > 1이라면 train.py처럼 리스트로 만들어야겠지만,
#             # 여기서는 시각화를 위해 0번 환경만 제어하거나, 모든 환경을 제어하되 
#             # convert_batch_obs_to_graph_list 로직을 가져와야 함.
            
#             # 간단하게 구현하기 위해 '모든 환경'을 처리하는 train.py 방식을 차용
#             graph_list = []
#             keys = list(obs_dict['policy'].keys())
#             for i in range(env.num_envs):
#                 single_env_obs = {k: obs_dict['policy'][k][i] for k in keys}
#                 graph_list.append(convert_state_to_graph(single_env_obs))
            
#             # 배치 그래프 생성
#             from torch_geometric.data import Batch
#             batch_graph = Batch.from_data_list(graph_list).to(device)

#             with torch.no_grad():
#                 actions_tensor = agent.actor(batch_graph) # Output: [Total_Nodes, 7]
                
#                 # --- [수정 시작] ---
#                 # GNN은 로봇 노드와 태스크 노드 모두에 대해 값을 출력하므로,
#                 # 로봇 노드(앞쪽 2개)만 슬라이싱해서 가져와야 합니다.
                
#                 num_robots = 2  # 현재 환경의 로봇 수
#                 action_dim = 7  # 로봇 당 액션 차원
                
#                 # 1. 배치 차원 복원: [Batch_Size, Num_Nodes_Per_Graph, Action_Dim]
#                 # actions_tensor.shape[0]는 (Batch * Num_Nodes)입니다.
#                 # 현재 Batch=1이므로 Num_Nodes=4 (Robot 2 + Task 2)가 됩니다.
#                 num_nodes_per_graph = actions_tensor.shape[0] // env.num_envs
#                 actions_reshaped = actions_tensor.view(env.num_envs, num_nodes_per_graph, action_dim)
                
#                 # 2. 로봇 노드만 추출 (graph_converter에서 로봇 노드를 앞쪽에 배치했음)
#                 robot_actions = actions_reshaped[:, :num_robots, :] # [Batch, 2, 7]
                
#                 # 3. 환경 입력 형태인 [Batch, 14]로 변환
#                 env_actions_tensor = robot_actions.reshape(env.num_envs, -1)
#                 # --- [수정 끝] ---

#                 # Test 시에는 탐험 노이즈를 추가하지 않음! (Pure Policy)
#                 # env_actions_tensor = env_actions_tensor.clamp(-1.0, 1.0)

#             # (3) 환경 스텝- env.step은 5개의 반환을 갖는다.
#             obs_dict, rewards, terminated, truncated, extras = env.step(env_actions_tensor)
#             dones = terminated | truncated #(terminated: 성공/실패로 끝남, truncated: 시간 초과)
#             # (4) 종료 확인
#             # if dones.any():
#             #     print(f"Episode Finished. Reward: {torch.mean(rewards).item():.4f}")
#             #     # 엔터키를 누르면 다음 에피소드, 아니면 그냥 계속 진행 등
#             #     # 여기서는 자동으로 리셋되므로 루프 계속 돔
                
#             #     # 만약 한 에피소드만 보고 싶다면 break
#             #     # break
#             # (4) 종료 확인 및 로그 출력
#             if dones.any():
#                 # 테스트는 보통 1개 환경(env 0)에서 진행하므로 0번 인덱스 기준
#                 # (여러 환경이면 반복문 사용)
#                 env_idx = 0
                
#                 # Extras에서 정보 가져오기 (0.5보다 크면 True)
#                 is_reached = extras["log/is_reached"][env_idx].item() > 0.5
#                 is_violated = extras["log/violation"][env_idx].item() > 0.5
#                 is_success = extras["log/success"][env_idx].item() > 0.5
#                 final_reward = rewards[env_idx].item()

#                 # 상황별 메시지 결정
#                 if is_success:
#                     status_icon = "🏆"
#                     status_msg = "Perfect Success (Reached & Safe)"
#                 elif is_reached and is_violated:
#                     status_icon = "⚠️"
#                     status_msg = "Reached but Violated (Stigma)"
#                 elif not is_reached and is_violated:
#                     status_icon = "❌"
#                     status_msg = "Failed (Violated & Not Reached)"
#                 else: # not reached, not violated
#                     status_icon = "⏳"
#                     status_msg = "Time Out (Safe but Not Reached)"

#                 print("-" * 50)
#                 print(f"Episode Finished!")
#                 print(f"Total Reward : {final_reward:.4f}")
#                 print(f"Status       : {status_icon} {status_msg}")
#                 print(f"Details      : Reached={is_reached}, Violated={is_violated}")
#                 print("-" * 50)

#     env.close()
#     simulation_app.close()

# if __name__ == "__main__":
#     main()

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
            for i in range(env.num_envs):
                is_currently_violated = currently_violating_tensor[i].item()

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