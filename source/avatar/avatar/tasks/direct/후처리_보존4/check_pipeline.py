# # check_pipeline.py

# import torch
# import numpy as np
# from torch_geometric.data import Batch

# # --- 1. 프로젝트의 핵심 컴포넌트들을 임포트합니다. ---
# # (파일 이름이 실제와 다른 경우, 이 부분을 수정해야 합니다.)
# try:
#     from graph_converter import (
#         convert_state_to_graph,
#         NODE_FEATURE_DIM, EDGE_FEATURE_DIM, GLOBAL_FEATURE_DIM,
#         RAW_ROBOT_DIM, RAW_TASK_DIM, RAW_OBSTACLE_DIM
#     )
#     from replay_buffer import GraphReplayBuffer
#     from agent import TD3 as RoboBallet_TD3
# except ImportError as e:
#     print(f"Import Error: {e}")
#     print("스크립트 임포트에 실패했습니다. 파일 이름과 위치를 확인해주세요.")
#     print("예상 파일: graph_converter.py, replay_buffer.py, agent.py")
#     exit()

# # --- 2. 예측 가능한 Mock Environment ---
# # main.py의 MockEnv를 개선하여, 행동에 따라 상태가 변하고 명확한 보상 체계를 갖도록 합니다.
# class MockEnv:
#     def __init__(self, n_robots=2, n_tasks=1, max_steps=50, action_dim=7):
#         self.n_robots = n_robots
#         self.n_tasks = n_tasks
#         self.action_dim = action_dim
#         self.step_count = 0
#         self.max_steps = max_steps
#         self.tolerance = 0.05 # 목표 도달 허용 오차

#         # 가상 환경의 상태
#         self.robot_state = torch.zeros(self.n_robots, RAW_ROBOT_DIM)
#         self.task_state = torch.zeros(self.n_tasks, RAW_TASK_DIM)
#         self.obstacle_state = torch.zeros(1, RAW_OBSTACLE_DIM)
#         self.goal_pos = torch.zeros(3)

#     def _get_raw_state(self):
#         return {
#             'robots': self.robot_state.clone(),
#             'tasks': self.task_state.clone(),
#             'obstacles': self.obstacle_state.clone(),
#             'globals': torch.tensor([self.step_count / self.max_steps, 0.0])
#         }

#     def reset(self):
#         self.step_count = 0
#         self.robot_state.zero_()
#         # 로봇의 위치(15:18)를 무작위로 설정
#         self.robot_state[:, 15:18] = torch.rand(self.n_robots, 3) * 2 - 1
        
#         # 목표(태스크) 위치를 무작위로 설정
#         self.goal_pos = torch.rand(3) * 2 - 1
#         self.task_state[:, 0:3] = self.goal_pos
        
#         achieved_goal = self.robot_state[0, 15:18].clone()
#         info = {'achieved_goal': achieved_goal, 'desired_goal': self.goal_pos.clone()}
#         return self._get_raw_state(), info

#     def step(self, action: np.ndarray):
#         self.step_count += 1
        
#         # 행동(속도 제어)에 따라 로봇 위치를 업데이트
#         action_tensor = torch.from_numpy(action).float()
#         # 7-dof 중 앞의 3개만 위치 제어에 사용한다고 가정
#         self.robot_state[:, 15:18] += action_tensor[:, 0:3] * 0.1 
        
#         next_raw_state = self._get_raw_state()
        
#         # 목표 도달 여부 확인 (0번 로봇 기준)
#         achieved_goal = self.robot_state[0, 15:18].clone()
#         distance = torch.norm(achieved_goal - self.goal_pos)
        
#         done = (distance < self.tolerance) or (self.step_count >= self.max_steps)
#         reward = 1.0 if (distance < self.tolerance) else 0.0
        
#         info = {'achieved_goal': achieved_goal, 'desired_goal': self.goal_pos.clone()}
        
#         return next_raw_state, reward, done, info

# # --- 3. 파이프라인 검증 메인 함수 ---
# def run_pipeline_check():
#     print("--- RoboBallet 코드 파이프라인 검증 시작 ---")

#     # --- 설정 ---
#     N_ROBOTS = 2
#     ACTION_DIM = 7
#     MAX_ACTION = 1.0
#     BATCH_SIZE = 4

#     # --- 단계 1: 모든 컴포넌트 초기화 ---
#     try:
#         env = MockEnv(n_robots=N_ROBOTS, action_dim=ACTION_DIM)
#         replay_buffer = GraphReplayBuffer(capacity=1000)
#         gnn_params = {
#             'node_dim': NODE_FEATURE_DIM, 'edge_dim': EDGE_FEATURE_DIM,
#             'global_dim': GLOBAL_FEATURE_DIM, 'action_dim': ACTION_DIM
#         }
#         agent = RoboBallet_TD3(gnn_params=gnn_params, max_action=MAX_ACTION)
#         print("✅ [1/5] 모든 컴포넌트 초기화 성공")
#     except Exception as e:
#         print(f"❌ [1/5] 컴포넌트 초기화 중 오류 발생: {e}")
#         return

#     # --- 단계 2: 환경 리셋 및 그래프 변환 ---
#     raw_state, _ = env.reset()
#     state_graph = convert_state_to_graph(raw_state)
#     print("✅ [2/5] 환경 리셋 및 그래프 변환 성공")
#     print(f"    - 변환된 State Graph: {state_graph}")
#     print(f"    - 노드 수: {state_graph.num_nodes}, 엣지 수: {state_graph.num_edges}")

#     # --- 단계 3: 행동 선택 (Action Selection) ---
#     action_tensor = agent.select_action(state_graph)
#     robot_actions = action_tensor[0:N_ROBOTS]
#     print("✅ [3/5] 에이전트 행동 선택 성공")
#     print(f"    - Actor가 출력한 전체 Action Tensor Shape: {action_tensor.shape}")
#     print(f"    - 환경에 전달될 Robot Action Shape: {robot_actions.shape}")
#     assert robot_actions.shape == (N_ROBOTS, ACTION_DIM), "로봇 액션 Shape 불일치"

#     # --- 단계 4: 리플레이 버퍼 채우기 ---
#     for _ in range(BATCH_SIZE * 2):
#         next_raw_state, reward, done, _ = env.step(robot_actions)
#         next_state_graph = convert_state_to_graph(next_raw_state)
#         replay_buffer.add(state_graph, torch.from_numpy(robot_actions).float(), next_state_graph, reward, done)
#         state_graph = next_state_graph
#         if done:
#             raw_state, _ = env.reset()
#             state_graph = convert_state_to_graph(raw_state)
#         # 다음 스텝을 위해 새로운 액션 선택
#         action_tensor = agent.select_action(state_graph)
#         robot_actions = action_tensor[0:N_ROBOTS]

#     print(f"✅ [4/5] 리플레이 버퍼에 데이터 추가 성공 (현재 크기: {len(replay_buffer)})")

#     # --- 단계 5: 에이전트 훈련 (가장 중요한 검증) ---
#     print("⏳ [5/5] agent.train() 1회 실행 시도...")
#     try:
#         agent.train(replay_buffer, BATCH_SIZE)
#         print("✅ [5/5] agent.train() 1회 실행 성공!")
#     except Exception as e:
#         print(f"❌ [5/5] agent.train() 실행 중 심각한 오류 발생!")
#         print("\n--- ERROR ---")
#         print(e)
#         print("\n--- DEBUG INFO ---")
#         print("오류를 유발했을 가능성이 있는 샘플 데이터를 출력합니다.")
#         s, a, ns, r, nd = replay_buffer.sample(BATCH_SIZE)
#         print("State Batch:", s)
#         print("Action Batch Shape:", a.shape)
#         print("Next State Batch:", ns)
#         print("Reward Batch Shape:", r.shape)
#         return

#     print("\n🎉 --- 파이프라인 검증 완료: 핵심 로직에 심각한 오류가 발견되지 않았습니다. --- 🎉")

# if __name__ == "__main__":
#     run_pipeline_check()

# check_pipeline.py
import torch
import numpy as np
from torch_geometric.data import Batch

try:
    from graph_converter import (
        convert_state_to_graph, NODE_FEATURE_DIM, EDGE_FEATURE_DIM, GLOBAL_FEATURE_DIM
    )
    from replay_buffer import GraphReplayBuffer
    from agent import TD3
except ImportError as e:
    print(f"Import Error: {e}")
    exit()

class MockEnv:
    def __init__(self, n_robots=2):
        self.n_robots = n_robots
        
    def _get_obs(self):
        # graph_converter가 기대하는 딕셔너리 구조 생성
        return {
            'robot_nodes': torch.rand(1, self.n_robots, 14), # [B, N, 14]
            'current_ee_poses': torch.rand(1, self.n_robots, 7),
            'goal_poses': torch.rand(1, self.n_robots, 7), # Task node count = robot count (simplified)
            'base_poses': torch.rand(1, self.n_robots, 7),
            'target_rel_pose': torch.rand(1, 7)
        }

    def reset(self):
        return self._get_obs(), {}

    def step(self, action):
        return self._get_obs(), 1.0, False, {}

def run_pipeline_check():
    print("--- RoboBallet Pipeline Check ---")
    
    N_ROBOTS = 2
    ACTION_DIM = 7
    
    # 1. Init
    env = MockEnv(N_ROBOTS)
    rb = GraphReplayBuffer(100)
    gnn_params = {
        'node_dim': NODE_FEATURE_DIM, 
        'edge_dim': EDGE_FEATURE_DIM,
        'global_dim': GLOBAL_FEATURE_DIM, 
        'action_dim': ACTION_DIM
    }
    agent = TD3(gnn_params, max_action=1.0)
    
    # 2. Reset & Graph
    raw_obs, _ = env.reset()
    graph_state = convert_state_to_graph(raw_obs)
    print(f"✅ Graph Converted: Nodes={graph_state.num_nodes}, Edges={graph_state.num_edges}")
    
    # 3. Action
    action = agent.select_action(graph_state)
    print(f"✅ Action Selected: {action.shape}") # Expect (2, 7)
    
    # 4. Buffer Add
    next_obs, r, d, _ = env.step(action)
    next_graph = convert_state_to_graph(next_obs)
    rb.add(graph_state, torch.tensor(action), next_graph, r, d)
    
    # Fill buffer
    for _ in range(10):
        rb.add(graph_state, torch.tensor(action), next_graph, r, d)
        
    # 5. Train
    print("⏳ Training...")
    agent.train(rb, batch_size=4)
    print("🎉 파이프라인 검증 완료")

if __name__ == "__main__":
    run_pipeline_check()