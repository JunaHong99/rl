import torch
import random
from collections import deque
from torch_geometric.data import Data, Batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GraphReplayBuffer:
    """
    GNN을 위한 리플레이 버퍼.
    PyG Data 객체를 저장하고 PyG Batch 객체로 샘플링합니다.
    """
    def __init__(self, capacity: int):
        # deque는 지정된 용량(capacity)이 찼을 때
        # 가장 오래된 항목을 자동으로 삭제하는 효율적인 자료구조입니다.
        self.buffer = deque(maxlen=capacity)

    def add(self, state_graph: Data, action, next_state_graph: Data, reward: float, done: bool):
        """
        하나의 경험(transition)을 버퍼에 저장합니다.
        
        Args:
            state_graph (Data): 현재 상태의 PyG Data 객체
            action: 수행한 행동 (Tensor)
            next_state_graph (Data): 다음 상태의 PyG Data 객체
            reward (float): 받은 보상
            done (bool): 에피소드 종료 여부
        """
        # (state, action, next_state, reward, done) 튜플로 저장
        experience = (state_graph, action, next_state_graph, reward, done)
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        """
        버퍼에서 'batch_size'만큼의 경험을 무작위로 샘플링하고,
        GNN 훈련에 적합한 PyG Batch 객체로 변환하여 반환합니다.
        """
        # 버퍼에서 batch_size만큼의 경험을 무작위로 샘플링
        experiences = random.sample(self.buffer, k=batch_size)
        
        # 샘플링된 튜플 리스트를 각 항목별 리스트로 분리 (unzip)
        # state_graphs: [Data, Data, ..., Data] (길이: batch_size)
        state_graphs, actions, next_state_graphs, rewards, dones = zip(*experiences)

        # --- GNN을 위한 핵심 처리 ---
        # 1. state_graph 리스트를 하나의 PyG 'Batch' 객체로 변환
        state_batch = Batch.from_data_list(state_graphs).to(device)
        
        # 2. next_state_graph 리스트도 하나의 PyG 'Batch' 객체로 변환
        next_state_batch = Batch.from_data_list(next_state_graphs).to(device)
        # -------------------------

        # 3. 나머지 데이터(action, reward, done)를 Tensor로 변환
        # (action이 이미 텐서가 아니라면 변환 필요)
        # 이 예제에서는 action이 이미 텐서라고 가정
        action_batch = torch.stack(actions).to(device) 
        
        reward_batch = torch.tensor(rewards, dtype=torch.float).unsqueeze(1).to(device)
        not_done_batch = torch.tensor([1.0 - d for d in dones], dtype=torch.float).unsqueeze(1).to(device)

        return state_batch, action_batch, next_state_batch, reward_batch, not_done_batch

    def __len__(self):
        """현재 버퍼에 저장된 경험의 수를 반환"""
        return len(self.buffer)

## --------------------------------------------------------------------
## 2. 사용 예시
## --------------------------------------------------------------------
# if __name__ == "__main__":
    
#     # --- 가상의 '변환기'와 '환경' 데이터 생성 ---
#     # (state_to_graph_converter.py가 있다고 가정)
    
#     def get_mock_graph_state(num_robots=2, num_tasks=3):
#         """테스트용 가상 PyG Data 객체를 생성"""
#         total_nodes = num_robots + num_tasks
        
#         # (실제로는 state_to_graph_converter가 이 작업을 수행)
#         return Data(
#             x=torch.rand(total_nodes, 18), # 18차원 노드 피쳐
#             edge_index=torch.randint(0, total_nodes, (2, 10)), # 10개 엣지
#             edge_attr=torch.rand(10, 9), # 9차원 엣지 피쳐
#             u=torch.rand(1, 2) # 2차원 글로벌 피쳐
#         )

#     def get_mock_action(num_robots=2, action_dim=7):
#         """테스트용 가상 행동 텐서 생성"""
#         # (참고: GNN Actor는 [total_nodes, action_dim]을 출력할 수 있으므로,
#         #  버퍼 저장 시 로봇 노드에 해당하는 action만 잘라내 저장해야 함)
#         return torch.rand(num_robots, action_dim)

#     # --- 리플레이 버퍼 초기화 및 데이터 추가 ---
    
#     BUFFER_CAPACITY = 1000
#     BATCH_SIZE = 4
    
#     replay_buffer = GraphReplayBuffer(capacity=BUFFER_CAPACITY)
    
#     print(f"버퍼 생성 (용량: {BUFFER_CAPACITY})")
    
#     # 100개의 가상 경험을 버퍼에 추가
#     for _ in range(100):
#         s_graph = get_mock_graph_state()
#         a = get_mock_action()
#         ns_graph = get_mock_graph_state()
#         r = random.random()
#         d = False
        
#         replay_buffer.add(s_graph, a, ns_graph, r, d)
        
#     print(f"현재 버퍼 크기: {len(replay_buffer)}")

#     # --- 버퍼에서 샘플링 ---
    
#     print(f"\n--- {BATCH_SIZE}개 배치 샘플링 테스트 ---")
    
#     state_batch, action_batch, next_state_batch, reward_batch, not_done_batch = \
#         replay_buffer.sample(batch_size=BATCH_SIZE)

#     print("\n샘플링된 배치 타입:")
#     print(f"state_batch:    {type(state_batch)}")
#     print(f"action_batch:   {type(action_batch)}")
#     print(f"reward_batch:   {type(reward_batch)}")

#     print("\n샘플링된 배치 상세 정보 (State Batch):")
#     print(state_batch)
    
#     # (2 R + 3 T) * 4 Batch = 20개 노드
#     expected_nodes = (2 + 3) * BATCH_SIZE 
#     print(f"\nState Batch의 총 노드 수: {state_batch.num_nodes} (기대값: {expected_nodes})")
#     print(f"State Batch의 그래프 수: {state_batch.num_graphs} (기대값: {BATCH_SIZE})")
#     print(f"State Batch의 'batch' 텐서 (노드-그래프 매핑):\n{state_batch.batch}")
    
#     print(f"\nAction Batch Shape: {action_batch.shape}")