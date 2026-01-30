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


class VectorizedGraphReplayBuffer:
    """
    [Optimized] 고정된 위상(Topology)을 가진 그래프를 위한 텐서 기반 리플레이 버퍼.
    Python List/Object 오버헤드 없이 PyTorch Tensor만으로 데이터를 관리합니다.
    """
    def __init__(self, capacity: int, num_envs: int, 
                 node_dim: int, edge_dim: int, global_dim: int, action_dim: int,
                 template_graph: Batch, device: str = "cpu"):
        
        self.capacity = capacity
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.ptr = 0
        self.size = 0
        
        # 그래프 구조 정보 (모든 샘플이 공유)
        # template_graph는 배치 크기 1인 단일 환경 그래프여야 함 (또는 num_envs 전체일 수도 있지만 여기선 구조만 따옴)
        # 여기서는 num_envs 크기의 배치를 처리할 때 사용할 'Template'을 저장합니다.
        # 실제로는 edge_index가 고정이라 가정하므로, 샘플링 시 배치를 구성할 때 재사용합니다.
        
        # 템플릿 정보 추출 (Single Env 기준)
        # 입력된 template_graph가 [num_envs] 배치라면, 하나만 떼어내서 저장
        if hasattr(template_graph, 'num_graphs') and template_graph.num_graphs > 1:
            # 첫 번째 그래프만 추출 (슬라이싱이 복잡하므로 구조적 가정 사용)
            # 가정: 모든 환경은 동일한 노드 수/에지 수를 가짐
            self.num_nodes_per_env = template_graph.x.shape[0] // template_graph.num_graphs
            self.num_edges_per_env = template_graph.edge_index.shape[1] // template_graph.num_graphs
        else:
            self.num_nodes_per_env = template_graph.x.shape[0]
            self.num_edges_per_env = template_graph.edge_index.shape[1]

        # 저장소 사전 할당 (Capacity만큼)
        # 주의: Capacity는 '환경 스텝' 단위가 아니라 '배치 스텝' 단위로 관리할 수도 있고, 
        #      개별 '환경' 단위로 관리할 수도 있음.
        #      TD3는 무작위 샘플링이 필요하므로, 개별 환경 단위(Transition)로 Flatten해서 저장하는 게 샘플링에 유리함.
        #      즉, buffer size = capacity (예: 1,000,000 transitions)
        
        self.states_x = torch.zeros((capacity, self.num_nodes_per_env, node_dim), dtype=torch.float32, device=self.device)
        self.states_edge_attr = torch.zeros((capacity, self.num_edges_per_env, edge_dim), dtype=torch.float32, device=self.device)
        self.states_u = torch.zeros((capacity, global_dim), dtype=torch.float32, device=self.device)
        
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=self.device)
        
        self.next_states_x = torch.zeros((capacity, self.num_nodes_per_env, node_dim), dtype=torch.float32, device=self.device)
        self.next_states_edge_attr = torch.zeros((capacity, self.num_edges_per_env, edge_dim), dtype=torch.float32, device=self.device)
        self.next_states_u = torch.zeros((capacity, global_dim), dtype=torch.float32, device=self.device)
        
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device)

        # Edge Index 템플릿 (샘플링 시 배치 구성을 위해 필요)
        # template_graph의 edge_index는 (2, E) 형태. 
        # 0번 환경 기준의 edge_index를 저장해둠.
        # 만약 template_graph가 배치라면 0번 환경것만 가져와야 함.
        # 여기서는 간단히: 0~N-1 노드 인덱스를 가지는 edge_index
        if template_graph.edge_index.max() >= self.num_nodes_per_env:
             # 입력이 배치라면 0번 그래프만 추출하는 로직 필요.
             # 단순히 0번 환경의 에지만 잘라냄 (가정: 앞쪽이 0번)
             self.template_edge_index = template_graph.edge_index[:, :self.num_edges_per_env].clone().to(self.device)
        else:
             self.template_edge_index = template_graph.edge_index.clone().to(self.device)

    def __len__(self):
        return self.size

    def add_batch(self, state_batch: Batch, action: torch.Tensor, next_state_batch: Batch, reward: torch.Tensor, done: torch.Tensor):
        """
        배치 데이터를 한 번에 버퍼에 추가합니다.
        state_batch: [Num_Envs * Nodes, Dim]을 가진 PyG Batch
        action: [Num_Envs, Action_Dim]
        """
        batch_size = action.shape[0] # Num_Envs
        
        # 버퍼에 들어갈 공간 확보
        # Circular Buffer 로직: 끝에 도달하면 처음으로 돌아감
        # 만약 배치가 남은 공간보다 크면 나눠서 넣거나(구현 복잡), 
        # 그냥 남은 공간만큼만 넣고 나머진 처음부터 넣음 (간단).
        
        indices = torch.arange(self.ptr, self.ptr + batch_size, device=self.device) % self.capacity
        
        # 1. State Deconstruction
        # Batch(x=[B*N, F]) -> Tensor([B, N, F])
        x = state_batch.x.view(batch_size, self.num_nodes_per_env, -1)
        edge_attr = state_batch.edge_attr.view(batch_size, self.num_edges_per_env, -1)
        u = state_batch.u.view(batch_size, -1) # u는 [B, Global_Dim] 형태임 (이미 배치)
        
        next_x = next_state_batch.x.view(batch_size, self.num_nodes_per_env, -1)
        next_edge_attr = next_state_batch.edge_attr.view(batch_size, self.num_edges_per_env, -1)
        next_u = next_state_batch.u.view(batch_size, -1)
        
        # 2. Store to Tensor
        self.states_x[indices] = x
        self.states_edge_attr[indices] = edge_attr
        self.states_u[indices] = u
        
        self.next_states_x[indices] = next_x
        self.next_states_edge_attr[indices] = next_edge_attr
        self.next_states_u[indices] = next_u
        
        self.actions[indices] = action
        self.rewards[indices] = reward.view(-1, 1).float()
        self.dones[indices] = done.view(-1, 1).float()
        
        # Update Pointer & Size
        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size: int):
        """
        랜덤 샘플링 후 PyG Batch 객체 재조립 (GPU 유지)
        """
        # 1. Random Indices
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        # 2. Retrieve Data
        x = self.states_x[indices]         # [B, N, F]
        ea = self.states_edge_attr[indices]# [B, E, F]
        u = self.states_u[indices]         # [B, G]
        
        nx = self.next_states_x[indices]
        nea = self.next_states_edge_attr[indices]
        nu = self.next_states_u[indices]
        
        act = self.actions[indices]
        rew = self.rewards[indices]
        don = self.dones[indices]
        
        # 3. Reconstruct PyG Batch
        # (A) State Batch
        state_batch = self._reconstruct_batch(x, ea, u, batch_size)
        
        # (B) Next State Batch
        next_state_batch = self._reconstruct_batch(nx, nea, nu, batch_size)
        
        return state_batch, act, next_state_batch, rew, 1.0 - don

    def _reconstruct_batch(self, x, ea, u, batch_size):
        # x: [B, N, F] -> [B*N, F]
        flat_x = x.view(-1, x.shape[-1])
        flat_ea = ea.view(-1, ea.shape[-1])
        
        # Edge Index Reconstruction
        # Template: [2, E] -> Repeat [B, 2, E] -> Add Offsets
        base_edge_index = self.template_edge_index # [2, E]
        
        # Offset: [0, N, 2N, ...]
        offsets = torch.arange(batch_size, device=self.device) * self.num_nodes_per_env
        offsets = offsets.view(batch_size, 1, 1) # [B, 1, 1]
        
        # [1, 2, E]
        base_expanded = base_edge_index.unsqueeze(0)
        
        # [B, 2, E] + [B, 1, 1] (Broadcasting)
        batch_edge_index = base_expanded + offsets
        
        # Flatten -> [2, B*E]
        # (dim 1인 axis 2와 axis 0을 permute하고 합쳐야 함)
        # Target: [2, Total_Edges]
        # batch_edge_index: [Batch, 2, Edges]
        # permute -> [2, Batch, Edges] -> reshape -> [2, Batch*Edges]
        flat_edge_index = batch_edge_index.permute(1, 0, 2).reshape(2, -1)
        
        # Batch Vector: [0...0, 1...1, ...]
        batch_vec = torch.arange(batch_size, device=self.device).repeat_interleave(self.num_nodes_per_env)
        
        batch_obj = Batch(
            x=flat_x,
            edge_index=flat_edge_index,
            edge_attr=flat_ea,
            u=u,
            batch=batch_vec
        )
        batch_obj.num_graphs = batch_size
        return batch_obj