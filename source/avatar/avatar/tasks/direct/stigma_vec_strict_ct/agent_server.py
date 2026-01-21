
# agent.py
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

import gnn_core as core  # 위에서 저장한 gnn_core.py 임포트

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorNetwork(nn.Module):
    def __init__(self, node_dim, edge_dim, global_dim, action_dim, max_action, num_gnn_steps=1):
        super().__init__()
        self.max_action = max_action
        self.num_gnn_steps = num_gnn_steps

        # 1. Embedding Layers (Raw Feature -> Latent)
        self.node_embedder = core.MLP(node_dim, 512, num_layers=3) #5
        self.edge_embedder = core.MLP(edge_dim, 256, num_layers=3) #5
        self.global_embedder = core.MLP(global_dim, 512, num_layers=3) #5

        # 2. GNN Block
        self.gnn = core.RoboBalletGNNBlock(node_dim=512, edge_dim=256, global_dim=512, hidden_dim=256)

        # 3. Action Head
        self.action_head = core.MLP(512, 256, num_layers=2, output_dim=action_dim)

    def forward(self, graph_batch):
        x, edge_index, edge_attr, u, batch = \
            graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr, graph_batch.u, graph_batch.batch

        # Embedding
        x = self.node_embedder(x)
        edge_attr = self.edge_embedder(edge_attr)
        u = self.global_embedder(u)

        # GNN Pass
        for _ in range(self.num_gnn_steps):
            x, edge_attr, u = self.gnn(x, edge_index, edge_attr, u, batch)

        # Action Decoding (모든 노드에 대해 계산 후 로봇 노드만 사용)
        # tanh로 -1~1 범위 조정 후 max_action 스케일링
        return self.max_action * torch.tanh(self.action_head(x))


class CriticNetwork(nn.Module):
    def __init__(self, node_dim, edge_dim, global_dim, action_dim, num_gnn_steps=1):
        super().__init__()
        self.num_gnn_steps = num_gnn_steps
        
        # Critic Node Input = Node Feature + Action
        critic_node_dim = node_dim + action_dim

        # Q1 Network
        self.n_emb1 = core.MLP(critic_node_dim, 512, num_layers=3)
        self.e_emb1 = core.MLP(edge_dim, 256, num_layers=3)
        self.g_emb1 = core.MLP(global_dim, 512, num_layers=3)
        self.gnn1 = core.RoboBalletGNNBlock(512, 256, 512, 256)
        self.head1 = core.MLP(512, 256, num_layers=2, output_dim=1) # Global feature -> Value

        # Q2 Network
        self.n_emb2 = core.MLP(critic_node_dim, 512, num_layers=3)
        self.e_emb2 = core.MLP(edge_dim, 256, num_layers=3)
        self.g_emb2 = core.MLP(global_dim, 512, num_layers=3)
        self.gnn2 = core.RoboBalletGNNBlock(512, 256, 512, 256)
        self.head2 = core.MLP(512, 256, num_layers=2, output_dim=1)


    def _forward_single(self, x, edge_index, edge_attr, u, batch, action, n_emb, e_emb, g_emb, gnn, head):
        """
        x: [Total_Nodes, Node_Dim] (예: [4*B, 14])
        action: [Batch, Total_Action_Dim] (예: [B, 14]) -> 이걸 쪼개서 로봇 노드에만 붙여야 함
        """
        # 1. 차원 정보 계산
        batch_size = action.shape[0]
        total_action_dim = action.shape[1] # 14
        num_robots = 2 
        action_dim_per_node = total_action_dim // num_robots # 7
        
        # 2. 액션을 로봇별로 분리: [Batch, 14] -> [Batch, 2, 7]
        robot_actions = action.reshape(batch_size, num_robots, action_dim_per_node)
        
        # 3. 그래프당 나머지 노드(태스크, 장애물 등) 개수 계산
        # x.shape[0]는 전체 노드 수, batch_size로 나누면 그래프당 노드 수
        num_nodes_per_graph = x.shape[0] // batch_size
        num_others = num_nodes_per_graph - num_robots
        
        # 4. 나머지 노드를 위한 더미 액션(0) 생성: [Batch, N_others, 7]
        other_actions = torch.zeros(batch_size, num_others, action_dim_per_node, device=x.device)
        
        # 5. 순서대로 합치기 (로봇이 먼저 온다고 가정): [Batch, N_total, 7]
        full_action_per_graph = torch.cat([robot_actions, other_actions], dim=1)
        
        # 6. 노드 피쳐와 합치기 위해 플래튼: [Total_Nodes, 7]
        flat_action = full_action_per_graph.view(-1, action_dim_per_node)

        # 7. 노드 피쳐와 결합: [Total_Nodes, 14] + [Total_Nodes, 7] = [Total_Nodes, 21]
        x_cat = torch.cat([x, flat_action], dim=1)

        # 8. GNN Pass
        x_h = n_emb(x_cat)
        e_h = e_emb(edge_attr)
        u_h = g_emb(u)

        for _ in range(self.num_gnn_steps):
            x_h, e_h, u_h = gnn(x_h, edge_index, e_h, u_h, batch)
        
        # Global Feature -> Q Value
        return head(u_h)

    def forward(self, graph_batch, action):
        x, ei, ea, u, b = graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr, graph_batch.u, graph_batch.batch
        
        q1 = self._forward_single(x, ei, ea, u, b, action, self.n_emb1, self.e_emb1, self.g_emb1, self.gnn1, self.head1)
        q2 = self._forward_single(x, ei, ea, u, b, action, self.n_emb2, self.e_emb2, self.g_emb2, self.gnn2, self.head2)
        return q1, q2

    def Q1(self, graph_batch, action):
        x, ei, ea, u, b = graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr, graph_batch.u, graph_batch.batch
        return self._forward_single(x, ei, ea, u, b, action, self.n_emb1, self.e_emb1, self.g_emb1, self.gnn1, self.head1)


class TD3(object):
    def __init__(self, gnn_params, max_action, lr=3e-4):
        # gnn_params: {node_dim, edge_dim, global_dim, action_dim}
        self.actor = ActorNetwork(**gnn_params, max_action=max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = CriticNetwork(**gnn_params).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.criterion = nn.MSELoss()

    def select_action(self, graph_state):
        # Single Graph -> Batch
        batch = Batch.from_data_list([graph_state]).to(device)
        self.actor.eval()
        with torch.no_grad():
            # Output: [Total Nodes, Action Dim]
            full_actions = self.actor(batch)
            
            # 로봇 노드(앞부분)만 잘라서 반환 (가정: 로봇 2대)
            # 실제 환경에서는 robot node indices를 정확히 알아야 함
            # 여기서는 graph_converter 로직에 따라 앞쪽 2개가 로봇이라고 가정
            robot_actions = full_actions[:2] 
            
        self.actor.train()
        return robot_actions.cpu().numpy()

    def train(self, replay_buffer, batch_size=64):
        # 1. Sample
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # --- 상수 정의 ---
        num_robots = 2
        action_dim_per_node = 7
        # ----------------

        # 2. Target Q Calculation
        with torch.no_grad():
            # Actor Target 출력: [Total_Nodes, 7]
            next_full_action = self.actor_target(next_state)
            
            # [수정 핵심] Actor 출력을 [Batch, 14]로 변환하여 Critic에게 전달
            # A. [Batch, Nodes_Per_Graph, 7]로 Reshape
            next_actions_reshaped = next_full_action.reshape(batch_size, -1, action_dim_per_node)
            
            # B. 로봇 노드(앞 2개)만 슬라이싱: [Batch, 2, 7]
            next_robot_actions = next_actions_reshaped[:, :num_robots, :]
            
            # C. 다시 [Batch, 14]로 합침 (Replay Buffer의 action과 동일한 형태)
            next_action_flat = next_robot_actions.reshape(batch_size, -1)

            # Add Noise
            noise = (torch.randn_like(next_action_flat) * 0.2).clamp(-0.5, 0.5) #0.02 -> 0.0005
            next_action = (next_action_flat + noise).clamp(-self.max_action, self.max_action)

            # Target Q (이제 next_action은 [Batch, 14]이므로 Shape Mismatch 없음)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * 0.99 * target_Q

        # 3. Critic Update
        # action은 이미 [Batch, 14] 형태임
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = self.criterion(current_Q1, target_Q) + self.criterion(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 4. Actor Update
        # Actor Current 출력: [Total_Nodes, 7]
        full_actor_action = self.actor(state)
        
        # 위와 동일하게 [Batch, 14]로 변환
        actor_actions_reshaped = full_actor_action.reshape(batch_size, -1, action_dim_per_node)
        actor_robot_actions = actor_actions_reshaped[:, :num_robots, :] # [Batch, 2, 7]
        actor_action_flat = actor_robot_actions.reshape(batch_size, -1) # [Batch, 14]
        
        # Critic에게 [Batch, 14] 전달
        actor_loss = -self.critic.Q1(state, actor_action_flat).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 5. Target Update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)

    def save(self, filename):
        """모델과 옵티마이저 저장"""
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        """모델과 옵티마이저 불러오기"""
        self.critic.load_state_dict(torch.load(filename + "_critic", map_location=device))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer", map_location=device))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor", map_location=device))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer", map_location=device))
        self.actor_target = copy.deepcopy(self.actor)
