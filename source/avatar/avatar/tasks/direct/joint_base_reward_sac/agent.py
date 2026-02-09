
# agent.py
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

import gnn_core as core  # 위에서 저장한 gnn_core.py 임포트

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GaussianActorNetwork(nn.Module):
    def __init__(self, node_dim, edge_dim, global_dim, action_dim, max_action, num_gnn_steps=1):
        super().__init__()
        self.max_action = max_action
        self.num_gnn_steps = num_gnn_steps
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

        # 1. Embedding Layers (Raw Feature -> Latent)
        self.node_embedder = core.MLP(node_dim, 512, num_layers=7)
        self.edge_embedder = core.MLP(edge_dim, 256, num_layers=6)
        self.global_embedder = core.MLP(global_dim, 512, num_layers=7)

        # 2. GNN Block
        self.gnn = core.RoboBalletGNNBlock(node_dim=512, edge_dim=256, global_dim=512, hidden_dim=256)

        # 3. Action Head (Mean and Log Std)
        self.mean_head = core.MLP(512, 64, num_layers=2, output_dim=action_dim)
        self.log_std_head = core.MLP(512, 64, num_layers=2, output_dim=action_dim)

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

        mu = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

        return mu, log_std

    def sample(self, graph_batch):
        mu, log_std = self.forward(graph_batch)
        std = log_std.exp()
        
        # Reparameterization trick
        dist = torch.distributions.Normal(mu, std)
        x_t = dist.rsample()  # for reparameterization trick (mean + std * N(0,1))
        
        # Enforcing Action Bound
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        
        # Log Probability Calculation
        log_prob = dist.log_prob(x_t)
        
        # Enforcing Action Bound Correction
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        # Mean action for evaluation
        mean_action = torch.tanh(mu) * self.max_action
        
        return action, log_prob, mean_action


class CriticNetwork(nn.Module):
    def __init__(self, node_dim, edge_dim, global_dim, action_dim, num_gnn_steps=1):
        super().__init__()
        self.num_gnn_steps = num_gnn_steps
        
        # Critic Node Input = Node Feature + Action
        critic_node_dim = node_dim + action_dim

        # Q1 Network
        self.n_emb1 = core.MLP(critic_node_dim, 512, num_layers=7)
        self.e_emb1 = core.MLP(edge_dim, 256, num_layers=6)
        self.g_emb1 = core.MLP(global_dim, 512, num_layers=7)
        self.gnn1 = core.RoboBalletGNNBlock(512, 256, 512, 256)
        self.head1 = core.MLP(512, 64, num_layers=2, output_dim=1) # Global feature -> Value

        # Q2 Network
        self.n_emb2 = core.MLP(critic_node_dim, 512, num_layers=7)
        self.e_emb2 = core.MLP(edge_dim, 256, num_layers=6)
        self.g_emb2 = core.MLP(global_dim, 512, num_layers=7)
        self.gnn2 = core.RoboBalletGNNBlock(512, 256, 512, 256)
        self.head2 = core.MLP(512, 64, num_layers=2, output_dim=1)


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

        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer", map_location=device))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor", map_location=device))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer", map_location=device))
        self.actor_target = copy.deepcopy(self.actor)


class SAC(object):
    def __init__(self, gnn_params, max_action, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, automatic_entropy_tuning=True):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning

        # Actor (Gaussian)
        self.actor = GaussianActorNetwork(**gnn_params, max_action=max_action).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # Critic (Q1, Q2)
        self.critic = CriticNetwork(**gnn_params).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        # Target Critic
        self.critic_target = copy.deepcopy(self.critic)
        
        self.max_action = max_action

        # Automatic Entropy Tuning
        if self.automatic_entropy_tuning:
            # We want to maintain a minimum entropy. 
            # For a 14-dim action space (2 robots * 7 joints), a heuristic is -dim(A).
            # However, since the actor outputs per-node actions (7 dim) and we sum/mean later,
            # we need to consider how `log_prob` is computed.
            # In `GaussianActorNetwork.sample`, log_prob is per node (summed over 7 dim).
            # If we treat the whole graph as one "action" composed of multiple node actions,
            # target entropy should be roughly -target_action_dim.
            # Since the code processes per-node but we minimize scalar loss, 
            # we'll target -action_dim (7) per robot node.
            
            self.target_entropy = -2 * gnn_params['action_dim']
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

    def select_action(self, graph_state, evaluate=False):
        batch = Batch.from_data_list([graph_state]).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            if evaluate:
                _, _, action = self.actor.sample(batch)
            else:
                action, _, _ = self.actor.sample(batch)
            
            # Robot nodes only (assuming first 2 nodes are robots)
            robot_actions = action[:2]
            
        self.actor.train()
        return robot_actions.cpu().numpy()

    def train(self, replay_buffer, batch_size=256):
        # 1. Sample
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        
        # --- Constants ---
        num_robots = 2
        action_dim_per_node = 7
        # -----------------

        with torch.no_grad():
            # Target Actions (Sampled from Actor)
            next_action_full, next_log_prob_full, _ = self.actor.sample(next_state)
            
            # Reshape next_action to [Batch, 14] for Critic
            next_action_reshaped = next_action_full.reshape(batch_size, -1, action_dim_per_node)
            next_robot_actions = next_action_reshaped[:, :num_robots, :]
            next_action_flat = next_robot_actions.reshape(batch_size, -1) # [Batch, 14]

            # Reshape next_log_prob to [Batch, 1] (Sum of log probs of robot nodes)
            # next_log_prob_full shape: [Total_Nodes, 1]
            next_log_prob_reshaped = next_log_prob_full.reshape(batch_size, -1, 1)
            # Only consider robot nodes' log prob
            next_robot_log_prob = next_log_prob_reshaped[:, :num_robots, :] # [Batch, 2, 1]
            next_log_prob = next_robot_log_prob.sum(dim=1) # [Batch, 1]

            # Target Q-Values
            target_Q1, target_Q2 = self.critic_target(next_state, next_action_flat)
            target_Q = torch.min(target_Q1, target_Q2) - self.alpha * next_log_prob
            target_Q = reward + not_done * self.gamma * target_Q

        # 2. Critic Update
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 3. Actor Update
        # Resample actions for current state
        action_full, log_prob_full, _ = self.actor.sample(state)
        
        # Reshape for Critic
        action_reshaped = action_full.reshape(batch_size, -1, action_dim_per_node)
        robot_actions = action_reshaped[:, :num_robots, :]
        action_flat = robot_actions.reshape(batch_size, -1)
        
        # Reshape Log Prob
        log_prob_reshaped = log_prob_full.reshape(batch_size, -1, 1)
        robot_log_prob = log_prob_reshaped[:, :num_robots, :]
        log_prob = robot_log_prob.sum(dim=1)

        # Q-values for current policy
        actor_Q1, actor_Q2 = self.critic(state, action_flat)
        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = (self.alpha * log_prob - actor_Q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 4. Alpha Update
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.tensor(0.0)

        # 5. Target Update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        return {
            "loss/actor": actor_loss.item(),
            "loss/critic": critic_loss.item(),
            "loss/alpha": alpha_loss.item(),
            "train/alpha": self.alpha.item(),
            "train/mean_q": actor_Q.mean().item(),
            "train/entropy": -log_prob.mean().item()
        }

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        if self.automatic_entropy_tuning:
            torch.save(self.log_alpha, filename + "_log_alpha")
            torch.save(self.alpha_optimizer.state_dict(), filename + "_alpha_optimizer")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor", map_location=self.device))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer", map_location=self.device))
        
        self.critic.load_state_dict(torch.load(filename + "_critic", map_location=self.device))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer", map_location=self.device))
        self.critic_target = copy.deepcopy(self.critic)
        
        if self.automatic_entropy_tuning and os.path.exists(filename + "_log_alpha"):
            self.log_alpha = torch.load(filename + "_log_alpha", map_location=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer.load_state_dict(torch.load(filename + "_alpha_optimizer", map_location=self.device))