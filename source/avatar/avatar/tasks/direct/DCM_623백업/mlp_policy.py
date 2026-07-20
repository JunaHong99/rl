"""
MLP-based SAC agent (drop-in replacement for gnn_policy.GNNSACAgent).

진단 목적: GNN이 학습 실패의 원인인지 확인.
GNN과 동일한 interface 유지 → sac_trainer / train script 수정 불필요.

전략:
  - batch (PyG Batch) 입력에서 rod node feature만 추출 (5-node 중 idx=2)
  - rod feature + global feature → flat vector → MLP
  - Squashed Gaussian actor (SAC standard, GNN과 동일)
  - Twin Q critic
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torch.distributions import Normal

import graph_converter as gc


# ──────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────
def _extract_rod_and_global(batch):
    """PyG Batch → (rod_node_features, global_features)."""
    x = batch.x                                                # (B*N, F_node)
    u = batch.u                                                # (B, F_global)
    B = batch.num_graphs
    N = gc.NODES_PER_ENV
    rod_idx = gc.ROD_NODE_IDX + torch.arange(B, device=x.device) * N
    rod_x = x[rod_idx]                                         # (B, F_node)
    return rod_x, u


def _extract_all_and_global(batch):
    """PyG Batch → (flattened_all_node_features, global_features). 5-node 모두 사용."""
    x = batch.x
    u = batch.u
    B = batch.num_graphs
    N = gc.NODES_PER_ENV
    all_x = x.view(B, N * gc.NODE_FEATURE_DIM)                 # (B, N*F_node)
    return all_x, u


def _extract_lean_obstacle(batch):
    """rod 노드(32) + 각 장애물 요약[rod기준 상대pos(3)+dist(1)+radius(1)] + global.
    Stage 0의 깨끗한 rod+global 입력 유지(희석 X) + rod 라우팅용 장애물 정보만 compact 추가.
    장애물 노드 layout: [0:3]=pos, [3]=radius, [7]=dist_to_rod (graph_converter._obstacle_features)."""
    x = batch.x; u = batch.u
    B = batch.num_graphs; N = gc.NODES_PER_ENV
    ar = torch.arange(B, device=x.device)
    rod_x = x[gc.ROD_NODE_IDX + ar * N]                         # (B,32)
    rod_pos = rod_x[:, 0:3]
    parts = [rod_x]
    for k in range(gc.N_OBSTACLES):
        ox = x[(gc.OBSTACLE_NODE_OFFSET + k) + ar * N]          # (B,32)
        radius = ox[:, 3:4]                                     # 비활성 장애물은 radius==0
        active = (radius > 0).float()                           # (B,1) 활성 마스크
        rel = ox[:, 0:3] - rod_pos                              # 상대 pos (정규화 env-local)
        summary = torch.cat([rel, ox[:, 7:8], radius], dim=-1)  # rel(3)+dist(1)+radius(1)
        parts.append(summary * active)                          # 비활성은 0 벡터 (무의미 신호 제거)
    return torch.cat(parts, dim=-1), u                         # (B, 32 + 5*N_OBS), global


def _state_dim(use_full_state: bool, use_lean_obstacle: bool = False) -> int:
    if use_lean_obstacle:
        return gc.NODE_FEATURE_DIM + 5 * gc.N_OBSTACLES + gc.GLOBAL_FEATURE_DIM
    if use_full_state:
        return gc.NODES_PER_ENV * gc.NODE_FEATURE_DIM + gc.GLOBAL_FEATURE_DIM
    return gc.NODE_FEATURE_DIM + gc.GLOBAL_FEATURE_DIM


def _extract_state(batch, use_full_state: bool, use_lean_obstacle: bool = False):
    if use_lean_obstacle:
        return _extract_lean_obstacle(batch)
    if use_full_state:
        return _extract_all_and_global(batch)
    return _extract_rod_and_global(batch)


def _mlp(in_dim, hidden_dim, out_dim, num_hidden_layers=2, activation=nn.ReLU):
    layers = [nn.Linear(in_dim, hidden_dim), activation()]
    for _ in range(num_hidden_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), activation()]
    layers += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*layers)


# ──────────────────────────────────────────────────────────────────────
# Actor (Squashed Gaussian)
# ──────────────────────────────────────────────────────────────────────
class MLPActor(nn.Module):
    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0
    EPS = 1e-6

    def __init__(
        self,
        action_dim: int = 6,
        action_scale=None,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
        init_log_std: float = 0.0,
        use_full_state: bool = False,
        use_lean_obstacle: bool = False,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.use_full_state = use_full_state
        self.use_lean_obstacle = use_lean_obstacle

        in_dim = _state_dim(use_full_state, use_lean_obstacle)
        self.mean_head = _mlp(in_dim, hidden_dim, action_dim, num_hidden_layers)

        # State-independent log std
        self.log_std = nn.Parameter(torch.full((action_dim,), init_log_std))

        # action_scale buffer
        if action_scale is None:
            action_scale = torch.tensor([0.02] * 3 + [0.05] * 3)
        elif isinstance(action_scale, (int, float)):
            action_scale = torch.full((action_dim,), float(action_scale))
        elif isinstance(action_scale, (list, tuple)):
            action_scale = torch.tensor(action_scale, dtype=torch.float32)
        self.register_buffer("action_scale", action_scale.float())

    def forward(self, batch):
        s, u = _extract_state(batch, self.use_full_state, self.use_lean_obstacle)
        h = torch.cat([s, u], dim=-1)
        mean_raw = self.mean_head(h)
        log_std = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean_raw, log_std

    def get_action_and_log_prob(self, batch, deterministic: bool = False):
        mean_raw, log_std = self.forward(batch)
        if deterministic:
            action_norm = torch.tanh(mean_raw)
            return self.action_scale * action_norm, None, None
        std = torch.exp(log_std)
        dist = Normal(mean_raw, std)
        u_sample = dist.rsample()
        action_norm = torch.tanh(u_sample)
        log_prob = dist.log_prob(u_sample)
        log_prob = log_prob - torch.log(1.0 - action_norm.pow(2) + self.EPS)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        action = self.action_scale * action_norm
        return action, log_prob, None

    def evaluate_actions(self, batch, actions):
        mean_raw, log_std = self.forward(batch)
        std = torch.exp(log_std)
        action_norm = (actions / self.action_scale).clamp(-1.0 + self.EPS, 1.0 - self.EPS)
        u_sample = 0.5 * torch.log((1.0 + action_norm) / (1.0 - action_norm))
        dist = Normal(mean_raw, std)
        log_prob = dist.log_prob(u_sample)
        log_prob = log_prob - torch.log(1.0 - action_norm.pow(2) + self.EPS)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy


# ──────────────────────────────────────────────────────────────────────
# Critic Q(s, a)
# ──────────────────────────────────────────────────────────────────────
class MLPQCritic(nn.Module):
    def __init__(self, action_dim: int = 6, hidden_dim: int = 256, num_hidden_layers: int = 2,
                 use_full_state: bool = False, use_lean_obstacle: bool = False):
        super().__init__()
        self.use_full_state = use_full_state
        self.use_lean_obstacle = use_lean_obstacle
        in_dim = _state_dim(use_full_state, use_lean_obstacle) + action_dim
        self.q_head = _mlp(in_dim, hidden_dim, 1, num_hidden_layers)

    def forward(self, batch, action):
        s, u = _extract_state(batch, self.use_full_state, self.use_lean_obstacle)
        h = torch.cat([s, u, action], dim=-1)
        return self.q_head(h).squeeze(-1)


class MLPTwinQ(nn.Module):
    def __init__(self, action_dim: int = 6, hidden_dim: int = 256, num_hidden_layers: int = 2,
                 use_full_state: bool = False, use_lean_obstacle: bool = False):
        super().__init__()
        self.q1 = MLPQCritic(action_dim, hidden_dim, num_hidden_layers, use_full_state, use_lean_obstacle)
        self.q2 = MLPQCritic(action_dim, hidden_dim, num_hidden_layers, use_full_state, use_lean_obstacle)

    def forward(self, batch, action):
        return self.q1(batch, action), self.q2(batch, action)


# ──────────────────────────────────────────────────────────────────────
# SAC Agent — same interface as GNNSACAgent
# ──────────────────────────────────────────────────────────────────────
class MLPSACAgent(nn.Module):
    """Drop-in replacement for GNNSACAgent. Same interface (actor, q, q_target, soft_update_target)."""
    def __init__(
        self,
        action_dim: int = 6,
        num_rounds: int = 2,    # 호환용. MLP에선 무시.
        action_scale=None,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
        use_full_state: bool = False,
        use_lean_obstacle: bool = False,
    ):
        super().__init__()
        self.actor = MLPActor(
            action_dim=action_dim,
            action_scale=action_scale,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            use_full_state=use_full_state,
            use_lean_obstacle=use_lean_obstacle,
        )
        self.q = MLPTwinQ(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            use_full_state=use_full_state,
            use_lean_obstacle=use_lean_obstacle,
        )
        self.q_target = MLPTwinQ(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            use_full_state=use_full_state,
            use_lean_obstacle=use_lean_obstacle,
        )
        self.q_target.load_state_dict(self.q.state_dict())
        for p in self.q_target.parameters():
            p.requires_grad = False

    def soft_update_target(self, tau: float = 0.005):
        for p_target, p in zip(self.q_target.parameters(), self.q.parameters()):
            p_target.data.mul_(1 - tau)
            p_target.data.add_(tau * p.data)
