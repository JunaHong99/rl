"""
GNN Actor/Critic for SAC/PPO (Phase 3.3 current)

새 framework: cooperative impedance controller 위에 올라가는 GNN 정책.

Input:  PyG Batch (from graph_converter.convert_batch_state_to_graph)
Output:
    Actor:  squashed Gaussian policy over 6-dim object-pose delta
    Critic: V(s) scalar (PPO path) or Q(s, a) scalar (SAC path)

아키텍처:
  1. Per-type node embedding (robot/ee/rod/obstacle 4종 → 공통 dim)
  2. K rounds of RoboBalletGNNBlock (default K=2)
  3. Rod 노드만 추출 (객체 레벨 의사결정) + global feature concat
  4. Actor head: raw mean + global log_std parameter
  5. tanh squash + action_scale 적용
  6. Critic/Q head: 모든 노드 mean-pool + global → scalar

Curriculum stage 1 (현재):
  - action_dim = 6 (x_d_obj delta: pos 3 + rot axis-angle 3)
  - K_abs, K_rel은 controller default 고정
  - 향후 18-dim (+K_abs 6 + K_rel 6)로 확장 시 action_dim만 변경
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import gnn_core
import graph_converter as gc


# ──────────────────────────────────────────────────────────────────────
# Node type embedding (heterogeneous → homogeneous)
# ──────────────────────────────────────────────────────────────────────
class NodeTypeEmbedder(nn.Module):
    """
    Input: (N_total, NODE_FEATURE_DIM) — raw padded features + 4-d type one-hot
    Output: (N_total, embed_dim)

    Strategy: 각 type별 별도 linear projection이 이상적이나, 일단 단일 MLP로 통합
    (one-hot tag가 feature에 들어 있어 implicit type 처리됨).
    """
    def __init__(self, in_dim: int = gc.NODE_FEATURE_DIM, embed_dim: int = 128):
        super().__init__()
        self.proj = gnn_core.MLP(in_dim, embed_dim, num_layers=3, output_dim=embed_dim)

    def forward(self, x):
        return self.proj(x)


class EdgeEmbedder(nn.Module):
    def __init__(self, in_dim: int = gc.EDGE_FEATURE_DIM, embed_dim: int = 64):
        super().__init__()
        self.proj = gnn_core.MLP(in_dim, embed_dim, num_layers=2, output_dim=embed_dim)

    def forward(self, e):
        return self.proj(e)


class GlobalEmbedder(nn.Module):
    def __init__(self, in_dim: int = gc.GLOBAL_FEATURE_DIM, embed_dim: int = 128):
        super().__init__()
        self.proj = gnn_core.MLP(in_dim, embed_dim, num_layers=3, output_dim=embed_dim)

    def forward(self, u):
        return self.proj(u)


# ──────────────────────────────────────────────────────────────────────
# Shared GNN Backbone
# ──────────────────────────────────────────────────────────────────────
class GNNBackbone(nn.Module):
    """
    공통 backbone: embed → K rounds of RoboBalletGNNBlock.

    Output: dict { 'x', 'edge_attr', 'u' } — 마지막 라운드 결과.
    """
    def __init__(
        self,
        node_embed_dim: int = 128,
        edge_embed_dim: int = 64,
        global_embed_dim: int = 128,
        num_rounds: int = 2,
    ):
        super().__init__()
        self.node_emb = NodeTypeEmbedder(embed_dim=node_embed_dim)
        self.edge_emb = EdgeEmbedder(embed_dim=edge_embed_dim)
        self.global_emb = GlobalEmbedder(embed_dim=global_embed_dim)

        # 첫 번째 라운드: in node=node_embed_dim, in edge=edge_embed_dim
        # RoboBalletGNNBlock 내부 out_node_dim=512, out_edge_dim=256, out_global_dim=512
        # 둘째 라운드부터는 in node=512, in edge=256, in global=512
        self.rounds = nn.ModuleList()
        cur_node, cur_edge, cur_global = node_embed_dim, edge_embed_dim, global_embed_dim
        for _ in range(num_rounds):
            block = gnn_core.RoboBalletGNNBlock(
                node_dim=cur_node, edge_dim=cur_edge, global_dim=cur_global, hidden_dim=256
            )
            self.rounds.append(block)
            cur_node, cur_edge, cur_global = block.out_node_dim, block.out_edge_dim, block.out_global_dim

        self.out_node_dim = cur_node
        self.out_edge_dim = cur_edge
        self.out_global_dim = cur_global

    def forward(self, batch):
        x = self.node_emb(batch.x)
        edge_attr = self.edge_emb(batch.edge_attr)
        u = self.global_emb(batch.u)
        for block in self.rounds:
            x, edge_attr, u = block(x, batch.edge_index, edge_attr, u, batch.batch)
        return x, edge_attr, u


# ──────────────────────────────────────────────────────────────────────
# Actor — Rod 노드에서 action 출력
# ──────────────────────────────────────────────────────────────────────
class GNNActor(nn.Module):
    """
    Rod embedding + global feature → action via Squashed Gaussian (SAC standard).

    Forward returns mean_raw (unbounded). Sampling:
        u ~ N(mean_raw, std)
        a_norm = tanh(u) ∈ (-1, 1)
        action = action_scale * a_norm
        log π(a) = log N(u) - Σ log(1 - tanh²(u))     ← Jacobian correction

    Squashed Gaussian + auto-alpha 조합이 정통 SAC. log_pi가 정상 범위(~-6)에
    들어와서 target_entropy=-action_dim과 매치되어 auto-alpha가 안정 작동.
    """
    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0
    EPS = 1e-6

    def __init__(
        self,
        backbone: GNNBackbone,
        action_dim: int = 6,
        action_scale=None,            # scalar OR (action_dim,) tensor — per-dim scale 허용
        init_log_std: float = 0.0,    # SAC + squashed: std=1 (tanh 후 분포가 [-1,1] 전체)
    ):
        super().__init__()
        self.backbone = backbone
        self.action_dim = action_dim

        # Per-dim action scale (6-dim 기본: pos 0.001, rot 0.0005)
        if action_scale is None:
            action_scale = torch.tensor([0.001, 0.001, 0.001, 0.0005, 0.0005, 0.0005])
        elif isinstance(action_scale, (int, float)):
            action_scale = torch.full((action_dim,), float(action_scale))
        elif isinstance(action_scale, (list, tuple)):
            action_scale = torch.tensor(action_scale, dtype=torch.float32)
        # register as buffer so .to(device) works
        self.register_buffer("action_scale", action_scale.float())

        # 2026-06-17 재설계: head에 raw rod + raw global skip 추가 (= 작동하는 MLP 입력 그대로).
        # MP embedding이 신호 망쳐도 raw가 학습 보장(strict superset), MP는 형태 맥락(일반화).
        head_in = (gc.NODE_FEATURE_DIM + gc.GLOBAL_FEATURE_DIM
                   + backbone.out_node_dim + backbone.out_global_dim)
        self.action_mean_head = gnn_core.MLP(
            head_in, hidden_dim=256, num_layers=2, output_dim=action_dim
        )

        # State-independent log std (PPO 일반적 설정)
        self.log_std = nn.Parameter(torch.full((action_dim,), init_log_std))

    def _extract_rod_features(self, x, num_envs):
        """
        Batch x (B*nodes_per_env, dim) → rod 노드만 추출 (B, dim).
        Rod is at index ROD_NODE_IDX per env (5-node refactor: idx=2).
        """
        nodes_per_env = gc.NODES_PER_ENV
        rod_idx = gc.ROD_NODE_IDX + torch.arange(num_envs, device=x.device) * nodes_per_env
        return x[rod_idx]

    def forward(self, batch):
        """Return RAW mean (unbounded) and log_std. Squashing happens in sample/evaluate."""
        x, _, u = self.backbone(batch)                         # message-passed
        B = batch.num_graphs
        rod_mp = self._extract_rod_features(x, B)              # (B, out_node_dim) MP
        rod_raw = self._extract_rod_features(batch.x, B)       # (B, NODE_FEATURE_DIM) raw skip
        # h = [raw_rod, raw_global, MP_rod, MP_global]
        h = torch.cat([rod_raw, batch.u, rod_mp, u], dim=-1)
        mean_raw = self.action_mean_head(h)                    # (B, action_dim)
        log_std = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean_raw, log_std

    def get_action_and_log_prob(self, batch, deterministic: bool = False):
        """Squashed Gaussian sample + Jacobian-corrected log prob.

        Returns:
            action: (B, A) — already scaled by action_scale, bounded in (-action_scale, action_scale)
            log_prob: (B, 1) — log π(a|s) with tanh Jacobian
            entropy: None (no closed-form for squashed Gaussian)
        """
        mean_raw, log_std = self.forward(batch)

        if deterministic:
            action_norm = torch.tanh(mean_raw)
            action = self.action_scale * action_norm
            return action, None, None

        std = torch.exp(log_std)
        dist = Normal(mean_raw, std)
        u = dist.rsample()                                     # reparameterized
        action_norm = torch.tanh(u)
        log_prob = dist.log_prob(u)                            # (B, A)
        log_prob = log_prob - torch.log(1.0 - action_norm.pow(2) + self.EPS)
        log_prob = log_prob.sum(dim=-1, keepdim=True)          # (B, 1)
        action = self.action_scale * action_norm
        return action, log_prob, None

    def evaluate_actions(self, batch, actions):
        """Stored action → log prob (PPO path only; SAC samples fresh each update)."""
        mean_raw, log_std = self.forward(batch)
        std = torch.exp(log_std)

        # Recover u = atanh(action / action_scale)
        action_norm = (actions / self.action_scale).clamp(-1.0 + self.EPS, 1.0 - self.EPS)
        u = 0.5 * torch.log((1.0 + action_norm) / (1.0 - action_norm))

        dist = Normal(mean_raw, std)
        log_prob = dist.log_prob(u)
        log_prob = log_prob - torch.log(1.0 - action_norm.pow(2) + self.EPS)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)     # Gaussian entropy (approx)
        return log_prob, entropy


# ──────────────────────────────────────────────────────────────────────
# Critic — Global mean-pool에서 V(s)
# ──────────────────────────────────────────────────────────────────────
class GNNCritic(nn.Module):
    def __init__(self, backbone: GNNBackbone):
        super().__init__()
        self.backbone = backbone
        head_in = backbone.out_node_dim + backbone.out_global_dim
        self.value_head = gnn_core.MLP(head_in, hidden_dim=256, num_layers=2, output_dim=1)

    def forward(self, batch):
        x, _, u = self.backbone(batch)
        # Mean pool over nodes per graph
        from torch_scatter import scatter_mean
        x_pool = scatter_mean(x, batch.batch, dim=0)            # (B, node_dim)
        h = torch.cat([x_pool, u], dim=-1)                      # (B, node_dim + global_dim)
        return self.value_head(h)                               # (B, 1)


# ──────────────────────────────────────────────────────────────────────
# Q-Critic for SAC — Q(s, a) instead of V(s)
# ──────────────────────────────────────────────────────────────────────
class GNNQCritic(nn.Module):
    """
    Q(s, a): graph state + action → scalar.

    Architecture:
      backbone → mean-pool nodes + global → concat with action → MLP → Q
    """
    def __init__(self, backbone: GNNBackbone, action_dim: int = 6):
        super().__init__()
        self.backbone = backbone
        # 재설계: raw rod + raw global skip + MP rod + MP global + action.
        head_in = (gc.NODE_FEATURE_DIM + gc.GLOBAL_FEATURE_DIM
                   + backbone.out_node_dim + backbone.out_global_dim + action_dim)
        self.q_head = gnn_core.MLP(head_in, hidden_dim=256, num_layers=2, output_dim=1)

    def forward(self, batch, action):
        x, _, u = self.backbone(batch)
        B = batch.num_graphs
        rod_idx = gc.ROD_NODE_IDX + torch.arange(B, device=x.device) * gc.NODES_PER_ENV
        rod_mp = x[rod_idx]                             # (B, out_node_dim) MP
        rod_raw = batch.x[rod_idx]                      # (B, NODE_FEATURE_DIM) raw skip
        h = torch.cat([rod_raw, batch.u, rod_mp, u, action], dim=-1)
        return self.q_head(h)


class GNNTwinQ(nn.Module):
    """SAC twin Q networks (Q1, Q2 fully separate)."""
    def __init__(self, action_dim: int = 6, num_rounds: int = 2):
        super().__init__()
        self.q1 = GNNQCritic(GNNBackbone(num_rounds=num_rounds), action_dim=action_dim)
        self.q2 = GNNQCritic(GNNBackbone(num_rounds=num_rounds), action_dim=action_dim)

    def forward(self, batch, action):
        return self.q1(batch, action), self.q2(batch, action)


class GNNSACAgent(nn.Module):
    """
    SAC agent: separate actor + twin Q + twin Q target.

    SAC standard: target Q networks (Polyak averaging) for stable bootstrapping.
    Each Q has its own backbone (no sharing) — initial SAC paper convention.
    """
    def __init__(self, action_dim: int = 6, num_rounds: int = 2, action_scale=None):
        super().__init__()
        actor_backbone = GNNBackbone(num_rounds=num_rounds)
        self.actor = GNNActor(actor_backbone, action_dim=action_dim, action_scale=action_scale)
        self.q = GNNTwinQ(action_dim=action_dim, num_rounds=num_rounds)
        self.q_target = GNNTwinQ(action_dim=action_dim, num_rounds=num_rounds)

        # Initialize target = main
        self.q_target.load_state_dict(self.q.state_dict())
        for p in self.q_target.parameters():
            p.requires_grad = False

    def soft_update_target(self, tau: float = 0.005):
        for p_target, p in zip(self.q_target.parameters(), self.q.parameters()):
            p_target.data.mul_(1 - tau)
            p_target.data.add_(tau * p.data)


# ──────────────────────────────────────────────────────────────────────
# Combined Actor-Critic (shared backbone optional)
# ──────────────────────────────────────────────────────────────────────
class GNNActorCritic(nn.Module):
    """
    Convenience: separate actor and critic (each with own backbone).

    PPO 표준은 separate networks. Compute 절약 위해 shared backbone도 가능하지만
    initial implementation은 분리 (안정성 우선).
    """
    def __init__(self, action_dim: int = 6, num_rounds: int = 2, action_scale=None):
        super().__init__()
        actor_backbone = GNNBackbone(num_rounds=num_rounds)
        critic_backbone = GNNBackbone(num_rounds=num_rounds)
        self.actor = GNNActor(actor_backbone, action_dim=action_dim, action_scale=action_scale)
        self.critic = GNNCritic(critic_backbone)

    def get_action_and_value(self, batch, deterministic: bool = False):
        action, log_prob, entropy = self.actor.get_action_and_log_prob(batch, deterministic)
        value = self.critic(batch)
        return action, log_prob, entropy, value

    def evaluate(self, batch, actions):
        log_prob, entropy = self.actor.evaluate_actions(batch, actions)
        value = self.critic(batch)
        return log_prob, entropy, value
