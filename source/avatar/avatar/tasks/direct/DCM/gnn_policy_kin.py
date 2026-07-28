"""Kinematic-graph GNN SAC (리더-팔로워 morphology 일반화용).

기존 gnn_policy의 backbone/GN block/embedder를 재사용하되:
  - 차원을 graph_converter_kin(17노드) 것으로 주입.
  - ★ Actor 출력 = 리더 joint 노드 7개에 **공유 head**(per-node) → 7D Δq (NerveNet식, DoF 무관).
  - Critic/Q = global mean-pool (노드 수 무관, 기존과 동일).

morphology 전이의 핵심: node-type embedder + edge-type message + per-joint 공유 output.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torch.distributions import Normal

import gnn_core
import graph_converter_kin as gk
from gnn_policy import NodeTypeEmbedder, EdgeEmbedder, GlobalEmbedder


class KinBackbone(nn.Module):
    """GNNBackbone의 kin 차원 버전 (embedder in_dim을 kin으로)."""
    def __init__(self, node_embed_dim=128, edge_embed_dim=64, global_embed_dim=128, num_rounds=8):
        super().__init__()
        self.node_emb = NodeTypeEmbedder(in_dim=gk.NODE_FEATURE_DIM, embed_dim=node_embed_dim)
        self.edge_emb = EdgeEmbedder(in_dim=gk.EDGE_FEATURE_DIM, embed_dim=edge_embed_dim)
        self.global_emb = GlobalEmbedder(in_dim=gk.GLOBAL_FEATURE_DIM, embed_dim=global_embed_dim)
        self.rounds = nn.ModuleList()
        cn, ce, cg = node_embed_dim, edge_embed_dim, global_embed_dim
        for _ in range(num_rounds):
            blk = gnn_core.RoboBalletGNNBlock(node_dim=cn, edge_dim=ce, global_dim=cg, hidden_dim=256)
            self.rounds.append(blk)
            cn, ce, cg = blk.out_node_dim, blk.out_edge_dim, blk.out_global_dim
        self.out_node_dim, self.out_edge_dim, self.out_global_dim = cn, ce, cg

    def forward(self, batch):
        x = self.node_emb(batch.x)
        e = self.edge_emb(batch.edge_attr)
        u = self.global_emb(batch.u)
        for blk in self.rounds:
            x, e, u = blk(x, batch.edge_index, e, u, batch.batch)
        return x, e, u


def _leader_joint_gather_idx(B, device):
    """리더 joint 노드(고정 idx 1~7)의 배치-오프셋 인덱스 → (B, 7)."""
    lead = torch.tensor(gk.LEADER_JOINT_IDX, device=device)          # (7,)
    return lead.unsqueeze(0) + torch.arange(B, device=device).unsqueeze(1) * gk.NODES_PER_ENV


class KinActor(nn.Module):
    """리더 joint 노드 7개 → 공유 head(per-node) → 7D Δq (squashed Gaussian)."""
    LOG_STD_MIN, LOG_STD_MAX, EPS = -5.0, 2.0, 1e-6

    def __init__(self, backbone: KinBackbone, action_scale, init_log_std=0.0):
        super().__init__()
        self.backbone = backbone
        self.action_dim = gk.N_ARM_JOINTS                            # 7
        if isinstance(action_scale, (int, float)):
            action_scale = torch.full((self.action_dim,), float(action_scale))
        elif isinstance(action_scale, (list, tuple)):
            action_scale = torch.tensor(action_scale, dtype=torch.float32)
        self.register_buffer("action_scale", action_scale.float())
        # 공유 per-joint head: [joint 노드 raw + MP embedding + global(goal 포함)] → 스칼라 Δq.
        #   ★goal은 global feature(u)로 이동 → backbone message passing이 모든 노드에 전파(actor/critic 자동).
        #     global을 head에도 skip(전파 보강 + MLP와 공정).
        head_in = gk.NODE_FEATURE_DIM + backbone.out_node_dim + gk.GLOBAL_FEATURE_DIM
        self.joint_head = gnn_core.MLP(head_in, hidden_dim=256, num_layers=2, output_dim=1)
        # per-joint log_std (7개, state-independent)
        self.log_std = nn.Parameter(torch.full((self.action_dim,), init_log_std))

    def forward(self, batch):
        x, _, _ = self.backbone(batch)                               # (B*17, node_dim)
        B = batch.num_graphs
        idx = _leader_joint_gather_idx(B, x.device)                  # (B,7)
        mp = x[idx]                                                  # (B,7,node_dim) MP
        raw = batch.x[idx]                                           # (B,7,NODE_FEATURE_DIM) joint raw skip
        gb = batch.u.unsqueeze(1).expand(B, 7, -1)                  # (B,7,global) — goal 포함
        h = torch.cat([raw, mp, gb], dim=-1)
        mean_raw = self.joint_head(h).squeeze(-1)                    # (B,7)
        log_std = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean_raw, log_std

    def get_action_and_log_prob(self, batch, deterministic=False):
        mean_raw, log_std = self.forward(batch)
        if deterministic:
            return self.action_scale * torch.tanh(mean_raw), None, None
        std = torch.exp(log_std)
        dist = Normal(mean_raw, std)
        u = dist.rsample()
        a_norm = torch.tanh(u)
        log_prob = dist.log_prob(u) - torch.log(1.0 - a_norm.pow(2) + self.EPS)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return self.action_scale * a_norm, log_prob, None


class KinQCritic(nn.Module):
    """Q(s,a): backbone → global mean-pool + global(goal 포함) + action → scalar (노드 수 무관).
    goal은 global(u)에 있어 backbone 통과 → mean-pool + u로 반영."""
    def __init__(self, backbone: KinBackbone, action_dim=7):
        super().__init__()
        self.backbone = backbone
        head_in = backbone.out_node_dim + backbone.out_global_dim + action_dim
        self.q_head = gnn_core.MLP(head_in, hidden_dim=256, num_layers=2, output_dim=1)

    def forward(self, batch, action):
        from torch_scatter import scatter_mean
        x, _, u = self.backbone(batch)
        x_pool = scatter_mean(x, batch.batch, dim=0)                 # (B, node_dim)
        return self.q_head(torch.cat([x_pool, u, action], dim=-1))


class KinTwinQ(nn.Module):
    def __init__(self, num_rounds=8):
        super().__init__()
        self.q1 = KinQCritic(KinBackbone(num_rounds=num_rounds))
        self.q2 = KinQCritic(KinBackbone(num_rounds=num_rounds))

    def forward(self, batch, action):
        return self.q1(batch, action), self.q2(batch, action)


class KinSACAgent(nn.Module):
    """SAC: KinActor + twin Q + target Q. action_dim=7(리더 관절)."""
    def __init__(self, num_rounds=8, action_scale=None):
        super().__init__()
        self.actor = KinActor(KinBackbone(num_rounds=num_rounds), action_scale=action_scale)
        self.q = KinTwinQ(num_rounds=num_rounds)
        self.q_target = KinTwinQ(num_rounds=num_rounds)
        self.q_target.load_state_dict(self.q.state_dict())
        for p in self.q_target.parameters():
            p.requires_grad = False

    def soft_update_target(self, tau: float = 0.005):
        for p_target, p in zip(self.q_target.parameters(), self.q.parameters()):
            p_target.data.mul_(1 - tau)
            p_target.data.add_(tau * p.data)


# ──────────────────────────────────────────────────────────────────────
# KinMLP — 공정 ablation: kin 그래프와 *같은 정보*(전 노드+전 엣지+global)를 flat하게 받되
#   message passing/그래프 구조 없음(순수 MLP). → "그래프 구조가 파지 일반화에 기여하나" 판별.
#   입력 = 17노드×10 + 34엣지×13 + global 11 = 623. 출력 = 리더 Δq 7 (KinActor와 동일).
#   ★goal은 global(11)에 포함 → GNN(u)·MLP(flat u) 둘 다 동일하게 goal 직접 접근(공정).
# ──────────────────────────────────────────────────────────────────────
def _flatten_kin_batch(batch):
    """kin Batch → (B, 623) flat 벡터 (전 노드 feat + 전 엣지 feat + global)."""
    B = batch.num_graphs
    x = batch.x.view(B, gk.NODES_PER_ENV * gk.NODE_FEATURE_DIM)          # (B, 170)
    e = batch.edge_attr.view(B, gk.N_EDGES_PER_ENV * gk.EDGE_FEATURE_DIM)  # (B, 442)
    return torch.cat([x, e, batch.u], dim=-1)                            # (B, 623)


_KIN_FLAT_DIM = (gk.NODES_PER_ENV * gk.NODE_FEATURE_DIM
                 + gk.N_EDGES_PER_ENV * gk.EDGE_FEATURE_DIM
                 + gk.GLOBAL_FEATURE_DIM)


class KinMLPActor(nn.Module):
    """flatten(kin batch) → MLP → 7D Δq. GNN과 정보 동일, 구조(message passing)만 없음."""
    LOG_STD_MIN, LOG_STD_MAX, EPS = -5.0, 2.0, 1e-6

    def __init__(self, action_scale, init_log_std=0.0, hidden_dim=512, num_layers=3):
        super().__init__()
        self.action_dim = gk.N_ARM_JOINTS
        if isinstance(action_scale, (int, float)):
            action_scale = torch.full((self.action_dim,), float(action_scale))
        elif isinstance(action_scale, (list, tuple)):
            action_scale = torch.tensor(action_scale, dtype=torch.float32)
        self.register_buffer("action_scale", action_scale.float())
        self.mean_head = gnn_core.MLP(_KIN_FLAT_DIM, hidden_dim=hidden_dim, num_layers=num_layers,
                                      output_dim=self.action_dim)
        self.log_std = nn.Parameter(torch.full((self.action_dim,), init_log_std))

    def forward(self, batch):
        mean_raw = self.mean_head(_flatten_kin_batch(batch))
        return mean_raw, self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)

    def get_action_and_log_prob(self, batch, deterministic=False):
        mean_raw, log_std = self.forward(batch)
        if deterministic:
            return self.action_scale * torch.tanh(mean_raw), None, None
        std = torch.exp(log_std)
        dist = Normal(mean_raw, std)
        u = dist.rsample()
        a_norm = torch.tanh(u)
        log_prob = dist.log_prob(u) - torch.log(1.0 - a_norm.pow(2) + self.EPS)
        return self.action_scale * a_norm, log_prob.sum(dim=-1, keepdim=True), None


class KinMLPQCritic(nn.Module):
    def __init__(self, action_dim=7, hidden_dim=512, num_layers=3):
        super().__init__()
        self.q_head = gnn_core.MLP(_KIN_FLAT_DIM + action_dim, hidden_dim=hidden_dim,
                                   num_layers=num_layers, output_dim=1)

    def forward(self, batch, action):
        return self.q_head(torch.cat([_flatten_kin_batch(batch), action], dim=-1))


class KinMLPTwinQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.q1 = KinMLPQCritic(); self.q2 = KinMLPQCritic()

    def forward(self, batch, action):
        return self.q1(batch, action), self.q2(batch, action)


class KinMLPSACAgent(nn.Module):
    """SAC ablation: KinMLP(flatten) actor + twin Q. KinSACAgent와 인터페이스 동일."""
    def __init__(self, action_scale=None, **kw):
        super().__init__()
        self.actor = KinMLPActor(action_scale=action_scale)
        self.q = KinMLPTwinQ()
        self.q_target = KinMLPTwinQ()
        self.q_target.load_state_dict(self.q.state_dict())
        for p in self.q_target.parameters():
            p.requires_grad = False

    def soft_update_target(self, tau: float = 0.005):
        for p_target, p in zip(self.q_target.parameters(), self.q.parameters()):
            p_target.data.mul_(1 - tau)
            p_target.data.add_(tau * p.data)
