"""
SAC Trainer (Phase 3.3-SAC, off-policy)

Soft Actor-Critic with GNN policy/Q-critic.

핵심:
  - Off-policy: ReplayBuffer에 transition 누적, 매 env step마다 batch 샘플링해 gradient update
  - Max-entropy: log_alpha 자동 튜닝 (target_entropy = -action_dim)
  - Twin Q + soft target update (Polyak τ)
  - Continuous action: stochastic policy (재param trick으로 gradient backprop)

PPO 대비 장점:
  - Sample efficient (replay buffer로 데이터 재사용)
  - Entropy bonus가 exploration 자동화 (우리 frozen policy 문제 해결 기대)

핵심 update step:
  Q loss:
    a' ~ π(s')
    y = r + γ(1-d)[min(Q1_t(s',a'), Q2_t(s',a')) − α·log π(a'|s')]
    Q_loss = MSE(Q1(s,a), y) + MSE(Q2(s,a), y)

  Actor loss:
    a_new ~ π(s) (reparam)
    actor_loss = E[α·log π(a_new|s) − min(Q1(s,a_new), Q2(s,a_new))]

  Alpha loss (entropy temperature):
    α_loss = −E[log α · (log π(a_new|s) + target_entropy)]

  Soft target: θ_target ← τ·θ + (1−τ)·θ_target
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch.distributions import Normal
from dataclasses import dataclass

import graph_converter as gc


# ──────────────────────────────────────────────────────────────────────
# SAC Hyperparameters
# ──────────────────────────────────────────────────────────────────────
@dataclass
class SACConfig:
    buffer_size: int = 1_000_000          # 1M (5-node 모델 작아져 메모리 여유)
    batch_size: int = 512                 # 256 → 512 (gradient 안정)
    gamma: float = 0.99
    tau: float = 0.005                    # Polyak target update
    lr_actor: float = 3e-4
    lr_q: float = 3e-4
    lr_alpha: float = 3e-4
    init_log_alpha: float = -1.0          # exp(-1) ≈ 0.37 initial entropy temp
    target_entropy: float = -6.0          # -action_dim (heuristic)
    auto_alpha: bool = True               # ★ Option A: Squashed Gaussian과 함께 정상 작동
    fixed_alpha: float = 0.05             # auto_alpha=False일 때 사용할 고정값
    alpha_min: float = 0.01               # ★ Phase 3.3: α collapse 방지 (4e-33 → 0.01 floor)
    warmup_steps: int = 25_000            # random actions before training (5k → 25k: buffer 다양성 ↑)
    updates_per_step: int = 4             # gradient updates per env step (1 → 4: vectorized sample efficiency)
    max_grad_norm: float = 1.0


# ──────────────────────────────────────────────────────────────────────
# ReplayBuffer — graph tensor 분해 저장
# ──────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    """
    Off-policy replay buffer.

    환경의 매 env.step에서 num_envs개 transition을 저장.
    Capacity = max transitions (FIFO when full).

    Each transition:
      state (x, edge_attr, u), action, reward, next_state, done

    Edge index is env-local static template라 sample 시 batch offset만 다시 붙인다.

    Memory consideration:
      current 5-node graph:
        x(5×32) + edge(10×4) + u(4) + action(6) + reward + done
        ≈ 160 + 40 + 4 + 6 + 2 ≈ 212 floats × 2 (s, s') = 424 floats
        ≈ 1.7KB / transition (float32 기준)
      1M buffer도 GPU에서 현실적인 범위다.
    """
    def __init__(self, capacity: int, num_envs: int, action_dim: int, device: torch.device):
        self.capacity = capacity
        self.num_envs = num_envs
        self.action_dim = action_dim
        self.device = device

        N, F_node = gc.NODES_PER_ENV, gc.NODE_FEATURE_DIM
        E, F_edge = gc.N_EDGES_PER_ENV, gc.EDGE_FEATURE_DIM
        F_global = gc.GLOBAL_FEATURE_DIM

        # State
        self.x = torch.zeros(capacity, N, F_node, device=device)
        self.edge_attr = torch.zeros(capacity, E, F_edge, device=device)
        self.u = torch.zeros(capacity, F_global, device=device)
        # Next state
        self.next_x = torch.zeros(capacity, N, F_node, device=device)
        self.next_edge_attr = torch.zeros(capacity, E, F_edge, device=device)
        self.next_u = torch.zeros(capacity, F_global, device=device)
        # Misc
        self.actions = torch.zeros(capacity, action_dim, device=device)
        self.rewards = torch.zeros(capacity, device=device)
        self.dones = torch.zeros(capacity, device=device)

        # Static edge indices (env-local)
        self._src = torch.tensor(gc._EDGE_SRC, device=device, dtype=torch.long)
        self._dst = torch.tensor(gc._EDGE_DST, device=device, dtype=torch.long)

        self.ptr = 0
        self.size = 0

    def add_batch(self, batch_state: Batch, action, reward, next_batch_state: Batch, done, valid_mask=None):
        """매 env.step마다 num_envs개 transition 추가.
        valid_mask: (B,) bool. 주어지면 True인 env만 저장 (settle transition 제외용)."""
        N, E = gc.NODES_PER_ENV, gc.N_EDGES_PER_ENV
        B = self.num_envs

        # Unflatten current state
        x_cur = batch_state.x.view(B, N, -1)
        e_cur = batch_state.edge_attr.view(B, E, -1)
        u_cur = batch_state.u

        x_next = next_batch_state.x.view(B, N, -1)
        e_next = next_batch_state.edge_attr.view(B, E, -1)
        u_next = next_batch_state.u

        if valid_mask is not None:
            sel = valid_mask.nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                return
            x_cur, e_cur, u_cur = x_cur[sel], e_cur[sel], u_cur[sel]
            x_next, e_next, u_next = x_next[sel], e_next[sel], u_next[sel]
            action, reward, done = action[sel], reward[sel], done[sel]

        M = x_cur.shape[0]
        # Circular insert: M transitions starting from self.ptr
        idxs = (self.ptr + torch.arange(M, device=self.device)) % self.capacity
        self.x[idxs] = x_cur
        self.edge_attr[idxs] = e_cur
        self.u[idxs] = u_cur
        self.next_x[idxs] = x_next
        self.next_edge_attr[idxs] = e_next
        self.next_u[idxs] = u_next
        self.actions[idxs] = action
        self.rewards[idxs] = reward
        self.dones[idxs] = done.float()

        self.ptr = (self.ptr + M) % self.capacity
        self.size = min(self.size + M, self.capacity)

    def sample(self, batch_size: int):
        """Sample minibatch and assemble PyG Batches for state, next_state."""
        idxs = torch.randint(0, self.size, (batch_size,), device=self.device)
        N, E = gc.NODES_PER_ENV, gc.N_EDGES_PER_ENV
        M = batch_size

        # Current batch
        x_flat = self.x[idxs].reshape(M * N, -1)
        e_flat = self.edge_attr[idxs].reshape(M * E, -1)
        u_flat = self.u[idxs]

        src_g = self._src.unsqueeze(0).expand(M, -1)
        dst_g = self._dst.unsqueeze(0).expand(M, -1)
        offsets = (torch.arange(M, device=self.device) * N).unsqueeze(-1)
        edge_index = torch.stack(
            [(src_g + offsets).reshape(-1), (dst_g + offsets).reshape(-1)], dim=0
        )
        batch_idx = torch.arange(M, device=self.device).repeat_interleave(N)

        cur_batch = Batch(x=x_flat, edge_index=edge_index, edge_attr=e_flat, u=u_flat, batch=batch_idx)
        cur_batch.num_graphs = M

        # Next batch (same edge structure)
        next_x_flat = self.next_x[idxs].reshape(M * N, -1)
        next_e_flat = self.next_edge_attr[idxs].reshape(M * E, -1)
        next_u_flat = self.next_u[idxs]

        next_batch = Batch(x=next_x_flat, edge_index=edge_index, edge_attr=next_e_flat,
                           u=next_u_flat, batch=batch_idx)
        next_batch.num_graphs = M

        return cur_batch, self.actions[idxs], self.rewards[idxs], next_batch, self.dones[idxs]


# ──────────────────────────────────────────────────────────────────────
# SAC Trainer
# ──────────────────────────────────────────────────────────────────────
class SACTrainer:
    def __init__(self, agent, cfg: SACConfig, device: torch.device):
        """
        agent: gnn_policy.GNNSACAgent  (actor + q + q_target)
        """
        self.agent = agent
        self.cfg = cfg
        self.device = device

        # Optimizers
        self.actor_opt = torch.optim.Adam(agent.actor.parameters(), lr=cfg.lr_actor)
        self.q_opt = torch.optim.Adam(agent.q.parameters(), lr=cfg.lr_q)

        # Entropy temperature
        if cfg.auto_alpha:
            self.log_alpha = torch.tensor(cfg.init_log_alpha, device=device, requires_grad=True)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.lr_alpha)
        else:
            # Fixed alpha (Option A: auto-tune 비활성, 폭주 방지)
            import math
            self.log_alpha = torch.tensor(math.log(cfg.fixed_alpha), device=device)
            self.alpha_opt = None

        # Replay buffer
        self.buffer = ReplayBuffer(
            capacity=cfg.buffer_size,
            num_envs=None,  # set when first added
            action_dim=6,
            device=device,
        )

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def update(self, batch_size: int):
        """One gradient update step."""
        cfg = self.cfg
        if self.buffer.size < batch_size:
            return None

        s, a, r, s_next, d = self.buffer.sample(batch_size)

        # ── Target Q ──
        with torch.no_grad():
            # next action sampled from current policy
            a_next, log_pi_next, _ = self.agent.actor.get_action_and_log_prob(s_next, deterministic=False)
            q1_t, q2_t = self.agent.q_target(s_next, a_next)
            q_min_t = torch.min(q1_t, q2_t).squeeze(-1)              # (B,)
            log_pi_next_flat = log_pi_next.squeeze(-1)
            y = r + cfg.gamma * (1.0 - d) * (q_min_t - self.alpha.detach() * log_pi_next_flat)

        # ── Q loss ──
        q1_cur, q2_cur = self.agent.q(s, a)
        q_loss = F.mse_loss(q1_cur.squeeze(-1), y) + F.mse_loss(q2_cur.squeeze(-1), y)

        self.q_opt.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.agent.q.parameters(), cfg.max_grad_norm)
        self.q_opt.step()

        # ── Actor loss ──
        a_new, log_pi_new, _ = self.agent.actor.get_action_and_log_prob(s, deterministic=False)
        q1_new, q2_new = self.agent.q(s, a_new)
        q_min_new = torch.min(q1_new, q2_new).squeeze(-1)
        log_pi_new_flat = log_pi_new.squeeze(-1)
        actor_loss = (self.alpha.detach() * log_pi_new_flat - q_min_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.agent.actor.parameters(), cfg.max_grad_norm)
        self.actor_opt.step()

        # ── Alpha loss (entropy temp 자동 튜닝) — auto_alpha 켰을 때만 ──
        if cfg.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_pi_new_flat.detach() + cfg.target_entropy)).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            # ★ Phase 3.3: α collapse 방지 — log_alpha ≥ log(alpha_min)
            import math as _math
            self.log_alpha.data.clamp_(min=_math.log(cfg.alpha_min))
            alpha_loss_val = alpha_loss.item()
        else:
            alpha_loss_val = 0.0  # 고정 alpha

        # ── Soft update target ──
        self.agent.soft_update_target(cfg.tau)

        return {
            "q_loss": q_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss_val,
            "alpha": self.alpha.item(),
            "q1_mean": q1_cur.mean().item(),
            "log_pi_mean": log_pi_new_flat.mean().item(),
        }
