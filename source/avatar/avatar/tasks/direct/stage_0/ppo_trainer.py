"""
PPO Trainer (Phase 3.3, cleanrl 스타일)

Custom GNN policy (gnn_policy.GNNActorCritic) + PyG batch state 위에서 동작.

핵심 설계:
  - Rollout buffer: graph tensor를 분해 (x, edge_attr, u) 형태로 저장 → 매 update 시 Batch 재조립
  - GAE (Generalized Advantage Estimation)
  - PPO clip update (K epochs × minibatches per rollout)
  - Standard hyperparameters (clip_eps 0.2, gae_lambda 0.95, gamma 0.99 등)

외부 dependency 최소화 — torch + torch_geometric.Batch + scatter만 사용.
"""

from __future__ import annotations
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from dataclasses import dataclass, field

import graph_converter as gc


# ──────────────────────────────────────────────────────────────────────
# Hyperparameter dataclass
# ──────────────────────────────────────────────────────────────────────
@dataclass
class PPOConfig:
    rollout_steps: int = 128             # T (steps per env per rollout)
    num_envs: int = 64                   # B (parallel envs)
    update_epochs: int = 4               # PPO epochs
    minibatch_size: int = 512            # minibatch within an update epoch
    lr: float = 3e-4
    clip_eps: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    vf_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float | None = None       # 0.015 같은 값으로 early stop 가능


# ──────────────────────────────────────────────────────────────────────
# Rollout buffer — graph는 tensor 분해 저장
# ──────────────────────────────────────────────────────────────────────
class RolloutBuffer:
    """
    T steps × B envs 단위로 데이터 저장.

    Graph state는 [x, edge_attr, u]로 분해해 저장.
    edge_index는 env-local 고정 template이므로 매번 저장 불필요 (gc._EDGE_*에서 재생성).

    Tensors shape:
        x:           (T, B, NODES_PER_ENV, NODE_FEATURE_DIM)
        edge_attr:   (T, B, N_EDGES_PER_ENV, EDGE_FEATURE_DIM)
        u:           (T, B, GLOBAL_FEATURE_DIM)
        actions:     (T, B, action_dim)
        log_probs:   (T, B, 1)
        values:      (T, B, 1)
        rewards:     (T, B)
        dones:       (T, B)
    """
    def __init__(self, T: int, B: int, action_dim: int, device: torch.device):
        self.T = T
        self.B = B
        self.action_dim = action_dim
        self.device = device

        # Storage tensors
        N = gc.NODES_PER_ENV
        E = gc.N_EDGES_PER_ENV
        self.x = torch.zeros(T, B, N, gc.NODE_FEATURE_DIM, device=device)
        self.edge_attr = torch.zeros(T, B, E, gc.EDGE_FEATURE_DIM, device=device)
        self.u = torch.zeros(T, B, gc.GLOBAL_FEATURE_DIM, device=device)
        self.actions = torch.zeros(T, B, action_dim, device=device)
        self.log_probs = torch.zeros(T, B, 1, device=device)
        self.values = torch.zeros(T, B, 1, device=device)
        self.rewards = torch.zeros(T, B, device=device)
        self.dones = torch.zeros(T, B, device=device)
        self.ptr = 0

        # Precompute static edge_index per single env (env-local)
        self._src = torch.tensor(gc._EDGE_SRC, device=device, dtype=torch.long)
        self._dst = torch.tensor(gc._EDGE_DST, device=device, dtype=torch.long)

    def add(self, batch: Batch, action, log_prob, value, reward, done):
        """매 step마다 호출. batch는 graph_converter 출력."""
        i = self.ptr
        # batch.x (B*N, F) → unflatten to (B, N, F)
        N, F_node = gc.NODES_PER_ENV, gc.NODE_FEATURE_DIM
        E, F_edge = gc.N_EDGES_PER_ENV, gc.EDGE_FEATURE_DIM
        self.x[i] = batch.x.view(self.B, N, F_node)
        self.edge_attr[i] = batch.edge_attr.view(self.B, E, F_edge)
        self.u[i] = batch.u
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.values[i] = value
        self.rewards[i] = reward
        self.dones[i] = done.float()
        self.ptr += 1

    def reset(self):
        self.ptr = 0

    def assemble_batch(self, idx: torch.Tensor) -> Batch:
        """
        idx: (M,) flat sample indices in [0, T*B).
        Each sample corresponds to a single (t, env) pair → one graph.
        Returns PyG Batch with M graphs.
        """
        M = idx.numel()
        N, E = gc.NODES_PER_ENV, gc.N_EDGES_PER_ENV

        # Flatten T,B → linear
        x_flat = self.x.view(self.T * self.B, N, gc.NODE_FEATURE_DIM)[idx]   # (M, N, F)
        e_flat = self.edge_attr.view(self.T * self.B, E, gc.EDGE_FEATURE_DIM)[idx]
        u_flat = self.u.view(self.T * self.B, gc.GLOBAL_FEATURE_DIM)[idx]    # (M, F_global)

        # Build flat x: (M*N, F)
        x_cat = x_flat.reshape(M * N, gc.NODE_FEATURE_DIM)
        e_cat = e_flat.reshape(M * E, gc.EDGE_FEATURE_DIM)

        # Edge indices with offset per graph
        src_g = self._src.unsqueeze(0).expand(M, -1)   # (M, E)
        dst_g = self._dst.unsqueeze(0).expand(M, -1)
        offsets = (torch.arange(M, device=self.device) * N).unsqueeze(-1)
        edge_index = torch.stack(
            [(src_g + offsets).reshape(-1), (dst_g + offsets).reshape(-1)], dim=0
        )

        # batch index per node
        batch_idx = torch.arange(M, device=self.device).repeat_interleave(N)

        out = Batch(
            x=x_cat, edge_index=edge_index, edge_attr=e_cat,
            u=u_flat, batch=batch_idx,
        )
        out.num_graphs = M
        return out


# ──────────────────────────────────────────────────────────────────────
# GAE
# ──────────────────────────────────────────────────────────────────────
def compute_gae(
    rewards, values, dones, last_value, gamma, gae_lambda
):
    """
    rewards: (T, B)
    values:  (T, B, 1)
    dones:   (T, B)
    last_value: (B, 1) — bootstrap from final state

    Returns: advantages (T, B), returns (T, B)
    """
    T, B = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(B, device=rewards.device)

    values_flat = values.squeeze(-1)              # (T, B)
    last_value_flat = last_value.squeeze(-1)      # (B,)

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = last_value_flat
        else:
            next_value = values_flat[t + 1]
        next_non_terminal = 1.0 - dones[t]

        delta = rewards[t] + gamma * next_value * next_non_terminal - values_flat[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values_flat
    return advantages, returns


# ──────────────────────────────────────────────────────────────────────
# PPO Trainer
# ──────────────────────────────────────────────────────────────────────
class PPOTrainer:
    def __init__(self, policy, cfg: PPOConfig, device: torch.device):
        self.policy = policy
        self.cfg = cfg
        self.device = device
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
        self.buffer = RolloutBuffer(cfg.rollout_steps, cfg.num_envs, action_dim=6, device=device)

    @torch.no_grad()
    def collect_rollout(self, env, current_batch):
        """
        env.step()을 T번 돌리며 rollout 수집.
        current_batch: 현재 시점의 graph batch (이전 step의 next_state).
        Returns: 마지막 graph batch, 마지막 value (bootstrap용).
        """
        self.buffer.reset()
        self.policy.eval()

        # Episode 통계 — env.episode_length_buf를 직접 사용해 rollout 간 누적 처리
        # (이전 버전의 running_length는 rollout 시작점부터만 셌어서 length 작게 보임)
        if not hasattr(self, "_running_reward"):
            self._running_reward = torch.zeros(self.cfg.num_envs, device=self.device)
        episode_rewards = []
        episode_lengths = []

        for t in range(self.cfg.rollout_steps):
            action, log_prob, _, value = self.policy.get_action_and_value(current_batch)

            # env.step 전의 episode_length_buf (이 step이 끝나면 +1 되어 그 값이 실제 episode 길이)
            ep_len_before = env.episode_length_buf.clone()

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated

            self.buffer.add(current_batch, action, log_prob, value, reward, done)

            self._running_reward += reward
            if done.any():
                done_mask = done.bool()
                # episode_length_buf는 reset 시점에 0으로 돌아가므로 step 전 값 + 1이 실제 길이
                actual_lengths = (ep_len_before[done_mask] + 1).float()
                episode_rewards.extend(self._running_reward[done_mask].cpu().tolist())
                episode_lengths.extend(actual_lengths.cpu().tolist())
                self._running_reward[done_mask] = 0.0

            current_batch = env._build_policy_batch()   # env가 직접 PyG Batch 반환 (helper)

        # Bootstrap value at final state
        _, _, _, last_value = self.policy.get_action_and_value(current_batch)

        rollout_stats = {
            "ep_reward_mean": (sum(episode_rewards) / len(episode_rewards)) if episode_rewards else 0.0,
            "ep_length_mean": (sum(episode_lengths) / len(episode_lengths)) if episode_lengths else 0.0,
            "n_episodes": len(episode_rewards),
        }
        return current_batch, last_value, rollout_stats

    def update(self, last_value):
        """PPO update over collected rollout."""
        cfg = self.cfg
        T, B = cfg.rollout_steps, cfg.num_envs

        # GAE
        advantages, returns = compute_gae(
            self.buffer.rewards, self.buffer.values, self.buffer.dones,
            last_value, cfg.gamma, cfg.gae_lambda
        )

        # Flatten T*B
        flat_adv = advantages.reshape(-1)
        flat_ret = returns.reshape(-1)
        flat_actions = self.buffer.actions.reshape(T * B, -1)
        flat_old_log_probs = self.buffer.log_probs.reshape(T * B, 1)
        flat_old_values = self.buffer.values.reshape(T * B, 1)

        # Advantage normalization (per-rollout)
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

        # Shuffle indices
        total_samples = T * B
        all_idx = torch.arange(total_samples, device=self.device)

        stats = {
            "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
            "approx_kl": 0.0, "clip_fraction": 0.0, "n_updates": 0,
        }

        self.policy.train()
        for epoch in range(cfg.update_epochs):
            perm = torch.randperm(total_samples, device=self.device)
            for start in range(0, total_samples, cfg.minibatch_size):
                idx = perm[start:start + cfg.minibatch_size]
                if idx.numel() == 0:
                    continue

                # Assemble PyG Batch for this minibatch
                mb_batch = self.buffer.assemble_batch(idx)
                mb_actions = flat_actions[idx]
                mb_old_log_probs = flat_old_log_probs[idx]
                mb_old_values = flat_old_values[idx]
                mb_adv = flat_adv[idx]
                mb_ret = flat_ret[idx]

                # Evaluate current policy
                log_prob, entropy, value = self.policy.evaluate(mb_batch, mb_actions)

                # PPO clip loss
                ratio = torch.exp(log_prob - mb_old_log_probs)
                surr1 = ratio * mb_adv.unsqueeze(-1)
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_adv.unsqueeze(-1)
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                v_clipped = mb_old_values + torch.clamp(
                    value - mb_old_values, -cfg.clip_eps, cfg.clip_eps
                )
                v_loss_unclipped = (value - mb_ret.unsqueeze(-1)).pow(2)
                v_loss_clipped = (v_clipped - mb_ret.unsqueeze(-1)).pow(2)
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                # Entropy bonus
                ent_loss = -entropy.mean()

                loss = policy_loss + cfg.vf_coef * value_loss + cfg.entropy_coef * ent_loss

                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                # Stats
                with torch.no_grad():
                    approx_kl = (mb_old_log_probs - log_prob).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip_eps).float().mean().item()
                    stats["policy_loss"] += policy_loss.item()
                    stats["value_loss"] += value_loss.item()
                    stats["entropy"] += -ent_loss.item()
                    stats["approx_kl"] += approx_kl
                    stats["clip_fraction"] += clip_frac
                    stats["n_updates"] += 1

            # Optional early stop on KL
            if cfg.target_kl is not None and (stats["approx_kl"] / max(stats["n_updates"], 1)) > cfg.target_kl:
                break

        n = max(stats["n_updates"], 1)
        for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"):
            stats[k] /= n
        return stats
