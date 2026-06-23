"""
HER (Hindsight Experience Replay) Buffer — Andrychowicz 2017.

핵심:
  - 매 env step에 episode buffer에 transition 임시 저장
  - Episode 종료 시:
    1) 원본 transitions를 main buffer에 추가
    2) k_future개의 virtual transitions 생성:
       - episode 내 random future timestep을 virtual goal로 relabel
       - state의 rod node feature에서 goal_pos, goal_quat, pos_err, rot_err 교체
       - virtual reward + done 재계산
    3) Virtual transitions도 main buffer에 추가
  - Sample 시 logic은 원본과 동일 (모든 transition이 (state, action, reward, next_state, done))

Memory:
  - 1M buffer × (1 original + k_future virtual) = effective buffer 1 + k 배 큼
  - 메모리는 그대로 1M slots (FIFO 순환)
"""
from __future__ import annotations
import math
import torch
from torch_geometric.data import Batch

import graph_converter as gc


# 정규화 상수 (graph_converter.py와 일치)
POS_NORM = gc.POS_NORM
ROT_ERR_NORM = gc.ROT_ERR_NORM
FEAT_CLIP = gc.FEAT_CLIP

# Rod node feature layout (graph_converter._rod_features 순서 따라)
# 0-2: rod_pos / POS_NORM
# 3-6: rod_quat
# 7-9: rod_lin_vel / VEL_LIN_NORM
# 10-12: rod_ang_vel / VEL_ANG_NORM
# 13-15: goal_pos / POS_NORM        ← HER에서 교체
# 16-19: goal_quat                   ← HER에서 교체
# 20-22: pos_err / POS_NORM          ← HER에서 재계산
# 23-25: rot_err_aa / ROT_ERR_NORM   ← HER에서 재계산
ROD_NODE_IDX = gc.ROD_NODE_IDX     # 5-node graph에서 rod 위치 (= 2)


def _quat_conj(q):
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


def _quat_to_axis_angle(q):
    w = q[..., 0:1]
    sign = torch.sign(w)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    q_signed = q * sign
    v = q_signed[..., 1:4]
    w_pos = q_signed[..., 0].clamp(min=-1.0, max=1.0)
    v_norm = torch.norm(v, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(v_norm.squeeze(-1), w_pos)
    axis = v / (v_norm + 1e-8)
    return axis * angle.unsqueeze(-1)


def replace_goal_in_node_features(x_per_env, rod_pos, rod_quat, new_goal_pos, new_goal_quat):
    """
    Graph의 rod node feature에서 goal 관련 부분을 새 goal로 교체.

    Args:
        x_per_env: (B, N, F) — full graph node features per env
        rod_pos: (B, 3) — current rod position (raw, env-local)
        rod_quat: (B, 4) — current rod quaternion (raw)
        new_goal_pos: (B, 3) — new (virtual) goal position
        new_goal_quat: (B, 4) — new goal quaternion

    Returns:
        x_modified: (B, N, F) — modified node features with new goal
    """
    x_modified = x_per_env.clone()
    rod_x = x_modified[:, ROD_NODE_IDX, :]   # (B, F=32)

    # 13-15: goal_pos (normalized)
    rod_x[:, 13:16] = new_goal_pos / POS_NORM
    # 16-19: goal_quat (raw [-1, 1])
    rod_x[:, 16:20] = new_goal_quat

    # 20-22: pos_err = goal_pos - rod_pos (normalized)
    pos_err = (new_goal_pos - rod_pos) / POS_NORM
    rod_x[:, 20:23] = pos_err

    # 23-25: rot_err_aa (normalized by π)
    q_err = _quat_mul(new_goal_quat, _quat_conj(rod_quat))
    rot_err_aa = _quat_to_axis_angle(q_err)
    rod_x[:, 23:26] = rot_err_aa / ROT_ERR_NORM

    # Clip
    rod_x[:, :26] = rod_x[:, :26].clamp(-FEAT_CLIP, FEAT_CLIP)
    x_modified[:, ROD_NODE_IDX, :] = rod_x

    return x_modified


def recompute_reward(rod_pos, rod_quat, new_goal_pos, new_goal_quat,
                     prev_dist, is_first,
                     pos_thresh=0.02, rot_thresh=0.1745,
                     progress_weight=0.0, success_bonus=100.0):
    """
    Virtual goal 기준 reward + reached 재계산.

    Returns:
        reward: (B,)
        is_reached: (B,) bool
        new_dist: (B,) — dist for next step's progress calc
    """
    pos_err_vec = new_goal_pos - rod_pos
    pos_err = torch.norm(pos_err_vec, dim=-1)
    q_err = _quat_mul(new_goal_quat, _quat_conj(rod_quat))
    rot_err_aa = _quat_to_axis_angle(q_err)
    rot_err = torch.norm(rot_err_aa, dim=-1)

    current_dist = pos_err + 0.1 * rot_err
    # Progress
    is_first_safe = is_first | torch.isinf(prev_dist)
    safe_prev = torch.where(is_first_safe, current_dist, prev_dist)
    r_progress = (safe_prev - current_dist) * progress_weight
    r_progress = torch.where(is_first_safe, torch.zeros_like(r_progress), r_progress)

    # Reached check
    is_reached = (pos_err < pos_thresh) & (rot_err < rot_thresh) & ~is_first_safe

    reward = r_progress + torch.where(is_reached, success_bonus, torch.zeros_like(r_progress))

    return reward, is_reached, current_dist


class HERReplayBuffer:
    """
    Episode-tracked HER replay buffer.

    HER strategy:
      - 'future' (기본 HER 논문): vg = rod 자신의 trajectory 미래 위치
        → 정지 정책일 경우 vg ≈ rod[t] trivial reach exploit 발생.
      - 'random_task' (권장, 우리 setup): vg = task goal distribution에서 random sampling
        → vg가 rod trajectory와 독립적이라 exploit 없음.
        → CachedPoseSampler의 (start, goal) offset 풀에서 sampling.

    Main storage: 동일 구조의 (state, action, reward, next_state, done) buffer (graph tensor 분해).
    Per-env episode buffer: 매 step transition + raw state (rod_pos, rod_quat, goal) 임시 저장.
    Done 시: 원본 + k_future개 virtual transitions를 main buffer에 push.

    Sample API는 ReplayBuffer와 동일 (interface 호환).
    """
    def __init__(self, capacity: int, num_envs: int, action_dim: int,
                 device: torch.device, k_future: int = 4, max_episode_len: int = 130,
                 strategy: str = "random_task",
                 goal_offset_pos_pool: torch.Tensor | None = None,
                 goal_offset_quat_pool: torch.Tensor | None = None):
        self.capacity = capacity
        self.num_envs = num_envs
        self.action_dim = action_dim
        self.device = device
        self.k_future = k_future
        self.max_episode_len = max_episode_len
        assert strategy in ("future", "random_task"), f"unknown HER strategy: {strategy}"
        self.strategy = strategy
        # (start→goal) offset pool — random_task strategy 전용
        self.goal_offset_pos_pool = goal_offset_pos_pool.to(device) if goal_offset_pos_pool is not None else None
        self.goal_offset_quat_pool = goal_offset_quat_pool.to(device) if goal_offset_quat_pool is not None else None
        if strategy == "random_task":
            assert self.goal_offset_pos_pool is not None and self.goal_offset_quat_pool is not None, \
                "random_task strategy requires goal_offset_*_pool"

        N, F_node = gc.NODES_PER_ENV, gc.NODE_FEATURE_DIM
        E, F_edge = gc.N_EDGES_PER_ENV, gc.EDGE_FEATURE_DIM
        F_global = gc.GLOBAL_FEATURE_DIM
        self.N, self.F_node = N, F_node
        self.E, self.F_edge = E, F_edge

        # Main buffer — same as ReplayBuffer
        self.x = torch.zeros(capacity, N, F_node, device=device)
        self.edge_attr = torch.zeros(capacity, E, F_edge, device=device)
        self.u = torch.zeros(capacity, F_global, device=device)
        self.next_x = torch.zeros(capacity, N, F_node, device=device)
        self.next_edge_attr = torch.zeros(capacity, E, F_edge, device=device)
        self.next_u = torch.zeros(capacity, F_global, device=device)
        self.actions = torch.zeros(capacity, action_dim, device=device)
        self.rewards = torch.zeros(capacity, device=device)
        self.dones = torch.zeros(capacity, device=device)

        # Static edge indices
        self._src = torch.tensor(gc._EDGE_SRC, device=device, dtype=torch.long)
        self._dst = torch.tensor(gc._EDGE_DST, device=device, dtype=torch.long)

        self.ptr = 0
        self.size = 0

        # Per-env episode buffers — tensor-based to allow vectorized processing
        # Each env: list of (x, edge_attr, u, action, next_x, next_edge_attr, next_u,
        #                    rod_pos, rod_quat, next_rod_pos, next_rod_quat,
        #                    goal_pos, goal_quat, is_first)
        # Capacity per env: max_episode_len
        self._init_episode_buffers()

    def _init_episode_buffers(self):
        N, F_node = self.N, self.F_node
        E, F_edge = self.E, self.F_edge
        L = self.max_episode_len
        B = self.num_envs
        d = self.device

        self.ep_x = torch.zeros(L, B, N, F_node, device=d)
        self.ep_edge = torch.zeros(L, B, E, F_edge, device=d)
        self.ep_u = torch.zeros(L, B, gc.GLOBAL_FEATURE_DIM, device=d)
        self.ep_action = torch.zeros(L, B, self.action_dim, device=d)
        self.ep_next_x = torch.zeros(L, B, N, F_node, device=d)
        self.ep_next_edge = torch.zeros(L, B, E, F_edge, device=d)
        self.ep_next_u = torch.zeros(L, B, gc.GLOBAL_FEATURE_DIM, device=d)
        self.ep_rod_pos = torch.zeros(L, B, 3, device=d)
        self.ep_rod_quat = torch.zeros(L, B, 4, device=d)
        self.ep_next_rod_pos = torch.zeros(L, B, 3, device=d)
        self.ep_next_rod_quat = torch.zeros(L, B, 4, device=d)
        self.ep_goal_pos = torch.zeros(L, B, 3, device=d)
        self.ep_goal_quat = torch.zeros(L, B, 4, device=d)
        # goal-무관 보상 (time+smooth+clearance+collision). HER relabel해도 보존돼야 함
        # (RoboBallet: R_collision은 goal 무관 → virtual goal에도 그대로 더함). 2026-06-18.
        self.ep_goal_indep = torch.zeros(L, B, device=d)

        # Per-env episode step pointer
        self.ep_step = torch.zeros(B, dtype=torch.long, device=d)

    def flush_pending_episodes(self):
        """현재 ep_buffer에 쌓인 미완료 episode들을 truncated transition으로 main buffer에 push.
        주로 warmup 종료 시점에 호출 — 그렇지 않으면 main buffer가 거의 비어 있어 SAC 첫 update가 불안정.
        Off-policy SAC에서는 trajectory continuity가 필요 없으므로 truncate해도 안전.
        """
        active = (self.ep_step > 0).nonzero(as_tuple=False).flatten()
        if active.numel() == 0:
            return 0
        before = self.size
        self._process_done_envs_vectorized(active)
        self.ep_step[active] = 0
        return self.size - before

    def add_step(self, batch_state: Batch, action, reward, next_batch_state: Batch, done,
                 rod_pos, rod_quat, next_rod_pos, next_rod_quat,
                 goal_pos, goal_quat, valid_mask=None, goal_indep_reward=None):
        """매 env.step마다 호출. Episode buffer에 임시 저장 + done env들 HER 처리.

        Args:
            valid_mask: (B,) bool. False인 env는 buffer 쓰기 / ep_step 증가 skip.
                env3 settle 중인 envs를 학습 데이터에서 제외 용도. None이면 모두 valid.
        """
        N, E = self.N, self.E
        B = self.num_envs

        if valid_mask is None:
            valid_idx = torch.arange(B, device=self.device)
            n_valid = B
        else:
            valid_idx = valid_mask.nonzero(as_tuple=True)[0]
            n_valid = int(valid_idx.numel())
            if n_valid == 0:
                # 모든 env가 settle 중 — 아무것도 push 안 함, ep_step도 그대로
                return

        x_cur = batch_state.x.view(B, N, -1)
        e_cur = batch_state.edge_attr.view(B, E, -1)
        u_cur = batch_state.u
        x_next = next_batch_state.x.view(B, N, -1)
        e_next = next_batch_state.edge_attr.view(B, E, -1)
        u_next = next_batch_state.u

        # Write to episode buffer (valid envs only)
        step_idx = self.ep_step.clamp(max=self.max_episode_len - 1)
        s_v = step_idx[valid_idx]
        self.ep_x[s_v, valid_idx] = x_cur[valid_idx]
        self.ep_edge[s_v, valid_idx] = e_cur[valid_idx]
        self.ep_u[s_v, valid_idx] = u_cur[valid_idx]
        self.ep_action[s_v, valid_idx] = action[valid_idx]
        self.ep_next_x[s_v, valid_idx] = x_next[valid_idx]
        self.ep_next_edge[s_v, valid_idx] = e_next[valid_idx]
        self.ep_next_u[s_v, valid_idx] = u_next[valid_idx]
        self.ep_rod_pos[s_v, valid_idx] = rod_pos[valid_idx]
        self.ep_rod_quat[s_v, valid_idx] = rod_quat[valid_idx]
        self.ep_next_rod_pos[s_v, valid_idx] = next_rod_pos[valid_idx]
        self.ep_next_rod_quat[s_v, valid_idx] = next_rod_quat[valid_idx]
        self.ep_goal_pos[s_v, valid_idx] = goal_pos[valid_idx]
        self.ep_goal_quat[s_v, valid_idx] = goal_quat[valid_idx]
        if goal_indep_reward is not None:
            self.ep_goal_indep[s_v, valid_idx] = goal_indep_reward[valid_idx]

        # Increment ep_step only for valid envs
        self.ep_step[valid_idx] += 1

        # Process done envs (only valid done envs — settle envs can't be done since
        # episode_length_buf is frozen during settle)
        if valid_mask is None:
            done_mask = done.bool()
        else:
            done_mask = done.bool() & valid_mask
        if done_mask.any():
            done_envs = torch.nonzero(done_mask, as_tuple=False).flatten()
            self._process_done_envs_vectorized(done_envs)
            self.ep_step[done_envs] = 0

    def _process_done_envs_vectorized(self, done_envs: torch.Tensor):
        """모든 done envs의 episodes를 batch 단위로 main buffer에 push (Python loop 제거).

        Layout 가정:
          - 평탄화된 (env, t) pair는 env-major (env_0의 t=0..T_0-1, env_1의 t=0..T_1-1, ...)
          - prev_dist는 episode 내부에서만 의미를 갖고, 경계는 is_first(=t==0) mask로 처리.
        """
        d = self.device
        D = int(done_envs.numel())
        if D == 0:
            return
        Ts = self.ep_step[done_envs]                                    # (D,)
        if int(Ts.sum().item()) == 0:
            return

        max_T = int(Ts.max().item())
        range_T = torch.arange(max_T, device=d).unsqueeze(0).expand(D, -1)  # (D, max_T)
        mask = range_T < Ts.unsqueeze(1)                                    # (D, max_T)

        t_flat = range_T[mask]                                              # (M_orig,)
        env_flat = done_envs.unsqueeze(1).expand(-1, max_T)[mask]
        T_per = Ts.unsqueeze(1).expand(-1, max_T)[mask]                     # (M_orig,)
        M_orig = int(t_flat.numel())

        # ----- Gather original transitions -----
        x_o = self.ep_x[t_flat, env_flat]
        e_o = self.ep_edge[t_flat, env_flat]
        u_o = self.ep_u[t_flat, env_flat]
        a_o = self.ep_action[t_flat, env_flat]
        nx_o = self.ep_next_x[t_flat, env_flat]
        ne_o = self.ep_next_edge[t_flat, env_flat]
        nu_o = self.ep_next_u[t_flat, env_flat]
        rod_p_o = self.ep_rod_pos[t_flat, env_flat]
        rod_q_o = self.ep_rod_quat[t_flat, env_flat]
        n_rod_p_o = self.ep_next_rod_pos[t_flat, env_flat]
        n_rod_q_o = self.ep_next_rod_quat[t_flat, env_flat]
        goal_p_o = self.ep_goal_pos[t_flat, env_flat]
        goal_q_o = self.ep_goal_quat[t_flat, env_flat]

        # ----- Original reward + done (vectorized recompute) -----
        pos_err_o = torch.norm(goal_p_o - n_rod_p_o, dim=-1)
        q_err_o = _quat_mul(goal_q_o, _quat_conj(n_rod_q_o))
        rot_err_o = torch.norm(_quat_to_axis_angle(q_err_o), dim=-1)
        cur_dist_o = pos_err_o + 0.1 * rot_err_o

        is_first_o = (t_flat == 0)
        prev_dist_o = torch.empty_like(cur_dist_o)
        prev_dist_o[0] = float('inf')
        if M_orig > 1:
            prev_dist_o[1:] = cur_dist_o[:-1]
        prev_dist_o = torch.where(is_first_o, torch.full_like(prev_dist_o, float('inf')), prev_dist_o)

        r_prog_o = (prev_dist_o - cur_dist_o) * 50.0
        r_prog_o = torch.where(is_first_o, torch.zeros_like(r_prog_o), r_prog_o)
        reached_o = (pos_err_o < 0.10) & (rot_err_o < 0.30) & ~is_first_o
        reward_o = r_prog_o + torch.where(reached_o, torch.full_like(r_prog_o, 100.0), torch.zeros_like(r_prog_o))
        # goal-무관 보상(충돌/clearance/time/smooth) 보존 — HER relabel과 무관하게 동일 state 페널티.
        gi_o = self.ep_goal_indep[t_flat, env_flat]
        reward_o = reward_o + gi_o
        done_o = reached_o | (t_flat == (T_per - 1))

        # ----- HER virtual transitions -----
        # Strategy 분기:
        #   future: (env, t) pair 중 future_t > t 있는 것만. vg = rod 자신의 미래 위치.
        #   random_task: 모든 (env, t). vg = ep_rod_start + (random offset from task pool).
        if self.strategy == "future":
            valid_mask = t_flat < (T_per - 1)
        else:  # random_task — 모든 transition을 HER 대상
            valid_mask = torch.ones_like(t_flat, dtype=torch.bool)

        if self.k_future > 0 and bool(valid_mask.any().item()):
            t_w = t_flat[valid_mask]
            env_w = env_flat[valid_mask]
            T_w = T_per[valid_mask]

            t_her = t_w.repeat_interleave(self.k_future)
            env_her = env_w.repeat_interleave(self.k_future)
            T_her = T_w.repeat_interleave(self.k_future)
            M_her = int(t_her.numel())

            if self.strategy == "future":
                # future_t ~ Uniform{t+1, ..., T_her - 1}
                range_size = (T_her - 1 - t_her).clamp(min=1).float()
                rand_off = (torch.rand(M_her, device=d) * range_size).long()
                future_t = (t_her + 1 + rand_off).clamp(max=T_her - 1)
                vg_pos = self.ep_next_rod_pos[future_t, env_her]
                vg_quat = self.ep_next_rod_quat[future_t, env_her]
            else:  # random_task
                P = self.goal_offset_pos_pool.shape[0]
                idx = torch.randint(0, P, (M_her,), device=d)
                offset_pos = self.goal_offset_pos_pool[idx]      # (M_her, 3)
                offset_quat = self.goal_offset_quat_pool[idx]    # (M_her, 4)
                # rod_start (이 episode 시작 시 rod pose, env-local frame)
                rod_start_pos = self.ep_rod_pos[0, env_her]
                rod_start_quat = self.ep_rod_quat[0, env_her]
                vg_pos = rod_start_pos + offset_pos
                vg_quat = _quat_mul(offset_quat, rod_start_quat)
                vg_quat = vg_quat / torch.norm(vg_quat, dim=-1, keepdim=True)

            x_h = self.ep_x[t_her, env_her]
            e_h = self.ep_edge[t_her, env_her]
            u_h = self.ep_u[t_her, env_her]
            a_h = self.ep_action[t_her, env_her]
            nx_h = self.ep_next_x[t_her, env_her]
            ne_h = self.ep_next_edge[t_her, env_her]
            nu_h = self.ep_next_u[t_her, env_her]
            rod_p_h = self.ep_rod_pos[t_her, env_her]
            rod_q_h = self.ep_rod_quat[t_her, env_her]
            n_rod_p_h = self.ep_next_rod_pos[t_her, env_her]
            n_rod_q_h = self.ep_next_rod_quat[t_her, env_her]

            vx_h = replace_goal_in_node_features(x_h, rod_p_h, rod_q_h, vg_pos, vg_quat)
            vnx_h = replace_goal_in_node_features(nx_h, n_rod_p_h, n_rod_q_h, vg_pos, vg_quat)

            # Virtual reward: prev_dist 정확히 (rod_p at t, virtual goal) 기준
            prev_pos_h = torch.norm(vg_pos - rod_p_h, dim=-1)
            prev_q_h = _quat_mul(vg_quat, _quat_conj(rod_q_h))
            prev_rot_h = torch.norm(_quat_to_axis_angle(prev_q_h), dim=-1)
            prev_dist_h = prev_pos_h + 0.1 * prev_rot_h
            is_first_h = torch.zeros(M_her, dtype=torch.bool, device=d)
            r_h, reached_h, _ = recompute_reward(
                n_rod_p_h, n_rod_q_h, vg_pos, vg_quat,
                prev_dist_h, is_first_h
            )
            # goal-무관 보상(충돌 등)은 virtual goal에도 동일하게 더함 (state 기반이므로 보존).
            r_h = r_h + self.ep_goal_indep[t_her, env_her]
            done_h = reached_h
        else:
            M_her = 0
            vx_h = e_h = u_h = a_h = vnx_h = ne_h = nu_h = r_h = done_h = None

        # ----- Batched push to main buffer -----
        if M_her > 0:
            x_all = torch.cat([x_o, vx_h], dim=0)
            e_all = torch.cat([e_o, e_h], dim=0)
            u_all = torch.cat([u_o, u_h], dim=0)
            a_all = torch.cat([a_o, a_h], dim=0)
            nx_all = torch.cat([nx_o, vnx_h], dim=0)
            ne_all = torch.cat([ne_o, ne_h], dim=0)
            nu_all = torch.cat([nu_o, nu_h], dim=0)
            r_all = torch.cat([reward_o, r_h], dim=0)
            d_all = torch.cat([done_o.float(), done_h.float()], dim=0)
        else:
            x_all, e_all, u_all, a_all = x_o, e_o, u_o, a_o
            nx_all, ne_all, nu_all = nx_o, ne_o, nu_o
            r_all = reward_o
            d_all = done_o.float()

        M_total = int(x_all.shape[0])
        idxs = (self.ptr + torch.arange(M_total, device=d)) % self.capacity
        self.x[idxs] = x_all
        self.edge_attr[idxs] = e_all
        self.u[idxs] = u_all
        self.actions[idxs] = a_all
        self.rewards[idxs] = r_all
        self.next_x[idxs] = nx_all
        self.next_edge_attr[idxs] = ne_all
        self.next_u[idxs] = nu_all
        self.dones[idxs] = d_all

        self.ptr = (self.ptr + M_total) % self.capacity
        self.size = min(self.size + M_total, self.capacity)

    def sample(self, batch_size: int):
        """Sample minibatch — ReplayBuffer.sample과 동일."""
        idxs = torch.randint(0, self.size, (batch_size,), device=self.device)
        N, E = self.N, self.E
        M = batch_size

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

        next_x_flat = self.next_x[idxs].reshape(M * N, -1)
        next_e_flat = self.next_edge_attr[idxs].reshape(M * E, -1)
        next_u_flat = self.next_u[idxs]

        next_batch = Batch(x=next_x_flat, edge_index=edge_index, edge_attr=next_e_flat,
                           u=next_u_flat, batch=batch_idx)
        next_batch.num_graphs = M

        return cur_batch, self.actions[idxs], self.rewards[idxs], next_batch, self.dones[idxs]
