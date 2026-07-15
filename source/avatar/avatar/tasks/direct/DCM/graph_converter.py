"""
Graph Converter — LEAN 3-node graph, POSE-only (RoboBallet 철학)

Isaac Lab의 raw observation dict → PyG Batch.

RoboBallet 철학: node feature는 최소/identity, geometry는 전부 상대-pose 엣지에.
로봇+손(hand)을 하나의 "Arm" 노드로 MERGE한 3-node 버전.
    - Arm 노드는 feature가 없다 (type one-hot으로만 구별). 단, EE/tip pose라는
      공간적 pose는 가지며, 이 pose가 상대-pose 엣지 계산에 쓰인다.
    - pose(spatial)는 morphology-agnostic → 향후 로봇 일반화 + 장애물 대비에 유리.

Graph 구조:
    Node ordering per env (3 nodes):
        [0] Arm1 (left robot+hand MERGED)   ← type arm
        [1] Rod  (object)                   ← ROD_NODE_IDX, type rod
        [2] Arm2 (right robot+hand MERGED)  ← type arm

    Edges (양방향):
        Grasp:       Arm1 ↔ Rod, Arm2 ↔ Rod   (edge type 0)
        Cooperative: Arm1 ↔ Arm2               (edge type 1)

Node feature (모든 노드 same dim, type별 padding + one-hot):
    - Arm: NO feature (identity only) — raw block 0-dim (padding 후 all-zeros)
    - Rod: pos_err(3) + rot_err_aa(3) = 6   (goal 오차만 — 절대 pos/quat 및 goal 절대값 없음)
    - Type one-hot (2 dim): [arm, rod]

본 구현은 padding 방식의 homogeneous graph다.

Output (PyG Batch):
    x:           [Total_Nodes, NODE_FEATURE_DIM]
    edge_index:  [2, Total_Edges]
    edge_attr:   [Total_Edges, EDGE_FEATURE_DIM]
    u:           [B, GLOBAL_FEATURE_DIM]
    batch:       [Total_Nodes] env index per node
"""

from __future__ import annotations
import math
import torch
from torch_geometric.data import Batch, Data


# ──────────────────────────────────────────────────────────────────────
# Normalization constants (각 feature의 대략적인 범위로 나눠 정규화)
# ──────────────────────────────────────────────────────────────────────
POS_NORM = 1.0          # env-local position [m] — sampler 0~1m 범위
ROT_ERR_NORM = math.pi  # axis-angle rotation error [rad]
FEAT_CLIP = 5.0         # 정규화 후 outlier clip


# ──────────────────────────────────────────────────────────────────────
# Constants — LEAN 3-node graph
# ──────────────────────────────────────────────────────────────────────
N_ARMS = 2

# 노드 인덱스 (env-local) — 3 nodes
ARM1_NODE_IDX = 0
ROD_NODE_IDX = 1
ARM2_NODE_IDX = 2
NODES_PER_ENV = 3

# Per-node-type raw feature 차원
ARM_RAW_DIM = 0         # Arm 노드는 feature 없음 (identity only)
ROD_RAW_DIM = 6         # pos_err(3) + rot_err_aa(3)

# 통합 padding dim — 모든 raw 중 최대
NODE_RAW_PADDED_DIM = max(ARM_RAW_DIM, ROD_RAW_DIM)  # 6
N_NODE_TYPES = 2  # arm, rod
NODE_FEATURE_DIM = NODE_RAW_PADDED_DIM + N_NODE_TYPES  # 6 + 2 = 8

# Edge type encoding
# 0 = grasp       (Arm ↔ Rod)
# 1 = cooperative (Arm1 ↔ Arm2, 양팔 협조 제약)
N_EDGE_TYPES = 2
# RoboBallet식 상대-pose 엣지: 타입 one-hot(2) + sender의 receiver-기준 상대 pose
#   상대위치(3) + 상대회전 6D(6).
EDGE_GEOM_DIM = 3 + 6  # rel_pos(3) + rel_rot_6d(6)
EDGE_FEATURE_DIM = N_EDGE_TYPES + EDGE_GEOM_DIM  # 2 + 9 = 11

# Global feature — normalized_time only
GLOBAL_FEATURE_DIM = 1  # normalized_time(1)


# ──────────────────────────────────────────────────────────────────────
# Quaternion helper (axis-angle for error)
# ──────────────────────────────────────────────────────────────────────
def _quat_conj(q):
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def _quat_apply(q, v):
    """Rotate vector v(...,3) by quat q(...,4) wxyz. Batched over leading dims."""
    qw = q[..., 0:1]
    qv = q[..., 1:4]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + qw * t + torch.cross(qv, t, dim=-1)


def _quat_to_6d(q):
    """quat(...,4) wxyz → 6D 회전표현(...,6) = 회전행렬의 첫 두 컬럼 (Zhou et al.)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # 회전행렬 col0, col1
    col0 = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], dim=-1)
    col1 = torch.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], dim=-1)
    return torch.cat([col0, col1], dim=-1)   # (...,6)


def _quat_to_axis_angle(q):
    """(B, 4) wxyz → (B, 3) axis-angle vector."""
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


# ──────────────────────────────────────────────────────────────────────
# Static edge index template (env-local)
# ──────────────────────────────────────────────────────────────────────
def build_edge_template():
    """
    3-node graph:
        Grasp:       Arm1↔Rod, Arm2↔Rod   (type 0)
        Cooperative: Arm1↔Arm2             (type 1)

    Returns:
        edge_src, edge_dst: list[int] (env-local node indices)
        edge_type: list[int] (0=grasp, 1=cooperative)
    """
    src, dst, etype = [], [], []

    def add_pair(a, b, t):
        src.append(a); dst.append(b); etype.append(t)
        src.append(b); dst.append(a); etype.append(t)

    # Grasp: Arm ↔ Rod
    add_pair(ARM1_NODE_IDX, ROD_NODE_IDX, 0)
    add_pair(ARM2_NODE_IDX, ROD_NODE_IDX, 0)

    # Cooperative constraint: Arm1 ↔ Arm2 (rod fixed joint로 양팔 제약)
    add_pair(ARM1_NODE_IDX, ARM2_NODE_IDX, 1)

    return src, dst, etype


_EDGE_SRC, _EDGE_DST, _EDGE_TYPE = build_edge_template()
N_EDGES_PER_ENV = len(_EDGE_SRC)


# ──────────────────────────────────────────────────────────────────────
# Per-node-type feature builders
# ──────────────────────────────────────────────────────────────────────
def _rod_features(rod_pos, rod_quat, goal_pos, goal_quat):
    """
    Rod 노드: goal 오차만 (pos_err + rot_err_aa). ★ rod_pos, goal_pos는 env-local 가정.

    All shapes (B, 3) or (B, 4).

    Returns: (B, 1, ROD_RAW_DIM=6) — normalized [-CLIP, CLIP]
        Schema: pos_err(3) + rot_err_aa(3).
    """
    # Raw 오차 (정규화 전)
    pos_err_raw = goal_pos - rod_pos                           # (B, 3)
    q_err = _quat_mul(goal_quat, _quat_conj(rod_quat))         # (B, 4)
    rot_err_aa_raw = _quat_to_axis_angle(q_err)                # (B, 3) rad

    # 정규화
    pe = pos_err_raw / POS_NORM
    re = rot_err_aa_raw / ROT_ERR_NORM

    feat = torch.cat([pe, re], dim=-1)                         # (B, 6)
    feat = torch.clamp(feat, -FEAT_CLIP, FEAT_CLIP)
    return feat.unsqueeze(1)                                   # (B, 1, 6)


# ──────────────────────────────────────────────────────────────────────
# Pad + add type one-hot
# ──────────────────────────────────────────────────────────────────────
def _assemble_nodes(rod_feat, B: int, device):
    """
    3-node ordering: [Arm1, Rod, Arm2].

    Arm 노드: raw block empty (0-dim) → padding 후 all-zeros + arm one-hot.
    rod_feat:   (B, 1, ROD_RAW_DIM=6)

    Returns: (B, NODES_PER_ENV=3, NODE_FEATURE_DIM=8)
    """
    type_id = {'arm': 0, 'rod': 1}

    def _wrap(raw, name):
        """Pad to NODE_RAW_PADDED_DIM + append type one-hot."""
        n_i = raw.shape[1]
        pad_dim = NODE_RAW_PADDED_DIM - raw.shape[-1]
        if pad_dim > 0:
            raw = torch.cat([raw, torch.zeros(B, n_i, pad_dim, device=device)], dim=-1)
        oh = torch.zeros(B, n_i, N_NODE_TYPES, device=device)
        oh[..., type_id[name]] = 1.0
        return torch.cat([raw, oh], dim=-1)                  # (B, n_i, NODE_FEATURE_DIM)

    # Arm 노드: raw block empty (0-dim) → wrap가 all-zeros로 padding
    arm_empty = torch.zeros(B, N_ARMS, ARM_RAW_DIM, device=device)  # (B, 2, 0)
    arm_padded = _wrap(arm_empty, 'arm')                     # (B, 2, 8) — raw all-zeros
    rod_padded = _wrap(rod_feat, 'rod')                      # (B, 1, 8)

    # 순서: Arm1, Rod, Arm2
    nodes = torch.stack([
        arm_padded[:, 0],       # Arm1
        rod_padded[:, 0],       # Rod
        arm_padded[:, 1],       # Arm2
    ], dim=1)                                                # (B, 3, 8)
    return nodes


# ──────────────────────────────────────────────────────────────────────
# Main API
# ──────────────────────────────────────────────────────────────────────
def convert_batch_state_to_graph(
    raw_state: dict,
    num_envs: int,
    goal_pos: torch.Tensor,
    goal_quat: torch.Tensor,
    normalized_time: torch.Tensor,
) -> Batch:
    """
    Raw observation dict + control state → PyG Batch (LEAN 3-node).

    Args:
        raw_state: env3._get_observations()["policy"] — 필요 키:
            'current_ee_poses' (B, 2, 7)  — Arm 노드의 tip(EE) pose (엣지 상대-pose용),
            'rod_pos' (B, 3), 'rod_quat' (B, 4)
        num_envs: B
        goal_pos: (B, 3)  ★ 실제 목표 (goal_rod_marker), env-local
        goal_quat: (B, 4)
        normalized_time: (B,) 0~1

    Returns:
        PyG Batch with x, edge_index, edge_attr, u, batch
    """
    current_ee_poses = raw_state['current_ee_poses']                   # (B, 2, 7)
    device = current_ee_poses.device
    B = num_envs

    rod_pos = raw_state['rod_pos']
    rod_quat = raw_state['rod_quat']

    # ── Node features (3-node): rod만 raw feature, Arm은 identity ──
    rod_feat = _rod_features(rod_pos, rod_quat, goal_pos, goal_quat)

    # node assembly (Arm1, Rod, Arm2 순서)
    x_per_env = _assemble_nodes(rod_feat, B, device)
    nodes_per_env = x_per_env.shape[1]

    # ── Edge index (env-local → global with batch offset) ──
    src_local = torch.tensor(_EDGE_SRC, device=device, dtype=torch.long)
    dst_local = torch.tensor(_EDGE_DST, device=device, dtype=torch.long)
    etype_local = torch.tensor(_EDGE_TYPE, device=device, dtype=torch.long)

    batch_src = src_local.unsqueeze(0).expand(B, -1)            # (B, E)
    batch_dst = dst_local.unsqueeze(0).expand(B, -1)
    offsets = (torch.arange(B, device=device) * nodes_per_env).unsqueeze(-1)
    src_global = (batch_src + offsets).reshape(-1)              # (B*E,)
    dst_global = (batch_dst + offsets).reshape(-1)
    edge_index = torch.stack([src_global, dst_global], dim=0)   # (2, B*E)

    # ── Edge feature: type one-hot(2) + 상대 pose(9) (RoboBallet식) ──
    # 노드 pose 조립 (env-local), 순서: Arm1, Rod, Arm2.
    #   Arm 노드의 공간적 pose = 해당 팔의 END-EFFECTOR(tip) pose (current_ee_poses).
    #   Rod 노드의 공간적 pose = rod pose.
    node_pos = torch.stack([
        current_ee_poses[:, 0, :3], rod_pos, current_ee_poses[:, 1, :3],
    ], dim=1)                                                              # (B,3,3)
    node_quat = torch.stack([
        current_ee_poses[:, 0, 3:7], rod_quat, current_ee_poses[:, 1, 3:7],
    ], dim=1)                                                              # (B,3,4)

    E = N_EDGES_PER_ENV
    p_src, q_src = node_pos[:, src_local], node_quat[:, src_local]         # (B,E,3),(B,E,4)
    p_dst, q_dst = node_pos[:, dst_local], node_quat[:, dst_local]
    q_dst_conj = _quat_conj(q_dst)
    rel_pos = _quat_apply(q_dst_conj, p_src - p_dst) / POS_NORM            # sender의 receiver-frame 상대위치
    rel_6d = _quat_to_6d(_quat_mul(q_dst_conj, q_src))                     # 상대회전 6D
    geom = torch.clamp(torch.cat([rel_pos, rel_6d], dim=-1), -FEAT_CLIP, FEAT_CLIP)  # (B,E,9)
    onehot = torch.zeros(B, E, N_EDGE_TYPES, device=device)
    onehot.scatter_(2, etype_local.view(1, E, 1).expand(B, E, 1), 1.0)
    edge_attr = torch.cat([onehot, geom], dim=-1).reshape(B * E, EDGE_FEATURE_DIM)

    # ── Global features (normalized_time only) ──
    u = normalized_time.reshape(B, 1)                          # (B, 1)

    # ── Assemble Batch ──
    x = x_per_env.reshape(B * nodes_per_env, NODE_FEATURE_DIM)
    batch_idx = torch.arange(B, device=device).repeat_interleave(nodes_per_env)

    out = Batch(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        u=u,
        batch=batch_idx,
    )
    out.num_graphs = B
    return out
