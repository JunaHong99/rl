"""
Graph Converter — 5-node graph, POSE-only (2026-07 실험)

Isaac Lab의 raw observation dict → PyG Batch.

3-node lean graph에서 로봇당 노드 1개를 다시 추가한 5-node 버전.
단, 로봇 노드는 관절각(q/dq/limit)이 아니라 ELBOW pose(pos+quat)를 쓴다.
    - joint-config는 특정 7-DoF 로봇에 그래프를 묶어버린다.
    - pose(spatial)는 morphology-agnostic → 향후 로봇 일반화 + 장애물 대비
      (팔의 공간적 존재를 표현)에 유리.

Graph 구조:
    Node ordering per env (5 nodes):
        [0] Robot1 (left arm, elbow pose)   ← type robot
        [1] EE_1   (left hand)              ← type ee
        [2] Rod    (object)                 ← ROD_NODE_IDX, type rod
        [3] EE_2   (right hand)             ← type ee
        [4] Robot2 (right arm, elbow pose)  ← type robot

    Edges (양방향):
        Kinematic:   Robot1 ↔ EE1, Robot2 ↔ EE2   (edge type 0)
        Grasp:       EE1 ↔ Rod, EE2 ↔ Rod         (edge type 1)
        Cooperative: EE1 ↔ EE2                     (edge type 2)

Node feature (모든 노드 same dim, type별 padding + one-hot):
    - Robot: elbow pos(3) + quat(4) = 7   (관절각 q/dq/limit 없음 — pose only)
    - EE:    pos(3) + quat(4) = 7
    - Rod:   pos(3) + quat(4) + goal_pos(3) + goal_quat(4) + pos_err(3) + rot_err_aa(3) = 20
    - Type one-hot (3 dim): [robot, ee, rod]

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
# Constants — 5-node pose-only graph (2026-07)
# ──────────────────────────────────────────────────────────────────────
N_ARMS = 2

# 노드 인덱스 (env-local) — 5 nodes
ROBOT1_NODE_IDX = 0
EE1_NODE_IDX = 1
ROD_NODE_IDX = 2
EE2_NODE_IDX = 3
ROBOT2_NODE_IDX = 4
NODES_PER_ENV = 5

# Per-node-type raw feature 차원
ROBOT_RAW_DIM = 7       # elbow pos(3) + quat(4)  ★ pose only, NO joint config
EE_RAW_DIM = 7          # pos(3) + quat(4)
ROD_RAW_DIM = 20        # pos(3) + quat(4) + goal_pos(3) + goal_quat(4) + pos_err(3) + rot_err_aa(3)

# 통합 padding dim — 모든 raw 중 최대
NODE_RAW_PADDED_DIM = max(ROBOT_RAW_DIM, EE_RAW_DIM, ROD_RAW_DIM)  # 20
N_NODE_TYPES = 3  # robot, ee, rod
NODE_FEATURE_DIM = NODE_RAW_PADDED_DIM + N_NODE_TYPES  # 20 + 3 = 23

# Edge type encoding
# 0 = kinematic   (Robot ↔ EE)
# 1 = grasp       (EE ↔ Rod)
# 2 = cooperative (EE1 ↔ EE2, 양손 협조 제약)
N_EDGE_TYPES = 3
# RoboBallet식 상대-pose 엣지: 타입 one-hot(3) + sender의 receiver-기준 상대 pose
#   상대위치(3) + 상대회전 6D(6).
EDGE_GEOM_DIM = 3 + 6  # rel_pos(3) + rel_rot_6d(6)
EDGE_FEATURE_DIM = N_EDGE_TYPES + EDGE_GEOM_DIM  # 3 + 9 = 12

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
    5-node graph:
        Kinematic:   Robot1↔EE1, Robot2↔EE2   (type 0)
        Grasp:       EE1↔Rod, EE2↔Rod          (type 1)
        Cooperative: EE1↔EE2                    (type 2)

    Returns:
        edge_src, edge_dst: list[int] (env-local node indices)
        edge_type: list[int] (0=kinematic, 1=grasp, 2=cooperative)
    """
    src, dst, etype = [], [], []

    def add_pair(a, b, t):
        src.append(a); dst.append(b); etype.append(t)
        src.append(b); dst.append(a); etype.append(t)

    # Kinematic: Robot ↔ EE (arm 구조 정보)
    add_pair(ROBOT1_NODE_IDX, EE1_NODE_IDX, 0)
    add_pair(ROBOT2_NODE_IDX, EE2_NODE_IDX, 0)

    # Grasp: EE ↔ Rod
    add_pair(EE1_NODE_IDX, ROD_NODE_IDX, 1)
    add_pair(EE2_NODE_IDX, ROD_NODE_IDX, 1)

    # Cooperative constraint: EE1 ↔ EE2 (rod fixed joint로 양손 제약)
    add_pair(EE1_NODE_IDX, EE2_NODE_IDX, 2)

    return src, dst, etype


_EDGE_SRC, _EDGE_DST, _EDGE_TYPE = build_edge_template()
N_EDGES_PER_ENV = len(_EDGE_SRC)


# ──────────────────────────────────────────────────────────────────────
# Per-node-type feature builders
# ──────────────────────────────────────────────────────────────────────
def _robot_features(robot_poses):
    """
    Robot 노드: elbow pose (pos+quat) only — ★ NO joint config (q/dq/limit).

    robot_poses: (B, 2, 7) — ★ env-local position 가정 (elbow_poses, base_poses fallback)

    Returns: (B, 2, ROBOT_RAW_DIM=7) — normalized [-CLIP, CLIP]
        Schema: pos(3) + quat(4).
    """
    pos = robot_poses[..., :3] / POS_NORM                      # (B, 2, 3)
    quat = robot_poses[..., 3:7]                               # (B, 2, 4) 이미 [-1, 1]
    feat = torch.cat([pos, quat], dim=-1)                      # (B, 2, 7)
    return torch.clamp(feat, -FEAT_CLIP, FEAT_CLIP)


def _ee_features(current_ee_poses):
    """
    current_ee_poses: (B, 2, 7) — ★ env-local position 가정

    Returns: (B, 2, EE_RAW_DIM=7) — normalized [-CLIP, CLIP]
        Schema: pos(3) + quat(4).
    """
    pos = current_ee_poses[..., :3] / POS_NORM                # (B, 2, 3)
    quat = current_ee_poses[..., 3:7]                          # (B, 2, 4) 이미 [-1, 1]
    feat = torch.cat([pos, quat], dim=-1)                      # (B, 2, 7)
    return torch.clamp(feat, -FEAT_CLIP, FEAT_CLIP)


def _rod_features(rod_pos, rod_quat, goal_pos, goal_quat):
    """
    All shapes (B, 3) or (B, 4). ★ rod_pos, goal_pos는 env-local 가정.

    Returns: (B, 1, ROD_RAW_DIM=20) — normalized [-CLIP, CLIP]
        Schema: pos(3) + quat(4) + goal_pos(3) + goal_quat(4) + pos_err(3) + rot_err_aa(3).
    """
    # Raw 오차 (정규화 전)
    pos_err_raw = goal_pos - rod_pos                           # (B, 3)
    q_err = _quat_mul(goal_quat, _quat_conj(rod_quat))         # (B, 4)
    rot_err_aa_raw = _quat_to_axis_angle(q_err)                # (B, 3) rad

    # 정규화
    rp = rod_pos / POS_NORM
    gp = goal_pos / POS_NORM
    pe = pos_err_raw / POS_NORM
    re = rot_err_aa_raw / ROT_ERR_NORM

    feat = torch.cat([
        rp, rod_quat,            # 현재 rod 상태 (7) — quat은 [-1, 1] 그대로
        gp, goal_quat,           # ★ 실제 목표 (7)
        pe, re                   # ★ 진짜 오차 (6)
    ], dim=-1)                                                 # (B, 20)
    feat = torch.clamp(feat, -FEAT_CLIP, FEAT_CLIP)
    return feat.unsqueeze(1)                                   # (B, 1, 20)


# ──────────────────────────────────────────────────────────────────────
# Pad + add type one-hot
# ──────────────────────────────────────────────────────────────────────
def _assemble_nodes(robot_feat, ee_feat, rod_feat, B: int, device):
    """
    5-node ordering: [Robot1, EE1, Rod, EE2, Robot2].

    robot_feat: (B, 2, ROBOT_RAW_DIM=7)
    ee_feat:    (B, 2, EE_RAW_DIM=7)
    rod_feat:   (B, 1, ROD_RAW_DIM=20)

    Returns: (B, NODES_PER_ENV=5, NODE_FEATURE_DIM=23)
    """
    type_id = {'robot': 0, 'ee': 1, 'rod': 2}

    def _wrap(raw, name):
        """Pad to NODE_RAW_PADDED_DIM + append type one-hot."""
        n_i = raw.shape[1]
        pad_dim = NODE_RAW_PADDED_DIM - raw.shape[-1]
        if pad_dim > 0:
            raw = torch.cat([raw, torch.zeros(B, n_i, pad_dim, device=device)], dim=-1)
        oh = torch.zeros(B, n_i, N_NODE_TYPES, device=device)
        oh[..., type_id[name]] = 1.0
        return torch.cat([raw, oh], dim=-1)                  # (B, n_i, NODE_FEATURE_DIM)

    robot_padded = _wrap(robot_feat, 'robot')                 # (B, 2, 23)
    ee_padded = _wrap(ee_feat, 'ee')                          # (B, 2, 23)
    rod_padded = _wrap(rod_feat, 'rod')                       # (B, 1, 23)

    # 순서: Robot1, EE1, Rod, EE2, Robot2
    nodes = torch.stack([
        robot_padded[:, 0],     # Robot1
        ee_padded[:, 0],        # EE1
        rod_padded[:, 0],       # Rod
        ee_padded[:, 1],        # EE2
        robot_padded[:, 1],     # Robot2
    ], dim=1)                                                # (B, 5, 23)
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
    Raw observation dict + control state → PyG Batch (5-node pose-only).

    Args:
        raw_state: env3._get_observations()["policy"] — 필요 키:
            'current_ee_poses' (B, 2, 7), 'rod_pos' (B, 3), 'rod_quat' (B, 4),
            'elbow_poses' (B, 2, 7)  (없으면 'base_poses' (B, 2, 7) fallback)
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

    # Robot 노드: elbow pose (없으면 base_poses fallback = 구 동작) — env-local
    robot_poses = raw_state.get('elbow_poses', raw_state['base_poses'])  # (B, 2, 7)

    # ── Node features (5-node) ──
    robot_feat = _robot_features(robot_poses)
    ee_feat = _ee_features(current_ee_poses)

    rod_pos = raw_state['rod_pos']
    rod_quat = raw_state['rod_quat']
    rod_feat = _rod_features(rod_pos, rod_quat, goal_pos, goal_quat)

    # node assembly (Robot1, EE1, Rod, EE2, Robot2 순서)
    x_per_env = _assemble_nodes(robot_feat, ee_feat, rod_feat, B, device)
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

    # ── Edge feature: type one-hot(3) + 상대 pose(9) (RoboBallet식) ──
    # 노드 pose 조립 (env-local), 순서: Robot1, EE1, Rod, EE2, Robot2.
    node_pos = torch.stack([
        robot_poses[:, 0, :3], current_ee_poses[:, 0, :3], rod_pos,
        current_ee_poses[:, 1, :3], robot_poses[:, 1, :3],
    ], dim=1)                                                              # (B,5,3)
    node_quat = torch.stack([
        robot_poses[:, 0, 3:7], current_ee_poses[:, 0, 3:7], rod_quat,
        current_ee_poses[:, 1, 3:7], robot_poses[:, 1, 3:7],
    ], dim=1)                                                              # (B,5,4)

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
