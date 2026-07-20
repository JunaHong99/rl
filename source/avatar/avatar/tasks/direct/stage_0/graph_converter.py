"""
Graph Converter (Phase 3.3, current)

Isaac Lab의 raw observation dict → PyG Batch.

현재 구현은 5-node graph를 사용한다.
이전 14-dim joint-vel TD3 버전(graph_converter_legacy.py.bak)과 호환되지 않는다.

Graph 구조:
    Node ordering per env:
        [0]      Robot_1   (joint state 요약)
        [1]      EE_1
        [2]      Rod
        [3]      EE_2
        [4]      Robot_2   (joint state 요약)
        [5..]    Obstacles (현재 0개; Phase 4에서 추가 시 끝에 append)

    Edges (양방향):
        Kinematic: Robot_i ↔ EE_i
        Grasp: EE_1 ↔ Rod, EE_2 ↔ Rod
        Cooperative: EE_1 ↔ EE_2
        Proximity (when obstacles): Rod ↔ Obstacle, EE ↔ Obstacle (Phase 4)

Node feature (모든 노드 same dim, type별 padding + one-hot):
    - Robot: q/dq/joint-limit margin
    - EE: pose/velocity/wrench
    - Rod: current state + goal + error
    - Type one-hot (4 dim): [robot, ee, rod, obstacle]

본 구현은 padding 방식의 homogeneous graph다.

Output (PyG Batch):
    x:           [Total_Nodes, NODE_FEATURE_DIM]
    edge_index:  [2, Total_Edges]
    u:           [B, GLOBAL_FEATURE_DIM]
    batch:       [Total_Nodes] env index per node
"""

from __future__ import annotations
import math
import torch
from torch_geometric.data import Batch, Data


# ──────────────────────────────────────────────────────────────────────
# Normalization constants (★ Phase 3.3: 각 feature의 대략적인 범위로 나눠 정규화)
# 학습 plateau 핵심 원인 fix: world coord + 스케일 mismatch 해결.
# ──────────────────────────────────────────────────────────────────────
POS_NORM = 1.0          # env-local position [m] — sampler 0~1m 범위
VEL_LIN_NORM = 1.0      # linear velocity [m/s]
VEL_ANG_NORM = 3.0      # angular velocity [rad/s]
WRENCH_LIN_NORM = 20.0  # force [N]
WRENCH_ANG_NORM = 5.0   # moment [Nm]
JOINT_Q_NORM = math.pi  # joint position [rad] — 대략 ±π
JOINT_DQ_NORM = 2.0     # joint velocity [rad/s] — vel_limit_sim=1.5
JOINT_TAU_NORM = 50.0   # joint torque [Nm] — effort_limit_sim=50
JOINT_MARGIN_NORM = math.pi  # margin [rad]
ROT_ERR_NORM = math.pi  # axis-angle rotation error [rad]
FEAT_CLIP = 5.0         # 정규화 후 outlier clip


# ──────────────────────────────────────────────────────────────────────
# Constants — 5-node refactor (2026-05-20)
# ──────────────────────────────────────────────────────────────────────
# 이전 17-node (14 joint + 2 EE + 1 rod) 구조는 action이 task-space(6-dim object delta)인데
# graph만 joint-level이라 표현 mismatch. EE 구조는 grasp/transport task에 직접적이므로 보존,
# joint는 robot node 하나로 압축. (3-node는 정보 손실 큼, 17-node는 over-engineered.)
N_ARM_JOINTS = 7
N_ARMS = 2
N_OBSTACLES = 0  # Phase 4에서 변경 가능

# 노드 인덱스 (env-local) — 5 nodes
ROBOT1_NODE_IDX = 0
EE1_NODE_IDX = 1
ROD_NODE_IDX = 2
EE2_NODE_IDX = 3
ROBOT2_NODE_IDX = 4
OBSTACLE_NODE_OFFSET = 5
NODES_PER_ENV = OBSTACLE_NODE_OFFSET + N_OBSTACLES     # 5 (+N_OBS)

# Per-node-type raw feature 차원
ROBOT_RAW_DIM = 28      # q(7) + dq(7) + margin_low(7) + margin_high(7)
EE_RAW_DIM = 19         # pos(3) + quat(4) + lin_vel(3) + ang_vel(3) + wrench(6)
ROD_RAW_DIM = 26        # pos(3) + quat(4) + lin_vel(3) + ang_vel(3) + goal_pos(3) + goal_quat(4) + pos_err(3) + rot_err_aa(3)
OBSTACLE_RAW_DIM = 8    # pos(3) + radius(1) + lin_vel(3) + dist_to_rod(1)

# 통합 padding dim — 모든 raw 중 최대
NODE_RAW_PADDED_DIM = max(ROBOT_RAW_DIM, EE_RAW_DIM, ROD_RAW_DIM, OBSTACLE_RAW_DIM)  # 28
N_NODE_TYPES = 4  # robot, ee, rod, obstacle
NODE_FEATURE_DIM = NODE_RAW_PADDED_DIM + N_NODE_TYPES  # 28 + 4 = 32

# Edge type encoding
# 0 = kinematic (Robot ↔ EE)
# 1 = grasp     (EE ↔ Rod)
# 2 = cooperative (EE1 ↔ EE2, 양손 협조 제약)
# 3 = proximity (Phase 4)
N_EDGE_TYPES = 4
EDGE_FEATURE_DIM = N_EDGE_TYPES  # 1-hot

# Global feature (target_x_rel + episode time + 미래용)
GLOBAL_FEATURE_DIM = 3 + 1  # target_x_rel(3) + normalized_time(1)


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
    5-node graph: Robot1 ↔ EE1 ↔ Rod ↔ EE2 ↔ Robot2 + cooperative EE1 ↔ EE2.

    Returns:
        edge_src, edge_dst: list[int] (env-local node indices)
        edge_type: list[int] (0=kinematic, 1=grasp, 2=cooperative, 3=proximity)
    """
    src, dst, etype = [], [], []

    def add_pair(a, b, t):
        src.append(a); dst.append(b); etype.append(t)
        src.append(b); dst.append(a); etype.append(t)

    # Kinematic: Robot ↔ EE per arm
    add_pair(ROBOT1_NODE_IDX, EE1_NODE_IDX, 0)
    add_pair(ROBOT2_NODE_IDX, EE2_NODE_IDX, 0)

    # Grasp: EE ↔ Rod
    add_pair(EE1_NODE_IDX, ROD_NODE_IDX, 1)
    add_pair(EE2_NODE_IDX, ROD_NODE_IDX, 1)

    # Cooperative constraint: EE1 ↔ EE2 (rod fixed joint로 양손 제약)
    add_pair(EE1_NODE_IDX, EE2_NODE_IDX, 2)

    # Proximity edges to obstacles — Phase 4에서 동적 추가 (현재 N_OBSTACLES=0)
    for k in range(N_OBSTACLES):
        add_pair(ROD_NODE_IDX, OBSTACLE_NODE_OFFSET + k, 3)
        add_pair(EE1_NODE_IDX, OBSTACLE_NODE_OFFSET + k, 3)
        add_pair(EE2_NODE_IDX, OBSTACLE_NODE_OFFSET + k, 3)

    return src, dst, etype


_EDGE_SRC, _EDGE_DST, _EDGE_TYPE = build_edge_template()
N_EDGES_PER_ENV = len(_EDGE_SRC)


# ──────────────────────────────────────────────────────────────────────
# Per-node-type feature builders
# ──────────────────────────────────────────────────────────────────────
def _robot_features(robot_nodes, joint_limits_low, joint_limits_high):
    """
    Robot 노드 1개당 자체 joint state를 통합한 vector (5-node refactor).

    robot_nodes: (B, N_ARMS, 14) [q(7), dq(7)] — env3._get_observations로부터
    joint_limits_low/high: (N_ARMS, 7)

    Returns: (B, N_ARMS=2, ROBOT_RAW_DIM=28) — normalized [-CLIP, CLIP]
        Schema: q(7) + dq(7) + margin_low(7) + margin_high(7)
        (joint_torque은 effort_mode에서 항상 외부 명령이라 측정값으로 의미 약해 제외)
    """
    q_raw = robot_nodes[..., :7]                                          # (B, 2, 7)
    dq_raw = robot_nodes[..., 7:14]
    margin_low_raw = q_raw - joint_limits_low.unsqueeze(0)
    margin_high_raw = joint_limits_high.unsqueeze(0) - q_raw

    # 정규화
    q = q_raw / JOINT_Q_NORM
    dq = dq_raw / JOINT_DQ_NORM
    margin_low = margin_low_raw / JOINT_MARGIN_NORM
    margin_high = margin_high_raw / JOINT_MARGIN_NORM

    # (B, 2, 7) × 4 → (B, 2, 28)
    feat = torch.cat([q, dq, margin_low, margin_high], dim=-1)
    feat = torch.clamp(feat, -FEAT_CLIP, FEAT_CLIP)
    return feat                                                            # (B, 2, 28)


def _ee_features(current_ee_poses, ee_lin_vel, ee_ang_vel, wrench_panda_1, wrench_panda_2):
    """
    current_ee_poses: (B, 2, 7) — ★ env-local position 가정
    ee_lin_vel, ee_ang_vel: (B, 2, 3) each
    wrench_panda_*: (B, 6) per arm

    Returns: (B, 2, EE_RAW_DIM=19) — normalized [-CLIP, CLIP]
    """
    pos = current_ee_poses[..., :3] / POS_NORM                # (B, 2, 3)
    quat = current_ee_poses[..., 3:7]                          # (B, 2, 4) 이미 [-1, 1]
    lin = ee_lin_vel / VEL_LIN_NORM
    ang = ee_ang_vel / VEL_ANG_NORM

    wrenches = torch.stack([wrench_panda_1, wrench_panda_2], dim=1)  # (B, 2, 6)
    f_lin = wrenches[..., :3] / WRENCH_LIN_NORM
    f_ang = wrenches[..., 3:] / WRENCH_ANG_NORM

    feat = torch.cat([pos, quat, lin, ang, f_lin, f_ang], dim=-1)    # (B, 2, 19)
    return torch.clamp(feat, -FEAT_CLIP, FEAT_CLIP)


def _rod_features(rod_pos, rod_quat, rod_lin, rod_ang, goal_pos, goal_quat):
    """
    All shapes (B, 3) or (B, 4). ★ rod_pos, goal_pos는 env-local 가정.

    Returns: (B, 1, ROD_RAW_DIM=26) — normalized [-CLIP, CLIP]
    """
    # Raw 오차 (정규화 전)
    pos_err_raw = goal_pos - rod_pos                            # (B, 3)
    q_err = _quat_mul(goal_quat, _quat_conj(rod_quat))          # (B, 4)
    rot_err_aa_raw = _quat_to_axis_angle(q_err)                 # (B, 3) rad

    # 정규화
    rp = rod_pos / POS_NORM
    rl = rod_lin / VEL_LIN_NORM
    ra = rod_ang / VEL_ANG_NORM
    gp = goal_pos / POS_NORM
    pe = pos_err_raw / POS_NORM
    re = rot_err_aa_raw / ROT_ERR_NORM

    feat = torch.cat([
        rp, rod_quat, rl, ra,    # 현재 rod 상태 (13) — quat은 [-1, 1] 그대로
        gp, goal_quat,           # ★ 실제 목표 (7)
        pe, re                   # ★ 진짜 오차 (6)
    ], dim=-1)                                                  # (B, 26)
    feat = torch.clamp(feat, -FEAT_CLIP, FEAT_CLIP)
    return feat.unsqueeze(1)                                    # (B, 1, 26)


# ──────────────────────────────────────────────────────────────────────
# Pad + add type one-hot
# ──────────────────────────────────────────────────────────────────────
def _assemble_nodes(robot_feat, ee_feat, rod_feat, obstacle_feat, B: int, device):
    """
    5-node ordering: [Robot1, EE1, Rod, EE2, Robot2, (Obstacles...)].

    robot_feat:    (B, 2, ROBOT_RAW_DIM=28)
    ee_feat:       (B, 2, EE_RAW_DIM=19)
    rod_feat:      (B, 1, ROD_RAW_DIM=26)
    obstacle_feat: (B, N_OBS, OBSTACLE_RAW_DIM=8) or None

    Returns: (B, NODES_PER_ENV, NODE_FEATURE_DIM=32)
    """
    type_id = {'robot': 0, 'ee': 1, 'rod': 2, 'obstacle': 3}

    def _wrap(raw, name):
        """Pad to NODE_RAW_PADDED_DIM + append type one-hot."""
        n_i = raw.shape[1]
        pad_dim = NODE_RAW_PADDED_DIM - raw.shape[-1]
        if pad_dim > 0:
            raw = torch.cat([raw, torch.zeros(B, n_i, pad_dim, device=device)], dim=-1)
        oh = torch.zeros(B, n_i, N_NODE_TYPES, device=device)
        oh[..., type_id[name]] = 1.0
        return torch.cat([raw, oh], dim=-1)                  # (B, n_i, NODE_FEATURE_DIM)

    robot_padded = _wrap(robot_feat, 'robot')                # (B, 2, 32)
    ee_padded = _wrap(ee_feat, 'ee')                          # (B, 2, 32)
    rod_padded = _wrap(rod_feat, 'rod')                       # (B, 1, 32)

    # 순서: Robot1, EE1, Rod, EE2, Robot2
    nodes = torch.stack([
        robot_padded[:, 0],     # Robot1
        ee_padded[:, 0],        # EE1
        rod_padded[:, 0],       # Rod
        ee_padded[:, 1],        # EE2
        robot_padded[:, 1],     # Robot2
    ], dim=1)                                                # (B, 5, 32)

    if obstacle_feat is not None and obstacle_feat.shape[1] > 0:
        obs_padded = _wrap(obstacle_feat, 'obstacle')
        nodes = torch.cat([nodes, obs_padded], dim=1)        # (B, 5+N_OBS, 32)

    return nodes


# ──────────────────────────────────────────────────────────────────────
# Main API
# ──────────────────────────────────────────────────────────────────────
def convert_batch_state_to_graph(
    raw_state: dict,
    num_envs: int,
    goal_pos: torch.Tensor,
    goal_quat: torch.Tensor,
    target_x_rel: torch.Tensor,
    normalized_time: torch.Tensor,
    joint_limits_low: torch.Tensor,
    joint_limits_high: torch.Tensor,
    joint_torque: torch.Tensor | None = None,
) -> Batch:
    """
    Raw observation dict + control state → PyG Batch.

    Args:
        raw_state: env3._get_observations()["policy"]
        num_envs: B
        goal_pos: (B, 3)  ★ 실제 목표 (goal_rod_marker)
        goal_quat: (B, 4)
        target_x_rel: (B, 3)
        normalized_time: (B,) 0~1
        joint_limits_low, joint_limits_high: (2, 7)
        joint_torque: (B, 2, 7) optional

    Returns:
        PyG Batch with x, edge_index, edge_attr, u, batch
    """
    device = raw_state['robot_nodes'].device
    B = num_envs

    # ── Node features (5-node) ──
    robot_feat = _robot_features(
        raw_state['robot_nodes'],
        joint_limits_low.to(device), joint_limits_high.to(device)
    )

    # EE features (env3._get_observations가 이미 ee_lin_vel/ee_ang_vel 노출)
    current_ee_poses = raw_state['current_ee_poses']                    # (B, 2, 7)
    ee_lin_vel = raw_state.get('ee_lin_vel', torch.zeros(B, N_ARMS, 3, device=device))
    ee_ang_vel = raw_state.get('ee_ang_vel', torch.zeros(B, N_ARMS, 3, device=device))
    wrench_1 = raw_state.get('wrench_panda_1', torch.zeros(B, 6, device=device))
    wrench_2 = raw_state.get('wrench_panda_2', torch.zeros(B, 6, device=device))
    ee_feat = _ee_features(current_ee_poses, ee_lin_vel, ee_ang_vel, wrench_1, wrench_2)

    # Rod features (env3가 'rod_pos' 등 직접 노출)
    rod_pos = raw_state['rod_pos']
    rod_quat = raw_state['rod_quat']
    rod_lin = raw_state['rod_lin_vel']
    rod_ang = raw_state['rod_ang_vel']
    rod_feat = _rod_features(rod_pos, rod_quat, rod_lin, rod_ang, goal_pos, goal_quat)

    # 5-node assembly (Robot1, EE1, Rod, EE2, Robot2 순서)
    x_per_env = _assemble_nodes(robot_feat, ee_feat, rod_feat, None, B, device)
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

    # Edge feature: type one-hot
    etype_batch = etype_local.unsqueeze(0).expand(B, -1).reshape(-1)  # (B*E,)
    edge_attr = torch.zeros(B * N_EDGES_PER_ENV, EDGE_FEATURE_DIM, device=device)
    edge_attr.scatter_(1, etype_batch.unsqueeze(-1), 1.0)

    # ── Global features ──
    u = torch.cat([target_x_rel, normalized_time.unsqueeze(-1)], dim=-1)  # (B, 4)

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
