"""Kinematic Graph Converter — 17-node (base·joint) graph, 리더-팔로워 morphology 일반화용.

lean 3-node(EE만)로는 파지/base 변이 시 관절 상태를 표현 못 함(정책이 리더 관절 소유).
→ 관절을 노드로: 파지 폭/각도는 grasp 엣지, base 간격은 base-rel 엣지, 관절 feasibility는 joint 노드.

노드 순서 (env-local, 고정 인덱스):
    [0]      base_L        (리더 base)        type base
    [1..7]   joint_L 1~7   (리더 관절)         type joint   ← ★ 출력 head (리더 Δq 7)
    [8]      rod           (object)           type object
    [9]      base_F        (팔로워 base)       type base
    [10..16] joint_F 1~7   (팔로워 관절)        type joint   ← backbone 참여, 출력 X(IK 결정)
    = 17 nodes/env

엣지 타입 (4, 전부 양방향, receiver-frame 상대 pose = trans3+rot6d=9):
    0 base→first   base ↔ joint1        (mount)
    1 joint-chain  joint_i ↔ joint_{i+1} (DH 고정변환 = 링크길이 인코딩)
    2 grasp        joint7(EE) ↔ rod      (양팔) ← 파지 폭+각도
    3 base-rel     base_L ↔ base_F       ← base 간격

노드 feature (타입별 raw + one-hot):
    base:   0 (identity)
    joint:  축(3) + 값 sin/cos(2) + margin(min·max, 2) + 0(type은 one-hot) = 7
    object: goal오차 base-frame pos(3) + rot6d(6) = 9
    → padded max(9) + type one-hot(3) = NODE_FEATURE_DIM 12

env는 각 노드의 공간 pose(엣지용)와 raw feature를 dict로 넘긴다(아래 convert 시그니처 참고).
"""
from __future__ import annotations
import torch
from torch_geometric.data import Batch
from graph_converter import _quat_conj, _quat_mul, _quat_apply, _quat_to_6d  # 재사용

POS_NORM = 1.0
FEAT_CLIP = 5.0

N_ARM_JOINTS = 7
NODES_PER_ENV = 17

# ★ 관절 회전축 = **부모링크 frame 기준 상수**(DH alpha 도출). NerveNet식: 노드는 국소(부모기준),
#   공간관계는 엣지. base/자세/로봇배치 완전 불변 → morphology 전이 최강. GNN 학습도 국소라 쉬움.
#   Modified DH: 부모frame 기준 joint_i 축 = Rx(alpha_i)·[0,0,1] = [0, -sin α, cos α].
#   alpha = [0, -π/2, π/2, π/2, -π/2, π/2, π/2] (franka_tensor_ik DH).
PARENT_FRAME_JOINT_AXES = [
    [0.0,  0.0, 1.0],   # j1
    [0.0,  1.0, 0.0],   # j2
    [0.0, -1.0, 0.0],   # j3
    [0.0, -1.0, 0.0],   # j4
    [0.0,  1.0, 0.0],   # j5
    [0.0, -1.0, 0.0],   # j6
    [0.0, -1.0, 0.0],   # j7
]
# 고정 인덱스
BASE_L_IDX = 0
JOINT_L_IDX = list(range(1, 8))      # 1..7  ← 출력 head 대상
ROD_IDX = 8
BASE_F_IDX = 9
JOINT_F_IDX = list(range(10, 17))    # 10..16
LEADER_JOINT_IDX = JOINT_L_IDX       # gather 대상 (고정)

# 노드 타입
N_NODE_TYPES = 3                     # base, joint, object
TYPE_BASE, TYPE_JOINT, TYPE_OBJECT = 0, 1, 2
BASE_RAW_DIM = 0
JOINT_RAW_DIM = 7                    # 축(3)+sincos(2)+margin(2)
OBJECT_RAW_DIM = 9                   # goal pos(3)+rot6d(6)
NODE_RAW_PADDED_DIM = max(BASE_RAW_DIM, JOINT_RAW_DIM, OBJECT_RAW_DIM)  # 9
NODE_FEATURE_DIM = NODE_RAW_PADDED_DIM + N_NODE_TYPES  # 9 + 3 = 12

# 엣지 타입
N_EDGE_TYPES = 4
E_BASE_FIRST, E_CHAIN, E_GRASP, E_BASEREL = 0, 1, 2, 3
EDGE_GEOM_DIM = 3 + 6
EDGE_FEATURE_DIM = N_EDGE_TYPES + EDGE_GEOM_DIM  # 4 + 9 = 13

GLOBAL_FEATURE_DIM = 2               # time(1) + f_int(1)


# ──────────────────────────────────────────────────────────────────────
# Static edge template (env-local, 고정)
# ──────────────────────────────────────────────────────────────────────
def _build_edge_template():
    src, dst, etype = [], [], []
    def add(a, b, t):   # 양방향
        src.append(a); dst.append(b); etype.append(t)
        src.append(b); dst.append(a); etype.append(t)
    # 리더 팔: base→joint1, joint chain 1-2-...-7
    add(BASE_L_IDX, JOINT_L_IDX[0], E_BASE_FIRST)
    for i in range(N_ARM_JOINTS - 1):
        add(JOINT_L_IDX[i], JOINT_L_IDX[i + 1], E_CHAIN)
    # 팔로워 팔
    add(BASE_F_IDX, JOINT_F_IDX[0], E_BASE_FIRST)
    for i in range(N_ARM_JOINTS - 1):
        add(JOINT_F_IDX[i], JOINT_F_IDX[i + 1], E_CHAIN)
    # grasp: 양팔 EE(joint7) ↔ rod
    add(JOINT_L_IDX[-1], ROD_IDX, E_GRASP)
    add(JOINT_F_IDX[-1], ROD_IDX, E_GRASP)
    # base-rel: 두 base
    add(BASE_L_IDX, BASE_F_IDX, E_BASEREL)
    return src, dst, etype


_EDGE_SRC, _EDGE_DST, _EDGE_TYPE = _build_edge_template()
N_EDGES_PER_ENV = len(_EDGE_SRC)


# ──────────────────────────────────────────────────────────────────────
# Main API
# ──────────────────────────────────────────────────────────────────────
def convert_kin_graph(raw: dict, num_envs: int) -> Batch:
    """
    raw dict (전부 (B, ...) 텐서, world 좌표는 env-local 권장):
      노드 공간 pose (엣지 상대pose 계산용, 순서=위 17노드):
        'node_pos'  (B, 17, 3)
        'node_quat' (B, 17, 4) wxyz
      joint 노드 raw feature 재료:
        'joint_axis'   (B, 14, 3)   리더7+팔로워7 관절축(부모링크 frame or world 일관되게)
        'joint_val'    (B, 14)      관절각 [rad]
        'joint_margin' (B, 14, 2)   (q-q_min)/range, (q_max-q)/range  ∈[0,1]
      object 노드 raw:
        'obj_goal_pos6d' (B, 9)     base-frame goal오차 pos(3)+rot6d(6)
      global:
        'time' (B,), 'f_int' (B,)
    """
    device = raw['node_pos'].device
    B = num_envs
    node_pos = raw['node_pos']                                   # (B,17,3)
    node_quat = raw['node_quat']                                 # (B,17,4)

    # ── 노드 raw feature 조립 (B,17,NODE_RAW_PADDED_DIM) + type one-hot ──
    raw_feat = torch.zeros(B, NODES_PER_ENV, NODE_RAW_PADDED_DIM, device=device)
    onehot = torch.zeros(B, NODES_PER_ENV, N_NODE_TYPES, device=device)
    # base 노드 (0, 9): identity → raw 0, type one-hot
    onehot[:, BASE_L_IDX, TYPE_BASE] = 1.0
    onehot[:, BASE_F_IDX, TYPE_BASE] = 1.0
    # joint 노드: 축(3)+sincos(2)+margin(2) = 7
    jaxis = raw['joint_axis']                                    # (B,14,3)
    jval = raw['joint_val']                                      # (B,14)
    jmargin = raw['joint_margin']                                # (B,14,2)
    jfeat = torch.cat([jaxis, torch.sin(jval).unsqueeze(-1), torch.cos(jval).unsqueeze(-1),
                       jmargin], dim=-1)                          # (B,14,7)
    all_joint_idx = JOINT_L_IDX + JOINT_F_IDX
    for k, nidx in enumerate(all_joint_idx):
        raw_feat[:, nidx, :JOINT_RAW_DIM] = jfeat[:, k]
        onehot[:, nidx, TYPE_JOINT] = 1.0
    # object 노드: base-frame goal오차 (9)
    raw_feat[:, ROD_IDX, :OBJECT_RAW_DIM] = raw['obj_goal_pos6d']
    onehot[:, ROD_IDX, TYPE_OBJECT] = 1.0

    x_per_env = torch.clamp(torch.cat([raw_feat, onehot], dim=-1), -FEAT_CLIP, FEAT_CLIP)  # (B,17,12)

    # ── 엣지 index (batch offset) ──
    src = torch.tensor(_EDGE_SRC, device=device, dtype=torch.long)
    dst = torch.tensor(_EDGE_DST, device=device, dtype=torch.long)
    etype = torch.tensor(_EDGE_TYPE, device=device, dtype=torch.long)
    E = N_EDGES_PER_ENV
    offs = (torch.arange(B, device=device) * NODES_PER_ENV).unsqueeze(-1)
    edge_index = torch.stack([(src.unsqueeze(0) + offs).reshape(-1),
                              (dst.unsqueeze(0) + offs).reshape(-1)], dim=0)

    # ── 엣지 feature: type one-hot(4) + receiver-frame 상대pose(9) ──
    p_src, q_src = node_pos[:, src], node_quat[:, src]           # (B,E,3),(B,E,4)
    p_dst, q_dst = node_pos[:, dst], node_quat[:, dst]
    q_dst_conj = _quat_conj(q_dst)
    rel_pos = _quat_apply(q_dst_conj, p_src - p_dst) / POS_NORM
    rel_6d = _quat_to_6d(_quat_mul(q_dst_conj, q_src))
    geom = torch.clamp(torch.cat([rel_pos, rel_6d], dim=-1), -FEAT_CLIP, FEAT_CLIP)   # (B,E,9)
    eoh = torch.zeros(B, E, N_EDGE_TYPES, device=device)
    eoh.scatter_(2, etype.view(1, E, 1).expand(B, E, 1), 1.0)
    edge_attr = torch.cat([eoh, geom], dim=-1).reshape(B * E, EDGE_FEATURE_DIM)

    # ── global ──
    u = torch.stack([raw['time'], 0.01 * raw['f_int']], dim=-1)  # (B,2)

    # ── assemble (기존 lean converter와 동일 방식) ──
    x = x_per_env.reshape(B * NODES_PER_ENV, NODE_FEATURE_DIM)
    batch_idx = torch.arange(B, device=device).repeat_interleave(NODES_PER_ENV)
    out = Batch(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u, batch=batch_idx)
    out.num_graphs = B
    return out
