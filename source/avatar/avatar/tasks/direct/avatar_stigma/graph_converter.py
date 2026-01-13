
import torch
from torch_geometric.data import Data

## --------------------------------------------------------------------
## 1. 그래프 피쳐 차원 정의
## --------------------------------------------------------------------

# --- Node Feature Dims ---
RAW_ROBOT_DIM = 14
# [수정됨] 태스크 노드 차원 0 (절대 위치 정보 제거 -> 완전한 상대성 확보)
RAW_TASK_DIM = 0 
RAW_OBSTACLE_DIM = 0

# --- GNN Input Dims ---
# 모든 노드는 이 차원으로 패딩됨 (로봇 차원인 14로 통일)
NODE_FEATURE_DIM = 14 
EDGE_FEATURE_DIM = 9

# 글로벌 차원: Current Rel(9) + Target Rel(9) + Time(1) = 19
GLOBAL_FEATURE_DIM = 19


## --------------------------------------------------------------------
## 2. 헬퍼 함수
## --------------------------------------------------------------------
def quat_to_rotmat(quat):
    """Quat(wxyz) -> RotMat(3x3)"""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    
    row0 = torch.stack([1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy)], dim=-1)
    row1 = torch.stack([2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx)], dim=-1)
    row2 = torch.stack([2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)

def rotmat_to_6d(rotmat):
    """RotMat(3x3) -> 6D Vector"""
    r1 = rotmat[..., :, 0]
    r2 = rotmat[..., :, 1]
    return torch.cat([r1, r2], dim=-1)

def quat_to_6d(quat):
    """Quat -> 6D Helper"""
    R = quat_to_rotmat(quat)
    return rotmat_to_6d(R)

def calculate_relative_pose(pos_sender, quat_sender, pos_receiver, quat_receiver):
    """Edge Feature 계산 (Relative Pose in Receiver Frame)"""
    R_s = quat_to_rotmat(quat_sender)
    R_r = quat_to_rotmat(quat_receiver)
    
    # Relative Position
    p_diff = pos_sender - pos_receiver
    p_rel = torch.matmul(R_r.transpose(-1, -2), p_diff.unsqueeze(-1)).squeeze(-1)
    
    # Relative Rotation
    R_rel = torch.matmul(R_r.transpose(-1, -2), R_s)
    rot_6d = rotmat_to_6d(R_rel)
    
    return torch.cat([p_rel, rot_6d], dim=-1) # [9]


## --------------------------------------------------------------------
## 3. 상태-그래프 변환기
## --------------------------------------------------------------------

def convert_state_to_graph(raw_state: dict) -> Data:
    
    # --- Input Unpacking ---
    robot_nodes = raw_state['robot_nodes']
    current_ee_poses = raw_state['current_ee_poses'] 
    goal_poses = raw_state['goal_poses']             
    base_poses = raw_state['base_poses']             
    
    # globals 처리 (Time info + Constraints)
    if 'globals' in raw_state:
        raw_globals = raw_state['globals'] # [15]
    else:
        raw_globals = torch.zeros(15, device=robot_nodes.device)

    # 차원 정리 (Single Env Slicing)
    if robot_nodes.ndim == 3:
        idx = 0
        r_nodes = robot_nodes[idx]
        curr_ee = current_ee_poses[idx]
        g_poses = goal_poses[idx]
        b_poses = base_poses[idx]
        g_vec = raw_globals[idx]
        device = r_nodes.device
    else:
        r_nodes = robot_nodes
        curr_ee = current_ee_poses
        g_poses = goal_poses
        b_poses = base_poses
        g_vec = raw_globals
        device = r_nodes.device

    num_robots = r_nodes.shape[0]
    num_tasks = g_poses.shape[0]
    total_nodes = num_robots + num_tasks

    # -------------------------------------------------------
    # 1. Node Features (x)
    # -------------------------------------------------------
    # 0으로 초기화 (Task Node는 0으로 남음)
    x = torch.zeros(total_nodes, NODE_FEATURE_DIM, dtype=torch.float, device=device)
    
    # Robot Nodes: Joint Info (유지)
    x[0:num_robots, 0:RAW_ROBOT_DIM] = r_nodes
    
    # Task Nodes: [수정됨] RAW_TASK_DIM=0 이므로 아무것도 할당하지 않음
    # x[num_robots:] 부분은 0 벡터(Zero Vector)로 유지됨.
    # GNN은 이 0 벡터를 "이것은 태스크 노드다"라는 신호로 받아들여 학습함.

    # -------------------------------------------------------
    # 2. Global Features (u) - [6D 변환 적용]
    # -------------------------------------------------------
    # g_vec 구조: [Curr_Pos(3), Curr_Quat(4), Targ_Pos(3), Targ_Quat(4), Time(1)]
    
    curr_rel_pos = g_vec[0:3]
    curr_rel_quat = g_vec[3:7]
    curr_rel_6d = quat_to_6d(curr_rel_quat) # [6]
    
    targ_rel_pos = g_vec[7:10]
    targ_rel_quat = g_vec[10:14]
    targ_rel_6d = quat_to_6d(targ_rel_quat) # [6]
    
    time_val = g_vec[14:15]
    
    # New Global: [3 + 6 + 3 + 6 + 1] = 19
    u = torch.cat([curr_rel_pos, curr_rel_6d, targ_rel_pos, targ_rel_6d, time_val], dim=-1).unsqueeze(0)

    # -------------------------------------------------------
    # 3. Edges
    # -------------------------------------------------------
    edge_indices = []
    edge_attrs = []

    # Type 1: Robot <-> Robot
    for i in range(num_robots): 
        for j in range(num_robots): 
            if i == j: continue
            edge_indices.append([j, i])
            rel_attr = calculate_relative_pose(
                pos_sender=b_poses[j, :3], quat_sender=b_poses[j, 3:],
                pos_receiver=curr_ee[i, :3], quat_receiver=curr_ee[i, 3:]
            )
            edge_attrs.append(rel_attr)

    # Type 2: Task -> Robot
    for t_idx in range(num_tasks): 
        task_node_idx = num_robots + t_idx
        for r_idx in range(num_robots): 
            robot_node_idx = r_idx
            edge_indices.append([task_node_idx, robot_node_idx])
            
            # [중요] Node Feature에서는 뺐지만, Edge Feature 계산에는 g_poses를 사용함 (필수)
            rel_attr = calculate_relative_pose(
                pos_sender=g_poses[t_idx, :3], quat_sender=g_poses[t_idx, 3:],
                pos_receiver=curr_ee[r_idx, :3], quat_receiver=curr_ee[r_idx, 3:]
            )
            edge_attrs.append(rel_attr)

    if not edge_indices:
        edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
        edge_attr = torch.empty(0, EDGE_FEATURE_DIM, dtype=torch.float, device=device)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long, device=device).t().contiguous()
        edge_attr = torch.stack(edge_attrs)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u)