
import torch
from torch_geometric.data import Data, Batch

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
EDGE_FEATURE_DIM = 16 # [Modified] 21 -> 16

# 글로벌 차원: Rel Pos Error(3) + Rel Rot 6D Error(6) + Time(1) = 10
GLOBAL_FEATURE_DIM = 10


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

def calculate_enhanced_edge_features(pos_sender, quat_sender, pos_receiver, quat_receiver, joint_sender, joint_receiver, edge_type_mask):
    """
    Enhanced Edge Feature Calculation
    Args:
        pos/quat: Cartesian info [N, 3], [N, 4]
        joint_sender: [N, 7] (Target joint for Task nodes, 0 or Current for Robot nodes)
        joint_receiver: [N, 7] (Current joint for Robot nodes)
        edge_type_mask: [N] 1.0 if Task->Robot edge, 0.0 otherwise
    Returns:
        [N, 21] Features
    """
    # 1. Standard Relative Pose (9 dims)
    R_s = quat_to_rotmat(quat_sender)
    R_r = quat_to_rotmat(quat_receiver)
    
    p_diff = pos_sender - pos_receiver
    p_rel = torch.matmul(R_r.transpose(-1, -2), p_diff.unsqueeze(-1)).squeeze(-1)
    
    R_rel = torch.matmul(R_r.transpose(-1, -2), R_s)
    rot_6d = rotmat_to_6d(R_rel)
    
    basic_feat = torch.cat([p_rel, rot_6d], dim=-1) # [N, 9]
    
    # 2. Joint Difference (7 dims)
    # Only meaningful for Task->Robot edges. For Robot->Robot, we want 0.
    # Diff = Target - Current
    joint_diff = (joint_sender - joint_receiver) * edge_type_mask.unsqueeze(-1)
    
    # Concatenate only basic and joint diff
    # [9] + [7] = 16
    full_feat = torch.cat([basic_feat, joint_diff], dim=-1)
    
    return full_feat


## --------------------------------------------------------------------
## 3. 상태-그래프 변환기
## --------------------------------------------------------------------

def convert_state_to_graph(raw_state: dict) -> Data:
    # Single env version not maintained for batch update logic consistency
    # Assuming vectorized environment always calls convert_batch_state_to_graph
    raise NotImplementedError("Use convert_batch_state_to_graph")


def convert_batch_state_to_graph(raw_state: dict, num_envs: int) -> Batch:
    """
    [Vectorized Version]
    Isaac Lab의 Batch Tensor를 Loop 없이 한 번에 PyG Batch 객체로 변환합니다.
    """
    
    # --- 1. Input Unpacking ---
    robot_nodes = raw_state['robot_nodes'] # [B, 2, 14] (Pos:0~7, Vel:7~14)
    current_ee_poses = raw_state['current_ee_poses']
    goal_poses = raw_state['goal_poses']
    base_poses = raw_state['base_poses']
    
    # [NEW] Target Joint Pos extraction
    if 'target_joint_pos' in raw_state:
        target_joints_flat = raw_state['target_joint_pos'] # [B, 14]
    else:
        target_joints_flat = torch.zeros(num_envs, 14, device=robot_nodes.device)
        
    target_joints = target_joints_flat.view(num_envs, 2, 7) # [B, 2, 7]
    current_joints = robot_nodes[:, :, :7] # [B, 2, 7]
    
    # [B, 15] or [B, 19]
    if 'globals' in raw_state:
        raw_globals = raw_state['globals']
    else:
        raw_globals = torch.zeros(num_envs, 15, device=robot_nodes.device)

    device = robot_nodes.device
    
    # Dimensions
    num_robots = robot_nodes.shape[1] # 2
    num_tasks = goal_poses.shape[1]   # 2
    nodes_per_env = num_robots + num_tasks # 4
    total_nodes = num_envs * nodes_per_env

    # -------------------------------------------------------
    # 2. Node Features (X) - [Total_Nodes, Node_Dim]
    # -------------------------------------------------------
    x_batch = torch.zeros(num_envs, nodes_per_env, NODE_FEATURE_DIM, dtype=torch.float, device=device)
    x_batch[:, :num_robots, :RAW_ROBOT_DIM] = robot_nodes
    x = x_batch.view(total_nodes, NODE_FEATURE_DIM)

    # -------------------------------------------------------
    # 3. Global Features (u) - Error-based (Target - Current)
    # -------------------------------------------------------
    # Current Rel (B, 3) / (B, 4)
    curr_rel_pos = raw_globals[:, 0:3]
    curr_rel_quat = raw_globals[:, 3:7]
    R_curr = quat_to_rotmat(curr_rel_quat)
    
    # Target Rel (B, 3) / (B, 4)
    targ_rel_pos = raw_globals[:, 7:10]
    targ_rel_quat = raw_globals[:, 10:14]
    R_targ = quat_to_rotmat(targ_rel_quat)
    
    # Position Error: Target - Current
    pos_error = targ_rel_pos - curr_rel_pos
    
    # Rotation Error Matrix: R_targ @ R_curr^T (The rotation needed to reach target)
    R_err = torch.matmul(R_targ, R_curr.transpose(-1, -2))
    rot_error_6d = rotmat_to_6d(R_err)
    
    # Time
    time_val = raw_globals[:, 14:15]

    # Combined Global: [3] + [6] + [1] = 10
    u = torch.cat([pos_error, rot_error_6d, time_val], dim=-1)

    # -------------------------------------------------------
    # 4. Edge Indices (Topology)
    # -------------------------------------------------------
    src_list = []
    dst_list = []
    type_mask_list = [] # 1.0 for Task->Robot, 0.0 for Robot<->Robot
    
    # Type 1: Robot <-> Robot
    for i in range(num_robots):
        for j in range(num_robots):
            if i == j: continue
            src_list.append(j); dst_list.append(i); type_mask_list.append(0.0)
            
    # Type 2: Task -> Robot
    for t_idx in range(num_tasks):
        task_node_local_idx = num_robots + t_idx
        for r_idx in range(num_robots):
            src_list.append(task_node_local_idx); dst_list.append(r_idx); type_mask_list.append(1.0)
            
    base_src = torch.tensor(src_list, dtype=torch.long, device=device)
    base_dst = torch.tensor(dst_list, dtype=torch.long, device=device)
    base_mask = torch.tensor(type_mask_list, dtype=torch.float, device=device)
    
    batch_src = base_src.unsqueeze(0).repeat(num_envs, 1)
    batch_dst = base_dst.unsqueeze(0).repeat(num_envs, 1)
    batch_mask = base_mask.unsqueeze(0).repeat(num_envs, 1)
    
    offsets = (torch.arange(num_envs, device=device) * nodes_per_env).unsqueeze(-1)
    
    batch_src = batch_src + offsets
    batch_dst = batch_dst + offsets
    
    edge_index = torch.stack([batch_src.view(-1), batch_dst.view(-1)], dim=0)
    flat_type_mask = batch_mask.view(-1) # [Total_Edges]

    # -------------------------------------------------------
    # 5. Edge Attributes (Enhanced)
    # -------------------------------------------------------
    # Prepare Pose Arrays [B, N_per_env, 7]
    sender_poses = torch.cat([base_poses, goal_poses], dim=1) 
    flat_sender_poses = sender_poses.view(-1, 7)
    
    receiver_poses = torch.cat([current_ee_poses, torch.zeros_like(goal_poses)], dim=1)
    flat_receiver_poses = receiver_poses.view(-1, 7)
    
    # Prepare Joint Arrays [B, N_per_env, 7]
    # Sender: Robot(Base)->Zero?, Robot(Joint)->Joint?, Task->Target Joint
    # Robot Sender Logic: Robot<->Robot edges use base pose, so joint info is ambiguous. 
    # But calculate_relative_pose expects joint_sender.
    # For Robot->Robot, joint diff is 0, so input doesn't matter (masked out).
    # For Task->Robot, Sender is Task (Target Joint).
    sender_joints = torch.cat([torch.zeros_like(current_joints), target_joints], dim=1) # [B, 4, 7]
    flat_sender_joints = sender_joints.view(-1, 7)
    
    # Receiver: Robot -> Current Joint
    receiver_joints = torch.cat([current_joints, torch.zeros_like(target_joints)], dim=1)
    flat_receiver_joints = receiver_joints.view(-1, 7)
    
    # Gather
    src_p_q = flat_sender_poses[edge_index[0]]
    dst_p_q = flat_receiver_poses[edge_index[1]]
    src_j = flat_sender_joints[edge_index[0]]
    dst_j = flat_receiver_joints[edge_index[1]]
    
    # Calculate
    edge_attr = calculate_enhanced_edge_features(
        pos_sender=src_p_q[:, :3], quat_sender=src_p_q[:, 3:],
        pos_receiver=dst_p_q[:, :3], quat_receiver=dst_p_q[:, 3:],
        joint_sender=src_j,
        joint_receiver=dst_j,
        edge_type_mask=flat_type_mask
    )

    # -------------------------------------------------------
    # 6. Batch Construction
    # -------------------------------------------------------
    batch = torch.arange(num_envs, device=device).repeat_interleave(nodes_per_env)
    
    out_batch = Batch(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        u=u,
        batch=batch,
        ptr=None
    )
    out_batch.num_graphs = num_envs
    
    return out_batch