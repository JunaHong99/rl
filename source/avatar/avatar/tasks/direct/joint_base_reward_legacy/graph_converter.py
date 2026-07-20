
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


def convert_batch_state_to_graph(raw_state: dict, num_envs: int) -> Batch:
    """
    [Vectorized Version]
    Isaac Lab의 Batch Tensor를 Loop 없이 한 번에 PyG Batch 객체로 변환합니다.
    """
    
    # --- 1. Input Unpacking ---
    # [B, 2, 14]
    robot_nodes = raw_state['robot_nodes'] 
    # [B, 2, 7]
    current_ee_poses = raw_state['current_ee_poses']
    # [B, 2, 7]
    goal_poses = raw_state['goal_poses']
    # [B, 2, 7]
    base_poses = raw_state['base_poses']
    
    # [B, 15] or [B, 19] depending on env version
    if 'globals' in raw_state:
        raw_globals = raw_state['globals']
    else:
        # Fallback (Should not happen if env is correct)
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
    # 초기화: [B, N_per_env, Node_Dim]
    # Task Node 부분은 0으로 유지됨
    x_batch = torch.zeros(num_envs, nodes_per_env, NODE_FEATURE_DIM, dtype=torch.float, device=device)
    
    # Robot Features 채우기 (앞쪽 num_robots 개)
    # robot_nodes: [B, 2, 14] -> x_batch[:, :2, :14]
    x_batch[:, :num_robots, :RAW_ROBOT_DIM] = robot_nodes

    # Flatten -> [B*N, Node_Dim]
    x = x_batch.view(total_nodes, NODE_FEATURE_DIM)

    # -------------------------------------------------------
    # 3. Global Features (u) - [Batch_Size, Global_Dim]
    # -------------------------------------------------------
    # raw_globals: [B, 15] 
    #   [0:3] Curr Rel Pos
    #   [3:7] Curr Rel Quat
    #   [7:10] Targ Rel Pos
    #   [10:14] Targ Rel Quat
    #   [14] Time
    
    curr_rel_pos = raw_globals[:, 0:3]
    curr_rel_quat = raw_globals[:, 3:7]
    curr_rel_6d = quat_to_6d(curr_rel_quat) # [B, 6]
    
    targ_rel_pos = raw_globals[:, 7:10]
    targ_rel_quat = raw_globals[:, 10:14]
    targ_rel_6d = quat_to_6d(targ_rel_quat) # [B, 6]
    
    time_val = raw_globals[:, 14:15]

    # [B, 3+6+3+6+1] = [B, 19]
    u = torch.cat([curr_rel_pos, curr_rel_6d, targ_rel_pos, targ_rel_6d, time_val], dim=-1)

    # -------------------------------------------------------
    # 4. Edge Indices (Topology)
    # -------------------------------------------------------
    # 모든 환경은 동일한 그래프 구조를 가집니다.
    # 따라서 "Base Template"을 하나 만들고 복제+오프셋 하면 됩니다.
    
    src_list = []
    dst_list = []
    
    # Type 1: Robot <-> Robot (Fully Connected, no self-loop)
    for i in range(num_robots):
        for j in range(num_robots):
            if i == j: continue
            src_list.append(j) # Sender
            dst_list.append(i) # Receiver
            
    # Type 2: Task -> Robot (Fully Connected)
    # Task Node Indices within an env: [num_robots, num_robots+1, ...]
    for t_idx in range(num_tasks):
        task_node_local_idx = num_robots + t_idx
        for r_idx in range(num_robots):
            src_list.append(task_node_local_idx)
            dst_list.append(r_idx)
            
    # Base Template: [2, Num_Edges_Per_Env]
    base_src = torch.tensor(src_list, dtype=torch.long, device=device)
    base_dst = torch.tensor(dst_list, dtype=torch.long, device=device)
    num_edges_per_env = len(src_list)
    
    # Expand for Batch
    # [Num_Envs, Num_Edges_Per_Env]
    batch_src = base_src.unsqueeze(0).repeat(num_envs, 1)
    batch_dst = base_dst.unsqueeze(0).repeat(num_envs, 1)
    
    # Offset Calculation: [[0], [4], [8], ...] -> [B, 1]
    # 각 환경의 노드 인덱스 시작점
    offsets = torch.arange(num_envs, device=device) * nodes_per_env
    offsets = offsets.unsqueeze(-1) # [B, 1]
    
    # Add Offsets
    batch_src = batch_src + offsets
    batch_dst = batch_dst + offsets
    
    # Flatten -> [2, Total_Edges]
    edge_index = torch.stack([batch_src.view(-1), batch_dst.view(-1)], dim=0)

    # -------------------------------------------------------
    # 5. Edge Attributes (Features)
    # -------------------------------------------------------
    # Edge Attr 계산을 위해 Source Node와 Dest Node의 물리적 포즈(Pos, Quat)가 필요합니다.
    # 이를 위해 전체 노드의 Pose 정보를 담은 텐서를 미리 준비합니다.
    
    # A. Prepare All Node Poses: [B, N_per_env, 7]
    # Robot Nodes: base_pose (Sender용) / ee_pose (Receiver용)
    # Task Nodes: goal_pose (Sender용)
    # (주의: Robot은 역할에 따라 쓰는 포즈가 다릅니다.
    #  - Robot이 Sender일 때: Base Pose
    #  - Robot이 Receiver일 때: Current EE Pose
    #  - Task가 Sender일 때: Goal Pose
    #  따라서, Sender용 Pose 배열과 Receiver용 Pose 배열을 따로 만드는 게 편합니다.)
    
    # --- Sender Pose Array ---
    # Robot(0~1) -> Base Poses, Task(2~3) -> Goal Poses
    sender_poses = torch.cat([base_poses, goal_poses], dim=1) # [B, 4, 7]
    # Flatten -> [Total_Nodes, 7]
    flat_sender_poses = sender_poses.view(-1, 7)
    
    # --- Receiver Pose Array ---
    # Robot(0~1) -> EE Poses, Task(2~3) -> Dummy(Not used as receiver)
    # Task는 Receiver가 되지 않으므로 아무 값이나 채워도 됩니다.
    receiver_poses = torch.cat([current_ee_poses, torch.zeros_like(goal_poses)], dim=1) # [B, 4, 7]
    flat_receiver_poses = receiver_poses.view(-1, 7)
    
    # B. Gather Poses using Edge Index
    # edge_index[0] -> Source Indices (Global)
    # edge_index[1] -> Dest Indices (Global)
    
    src_p_q = flat_sender_poses[edge_index[0]]   # [Total_Edges, 7]
    dst_p_q = flat_receiver_poses[edge_index[1]] # [Total_Edges, 7]
    
    # C. Calculate Relative Pose (Vectorized)
    # 기존 calculate_relative_pose 함수는 (..., 3), (..., 4) 입력을 받으므로 호환됨
    edge_attr = calculate_relative_pose(
        pos_sender=src_p_q[:, :3], quat_sender=src_p_q[:, 3:],
        pos_receiver=dst_p_q[:, :3], quat_receiver=dst_p_q[:, 3:]
    ) # [Total_Edges, 9]

    # -------------------------------------------------------
    # 6. Batch Vector
    # -------------------------------------------------------
    # [0, 0, 0, 0, 1, 1, 1, 1, ..., B-1, B-1]
    batch = torch.arange(num_envs, device=device).repeat_interleave(nodes_per_env)
    
    # -------------------------------------------------------
    # 7. Construct PyG Batch
    # -------------------------------------------------------
    # Data 객체 대신 Batch 객체를 직접 생성하거나, Data 객체에 batch 속성을 넣어줍니다.
    # PyG의 Batch 객체는 사실 Data 객체를 상속받으며 batch, ptr 등의 속성을 추가로 가집니다.
    # 여기서는 간단히 Data 객체에 batch를 넣어 리턴해도 모델에서는 문제없이 돌아갑니다.
    # 하지만 엄격한 타입을 위해 Batch.from_data_list 없이 직접 구성합니다.
    
    out_batch = Batch(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        u=u,
        batch=batch,
        ptr=None # ptr은 필요하면 계산, 보통 자동 처리되거나 없어도 됨
    )
    # Batch size 명시 (Global Feature 처리에 중요)
    out_batch.num_graphs = num_envs
    
    return out_batch