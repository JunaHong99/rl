import torch
import numpy as np
import argparse
from franka_tensor_ik import FrankaTensorIK

def quat_to_matrix(quats):
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    
    rot_mats = torch.zeros((quats.shape[0], 3, 3), device=quats.device)
    rot_mats[:, 0, 0] = 1 - 2 * (yy + zz); rot_mats[:, 0, 1] = 2 * (xy - wz); rot_mats[:, 0, 2] = 2 * (xz + wy)
    rot_mats[:, 1, 0] = 2 * (xy + wz); rot_mats[:, 1, 1] = 1 - 2 * (xx + zz); rot_mats[:, 1, 2] = 2 * (yz - wx)
    rot_mats[:, 2, 0] = 2 * (xz - wy); rot_mats[:, 2, 1] = 2 * (yz + wx); rot_mats[:, 2, 2] = 1 - 2 * (xx + yy)
    return rot_mats

def euler_to_quat(roll, pitch, yaw, device):
    cR = torch.cos(roll/2); sR = torch.sin(roll/2)
    cP = torch.cos(pitch/2); sP = torch.sin(pitch/2)
    cY = torch.cos(yaw/2); sY = torch.sin(yaw/2)
    w = cR*cP*cY + sR*sP*sY
    x = sR*cP*cY - cR*sP*sY
    y = cR*sP*cY + sR*cP*sY
    z = cR*cP*sY - sR*sP*cY
    return torch.stack([w, x, y, z], dim=1)

def get_ee_pose_world(obj_pos, obj_quat, offset_pos, offset_quat):
    R_obj = quat_to_matrix(obj_quat)
    R_off = quat_to_matrix(offset_quat)
    R_ee = torch.bmm(R_obj, R_off)
    p_off = offset_pos.unsqueeze(2)
    p_ee = torch.bmm(R_obj, p_off).squeeze(2) + obj_pos
    return p_ee, R_ee

def main():
    device = "cuda:0"
    data_path = "test_dataset_strict.pt"
    
    print(f"📂 Loading {data_path}...")
    data = torch.load(data_path, map_location=device)
    
    num_samples = data["base_pose_1"].shape[0]
    print(f"📊 Verifying {num_samples} samples...")
    
    ik_solver = FrankaTensorIK(device=device)
    
    # --- Reconstruct Start Targets ---
    # Sampler Logic Hardcoded
    width = 0.8; half_width = width / 2.0
    
    # Offset 1 (Left): [-0.4, 0, 0], Euler(0, pi/2, 0)
    off_p1 = torch.tensor([-half_width, 0.0, 0.0], device=device).repeat(num_samples, 1)
    off_q1 = euler_to_quat(torch.zeros(num_samples, device=device), 
                           torch.ones(num_samples, device=device)*np.pi/2, 
                           torch.zeros(num_samples, device=device), device)
                           
    # Offset 2 (Right): [0.4, 0, 0], Euler(0, -pi/2, 0)
    off_p2 = torch.tensor([half_width, 0.0, 0.0], device=device).repeat(num_samples, 1)
    off_q2 = euler_to_quat(torch.zeros(num_samples, device=device), 
                           torch.ones(num_samples, device=device)*(-np.pi/2), 
                           torch.zeros(num_samples, device=device), device)
    
    s_obj_p = data["start_obj_pose"][:, :3]
    s_obj_q = data["start_obj_pose"][:, 3:]
    
    # Calculate Expected Start EE World Pose
    s_ee1_target_p, s_ee1_target_R = get_ee_pose_world(s_obj_p, s_obj_q, off_p1, off_q1)
    s_ee2_target_p, s_ee2_target_R = get_ee_pose_world(s_obj_p, s_obj_q, off_p2, off_q2)
    
    # --- Verify Start IK ---
    # Need to convert World FK to Local Base Frame to compare? 
    # Or convert FK Result(Local) to World?
    # Easier: Convert Target World -> Local, then compare with FK(Local)
    
    def to_local(world_p, world_R, base_p, base_q):
        R_base = quat_to_matrix(base_q)
        R_base_T = R_base.transpose(1, 2)
        loc_p = torch.bmm(R_base_T, (world_p - base_p).unsqueeze(2)).squeeze(2)
        loc_R = torch.bmm(R_base_T, world_R)
        return loc_p, loc_R
        
    def check(q, target_p, target_R, name):
        curr_p, curr_R = ik_solver.forward_kinematics(q)
        pos_err = torch.norm(curr_p - target_p, dim=1)
        
        # Rot Error
        R_diff = torch.bmm(curr_R, target_R.transpose(1, 2))
        tr = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
        rot_err = torch.abs(3.0 - tr)
        
        max_p = pos_err.max().item()
        mean_p = pos_err.mean().item()
        max_r = rot_err.max().item()
        
        print(f"[{name}] Pos Err: Max {max_p*1000:.2f}mm, Mean {mean_p*1000:.2f}mm | Rot Err Score: {max_r:.4f}")
        return max_p

    print("\n--- 1. Start Pose Verification ---")
    
    # Robot 1
    s1_loc_p, s1_loc_R = to_local(s_ee1_target_p, s_ee1_target_R, data["base_pose_1"][:,:3], data["base_pose_1"][:,3:])
    check(data["q_start_1"], s1_loc_p, s1_loc_R, "Robot 1 Start")
    
    # Robot 2
    s2_loc_p, s2_loc_R = to_local(s_ee2_target_p, s_ee2_target_R, data["base_pose_2"][:,:3], data["base_pose_2"][:,3:])
    check(data["q_start_2"], s2_loc_p, s2_loc_R, "Robot 2 Start")
    
    print("\n--- 2. Goal Pose Verification ---")
    
    # Goal Targets are stored in dataset (EE Pose World)
    g_ee1_p = data["goal_ee1_pose"][:, :3]
    g_ee1_q = data["goal_ee1_pose"][:, 3:]
    g_ee1_R = quat_to_matrix(g_ee1_q)
    
    g_ee2_p = data["goal_ee2_pose"][:, :3]
    g_ee2_q = data["goal_ee2_pose"][:, 3:]
    g_ee2_R = quat_to_matrix(g_ee2_q)
    
    # Robot 1
    g1_loc_p, g1_loc_R = to_local(g_ee1_p, g_ee1_R, data["base_pose_1"][:,:3], data["base_pose_1"][:,3:])
    check(data["q_goal_1"], g1_loc_p, g1_loc_R, "Robot 1 Goal")
    
    # Robot 2
    g2_loc_p, g2_loc_R = to_local(g_ee2_p, g_ee2_R, data["base_pose_2"][:,:3], data["base_pose_2"][:,3:])
    check(data["q_goal_2"], g2_loc_p, g2_loc_R, "Robot 2 Goal")

if __name__ == "__main__":
    main()
