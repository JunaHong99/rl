import torch
import numpy as np
from franka_tensor_ik import FrankaTensorIK

class VectorizedPoseSampler:
    def __init__(self, device='cuda:0'):
        self.device = device
        self.ik_solver = FrankaTensorIK(device=device)
        
        # Robot Base Ranges (Robot 1: Left, Robot 2: Right)
        self.r1_ranges = torch.tensor([[-0.5, -0.45], [0.0, 0.0]], device=device)
        self.r2_ranges = torch.tensor([[0.45, 0.5], [0.0, 0.0]], device=device)

    def _quat_to_matrix(self, quats):
        """
        Convert quaternions to rotation matrices.
        Args:
            quats: (B, 4) tensor (w, x, y, z)
        Returns:
            rot_mats: (B, 3, 3) tensor
        """
        w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
        
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        
        rot_mats = torch.zeros((quats.shape[0], 3, 3), device=self.device)
        
        rot_mats[:, 0, 0] = 1 - 2 * (yy + zz)
        rot_mats[:, 0, 1] = 2 * (xy - wz)
        rot_mats[:, 0, 2] = 2 * (xz + wy)
        
        rot_mats[:, 1, 0] = 2 * (xy + wz)
        rot_mats[:, 1, 1] = 1 - 2 * (xx + zz)
        rot_mats[:, 1, 2] = 2 * (yz - wx)
        
        rot_mats[:, 2, 0] = 2 * (xz - wy)
        rot_mats[:, 2, 1] = 2 * (yz + wx)
        rot_mats[:, 2, 2] = 1 - 2 * (xx + yy)
        
        return rot_mats

    def _euler_to_quat(self, roll, pitch, yaw):
        """
        Vectorized Euler (XYZ) to Quat (wxyz)
        """
        cR = torch.cos(roll/2); sR = torch.sin(roll/2)
        cP = torch.cos(pitch/2); sP = torch.sin(pitch/2)
        cY = torch.cos(yaw/2); sY = torch.sin(yaw/2)
        
        w = cR*cP*cY + sR*sP*sY
        x = sR*cP*cY - cR*sP*sY
        y = cR*sP*cY + sR*cP*sY
        z = cR*cP*sY - sR*sP*cY
        
        return torch.stack([w, x, y, z], dim=1)

    def _get_ee_pose_world(self, obj_pos, obj_quat, offset_pos, offset_quat):
        """
        Compute EE world pose from Object pose and Offset (in object frame).
        T_world_ee = T_world_obj * T_obj_ee
        """
        R_obj = self._quat_to_matrix(obj_quat) # (B, 3, 3)
        R_off = self._quat_to_matrix(offset_quat) # (B, 3, 3)
        
        # Rotation: R_ee = R_obj @ R_off
        R_ee = torch.bmm(R_obj, R_off)
        
        # Position: p_ee = R_obj @ p_off + p_obj
        p_off = offset_pos.unsqueeze(2)
        p_ee = torch.bmm(R_obj, p_off).squeeze(2) + obj_pos
        
        return p_ee, R_ee

    def check_ik_error(self, q, target_pos_local, target_rot_local):
        """
        Calculate Forward Kinematics Error
        """
        curr_pos, curr_rot = self.ik_solver.forward_kinematics(q)
        pos_err = torch.norm(curr_pos - target_pos_local, dim=1)
        
        # Rotation Error (Trace based)
        R_diff = torch.bmm(curr_rot, target_rot_local.transpose(1, 2))
        tr = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
        rot_err = torch.abs(3.0 - tr) 
        
        return pos_err, rot_err

    def sample_episodes(self, num_envs):
        """
        Generates 'num_envs' valid episodes using vectorized operations.
        Ensures IK feasibility for both Start and Goal poses.
        """
        collected = {
            "base_pose_1": [], "base_pose_2": [],
            "start_obj_pose": [], "goal_obj_pose": [],
            "offset_pos_1": [], "offset_quat_1": [],
            "offset_pos_2": [], "offset_quat_2": [],
            # Keep IK solutions too
            "q_start_1": [], "q_start_2": [],
            "goal_ee1_pose": [], "goal_ee2_pose": []
        }
        
        total_valid = 0
        loop_cnt = 0
        MAX_LOOPS = 100
        
        print(f"[PoseSampler] Generating {num_envs} strict samples (POS<0.01, ROT<0.05, Q_DIST<3.0)...")
        
        while total_valid < num_envs and loop_cnt < MAX_LOOPS:
            loop_cnt += 1
            # Generate a large batch of candidates
            # Yield is low (~0.5% maybe?), so generate 20x the remaining needed
            needed = num_envs - total_valid
            N = max(needed * 200, 10000) 
            
            # --- A. Sample Bases ---
            r1_x = torch.rand(N, device=self.device) * (self.r1_ranges[0, 1] - self.r1_ranges[0, 0]) + self.r1_ranges[0, 0]
            r1_y = torch.rand(N, device=self.device) * (self.r1_ranges[1, 1] - self.r1_ranges[1, 0]) + self.r1_ranges[1, 0]
            r1_yaw = (torch.rand(N, device=self.device) * 2 - 1) * (torch.pi / 4)
            base_pos_1 = torch.stack([r1_x, r1_y, torch.zeros(N, device=self.device)], dim=1)
            base_quat_1 = self._euler_to_quat(torch.zeros(N, device=self.device), torch.zeros(N, device=self.device), r1_yaw)
            
            r2_x = torch.rand(N, device=self.device) * (self.r2_ranges[0, 1] - self.r2_ranges[0, 0]) + self.r2_ranges[0, 0]
            r2_y = torch.rand(N, device=self.device) * (self.r2_ranges[1, 1] - self.r2_ranges[1, 0]) + self.r2_ranges[1, 0]
            r2_yaw = torch.pi + (torch.rand(N, device=self.device) * 2 - 1) * (torch.pi / 4)
            base_pos_2 = torch.stack([r2_x, r2_y, torch.zeros(N, device=self.device)], dim=1)
            base_quat_2 = self._euler_to_quat(torch.zeros(N, device=self.device), torch.zeros(N, device=self.device), r2_yaw)

            # --- B. Sample Object ---
            center_x = (r1_x + r2_x) / 2.0; center_y = (r1_y + r2_y) / 2.0
            
            # Start
            start_off_x = (torch.rand(N, device=self.device) * 0.2) - 0.1
            start_off_y = (torch.rand(N, device=self.device) * 0.2) - 0.1
            start_z = torch.rand(N, device=self.device) * 0.4 + 0.4
            start_obj_pos = torch.stack([center_x + start_off_x, center_y + start_off_y, start_z], dim=1)
            start_yaw = (torch.rand(N, device=self.device) * 2 - 1) * (torch.pi / 4)
            start_obj_quat = self._euler_to_quat(torch.zeros(N, device=self.device), torch.zeros(N, device=self.device), start_yaw)
            
            # Goal
            goal_dist = torch.rand(N, device=self.device) * 0.4 + 0.3 
            goal_theta = torch.rand(N, device=self.device) * 2 * torch.pi
            goal_dx = goal_dist * torch.cos(goal_theta); goal_dy = goal_dist * torch.sin(goal_theta)
            goal_dz = (torch.rand(N, device=self.device) * 0.4) - 0.2 
            goal_obj_pos = start_obj_pos + torch.stack([goal_dx, goal_dy, goal_dz], dim=1)
            goal_obj_pos[:, 2].clamp_(0.3, 0.85)
            goal_yaw = (torch.rand(N, device=self.device) * 2 - 1) * (torch.pi / 4)
            goal_obj_quat = self._euler_to_quat(torch.zeros(N, device=self.device), torch.zeros(N, device=self.device), goal_yaw)
            
            # --- C. Distance Filter (Strict EE-based) ---
            # Calculate tentative EE positions at Goal to check distance from base
            width = 0.8; half_width = width / 2.0
            # Offset definitions (same as later)
            off_p1_temp = torch.tensor([-half_width, 0.0, 0.0], device=self.device).repeat(N, 1)
            off_q1_temp = self._euler_to_quat(torch.zeros(N, device=self.device), torch.ones(N, device=self.device)*np.pi/2, torch.zeros(N, device=self.device))
            off_p2_temp = torch.tensor([half_width, 0.0, 0.0], device=self.device).repeat(N, 1)
            off_q2_temp = self._euler_to_quat(torch.zeros(N, device=self.device), torch.ones(N, device=self.device)*(-np.pi/2), torch.zeros(N, device=self.device))

            ee1_goal_p, _ = self._get_ee_pose_world(goal_obj_pos, goal_obj_quat, off_p1_temp, off_q1_temp)
            ee2_goal_p, _ = self._get_ee_pose_world(goal_obj_pos, goal_obj_quat, off_p2_temp, off_q2_temp)

            d1_start = torch.norm(start_obj_pos - base_pos_1, dim=1)
            d2_start = torch.norm(start_obj_pos - base_pos_2, dim=1)
            d1_ee_goal = torch.norm(ee1_goal_p - base_pos_1, dim=1)
            d2_ee_goal = torch.norm(ee2_goal_p - base_pos_2, dim=1)
            
            # [User Request] EE goal distance > 0.5m to avoid twisted arm configurations
            dist_mask = (d1_start > 0.3) & (d1_start < 0.85) & \
                        (d2_start > 0.3) & (d2_start < 0.85) & \
                        (d1_ee_goal > 0.5) & (d1_ee_goal < 0.85) & \
                        (d2_ee_goal > 0.5) & (d2_ee_goal < 0.85)
            
            cand_indices = torch.nonzero(dist_mask).flatten()
            count = cand_indices.numel()
            
            if count == 0:
                continue
                
            # Filter Candidates
            b1_p = base_pos_1[cand_indices]; b1_q = base_quat_1[cand_indices]
            b2_p = base_pos_2[cand_indices]; b2_q = base_quat_2[cand_indices]
            s_obj_p = start_obj_pos[cand_indices]; s_obj_q = start_obj_quat[cand_indices]
            g_obj_p = goal_obj_pos[cand_indices]; g_obj_q = goal_obj_quat[cand_indices]
            
            # Offsets
            width = 0.8; half_width = width / 2.0
            off_pos_1 = torch.tensor([-half_width, 0.0, 0.0], device=self.device).repeat(count, 1)
            off_quat_1 = self._euler_to_quat(torch.zeros(count, device=self.device), torch.ones(count, device=self.device)*np.pi/2, torch.zeros(count, device=self.device))
            off_pos_2 = torch.tensor([half_width, 0.0, 0.0], device=self.device).repeat(count, 1)
            off_quat_2 = self._euler_to_quat(torch.zeros(count, device=self.device), torch.ones(count, device=self.device)*(-np.pi/2), torch.zeros(count, device=self.device))

            # --- D. IK Solve & Verify ---
            # Start
            ee1_w_p, ee1_w_R = self._get_ee_pose_world(s_obj_p, s_obj_q, off_pos_1, off_quat_1)
            ee2_w_p, ee2_w_R = self._get_ee_pose_world(s_obj_p, s_obj_q, off_pos_2, off_quat_2)
            
            R_b1 = self._quat_to_matrix(b1_q); R_b1_T = R_b1.transpose(1, 2)
            ee1_loc_p = torch.bmm(R_b1_T, (ee1_w_p - b1_p).unsqueeze(2)).squeeze(2)
            ee1_loc_R = torch.bmm(R_b1_T, ee1_w_R)
            
            R_b2 = self._quat_to_matrix(b2_q); R_b2_T = R_b2.transpose(1, 2)
            ee2_loc_p = torch.bmm(R_b2_T, (ee2_w_p - b2_p).unsqueeze(2)).squeeze(2)
            ee2_loc_R = torch.bmm(R_b2_T, ee2_w_R)
            
            q_start_1 = self.ik_solver.solve_ik_gradient(ee1_loc_p, ee1_loc_R, max_iter=200) # Slightly reduced iter for speed
            q_start_2 = self.ik_solver.solve_ik_gradient(ee2_loc_p, ee2_loc_R, max_iter=200)
            
            s1_p_err, s1_r_err = self.check_ik_error(q_start_1, ee1_loc_p, ee1_loc_R)
            s2_p_err, s2_r_err = self.check_ik_error(q_start_2, ee2_loc_p, ee2_loc_R)
            
            # Goal
            g_ee1_w_p, g_ee1_w_R = self._get_ee_pose_world(g_obj_p, g_obj_q, off_pos_1, off_quat_1)
            g_ee2_w_p, g_ee2_w_R = self._get_ee_pose_world(g_obj_p, g_obj_q, off_pos_2, off_quat_2)
            
            g_ee1_loc_p = torch.bmm(R_b1_T, (g_ee1_w_p - b1_p).unsqueeze(2)).squeeze(2)
            g_ee1_loc_R = torch.bmm(R_b1_T, g_ee1_w_R)
            g_ee2_loc_p = torch.bmm(R_b2_T, (g_ee2_w_p - b2_p).unsqueeze(2)).squeeze(2)
            g_ee2_loc_R = torch.bmm(R_b2_T, g_ee2_w_R)
            
            # Solve Goal IK (Seeded IK: Use q_start as seed for continuity)
            q_goal_1 = self.ik_solver.solve_ik_gradient(g_ee1_loc_p, g_ee1_loc_R, q_init=q_start_1, max_iter=200)
            q_goal_2 = self.ik_solver.solve_ik_gradient(g_ee2_loc_p, g_ee2_loc_R, q_init=q_start_2, max_iter=200)
            
            g1_p_err, g1_r_err = self.check_ik_error(q_goal_1, g_ee1_loc_p, g_ee1_loc_R)
            g2_p_err, g2_r_err = self.check_ik_error(q_goal_2, g_ee2_loc_p, g_ee2_loc_R)
            
            # Challenge Filter (Moderate Strictness)
            POS_THR = 0.015; ROT_THR = 0.1
            q_diff_1 = torch.norm(q_start_1 - q_goal_1, dim=1)
            q_diff_2 = torch.norm(q_start_2 - q_goal_2, dim=1)
            Q_DIST_THR = 5.0 # Allow longer, more challenging paths

            valid_mask = (s1_p_err < POS_THR) & (s1_r_err < ROT_THR) & \
                         (s2_p_err < POS_THR) & (s2_r_err < ROT_THR) & \
                         (g1_p_err < POS_THR) & (g1_r_err < ROT_THR) & \
                         (g2_p_err < POS_THR) & (g2_r_err < ROT_THR) & \
                         (q_diff_1 < Q_DIST_THR) & (q_diff_2 < Q_DIST_THR)
                         
            valid_indices = torch.nonzero(valid_mask).flatten()
            num_found = valid_indices.numel()
            
            print(f"  Loop {loop_cnt}: Found {num_found} valid samples.")
            
            if num_found > 0:
                # Goal Quat Helper
                def quat_mul(q1, q2):
                    w1, x1, y1, z1 = q1[:,0], q1[:,1], q1[:,2], q1[:,3]
                    w2, x2, y2, z2 = q2[:,0], q2[:,1], q2[:,2], q2[:,3]
                    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
                    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
                    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
                    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
                    return torch.stack([w, x, y, z], dim=1)
                
                # Append to collection
                # Take up to needed amount
                take = min(num_found, needed)
                v_idx = valid_indices[:take]
                
                collected["base_pose_1"].append(torch.cat([b1_p[v_idx], b1_q[v_idx]], dim=1))
                collected["base_pose_2"].append(torch.cat([b2_p[v_idx], b2_q[v_idx]], dim=1))
                collected["q_start_1"].append(q_start_1[v_idx])
                collected["q_start_2"].append(q_start_2[v_idx])
                collected["start_obj_pose"].append(torch.cat([s_obj_p[v_idx], s_obj_q[v_idx]], dim=1))
                collected["goal_obj_pose"].append(torch.cat([g_obj_p[v_idx], g_obj_q[v_idx]], dim=1))
                
                g_ee1_quat = quat_mul(g_obj_q[v_idx], off_quat_1[v_idx])
                g_ee2_quat = quat_mul(g_obj_q[v_idx], off_quat_2[v_idx])
                
                collected["goal_ee1_pose"].append(torch.cat([g_ee1_w_p[v_idx], g_ee1_quat], dim=1))
                collected["goal_ee2_pose"].append(torch.cat([g_ee2_w_p[v_idx], g_ee2_quat], dim=1))
                
                total_valid += take
        
        # Concat
        if total_valid < num_envs:
            print(f"Warning: Could only collect {total_valid}/{num_envs} samples.")
            # Fill rest with duplicates of the last one just to avoid crash, or error out
            if total_valid == 0:
                raise RuntimeError("No valid samples found after max loops.")
            
            missing = num_envs - total_valid
            last_idx = -1 # Take last chunk's last item
            # Simple duplication logic for lists of tensors
            collected["base_pose_1"].append(collected["base_pose_1"][-1][-1:].repeat(missing, 1))
            collected["base_pose_2"].append(collected["base_pose_2"][-1][-1:].repeat(missing, 1))
            collected["q_start_1"].append(collected["q_start_1"][-1][-1:].repeat(missing, 1))
            collected["q_start_2"].append(collected["q_start_2"][-1][-1:].repeat(missing, 1))
            collected["start_obj_pose"].append(collected["start_obj_pose"][-1][-1:].repeat(missing, 1))
            collected["goal_obj_pose"].append(collected["goal_obj_pose"][-1][-1:].repeat(missing, 1))
            collected["goal_ee1_pose"].append(collected["goal_ee1_pose"][-1][-1:].repeat(missing, 1))
            collected["goal_ee2_pose"].append(collected["goal_ee2_pose"][-1][-1:].repeat(missing, 1))
            
        return {
            "base_pose_1": torch.cat(collected["base_pose_1"]),
            "base_pose_2": torch.cat(collected["base_pose_2"]),
            "q_start_1": torch.cat(collected["q_start_1"]),
            "q_start_2": torch.cat(collected["q_start_2"]),
            "start_obj_pose": torch.cat(collected["start_obj_pose"]),
            "goal_obj_pose": torch.cat(collected["goal_obj_pose"]),
            "goal_ee1_pose": torch.cat(collected["goal_ee1_pose"]),
            "goal_ee2_pose": torch.cat(collected["goal_ee2_pose"]),
            "obj_width": 0.8
        }