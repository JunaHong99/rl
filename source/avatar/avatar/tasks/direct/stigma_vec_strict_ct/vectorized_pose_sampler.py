
import torch
import numpy as np
from franka_tensor_ik import FrankaTensorIK

class VectorizedPoseSampler:
    def __init__(self, device='cuda:0'):
        self.device = device
        self.ik_solver = FrankaTensorIK(device=device)
        
        # Robot Base Ranges (from original sampler)
        self.r1_x_range = [-0.5, -0.45]
        self.r1_y_range = [-0.1, 0.1] # Original was 0.0, 0.0 but let's give it slack if needed, or stick to strict
        # Checking original: r1_pos_range = ([-0.5, -0.45], [0.0, 0.0])
        # It seems Y is fixed to 0.0? "[-0.5, -0.45], [0.0, 0.0]" suggests range[1] is [0,0].
        # I will strictly follow the original ranges.
        
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
        # Assuming roll, pitch, yaw are tensors of shape (B,)
        cR = torch.cos(roll/2); sR = torch.sin(roll/2)
        cP = torch.cos(pitch/2); sP = torch.sin(pitch/2)
        cY = torch.cos(yaw/2); sY = torch.sin(yaw/2)
        
        w = cR*cP*cY + sR*sP*sY
        x = sR*cP*cY - cR*sP*sY
        y = cR*sP*cY + sR*cP*sY
        z = cR*cP*sY - sR*sP*cY
        
        return torch.stack([w, x, y, z], dim=1)

    def _apply_transform(self, pos, quat, target_pos):
        """
        Apply transform (pos, quat) to target_pos
        T * p = R * p + t
        """
        # This function seems unused or needs clarification. 
        # We usually need local to world or world to local.
        pass

    def _get_ee_pose_world(self, obj_pos, obj_quat, offset_pos, offset_quat):
        """
        Compute EE world pose from Object pose and Offset (in object frame).
        T_world_ee = T_world_obj * T_obj_ee
        """
        B = obj_pos.shape[0]
        
        # Convert Quats to Matrices
        R_obj = self._quat_to_matrix(obj_quat) # (B, 3, 3)
        R_off = self._quat_to_matrix(offset_quat) # (B, 3, 3)
        
        # Rotation: R_ee = R_obj @ R_off
        R_ee = torch.bmm(R_obj, R_off)
        
        # Position: p_ee = R_obj @ p_off + p_obj
        # p_off needs to be (B, 3, 1)
        p_off = offset_pos.unsqueeze(2)
        p_ee = torch.bmm(R_obj, p_off).squeeze(2) + obj_pos
        
        return p_ee, R_ee

    def sample_episodes(self, num_envs):
        """
        Generates 'num_envs' valid episodes using vectorized operations.
        Accumulates valid samples across iterations until enough are found.
        """
        # Storage for accumulated valid samples
        collected = {
            "base_pose_1": [], "base_pose_2": [],
            "start_obj_pose": [], "goal_obj_pose": [],
            "goal_ee1_pose_world": [], "goal_ee1_rot_world": [], # Intermediate storage
            "goal_ee2_pose_world": [], "goal_ee2_rot_world": [],
            "offset_pos_1": [], "offset_quat_1": [],
            "offset_pos_2": [], "offset_quat_2": []
        }
        
        total_collected = 0
        attempt_count = 0
        
        while total_collected < num_envs:
            attempt_count += 1
            
            # 1. Oversample candidates
            # Calculate how many more we need, but keep a minimum batch size for efficiency
            needed = num_envs - total_collected
            # Heuristic: if success rate is ~2.5% (2000/80000), we need ~40x the needed count.
            # Let's be safe and sample plenty.
            factor = 40 
            N = max(needed * factor, 5000)
            
            # -------------------------------------------------------
            # A. Sample Bases
            # -------------------------------------------------------
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

            # -------------------------------------------------------
            # B. Sample Object (Start & Goal)
            # -------------------------------------------------------
            center_x = (r1_x + r2_x) / 2.0
            center_y = (r1_y + r2_y) / 2.0
            
            # Start Pose
            start_off_x = (torch.rand(N, device=self.device) * 0.2) - 0.1
            start_off_y = (torch.rand(N, device=self.device) * 0.2) - 0.1
            start_z = torch.rand(N, device=self.device) * 0.4 + 0.4
            
            start_obj_pos = torch.stack([center_x + start_off_x, center_y + start_off_y, start_z], dim=1)
            start_yaw = (torch.rand(N, device=self.device) * 2 - 1) * (torch.pi / 4)
            start_obj_quat = self._euler_to_quat(torch.zeros(N, device=self.device), torch.zeros(N, device=self.device), start_yaw)
            
            # Goal Pose
            goal_dist = torch.rand(N, device=self.device) * 0.4 + 0.3 
            goal_theta = torch.rand(N, device=self.device) * 2 * torch.pi
            
            goal_dx = goal_dist * torch.cos(goal_theta)
            goal_dy = goal_dist * torch.sin(goal_theta)
            goal_dz = (torch.rand(N, device=self.device) * 0.4) - 0.2 
            
            goal_obj_pos = start_obj_pos + torch.stack([goal_dx, goal_dy, goal_dz], dim=1)
            goal_obj_pos[:, 2].clamp_(0.3, 0.85)
            
            goal_yaw = (torch.rand(N, device=self.device) * 2 - 1) * (torch.pi / 4)
            goal_obj_quat = self._euler_to_quat(torch.zeros(N, device=self.device), torch.zeros(N, device=self.device), goal_yaw)
            
            # -------------------------------------------------------
            # C. Distance Filter
            # -------------------------------------------------------
            d1_start = torch.norm(start_obj_pos - base_pos_1, dim=1)
            d2_start = torch.norm(start_obj_pos - base_pos_2, dim=1)
            d1_goal = torch.norm(goal_obj_pos - base_pos_1, dim=1)
            d2_goal = torch.norm(goal_obj_pos - base_pos_2, dim=1)
            
            valid_mask = (d1_start > 0.35) & (d1_start < 0.80) & \
                        (d2_start > 0.35) & (d2_start < 0.80) & \
                        (d1_goal > 0.35) & (d1_goal < 0.80) & \
                        (d2_goal > 0.35) & (d2_goal < 0.80)
            
            indices = torch.nonzero(valid_mask).flatten()
            
            if indices.numel() > 0:
                # Prepare offsets for this batch (needed for storage)
                width = 0.8
                half_width = width / 2.0
                
                # [Fix] Add random noise to grasp orientation for robustness
                # Matches original sampler: +/- 0.05 rad noise on R, P, Y
                noise_range = 0.05
                count = indices.numel()
                
                # Robot 1 (Left): Base (0, pi/2, 0) + Noise
                r_noise = (torch.rand(count, device=self.device) * 2 - 1) * noise_range
                p_noise = (torch.rand(count, device=self.device) * 2 - 1) * noise_range
                y_noise = (torch.rand(count, device=self.device) * 2 - 1) * noise_range
                
                off_pos_1 = torch.tensor([-half_width, 0.0, 0.0], device=self.device).repeat(count, 1)
                off_quat_1 = self._euler_to_quat(
                    torch.zeros(count, device=self.device) + r_noise, 
                    torch.ones(count, device=self.device)*np.pi/2 + p_noise, 
                    torch.zeros(count, device=self.device) + y_noise
                )

                # Robot 2 (Right): Base (0, -pi/2, 0) + Noise
                r_noise = (torch.rand(count, device=self.device) * 2 - 1) * noise_range
                p_noise = (torch.rand(count, device=self.device) * 2 - 1) * noise_range
                y_noise = (torch.rand(count, device=self.device) * 2 - 1) * noise_range

                off_pos_2 = torch.tensor([half_width, 0.0, 0.0], device=self.device).repeat(count, 1)
                off_quat_2 = self._euler_to_quat(
                    torch.zeros(count, device=self.device) + r_noise, 
                    torch.ones(count, device=self.device)*(-np.pi/2) + p_noise, 
                    torch.zeros(count, device=self.device) + y_noise
                )

                # Extract valid data
                collected["base_pose_1"].append(torch.cat([base_pos_1[indices], base_quat_1[indices]], dim=1))
                collected["base_pose_2"].append(torch.cat([base_pos_2[indices], base_quat_2[indices]], dim=1))
                collected["start_obj_pose"].append(torch.cat([start_obj_pos[indices], start_obj_quat[indices]], dim=1))
                collected["goal_obj_pose"].append(torch.cat([goal_obj_pos[indices], goal_obj_quat[indices]], dim=1))
                
                # Store offsets to use later
                collected["offset_pos_1"].append(off_pos_1)
                collected["offset_quat_1"].append(off_quat_1)
                collected["offset_pos_2"].append(off_pos_2)
                collected["offset_quat_2"].append(off_quat_2)
                
                total_collected += indices.numel()
                # print(f"[Sampler] Collected {total_collected}/{num_envs}")

        # -------------------------------------------------------
        # D. Aggregate & IK Solve
        # -------------------------------------------------------
        # Concatenate all collected lists
        b1_pose = torch.cat(collected["base_pose_1"])[:num_envs]
        b2_pose = torch.cat(collected["base_pose_2"])[:num_envs]
        s_obj_pose = torch.cat(collected["start_obj_pose"])[:num_envs]
        g_obj_pose = torch.cat(collected["goal_obj_pose"])[:num_envs]
        
        off_p1 = torch.cat(collected["offset_pos_1"])[:num_envs]
        off_q1 = torch.cat(collected["offset_quat_1"])[:num_envs]
        off_p2 = torch.cat(collected["offset_pos_2"])[:num_envs]
        off_q2 = torch.cat(collected["offset_quat_2"])[:num_envs]

        # Unpack poses for IK
        b1_p = b1_pose[:, :3]; b1_q = b1_pose[:, 3:]
        b2_p = b2_pose[:, :3]; b2_q = b2_pose[:, 3:]
        s_obj_p = s_obj_pose[:, :3]; s_obj_q = s_obj_pose[:, 3:]
        g_obj_p = g_obj_pose[:, :3]; g_obj_q = g_obj_pose[:, 3:]
        
        # 2. Compute World EE Poses for Start
        ee1_world_pos, ee1_world_rot = self._get_ee_pose_world(s_obj_p, s_obj_q, off_p1, off_q1)
        ee2_world_pos, ee2_world_rot = self._get_ee_pose_world(s_obj_p, s_obj_q, off_p2, off_q2)
        
        # 3. Transform World EE -> Local EE (Base Frame)
        R_b1 = self._quat_to_matrix(b1_q)
        R_b1_T = R_b1.transpose(1, 2)
        ee1_loc_pos = torch.bmm(R_b1_T, (ee1_world_pos - b1_p).unsqueeze(2)).squeeze(2)
        ee1_loc_rot = torch.bmm(R_b1_T, ee1_world_rot)
        
        R_b2 = self._quat_to_matrix(b2_q)
        R_b2_T = R_b2.transpose(1, 2)
        ee2_loc_pos = torch.bmm(R_b2_T, (ee2_world_pos - b2_p).unsqueeze(2)).squeeze(2)
        ee2_loc_rot = torch.bmm(R_b2_T, ee2_world_rot)
        
        # 4. Solve IK
        q_start_1 = self.ik_solver.solve_ik_gradient(ee1_loc_pos, ee1_loc_rot, max_iter=50)
        q_start_2 = self.ik_solver.solve_ik_gradient(ee2_loc_pos, ee2_loc_rot, max_iter=50)
        
        # 5. Compute Goal EE Poses
        g_ee1_world_pos, g_ee1_world_rot = self._get_ee_pose_world(g_obj_p, g_obj_q, off_p1, off_q1)
        g_ee2_world_pos, g_ee2_world_rot = self._get_ee_pose_world(g_obj_p, g_obj_q, off_p2, off_q2)
        
        def quat_mul(q1, q2):
            w1, x1, y1, z1 = q1[:,0], q1[:,1], q1[:,2], q1[:,3]
            w2, x2, y2, z2 = q2[:,0], q2[:,1], q2[:,2], q2[:,3]
            w = w1*w2 - x1*x2 - y1*y2 - z1*z2
            x = w1*x2 + x1*w2 + y1*z2 - z1*y2
            y = w1*y2 - x1*z2 + y1*w2 + z1*x2
            z = w1*z2 + x1*y2 - y1*x2 + z1*w2
            return torch.stack([w, x, y, z], dim=1)
            
        g_ee1_quat = quat_mul(g_obj_q, off_q1)
        g_ee2_quat = quat_mul(g_obj_q, off_q2)
        
        return {
            "base_pose_1": b1_pose,
            "base_pose_2": b2_pose,
            "q_start_1": q_start_1,
            "q_start_2": q_start_2,
            "start_obj_pose": s_obj_pose,
            "goal_obj_pose": g_obj_pose,
            "goal_ee1_pose": torch.cat([g_ee1_world_pos, g_ee1_quat], dim=1),
            "goal_ee2_pose": torch.cat([g_ee2_world_pos, g_ee2_quat], dim=1),
            "obj_width": 0.8
        }

        # -------------------------------------------------------
        # D. IK Solve (Vectorized Gradient Descent)
        # -------------------------------------------------------
        # We need q_start_1, q_start_2 (for Start Pose)
        # We technically don't need q_goal_1, q_goal_2 unless we reset to goal?
        # The env usually resets to start.
        # But for 'target_ee_rel_poses' we might need valid goal EE poses.
        
        # 1. Define Offsets
        width = 0.8
        half_width = width / 2.0
        
        # Offset 1 (Left): [-0.4, 0, 0] rotated 90 deg around Y? 
        # Wait, original sampler used random grasp quats.
        # Let's fix them for stability or randomize them slightly.
        # Original: "0.0, pi/4, 0.0" noise range.
        # Let's use fixed standard grasp for now: Left side facing center, Right side facing center.
        # Left Side (Robot 1): Pos [-0.4, 0, 0] (in Obj frame). Rot: Hand points +X?
        # Robot Hand usually points +Z.
        # We need EE to be at [-0.4, 0, 0] and pointing towards +X.
        # Let's trust the logic from `_get_ee_pose_from_object`.
        
        # Construct Offset tensors (B, 3) and (B, 4)
        off_pos_1 = torch.tensor([-half_width, 0.0, 0.0], device=self.device).repeat(num_envs, 1)
        # Euler(0, pi/2, 0)
        off_quat_1 = self._euler_to_quat(torch.zeros(num_envs, device=self.device), 
                                         torch.ones(num_envs, device=self.device)*np.pi/2, 
                                         torch.zeros(num_envs, device=self.device)) #.repeat(num_envs, 1)

        off_pos_2 = torch.tensor([half_width, 0.0, 0.0], device=self.device).repeat(num_envs, 1)
        # Euler(0, -pi/2, 0)
        off_quat_2 = self._euler_to_quat(torch.zeros(num_envs, device=self.device), 
                                         torch.ones(num_envs, device=self.device)*(-np.pi/2), 
                                         torch.zeros(num_envs, device=self.device))
        
        # 2. Compute World EE Poses for Start
        ee1_world_pos, ee1_world_rot = self._get_ee_pose_world(s_obj_pos, s_obj_quat, off_pos_1, off_quat_1)
        ee2_world_pos, ee2_world_rot = self._get_ee_pose_world(s_obj_pos, s_obj_quat, off_pos_2, off_quat_2)
        
        # 3. Transform World EE -> Local EE (Base Frame)
        # T_base_world = [R_base^T, -R_base^T * p_base]
        # p_local = R_base^T * (p_world - p_base)
        # R_local = R_base^T * R_world
        
        # Base 1
        R_b1 = self._quat_to_matrix(b1_quat)
        R_b1_T = R_b1.transpose(1, 2)
        ee1_loc_pos = torch.bmm(R_b1_T, (ee1_world_pos - b1_pos).unsqueeze(2)).squeeze(2)
        ee1_loc_rot = torch.bmm(R_b1_T, ee1_world_rot)
        
        # Base 2
        R_b2 = self._quat_to_matrix(b2_quat)
        R_b2_T = R_b2.transpose(1, 2)
        ee2_loc_pos = torch.bmm(R_b2_T, (ee2_world_pos - b2_pos).unsqueeze(2)).squeeze(2)
        ee2_loc_rot = torch.bmm(R_b2_T, ee2_world_rot)
        
        # 4. Solve IK
        # This is the heavy part, but it's batched on GPU.
        q_start_1 = self.ik_solver.solve_ik_gradient(ee1_loc_pos, ee1_loc_rot, max_iter=50)
        q_start_2 = self.ik_solver.solve_ik_gradient(ee2_loc_pos, ee2_loc_rot, max_iter=50)
        
        # 5. Compute Goal EE Poses (for env target buffer)
        g_ee1_world_pos, g_ee1_world_rot = self._get_ee_pose_world(g_obj_pos, g_obj_quat, off_pos_1, off_quat_1)
        g_ee2_world_pos, g_ee2_world_rot = self._get_ee_pose_world(g_obj_pos, g_obj_quat, off_pos_2, off_quat_2)
        
        # We need Quaternions for Goal EE
        # Conversion Matrix -> Quat is annoying.
        # But wait, we have `g_obj_quat` and `off_quat`. 
        # q_ee = q_obj * q_off
        # Let's do Quaternion multiplication instead of Matrix->Quat
        # quaternion multiplication: q1 * q2
        # (w1, v1) * (w2, v2) = (w1w2 - v1.v2, w1v2 + w2v1 + v1 x v2)
        
        def quat_mul(q1, q2):
            w1, x1, y1, z1 = q1[:,0], q1[:,1], q1[:,2], q1[:,3]
            w2, x2, y2, z2 = q2[:,0], q2[:,1], q2[:,2], q2[:,3]
            w = w1*w2 - x1*x2 - y1*y2 - z1*z2
            x = w1*x2 + x1*w2 + y1*z2 - z1*y2
            y = w1*y2 - x1*z2 + y1*w2 + z1*x2
            z = w1*z2 + x1*y2 - y1*x2 + z1*w2
            return torch.stack([w, x, y, z], dim=1)
            
        g_ee1_quat = quat_mul(g_obj_quat, off_quat_1)
        g_ee2_quat = quat_mul(g_obj_quat, off_quat_2)
        
        # -------------------------------------------------------
        # E. Pack Data
        # -------------------------------------------------------
        return {
            "base_pose_1": torch.cat([b1_pos, b1_quat], dim=1),
            "base_pose_2": torch.cat([b2_pos, b2_quat], dim=1),
            "q_start_1": q_start_1,
            "q_start_2": q_start_2,
            "start_obj_pose": torch.cat([s_obj_pos, s_obj_quat], dim=1),
            "goal_obj_pose": torch.cat([g_obj_pos, g_obj_quat], dim=1),
            "goal_ee1_pose": torch.cat([g_ee1_world_pos, g_ee1_quat], dim=1),
            "goal_ee2_pose": torch.cat([g_ee2_world_pos, g_ee2_quat], dim=1),
            "obj_width": width
        }
