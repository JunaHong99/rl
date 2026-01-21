
import torch
import numpy as np
from franka_tensor_ik import FrankaTensorIK

class VectorizedPoseSampler:
    def __init__(self, device='cuda:0'):
        self.device = device
        self.ik_solver = FrankaTensorIK(device=device)
        
        # Robot Base Ranges
        self.r1_ranges = torch.tensor([[-0.5, -0.45], [0.0, 0.0]], device=device)
        self.r2_ranges = torch.tensor([[0.45, 0.5], [0.0, 0.0]], device=device)

    def _quat_to_matrix(self, quats):
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
        cR = torch.cos(roll/2); sR = torch.sin(roll/2)
        cP = torch.cos(pitch/2); sP = torch.sin(pitch/2)
        cY = torch.cos(yaw/2); sY = torch.sin(yaw/2)
        w = cR*cP*cY + sR*sP*sY
        x = sR*cP*cY - cR*sP*sY
        y = cR*sP*cY + sR*cP*sY
        z = cR*cP*sY - sR*sP*cY
        return torch.stack([w, x, y, z], dim=1)

    def _get_ee_pose_world(self, obj_pos, obj_quat, offset_pos, offset_quat):
        R_obj = self._quat_to_matrix(obj_quat)
        p_ee = torch.bmm(R_obj, offset_pos.unsqueeze(2)).squeeze(2) + obj_pos
        R_off = self._quat_to_matrix(offset_quat)
        R_ee = torch.bmm(R_obj, R_off)
        return p_ee, R_ee

    def sample_episodes(self, num_envs):
        collected = {
            "base_pose_1": [], "base_pose_2": [],
            "start_obj_pose": [], "goal_obj_pose": [],
            "q_start_1": [], "q_start_2": [],
            "goal_ee1_pose": [], "goal_ee2_pose": [],
            "offset_pos_1": [], "offset_quat_1": [],
            "offset_pos_2": [], "offset_quat_2": []
        }
        total_collected = 0
        attempt_count = 0
        
        while total_collected < num_envs:
            attempt_count += 1
            needed = num_envs - total_collected
            # For efficiency, if needed is small, still sample enough to get hits
            N = max(needed * 30, 1000) 
            
            # 1. Sample Bases & Object Poses (Start & Goal)
            r1_x = torch.rand(N, device=self.device) * (self.r1_ranges[0, 1] - self.r1_ranges[0, 0]) + self.r1_ranges[0, 0]
            r1_y = torch.rand(N, device=self.device) * (self.r1_ranges[1, 1] - self.r1_ranges[1, 0]) + self.r1_ranges[1, 0]
            r1_yaw = (torch.rand(N, device=self.device) * 2 - 1) * (torch.pi / 4)
            b1_p = torch.stack([r1_x, r1_y, torch.zeros(N, device=self.device)], dim=1)
            b1_q = self._euler_to_quat(torch.zeros(N, device=self.device), torch.zeros(N, device=self.device), r1_yaw)
            
            r2_x = torch.rand(N, device=self.device) * (self.r2_ranges[0, 1] - self.r2_ranges[0, 0]) + self.r2_ranges[0, 0]
            r2_y = torch.rand(N, device=self.device) * (self.r2_ranges[1, 1] - self.r2_ranges[1, 0]) + self.r2_ranges[1, 0]
            r2_yaw = torch.pi + (torch.rand(N, device=self.device) * 2 - 1) * (torch.pi / 4)
            b2_p = torch.stack([r2_x, r2_y, torch.zeros(N, device=self.device)], dim=1)
            b2_q = self._euler_to_quat(torch.zeros(N, device=self.device), torch.zeros(N, device=self.device), r2_yaw)

            center_x = (r1_x + r2_x) / 2.0; center_y = (r1_y + r2_y) / 2.0
            s_obj_p = torch.stack([center_x + (torch.rand(N, device=self.device)*0.2-0.1), center_y + (torch.rand(N, device=self.device)*0.2-0.1), torch.rand(N, device=self.device)*0.4+0.4], dim=1)
            s_obj_q = self._euler_to_quat(torch.zeros(N, device=self.device), torch.zeros(N, device=self.device), (torch.rand(N, device=self.device)*2-1)*(torch.pi/4))
            
            goal_dist = torch.rand(N, device=self.device) * 0.4 + 0.3 
            goal_theta = torch.rand(N, device=self.device) * 2 * torch.pi
            g_obj_p = s_obj_p + torch.stack([goal_dist*torch.cos(goal_theta), goal_dist*torch.sin(goal_theta), (torch.rand(N, device=self.device)*0.4)-0.2], dim=1)
            g_obj_p[:, 2].clamp_(0.3, 0.85)
            g_obj_q = self._euler_to_quat(torch.zeros(N, device=self.device), torch.zeros(N, device=self.device), (torch.rand(N, device=self.device)*2-1)*(torch.pi/4))
            
            # 2. Distance Filter
            width = 0.8; half_width = width / 2.0
            # Rough EE check for filtering
            ee1_start_rough = s_obj_p + torch.bmm(self._quat_to_matrix(s_obj_q), torch.tensor([-half_width, 0, 0], device=self.device).repeat(N, 1).unsqueeze(2)).squeeze(2)
            d1_ee = torch.norm(ee1_start_rough - b1_p, dim=1)
            
            valid_mask = (torch.norm(s_obj_p - b1_p, dim=1) < 0.75) & (torch.norm(g_obj_p - b1_p, dim=1) < 0.75) & (d1_ee > 0.28)
            indices = torch.nonzero(valid_mask).flatten()
            
            if indices.numel() > 0:
                idx = indices
                count = idx.numel()
                # Prepare Offsets with Noise
                noise = 0.05
                off_p1 = torch.tensor([-half_width, 0.0, 0.0], device=self.device).repeat(count, 1)
                off_q1 = self._euler_to_quat((torch.rand(count, device=self.device)*2-1)*noise, torch.ones(count, device=self.device)*np.pi/2+(torch.rand(count, device=self.device)*2-1)*noise, (torch.rand(count, device=self.device)*2-1)*noise)
                off_p2 = torch.tensor([half_width, 0.0, 0.0], device=self.device).repeat(count, 1)
                off_q2 = self._euler_to_quat((torch.rand(count, device=self.device)*2-1)*noise, torch.ones(count, device=self.device)*(-np.pi/2)+(torch.rand(count, device=self.device)*2-1)*noise, (torch.rand(count, device=self.device)*2-1)*noise)

                # World -> Local Frame Conversion
                R_b1_T = self._quat_to_matrix(b1_q[idx]).transpose(1, 2)
                R_b2_T = self._quat_to_matrix(b2_q[idx]).transpose(1, 2)

                # Solve Start IK
                s1_w_p, s1_w_r = self._get_ee_pose_world(s_obj_p[idx], s_obj_q[idx], off_p1, off_q1)
                s2_w_p, s2_w_r = self._get_ee_pose_world(s_obj_p[idx], s_obj_q[idx], off_p2, off_q2)
                q_s1 = self.ik_solver.solve_ik_gradient(torch.bmm(R_b1_T, (s1_w_p - b1_p[idx]).unsqueeze(2)).squeeze(2), torch.bmm(R_b1_T, s1_w_r), max_iter=40)
                q_s2 = self.ik_solver.solve_ik_gradient(torch.bmm(R_b2_T, (s2_w_p - b2_p[idx]).unsqueeze(2)).squeeze(2), torch.bmm(R_b2_T, s2_w_r), max_iter=40)
                
                # Solve Goal IK
                g1_w_p, g1_w_r = self._get_ee_pose_world(g_obj_p[idx], g_obj_q[idx], off_p1, off_q1)
                g2_w_p, g2_w_r = self._get_ee_pose_world(g_obj_p[idx], g_obj_q[idx], off_p2, off_q2)
                q_g1 = self.ik_solver.solve_ik_gradient(torch.bmm(R_b1_T, (g1_w_p - b1_p[idx]).unsqueeze(2)).squeeze(2), torch.bmm(R_b1_T, g1_w_r), max_iter=40)
                q_g2 = self.ik_solver.solve_ik_gradient(torch.bmm(R_b2_T, (g2_w_p - b2_p[idx]).unsqueeze(2)).squeeze(2), torch.bmm(R_b2_T, g2_w_r), max_iter=40)
                
                # Verify BOTH with FK
                fk_s1, _ = self.ik_solver.forward_kinematics(q_s1); fk_s2, _ = self.ik_solver.forward_kinematics(q_s2)
                fk_g1, _ = self.ik_solver.forward_kinematics(q_g1); fk_g2, _ = self.ik_solver.forward_kinematics(q_g2)
                
                err_s1 = torch.norm(fk_s1 - torch.bmm(R_b1_T, (s1_w_p - b1_p[idx]).unsqueeze(2)).squeeze(2), dim=1)
                err_s2 = torch.norm(fk_s2 - torch.bmm(R_b2_T, (s2_w_p - b2_p[idx]).unsqueeze(2)).squeeze(2), dim=1)
                err_g1 = torch.norm(fk_g1 - torch.bmm(R_b1_T, (g1_w_p - b1_p[idx]).unsqueeze(2)).squeeze(2), dim=1)
                err_g2 = torch.norm(fk_g2 - torch.bmm(R_b2_T, (g2_w_p - b2_p[idx]).unsqueeze(2)).squeeze(2), dim=1)
                
                ik_valid = (err_s1 < 0.015) & (err_s2 < 0.015) & (err_g1 < 0.015) & (err_g2 < 0.015)
                f_idx = torch.nonzero(ik_valid).flatten()
                
                if f_idx.numel() > 0:
                    sel = f_idx
                    collected["base_pose_1"].append(torch.cat([b1_p[idx][sel], b1_q[idx][sel]], dim=1))
                    collected["base_pose_2"].append(torch.cat([b2_p[idx][sel], b2_q[idx][sel]], dim=1))
                    collected["start_obj_pose"].append(torch.cat([s_obj_p[idx][sel], s_obj_q[idx][sel]], dim=1))
                    collected["goal_obj_pose"].append(torch.cat([g_obj_p[idx][sel], g_obj_q[idx][sel]], dim=1))
                    collected["q_start_1"].append(q_s1[sel]); collected["q_start_2"].append(q_s2[sel])
                    
                    # Goal EE Poses (Global)
                    def q_mul(q1, q2):
                        w1, x1, y1, z1 = q1[:,0], q1[:,1], q1[:,2], q1[:,3]
                        w2, x2, y2, z2 = q2[:,0], q2[:,1], q2[:,2], q2[:,3]
                        return torch.stack([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2], dim=1)
                    
                    collected["goal_ee1_pose"].append(torch.cat([g1_w_p[sel], q_mul(g_obj_q[idx][sel], off_q1[sel])], dim=1))
                    collected["goal_ee2_pose"].append(torch.cat([g2_w_p[sel], q_mul(g_obj_q[idx][sel], off_q2[sel])], dim=1))
                    
                    total_collected += sel.numel()
                    print(f"[Sampler] Verified Samples: {total_collected}/{num_envs}")

        # Concatenate and return
        res = {k: torch.cat(v)[:num_envs] for k, v in collected.items() if len(v) > 0}
        res["obj_width"] = 0.8
        return res
