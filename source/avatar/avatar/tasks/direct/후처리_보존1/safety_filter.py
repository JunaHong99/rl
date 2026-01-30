import torch
import isaaclab.utils.math as math_utils

class SafetyFilter:
    def __init__(self, env):
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        
        # Store robot references
        self.robot_1 = env.robot_1
        self.robot_2 = env.robot_2
        
        # EE Body Indices
        self.ee_idx_1 = env.ee_body_idx_1
        self.ee_idx_2 = env.ee_body_idx_2
        
        # Joint Indices
        self.joint_ids_1 = env.robot_1_joint_ids
        self.joint_ids_2 = env.robot_2_joint_ids
        
        # [Gains from 87.5% Success Version] -> 10, 10
        self.kp_pos = 20.0 
        self.kp_rot = 20.0 
        
        # Get limits
        self.q_vel_limit_1 = getattr(env, "vel_limit_1", None)
        self.q_vel_limit_2 = getattr(env, "vel_limit_2", None)
        
        self.debug_mode = True
        self.step_cnt = 0

    def _get_jacobians(self):
        """Retrieves World Frame Jacobians directly from simulator."""
        J1_full = self.robot_1.root_physx_view.get_jacobians() 
        J1_ee = J1_full[:, self.ee_idx_1, :, :]
        J1_arm = J1_ee[:, :, self.joint_ids_1]
        
        J2_full = self.robot_2.root_physx_view.get_jacobians()
        J2_ee = J2_full[:, self.ee_idx_2, :, :]
        J2_arm = J2_ee[:, :, self.joint_ids_2]
        
        return J1_arm, J2_arm

    def apply_filter(self, actions_1, actions_2):
        self.step_cnt += 1
        
        # 1. Get State
        J1, J2 = self._get_jacobians()
        state_1 = self.robot_1.data.body_state_w[:, self.ee_idx_1]
        p1 = state_1[:, 0:3]; q1 = state_1[:, 3:7]
        state_2 = self.robot_2.data.body_state_w[:, self.ee_idx_2]
        p2 = state_2[:, 0:3]; q2 = state_2[:, 3:7]
        
        # 2. Compute Drift (Error)
        target_rel = self.env.target_ee_rel_poses
        t_rel_pos = target_rel[:, 0:3]; t_rel_rot = target_rel[:, 3:7]
        
        q1_inv = math_utils.quat_conjugate(q1)
        curr_rel_pos_local = math_utils.quat_apply(q1_inv, p2 - p1)
        pos_err_world = math_utils.quat_apply(q1, curr_rel_pos_local - t_rel_pos)
        v_correction_pos = -self.kp_pos * pos_err_world
        
        curr_rel_rot = math_utils.quat_mul(q1_inv, q2)
        q_err = math_utils.quat_mul(curr_rel_rot, math_utils.quat_conjugate(t_rel_rot))
        q_err_v = q_err[:, 1:4]; q_err_w = q_err[:, 0:1]
        sign = torch.sign(q_err_w)
        rot_err_world = math_utils.quat_apply(q1, 2.0 * q_err_v * sign)
        v_correction_rot = -self.kp_rot * rot_err_world
        
        # 3. Construct Constraint Jacobian
        r12 = p2 - p1
        B = self.num_envs
        r_x = torch.zeros(B, 3, 3, device=self.device)
        r_x[:, 0, 1] = -r12[:, 2]; r_x[:, 0, 2] = r12[:, 1]
        r_x[:, 1, 0] = r12[:, 2]; r_x[:, 1, 2] = -r12[:, 0]
        r_x[:, 2, 0] = -r12[:, 1]; r_x[:, 2, 1] = r12[:, 0]
        
        Jv1, Jw1 = J1[:, :3, :], J1[:, 3:, :]
        Jv2, Jw2 = J2[:, :3, :], J2[:, 3:, :]
        
        # Constraint: v2 - v1 + r12 x w1 = 0 (and w2 - w1 = 0)
        C_lin_1 = -Jv1 + torch.bmm(r_x, Jw1)
        C_lin_2 = Jv2
        C_ang_1 = -Jw1
        C_ang_2 = Jw2
        
        J_c = torch.cat([
            torch.cat([C_lin_1, C_ang_1], dim=1),
            torch.cat([C_lin_2, C_ang_2], dim=1)
        ], dim=2)
        
        drift_corr = torch.cat([v_correction_pos, v_correction_rot], dim=1)
        
        # 4. Projection with Drift Correction
        JJT = torch.bmm(J_c, J_c.transpose(1, 2))
        damp = 1e-4 * torch.eye(6, device=self.device).unsqueeze(0)
        JJT_inv = torch.inverse(JJT + damp)
        J_dagger = torch.bmm(J_c.transpose(1, 2), JJT_inv) 
        
        q_nom = torch.cat([actions_1, actions_2], dim=1).unsqueeze(2)
        J_q_nom = torch.bmm(J_c, q_nom) 
        error_signal = J_q_nom - drift_corr.unsqueeze(2)
        dq_correction = torch.bmm(J_dagger, error_signal).squeeze(2)
        
        q_safe = torch.cat([actions_1, actions_2], dim=1) - dq_correction
        
        # 5. Simple Vector Scaling (preserving direction)
        def scale_velocity(q, limit):
            if limit is None: return q
            if not isinstance(limit, torch.Tensor): limit = torch.tensor(limit, device=self.device)
            abs_q = torch.abs(q)
            ratios = abs_q / (limit + 1e-6)
            max_ratios, _ = torch.max(ratios, dim=1)
            scale_factors = torch.clamp(max_ratios, min=1.0)
            return q / scale_factors.unsqueeze(1)

        q_final_1 = scale_velocity(q_safe[:, :7], self.q_vel_limit_1)
        q_final_2 = scale_velocity(q_safe[:, 7:], self.q_vel_limit_2)
            
        if self.debug_mode and self.step_cnt % 50 == 0:
            pos_err_mm = torch.norm(pos_err_world, dim=1).mean().item() * 1000
            rot_err_rad = torch.norm(rot_err_world, dim=1).mean().item()
            print(f"[Filter] PosErr: {pos_err_mm:.2f}mm, RotErr: {rot_err_rad:.4f}rad")

        return q_final_1, q_final_2