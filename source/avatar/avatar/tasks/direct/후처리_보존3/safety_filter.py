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
        self.ee_idx_1 = env.ee_body_idx_1
        self.ee_idx_2 = env.ee_body_idx_2
        self.joint_ids_1 = env.robot_1_joint_ids
        self.joint_ids_2 = env.robot_2_joint_ids
        
        # Gains
        self.kp_pos = 10.0 
        self.kp_rot = 10.0 
        
        self.q_vel_limit_1 = getattr(env, "vel_limit_1", None)
        self.q_vel_limit_2 = getattr(env, "vel_limit_2", None)
        
        self.debug_mode = True
        self.step_cnt = 0

    def _get_jacobians(self):
        J1_full = self.robot_1.root_physx_view.get_jacobians() 
        J1_ee = J1_full[:, self.ee_idx_1, :, :]
        J1_arm = J1_ee[:, :, self.joint_ids_1]
        J2_full = self.robot_2.root_physx_view.get_jacobians()
        J2_ee = J2_full[:, self.ee_idx_2, :, :]
        J2_arm = J2_ee[:, :, self.joint_ids_2]
        return J1_arm, J2_arm

    def apply_filter(self, actions_1, actions_2):
        self.step_cnt += 1
        
        # 1. State & Jacobian
        J1, J2 = self._get_jacobians()
        state_1 = self.robot_1.data.body_state_w[:, self.ee_idx_1]
        p1 = state_1[:, 0:3]; q1 = state_1[:, 3:7]
        state_2 = self.robot_2.data.body_state_w[:, self.ee_idx_2]
        p2 = state_2[:, 0:3]; q2 = state_2[:, 3:7]
        
        # 2. Errors & Correction
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
        
        drift_corr = torch.cat([v_correction_pos, v_correction_rot], dim=1)

        # 3. Constraint Jacobian
        r12 = p2 - p1
        B = self.num_envs
        r_x = torch.zeros(B, 3, 3, device=self.device)
        r_x[:, 0, 1] = -r12[:, 2]; r_x[:, 0, 2] = r12[:, 1]
        r_x[:, 1, 0] = r12[:, 2]; r_x[:, 1, 2] = -r12[:, 0]
        r_x[:, 2, 0] = -r12[:, 1]; r_x[:, 2, 1] = r12[:, 0]
        
        Jv1, Jw1 = J1[:, :3, :], J1[:, 3:, :]
        Jv2, Jw2 = J2[:, :3, :], J2[:, 3:, :]
        
        C_lin_1 = -Jv1 + torch.bmm(r_x, Jw1)
        C_lin_2 = Jv2
        C_ang_1 = -Jw1
        C_ang_2 = Jw2
        
        J_c = torch.cat([
            torch.cat([C_lin_1, C_ang_1], dim=1),
            torch.cat([C_lin_2, C_ang_2], dim=1)
        ], dim=2)
        
        # 4. Projection
        JJT = torch.bmm(J_c, J_c.transpose(1, 2))
        damp = 1e-4 * torch.eye(6, device=self.device).unsqueeze(0)
        JJT_inv = torch.inverse(JJT + damp)
        J_dagger = torch.bmm(J_c.transpose(1, 2), JJT_inv) 
        
        q_nom = torch.cat([actions_1, actions_2], dim=1).unsqueeze(2)

        # [Fix] Input Suppression based on Drift (Deadlock Resolution)
        drift_norm = torch.norm(drift_corr, dim=1, keepdim=True).unsqueeze(2) # [B, 1, 1]
        alpha = torch.clamp(1.0 - (drift_norm - 0.5) / 2.0, 0.0, 1.0)
        q_nom = q_nom * alpha
        
        J_q_nom = torch.bmm(J_c, q_nom) 
        error_signal = J_q_nom - drift_corr.unsqueeze(2)
        dq_correction = torch.bmm(J_dagger, error_signal).squeeze(2)
        
        q_safe = torch.cat([actions_1, actions_2], dim=1) - dq_correction
        
        # 5. [DEBUG] Deadlock Analysis
        if self.debug_mode and self.step_cnt % 20 == 0:
            # Check Env 0
            idx = 0
            
            # Magnitudes
            nom_mag = torch.norm(q_nom[idx]).item()
            safe_mag = torch.norm(q_safe[idx]).item()
            drift_mag = torch.norm(drift_corr[idx]).item()
            
            # Get alpha for env 0
            alpha_val = alpha[idx, 0, 0].item()

            print(f"[Step {self.step_cnt}] In: {nom_mag:.4f} | Out: {safe_mag:.4f} | Drift: {drift_mag:.4f} | Alpha: {alpha_val:.4f}")
            
            # Deadlock Condition: Input is large, Output is tiny
            if nom_mag > 0.1 and safe_mag < 0.05:
                print(f"⚠️ [Deadlock Potential]")
                
                # Check Alignment: Is q_nom parallel to Constraint Gradient?
                # Projection P = I - J+J
                # If q_nom is entirely in range(J^T), then P * q_nom = 0.
                # Let's check the cosine similarity between q_nom and the removed component (dq_correction)
                # If they are parallel, it means q_nom was PURE violation.
                
                # Correction direction (ignoring drift_corr for purity check)
                pure_violation = torch.bmm(J_dagger, J_q_nom).squeeze(2) # Part of q_nom that violates constraint
                
                sim = torch.cosine_similarity(q_nom[idx].flatten(), pure_violation[idx].flatten(), dim=0).item()
                print(f"  Alignment with Violation: {sim:.4f} (1.0 means pure violation)")
                
                if sim > 0.9:
                    print("  => CONFIRMED: Actor is trying to move ONLY in the violation direction.")
                else:
                    print("  => Reason unclear. Possibly complex constraint interaction.")

        # 6. Scaling
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
            
        return q_final_1, q_final_2
