import torch
import isaaclab.utils.math as math_utils
from franka_tensor_ik import FrankaTensorIK

class SafetyFilter:
    def __init__(self, env):
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.dt = env.cfg.sim.dt * env.cfg.decimation
        
        # IK Solver for FK predictions
        self.ik_solver = FrankaTensorIK(device=self.device)
        
        # Store robot references
        self.robot_1 = env.robot_1
        self.robot_2 = env.robot_2
        self.ee_idx_1 = env.ee_body_idx_1
        self.ee_idx_2 = env.ee_body_idx_2
        self.joint_ids_1 = env.robot_1_joint_ids
        self.joint_ids_2 = env.robot_2_joint_ids
        
        # Gains
        self.kp_pos = 30.0 
        self.kp_rot = 30.0 
        
        self.q_vel_limit_1 = getattr(env, "vel_limit_1", None)
        self.q_vel_limit_2 = getattr(env, "vel_limit_2", None)
        
        # Joint Limit Avoidance Params
        limit_data = env.robot_1.data.soft_joint_pos_limits[0, :7, :]
        self.q_min = limit_data[:, 0].to(self.device)
        self.q_max = limit_data[:, 1].to(self.device)
        
        self.k_limit = 5.0      
        self.limit_margin = 0.1 
        
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

    def _compute_drift_correction(self, p1, q1, p2, q2):
        """Helper to compute V_correction from current EE states."""
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
        
        return torch.cat([v_correction_pos, v_correction_rot], dim=1)

    def apply_filter(self, actions_1, actions_2):
        self.step_cnt += 1
        
        # 1. Current State & Jacobian
        J1, J2 = self._get_jacobians()
        state_1 = self.robot_1.data.body_state_w[:, self.ee_idx_1]
        p1 = state_1[:, 0:3]; q1 = state_1[:, 3:7]
        state_2 = self.robot_2.data.body_state_w[:, self.ee_idx_2]
        p2 = state_2[:, 0:3]; q2 = state_2[:, 3:7]
        
        joint_pos_1 = self.robot_1.data.joint_pos[:, self.joint_ids_1]
        joint_pos_2 = self.robot_2.data.joint_pos[:, self.joint_ids_2]
        q_curr = torch.cat([joint_pos_1, joint_pos_2], dim=1) # (B, 14)

        # 2. Constraint Jacobian (J_c)
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
        
        JJT = torch.bmm(J_c, J_c.transpose(1, 2))
        damp = 1e-4 * torch.eye(6, device=self.device).unsqueeze(0)
        JJT_inv = torch.inverse(JJT + damp)
        J_dagger = torch.bmm(J_c.transpose(1, 2), JJT_inv) 
        
        # 3. Initial Safe Action (Projected + Restoration + Governor)
        q_nom = torch.cat([actions_1, actions_2], dim=1).unsqueeze(2)
        q_nom_norm = torch.norm(q_nom.squeeze(2), dim=1, keepdim=True).unsqueeze(2)

        J_q_nom = torch.bmm(J_c, q_nom) 
        q_projected = q_nom - torch.bmm(J_dagger, J_q_nom)
        
        q_proj_norm = torch.norm(q_projected.squeeze(2), dim=1, keepdim=True).unsqueeze(2)
        
        # [NEW] Safety Governor: Throttle restoration based on drift severity
        # Calculate current drift correction vector to determine governor level.
        drift_corr_curr = self._compute_drift_correction(p1, q1, p2, q2)
        drift_norm = torch.norm(drift_corr_curr, dim=1, keepdim=True).unsqueeze(2)
        
        # Governor Logic:
        # drift_norm is roughly proportional to error * kp (kp=30)
        # Hard Safety Limit: If error > ~5cm (drift > 1.5), STOP completely.
        THRESHOLD_STOP = 1.5
        
        # Vectorized governor logic
        alpha = torch.where(drift_norm > THRESHOLD_STOP, torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device))
        
        # Apply restoration with governor
        q_safe_projected = q_projected * (q_nom_norm / (q_proj_norm + 1e-6)) * alpha

        # 4. [NEW] Iterative Refinement with Priority Scaling
        # We perform refinement on the components, but enforce priority at each step or at the end.
        # Since iterative refinement is complex with clamping inside, let's apply priority scaling 
        # to the accumulated result of the refinement loop.
        
        dq_drift_total = torch.zeros_like(q_curr)
        q_task_total = q_safe_projected.squeeze(2) # Initial task guess
        
        # We need to separate drift fix and task movement during refinement
        # Refinement Loop
        current_drift_fix = torch.bmm(J_dagger, drift_corr_curr.unsqueeze(2)).squeeze(2)
        dq_drift_total += current_drift_fix
        
        # Refine prediction
        NUM_ITERS = 3
        
        # Get base poses for world FK calculation
        b1_pos = self.robot_1.data.root_pos_w
        b1_quat = self.robot_1.data.root_quat_w
        b2_pos = self.robot_2.data.root_pos_w
        b2_quat = self.robot_2.data.root_quat_w

        for _ in range(NUM_ITERS):
            # Combine current best guess
            q_candidate = q_task_total + dq_drift_total
            
            # Predict next state
            q_next = q_curr + q_candidate * self.dt
            
            # FK
            p1_n, R1_n = self.ik_solver.forward_kinematics(q_next[:, :7])
            q1_n = math_utils.quat_from_matrix(R1_n)
            p1_w = math_utils.quat_apply(b1_quat, p1_n) + b1_pos
            q1_w = math_utils.quat_mul(b1_quat, q1_n)
            
            p2_n, R2_n = self.ik_solver.forward_kinematics(q_next[:, 7:])
            q2_n = math_utils.quat_from_matrix(R2_n)
            p2_w = math_utils.quat_apply(b2_quat, p2_n) + b2_pos
            q2_w = math_utils.quat_mul(b2_quat, q2_n)
            
            # Error
            drift_corr_n = self._compute_drift_correction(p1_w, q1_w, p2_w, q2_w)
            
            # Update Correction Term ONLY
            # We want to refine the correction to kill the error.
            dq_delta = torch.bmm(J_dagger, drift_corr_n.unsqueeze(2)).squeeze(2)
            dq_drift_total += dq_delta

        # 5. Priority-Based Scaling (The Last Line of Defense)
        # We have dq_drift_total (Priority 1) and q_task_total (Priority 2)
        
        # Helper for scaling
        def get_scale_factor(q, limit):
            if limit is None: return torch.ones(q.shape[0], 1, device=self.device)
            if not isinstance(limit, torch.Tensor): limit = torch.tensor(limit, device=self.device)
            # limit is (7,), q is (B, 7)
            limit_exp = limit.unsqueeze(0)
            abs_q = torch.abs(q)
            ratios = abs_q / (limit_exp + 1e-6)
            max_ratios, _ = torch.max(ratios, dim=1)
            # If max_ratio > 1.0, we need to scale down by 1.0/max_ratio
            scale = torch.clamp(1.0 / (max_ratios + 1e-6), max=1.0)
            return scale.unsqueeze(1)

        # Split limits
        lim1 = self.q_vel_limit_1
        lim2 = self.q_vel_limit_2
        
        # A. Scale Correction first
        dr1 = dq_drift_total[:, :7]
        dr2 = dq_drift_total[:, 7:]
        
        s1_corr = get_scale_factor(dr1, lim1)
        s2_corr = get_scale_factor(dr2, lim2)
        
        # If correction is too huge, we must scale it down (Physical Limit)
        dr1_scaled = dr1 * s1_corr
        dr2_scaled = dr2 * s2_corr
        
        # B. Calculate Remaining Budget
        # margin = limit - |dr_scaled|
        # If we used 100% budget, margin is 0.
        # Note: This simple subtraction is conservative (box constraints).
        if isinstance(lim1, torch.Tensor): lim1 = lim1.to(self.device)
        else: lim1 = torch.tensor(lim1, device=self.device)
        
        if isinstance(lim2, torch.Tensor): lim2 = lim2.to(self.device)
        else: lim2 = torch.tensor(lim2, device=self.device)
        
        margin1 = torch.clamp(lim1.unsqueeze(0) - torch.abs(dr1_scaled), min=0.0)
        margin2 = torch.clamp(lim2.unsqueeze(0) - torch.abs(dr2_scaled), min=0.0)
        
        # C. Scale Task to fit in Margin
        task1 = q_task_total[:, :7]
        task2 = q_task_total[:, 7:]
        
        # Scale factor for task based on margin
        # ratios = |task| / margin
        # scale = 1 / max(ratios)
        ratios1 = torch.abs(task1) / (margin1 + 1e-6)
        max_r1, _ = torch.max(ratios1, dim=1)
        s1_task = torch.clamp(1.0 / (max_r1 + 1e-6), max=1.0).unsqueeze(1)
        
        ratios2 = torch.abs(task2) / (margin2 + 1e-6)
        max_r2, _ = torch.max(ratios2, dim=1)
        s2_task = torch.clamp(1.0 / (max_r2 + 1e-6), max=1.0).unsqueeze(1)
        
        task1_scaled = task1 * s1_task
        task2_scaled = task2 * s2_task
        
        # D. Combine
        q_final_1 = dr1_scaled + task1_scaled
        q_final_2 = dr2_scaled + task2_scaled
        
        # 6. Joint Limit Avoidance (Null Space Repulsion) - Skipped for now or integrated?
        # Ideally, this should be part of the 'Task' or a separate priority layer.
        # Given the complexity, let's rely on the Governor and Drift Fix for now.
        # If needed, it can be added to q_task_total before scaling.
            
        return q_final_1, q_final_2