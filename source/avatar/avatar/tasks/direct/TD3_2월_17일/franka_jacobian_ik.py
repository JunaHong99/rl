import torch
import numpy as np
from franka_tensor_ik import FrankaTensorIK

class FrankaJacobianIK(FrankaTensorIK):
    def __init__(self, device='cpu'):
        super().__init__(device=device)

    def _quat_to_matrix(self, quats):
        """Helper: (B, 4) -> (B, 3, 3)"""
        w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
        rot_mats = torch.zeros((quats.shape[0], 3, 3), device=self.device)
        
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        
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

    def compute_jacobian(self, q, base_quat=None):
        """
        Computes Geometric Jacobian in WORLD FRAME.
        Args:
            q: (B, 7) Joint angles
            base_quat: (B, 4) Base orientation quaternions. If None, assumes Identity.
        Returns:
            J_world: (B, 6, 7)
            p_ee: (B, 3) World position of EE (relative to base position, but rotated)
        """
        B = q.shape[0]
        
        # 1. FK Implementation (Same as before)
        T_curr = torch.eye(4, device=self.device).unsqueeze(0).repeat(B, 1, 1)
        z_axes = [T_curr[:, :3, 2]] 
        p_vecs = [T_curr[:, :3, 3]] 
        
        for i in range(7):
            a = self.dh_params[i, 0]; d = self.dh_params[i, 1]
            alpha = self.dh_params[i, 2]; offset = self.dh_params[i, 3]
            theta = q[:, i] + offset
            ct = torch.cos(theta); st = torch.sin(theta)
            ca = torch.cos(alpha); sa = torch.sin(alpha)
            
            Ti = torch.zeros(B, 4, 4, device=self.device)
            Ti[:, 0, 0] = ct; Ti[:, 0, 1] = -st; Ti[:, 0, 2] = 0; Ti[:, 0, 3] = a
            Ti[:, 1, 0] = st*ca; Ti[:, 1, 1] = ct*ca; Ti[:, 1, 2] = -sa; Ti[:, 1, 3] = -sa*d
            Ti[:, 2, 0] = st*sa; Ti[:, 2, 1] = ct*sa; Ti[:, 2, 2] = ca; Ti[:, 2, 3] = ca*d
            Ti[:, 3, 3] = 1.0
            
            T_curr = torch.bmm(T_curr, Ti)
            if i < 6:
                z_axes.append(T_curr[:, :3, 2])
                p_vecs.append(T_curr[:, :3, 3])

        # Hand Offset
        angle = -np.pi / 4.0; c, s = np.cos(angle), np.sin(angle)
        T_hand = torch.eye(4, device=self.device).unsqueeze(0).repeat(B, 1, 1)
        T_hand[:, 0, 0] = c; T_hand[:, 0, 1] = -s
        T_hand[:, 1, 0] = s; T_hand[:, 1, 1] = c
        T_hand[:, 2, 3] = 0.107
        T_ee = torch.bmm(T_curr, T_hand)
        p_ee = T_ee[:, :3, 3]
        
        # 2. Jacobian in Base Frame
        J_base = torch.zeros(B, 6, 7, device=self.device)
        for i in range(7):
            z_prev = z_axes[i]
            p_prev = p_vecs[i]
            p_diff = p_ee - p_prev
            J_base[:, :3, i] = torch.cross(z_prev, p_diff, dim=1)
            J_base[:, 3:, i] = z_prev
            
        # 3. Transform to World Frame
        if base_quat is not None:
            R_base = self._quat_to_matrix(base_quat) # (B, 3, 3)
            
            # Rotate Linear Velocity Part
            # J_v_world = R * J_v_base
            J_base_v = J_base[:, :3, :]
            J_world_v = torch.bmm(R_base, J_base_v)
            
            # Rotate Angular Velocity Part
            # J_w_world = R * J_w_base
            J_base_w = J_base[:, 3:, :]
            J_world_w = torch.bmm(R_base, J_base_w)
            
            J_world = torch.cat([J_world_v, J_world_w], dim=1)
            
            # Also rotate p_ee for consistent usage outside
            p_ee_world = torch.bmm(R_base, p_ee.unsqueeze(2)).squeeze(2)
            
            return J_world, p_ee_world
        else:
            return J_base, p_ee

    def apply_relative_constraint_projection(self, q1, q2, base_quat1, base_quat2, p_base1, p_base2, action1, action2):
        """
        Projects actions to satisfy:
        1. w1 = w2 (Equal angular velocity -> Fixed Relative Orientation)
        2. v2 = v1 + w1 x (p2 - p1) (Rigid body kinematic constraint)
        
        All computations are in WORLD FRAME.
        """
        B = q1.shape[0]
        
        # 1. Compute World Jacobians & EE Positions
        J1, p_ee1_local = self.compute_jacobian(q1, base_quat1)
        J2, p_ee2_local = self.compute_jacobian(q2, base_quat2)
        
        # Add base positions to get absolute world positions
        p1 = p_ee1_local + p_base1
        p2 = p_ee2_local + p_base2
        
        # Relative Position Vector (World Frame)
        r12 = p2 - p1 
        
        # 2. Build Constraint Matrix J_c (6 x 14)
        # We want: 
        # (1) w2 - w1 = 0
        # (2) v2 - v1 - w1 x r12 = 0
        
        # J1 = [Jv1; Jw1], J2 = [Jv2; Jw2]
        Jv1 = J1[:, :3, :]
        Jw1 = J1[:, 3:, :]
        Jv2 = J2[:, :3, :]
        Jw2 = J2[:, 3:, :]
        
        # Constraint (2): v2 - v1 - w1 x r12 = 0
        # => Jv2*dq2 - Jv1*dq1 - (Jw1*dq1) x r12 = 0
        # Note: a x b = [a]_x * b. 
        # w1 x r12 = - r12 x w1 = - [r12]_x * Jw1 * dq1
        # So: Jv2*dq2 - Jv1*dq1 + [r12]_x * Jw1 * dq1 = 0
        # Terms for dq1: -Jv1 + [r12]_x * Jw1
        # Terms for dq2: Jv2
        
        # Skew-symmetric matrix for r12
        r_x = torch.zeros(B, 3, 3, device=self.device)
        r_x[:, 0, 1] = -r12[:, 2]; r_x[:, 0, 2] = r12[:, 1]
        r_x[:, 1, 0] = r12[:, 2]; r_x[:, 1, 2] = -r12[:, 0]
        r_x[:, 2, 0] = -r12[:, 1]; r_x[:, 2, 1] = r12[:, 0]
        
        # Block for dq1 (Linear Constraint)
        # C_lin_1 = -Jv1 + r_x @ Jw1
        C_lin_1 = -Jv1 + torch.bmm(r_x, Jw1)
        
        # Block for dq2 (Linear Constraint)
        C_lin_2 = Jv2
        
        # Constraint (1): w2 - w1 = 0
        # => Jw2*dq2 - Jw1*dq1 = 0
        # Block for dq1: -Jw1
        # Block for dq2: Jw2
        
        C_ang_1 = -Jw1
        C_ang_2 = Jw2
        
        # Stack to form full J_constraint
        # Row 1-3: Linear, Row 4-6: Angular
        J_c_1 = torch.cat([C_lin_1, C_ang_1], dim=1) # (B, 6, 7)
        J_c_2 = torch.cat([C_lin_2, C_ang_2], dim=1) # (B, 6, 7)
        
        J_c = torch.cat([J_c_1, J_c_2], dim=2) # (B, 6, 14)
        
        # 3. Projection
        # P = I - J_c^dagger * J_c
        JJT = torch.bmm(J_c, J_c.transpose(1, 2))
        damp = 1e-3 * torch.eye(6, device=self.device).unsqueeze(0)
        JJT_inv = torch.inverse(JJT + damp)
        
        J_dagger = torch.bmm(J_c.transpose(1, 2), JJT_inv) # (B, 14, 6)
        
        action_combined = torch.cat([action1, action2], dim=1).unsqueeze(2) # (B, 14, 1)
        
        # Violation velocity
        v_viol = torch.bmm(J_c, action_combined)
        
        # Correction
        dq_remove = torch.bmm(J_dagger, v_viol).squeeze(2)
        
        action_projected = torch.cat([action1, action2], dim=1) - dq_remove
        
        return action_projected[:, :7], action_projected[:, 7:]