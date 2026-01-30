
import torch
import numpy as np

class FrankaTensorIK:
    def __init__(self, device='cpu'):
        self.device = device
        
        # Franka Emika Panda DH Parameters (Modified DH)
        # a, d, alpha, theta_offset
        # Note: These are standard parameters. 
        self.dh_params = torch.tensor([
            [0,      0.333,  0,       0],       # Joint 1
            [0,      0,      -np.pi/2, 0],       # Joint 2
            [0,      0.316,  np.pi/2,  0],       # Joint 3
            [0.0825, 0,      np.pi/2,  0],       # Joint 4
            [-0.0825, 0.384,  -np.pi/2, 0],       # Joint 5
            [0,      0,      np.pi/2,  0],       # Joint 6
            [0.088,  0,      np.pi/2,  0]        # Joint 7
        ], device=device, dtype=torch.float32)
        
        # Joint Limits (approximate, slightly tighter for safety)
        self.q_min = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], device=device)
        self.q_max = torch.tensor([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], device=device)

    def forward_kinematics(self, q):
        """
        Batched Forward Kinematics
        Args:
            q: (B, 7) Joint angles
        Returns:
            ee_pos: (B, 3) End-effector position
            ee_rot_mat: (B, 3, 3) End-effector rotation matrix
        """
        B = q.shape[0]
        
        # Identity matrix for base transformation
        T = torch.eye(4, device=self.device).unsqueeze(0).repeat(B, 1, 1)
        
        for i in range(7):
            a = self.dh_params[i, 0]
            d = self.dh_params[i, 1]
            alpha = self.dh_params[i, 2]
            offset = self.dh_params[i, 3]
            theta = q[:, i] + offset
            
            # Modified DH Transformation Matrix
            # T_{i-1, i} = Rot_x(alpha) * Trans_x(a) * Rot_z(theta) * Trans_z(d)
            
            ct = torch.cos(theta)
            st = torch.sin(theta)
            ca = torch.cos(alpha)
            sa = torch.sin(alpha)
            
            # Build transformation matrix for this joint (B, 4, 4)
            Ti = torch.zeros(B, 4, 4, device=self.device)
            
            Ti[:, 0, 0] = ct
            Ti[:, 0, 1] = -st
            Ti[:, 0, 2] = 0
            Ti[:, 0, 3] = a # a is scalar, but broadcasting works
            
            Ti[:, 1, 0] = st * ca
            Ti[:, 1, 1] = ct * ca
            Ti[:, 1, 2] = -sa
            Ti[:, 1, 3] = -sa * d
            
            Ti[:, 2, 0] = st * sa
            Ti[:, 2, 1] = ct * sa
            Ti[:, 2, 2] = ca
            Ti[:, 2, 3] = ca * d
            
            Ti[:, 3, 3] = 1.0
            
            T = torch.bmm(T, Ti)
            
        
        # T_hand = RotZ(-pi/4) * TransZ(0.107)
        B = q.shape[0]
        T_hand = torch.eye(4, device=self.device).unsqueeze(0).repeat(B, 1, 1)
        
        angle = -np.pi / 4.0
        c = np.cos(angle)
        s = np.sin(angle)
        
        T_hand[:, 0, 0] = c
        T_hand[:, 0, 1] = -s
        T_hand[:, 1, 0] = s
        T_hand[:, 1, 1] = c
        T_hand[:, 2, 3] = 0.107 # 10.7cm offset
        
        T = torch.bmm(T, T_hand)
        
        return T[:, :3, 3], T[:, :3, :3]

    def solve_ik_gradient(self, target_pos, target_rot_mat, q_init=None, max_iter=100, lr=0.1, tol=1e-3):
        """
        Simple Gradient-Descent IK (Batched)
        Args:
            target_pos: (B, 3)
            target_rot_mat: (B, 3, 3)
        """
        B = target_pos.shape[0]
        if q_init is None:
            # [Fix] Initialize with Neutral Pose to avoid weird elbow configurations
            # Neutral Pose: [0, -pi/4, 0, -3pi/4, 0, pi/2, pi/4] approx
            neutral_pose = torch.tensor([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], device=self.device)
            q = neutral_pose.unsqueeze(0).repeat(B, 1)
            # Add small noise to avoid getting stuck in local minima if exactly at singularity (unlikely)
            q += (torch.rand_like(q) - 0.5) * 0.1
        else:
            q = q_init.clone()
            
        q.requires_grad_(True)
        optimizer = torch.optim.Adam([q], lr=lr)
        
        for i in range(max_iter):
            optimizer.zero_grad()
            
            curr_pos, curr_rot = self.forward_kinematics(q)
            
            # Position Error
            pos_err = torch.norm(curr_pos - target_pos, dim=1).mean()
            
            # Rotation Error (Trace of R_diff)
            # R_diff = R_current @ R_target.T
            # Trace should be 3 for perfect match
            R_diff = torch.bmm(curr_rot, target_rot_mat.transpose(1, 2))
            tr = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
            rot_err = (3.0 - tr).mean()
            
            loss = pos_err + 0.5 * rot_err 
            # Add joint limit penalty
            # penalty = torch.sum(torch.relu(q - self.q_max) + torch.relu(self.q_min - q))
            # loss += penalty
            
            loss.backward()
            optimizer.step()
            
            # Project to limits
            with torch.no_grad():
                q.clamp_(self.q_min, self.q_max)
                
            if loss.item() < tol:
                break
                
        return q.detach()

    def solve_ik_jacobian(self, target_pos, target_quat, q_init=None, max_iter=50, damp=1e-2):
        """
        Jacobian Damped Least Squares IK (Batched) - More robust than GD
        """
        # This requires calculating Jacobian manually or via autograd.
        # Autograd Jacobian is expensive for batches.
        # Let's stick to a very efficient 'Random Shooting + GD' or purely heuristic validation if this is too slow.
        pass
        
    def check_reachability(self, target_pos, base_pos=None):
        """
        Simple check: is target within 0.85m of base?
        """
        if base_pos is None:
            base_pos = torch.zeros_like(target_pos)
        
        dist = torch.norm(target_pos - base_pos, dim=1)
        return dist < 0.8
