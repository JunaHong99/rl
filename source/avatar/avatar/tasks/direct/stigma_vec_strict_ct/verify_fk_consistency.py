import torch
import numpy as np
import pinocchio as pin
import os
from franka_tensor_ik import FrankaTensorIK

def main():
    print("🚀 Verifying Forward Kinematics Consistency...")
    
    # 1. Setup Pinocchio (Ground Truth)
    # Using the same URDF path logic as in ik_solver.py
    # Assuming relative path from current dir
    urdf_path = "../../../asset/robot_urdf/franka_description/franka_panda.urdf"
    if not os.path.exists(urdf_path):
        # Fallback to absolute path search or common location if needed, 
        # but let's try the one from the user's project structure
        # /home/hjh/research/2nd/IsaacLab/scripts/avatar/source/avatar/avatar/asset/robot_urdf/franka_description/franka_panda.urdf
        base_path = "/home/hjh/research/2nd/IsaacLab/scripts/avatar/source/avatar/avatar/asset/robot_urdf/franka_description/franka_panda.urdf"
        if os.path.exists(base_path):
            urdf_path = base_path
        else:
            print(f"❌ URDF not found at {urdf_path} or {base_path}")
            return

    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    ee_frame_id = model.getFrameId("panda_hand") # Check this name! 
    
    # 2. Setup Tensor IK (Candidate)
    tensor_ik = FrankaTensorIK(device='cpu')
    
    # 3. Generate Random Configurations
    num_tests = 10
    q_rand = torch.rand(num_tests, 7) * (tensor_ik.q_max.cpu() - tensor_ik.q_min.cpu()) + tensor_ik.q_min.cpu()
    
    print(f"\nComparing {num_tests} random configurations...")
    print(f"{ 'Test':<5} | {'Pos Err (m)':<15} | {'Rot Err (rad)':<15}")
    print("---------------------------------------------")
    
    max_pos_err = 0.0
    
    for i in range(num_tests):
        q_7dof = q_rand[i].numpy()
        # Pinocchio needs 9 DOF (7 arm + 2 gripper)
        q_9dof = np.concatenate([q_7dof, np.zeros(2)])
        
        q_tensor = q_rand[i].unsqueeze(0) # (1, 7)
        
        # A. Pinocchio FK
        pin.forwardKinematics(model, data, q_9dof)
        pin.updateFramePlacements(model, data)
        pin_pose = data.oMf[ee_frame_id]
        pin_pos = pin_pose.translation
        pin_rot = pin_pose.rotation
        
        # B. Tensor FK
        tensor_pos, tensor_rot = tensor_ik.forward_kinematics(q_tensor)
        t_pos = tensor_pos.squeeze().numpy()
        t_rot = tensor_rot.squeeze().numpy()
        
        # Calc Errors
        pos_err = np.linalg.norm(pin_pos - t_pos)
        
        # Rotation Error (Geodesic distance on SO3)
        # R_diff = R1 * R2^T
        # angle = arccos((tr(R_diff) - 1) / 2)
        r_diff = pin_rot @ t_rot.T
        tr = np.trace(r_diff)
        # Clamp for numerical stability
        tr = np.clip(tr, -1.0, 3.0) 
        rot_err = np.arccos((tr - 1) / 2.0)
        
        print(f"{i:<5} | {pos_err:<15.6f} | {rot_err:<15.6f}")
        max_pos_err = max(max_pos_err, pos_err)

    print("---------------------------------------------")
    if max_pos_err > 0.01: # Allow 1cm tolerance? Actually should be < 1mm
        print(f"❌ FAIL: Maximum Position Error is {max_pos_err:.4f} m (> 1cm)")
        print("This suggests a mismatch in DH parameters, Joint Offsets, or EE Frame definition.")
    else:
        print(f"✅ PASS: Maximum Position Error is {max_pos_err:.6f} m")
        print("The custom DH parameters match the URDF reasonably well.")

if __name__ == "__main__":
    main()
