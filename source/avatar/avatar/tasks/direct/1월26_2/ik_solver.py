# ik_solver.py

import pinocchio as pin
import numpy as np
import os

class FrankaIKSolver:
    def __init__(self):
        # 1. Franka Panda 로봇 모델 로드 (URDF 직접 로드)
        # 사용자 환경의 URDF 경로
        current_dir = os.path.dirname(os.path.abspath(__file__))   
        urdf_path = os.path.join(current_dir, "../../../asset/robot_urdf/franka_description/franka_panda.urdf")  

        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF 파일을 찾을 수 없습니다: {urdf_path}")

        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        
        # 2. End-Effector 프레임 ID 찾기
        # (URDF에서 'panda_hand'가 EE 링크 이름임)
        self.ee_frame_id = self.model.getFrameId("panda_hand")
        
        # 3. 관절 한계 (Limits) 가져오기
        self.q_min = self.model.lowerPositionLimit
        self.q_max = self.model.upperPositionLimit
        
        # 4. 기본(Neutral) 자세 (초기값용)
        self.q_neutral = pin.neutral(self.model)

    def solve(self, target_pos, target_quat, seed_q=None, max_iter=1000, eps=1e-4, damping=1e-3):
        """
        주어진 목표 포즈(위치, 쿼터니언)에 대한 역기구학(IK)을 풉니다.
        
        Args:
            target_pos (np.array): [x, y, z] 목표 위치 (Robot Base Frame 기준)
            target_quat (np.array): [w, x, y, z] 목표 쿼터니언 (Isaac Lab 순서: w, x, y, z)
            seed_q (np.array, optional): 초기 관절 각도 추정값. 없으면 neutral pose 사용.
            max_iter (int): 최대 반복 횟수
            eps (float): 수렴 오차 허용 범위 (m)
            damping (float): DLS 댐핑 계수 (특이점 근처 안정성 확보용)

        Returns:
            q_sol (np.array): 해결된 7-DoF 관절 각도
            success (bool): 수렴 성공 여부
        """
        
        # --- 1. 좌표계 변환 (Isaac Lab [w,x,y,z] -> Pinocchio [x,y,z,w]) ---
        # Pinocchio는 쿼터니언 순서가 [x, y, z, w]입니다. 주의!
        w, x, y, z = target_quat
        pin_quat = np.array([x, y, z, w]) 
        
        # 목표 변환 행렬 (SE3) 생성
        target_rot = pin.Quaternion(pin_quat).matrix()
        oMdes = pin.SE3(target_rot, np.array(target_pos))

        # 초기값 설정
        q = seed_q.copy() if seed_q is not None else self.q_neutral.copy()

        success = False
        
        # --- 2. IK 루프 (Newton-Raphson Method) ---
        for i in range(max_iter):
            # (1) 순기구학 (FK) 계산: 현재 q에서의 EE 포즈
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            
            oMtool = self.data.oMf[self.ee_frame_id] # 현재 EE 포즈
            
            # (2) 오차 계산 (Local Frame 기준)
            # dMi: 목표 포즈와 현재 포즈 사이의 차이 변환 행렬
            dMi = oMdes.actInv(oMtool)
            # err: SE3 오차를 6D 벡터(위치+회전)로 변환 (Log map)
            err = pin.log(dMi).vector
            
            # (3) 수렴 확인
            if np.linalg.norm(err) < eps:
                success = True
                break
            
            # (4) 자코비안 계산 (Local Frame 기준)
            J = pin.computeFrameJacobian(self.model, self.data, q, self.ee_frame_id)
            
            # (5) 관절 업데이트 (Damped Least Squares)
            # dq = - (J^T * J + lambda * I)^-1 * J^T * err
            # np.linalg.solve를 사용하여 역행렬 직접 계산 회피
            # 식: (J*J.T + damping*I) * v = -err  (v는 task space velocity) -> J.T * v = dq
            # 하지만 여기서는 더 일반적인 J_dagger * err 형태를 씁니다.
            
            # 간단한 DLS 구현: v = -J.T * (J*J.T + damping*I)^-1 * err
            v = - J.T.dot(np.linalg.solve(J.dot(J.T) + damping * np.eye(6), err))
            
            # (6) q 업데이트 및 적분 (Lie Group manifold 고려)
            q = pin.integrate(self.model, q, v * 1.0) # step size = 1.0
            
            # (7) 관절 한계 클리핑 (Joint Limits)
            q = np.clip(q, self.q_min, self.q_max)
            
        return q, success

# --- 간단한 테스트 코드 ---
if __name__ == "__main__":
    solver = FrankaIKSolver()
    print("✅ IK Solver Initialized.")
    
    # 테스트 목표: 로봇 앞쪽, 약간 위
    target_pos = np.array([0.5, 0.0, 0.5]) 
    # 쿼터니언: 손바닥이 아래를 향함 (대략적인 값) -> w, x, y, z
    # (Pinocchio example robot의 기본 자세는 팔을 위로 뻗은 상태)
    target_quat = np.array([0.0, 1.0, 0.0, 0.0]) 
    
    print(f"Testing IK for Pos: {target_pos}, Quat(wxyz): {target_quat}")
    
    q_sol, success = solver.solve(target_pos, target_quat)
    
    if success:
        print("🎉 IK Success!")
        print(f"Solution q: {q_sol}")
    else:
        print("❌ IK Failed.")