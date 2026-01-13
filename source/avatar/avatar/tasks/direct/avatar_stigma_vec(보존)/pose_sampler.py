#일정 거리 이상
import numpy as np
import pinocchio as pin
from ik_solver import FrankaIKSolver

class DualArmPoseSampler:
    def __init__(self):
        self.ik_solver = FrankaIKSolver()
        # 베이스 위치 랜덤 범위 설정 (예시)
        # Robot 1 (Left): x=[-0.6, -0.4], y=[-0.1, 0.1]
        self.r1_pos_range = ([-0.5, -0.45], [0.0, 0.0]) 
        # Robot 2 (Right): x=[0.4, -0.1], y=[0.6, 0.1]
        self.r2_pos_range = ([0.45, 0.5], [0.0, 0.0])

    def _get_quat_from_euler(self, roll, pitch, yaw):
        """Euler(XYZ) -> Quat(wxyz)"""
        cR = np.cos(roll/2); sR = np.sin(roll/2)
        cP = np.cos(pitch/2); sP = np.sin(pitch/2)
        cY = np.cos(yaw/2); sY = np.sin(yaw/2)
        w = cR*cP*cY + sR*sP*sY
        x = sR*cP*cY - cR*sP*sY
        y = cR*sP*cY + sR*cP*sY
        z = cR*cP*sY - sR*sP*cY
        return np.array([w, x, y, z])

    def _to_local(self, world_pos, world_quat, base_pos, base_quat):
        """
        World Frame의 Pose를 움직이는 Base Frame 기준의 Local Pose로 변환
        T_local = T_base_world^(-1) * T_world_target
        """
        # 1. Base SE3
        w, x, y, z = base_quat
        b_rot = pin.Quaternion(np.array([x, y, z, w])).matrix()
        T_world_base = pin.SE3(b_rot, base_pos)
        
        # 2. Target(World) SE3
        w, x, y, z = world_quat
        t_rot = pin.Quaternion(np.array([x, y, z, w])).matrix()
        T_world_target = pin.SE3(t_rot, world_pos)
        
        # 3. Calc Inverse: T_base_target = T_world_base.inv() * T_world_target
        T_base_target = T_world_base.actInv(T_world_target)
        
        # 4. Extract
        loc_pos = T_base_target.translation
        loc_rot = pin.Quaternion(T_base_target.rotation).coeffs() # [x, y, z, w]
        
        return loc_pos, np.array([loc_rot[3], loc_rot[0], loc_rot[1], loc_rot[2]]) # wxyz

    def _get_ee_pose_from_object(self, obj_pos, obj_quat, offset_pos, offset_quat):
        """ (이전과 동일) Object Pose + Offset -> EE World Pose """
        w, x, y, z = obj_quat
        obj_rot = pin.Quaternion(np.array([x, y, z, w])).matrix()
        T_world_obj = pin.SE3(obj_rot, obj_pos)
        
        w, x, y, z = offset_quat
        offset_rot = pin.Quaternion(np.array([x, y, z, w])).matrix()
        T_obj_ee = pin.SE3(offset_rot, offset_pos)
        
        T_world_ee = T_world_obj * T_obj_ee
        
        ee_pos = T_world_ee.translation
        ee_rot_mat = T_world_ee.rotation
        ee_quat_pin = pin.Quaternion(ee_rot_mat).coeffs()
        
        return ee_pos, np.array([ee_quat_pin[3], ee_quat_pin[0], ee_quat_pin[1], ee_quat_pin[2]])

    def _get_random_grasp_quat(self, base_roll, base_pitch, base_yaw):
        """
        기본(Base) 자세에 약간의 랜덤 노이즈를 섞어서 반환
        """
        # 노이즈 범위 (라디안): 예) +/- 15도 (0.26 rad)
        noise_range = 0.05 
        
        # Roll, Pitch, Yaw에 각각 노이즈 추가
        # (주의: 파지 방향에 따라 노이즈를 주면 안 되는 축이 있을 수 있음.
        #  여기서는 3축 모두 약간씩 흔듦)
        r_noise = np.random.uniform(-noise_range, noise_range)
        p_noise = np.random.uniform(-noise_range, noise_range)
        y_noise = np.random.uniform(-noise_range, noise_range)
        
        return self._get_quat_from_euler(
            base_roll + r_noise, 
            base_pitch + p_noise, 
            base_yaw + y_noise
        )
    
    def sample_valid_episode(self, max_retries=100):
        """
        베이스 위치까지 랜덤 샘플링하여 유효한 에피소드 생성
        """
        for _ in range(max_retries):
            # ---------------------------------------------------
            # 1. [Random Base Pose] 베이스 위치/회전 먼저 결정
            # ---------------------------------------------------
            # Robot 1
            # range[0]은 X범위, range[1]은 Y범위입니다.
            r1_x = np.random.uniform(self.r1_pos_range[0][0], self.r1_pos_range[0][1])
            r1_y = np.random.uniform(self.r1_pos_range[1][0], self.r1_pos_range[1][1])
            
            # 회전(Orientation)은 여전히 랜덤입니다 (+/- 45도)
            # 만약 회전도 고정하고 싶다면 r1_yaw = 0.0 으로 바꾸세요.
            r1_yaw = np.random.uniform(-np.pi/4, np.pi/4) 
            
            base_pos_1 = np.array([r1_x, r1_y, 0.0])
            base_quat_1 = self._get_quat_from_euler(0.0, 0.0, r1_yaw)
            
            # Robot 2
            r2_x = np.random.uniform(self.r2_pos_range[0][0], self.r2_pos_range[0][1])
            r2_y = np.random.uniform(self.r2_pos_range[1][0], self.r2_pos_range[1][1])
            
            r2_yaw = np.random.uniform(np.pi - np.pi/4, np.pi + np.pi/4)
            
            base_pos_2 = np.array([r2_x, r2_y, 0.0])
            base_quat_2 = self._get_quat_from_euler(0.0, 0.0, r2_yaw)

            # ---------------------------------------------------
            # 2. 물체 및 파지 전략 설정
            # ---------------------------------------------------
            width = np.random.uniform(0.8, 0.8) #물체 너비 설정
            half_width = width / 2.0
            
            # [수정] 그립 자세 랜덤화
            # Robot 1 (왼쪽 면 잡기): 기본은 (0, 90도, 0)
            # 여기서 랜덤 노이즈를 추가해 "약간 비스듬히 잡기" 등을 구현
            off_pos_1 = np.array([-half_width, 0.0, 0.0])
            off_quat_1 = self._get_random_grasp_quat(0.0, np.pi/4, 0.0)

            # Robot 2 (오른쪽 면 잡기): 기본은 (0, -90도, 0)
            off_pos_2 = np.array([half_width, 0.0, 0.0])
            off_quat_2 = self._get_random_grasp_quat(0.0, -np.pi/4, 0.0)

            # # Robot 1 (Left Side of Object)
            # off_pos_1 = np.array([-half_width, 0.0, 0.0])
            # off_quat_1 = self._get_quat_from_euler(0.0, np.pi/2, 0.0) 
            # # Robot 2 (Right Side of Object)
            # off_pos_2 = np.array([half_width, 0.0, 0.0])
            # off_quat_2 = self._get_quat_from_euler(0.0, -np.pi/2, 0.0)

            # ---------------------------------------------------
            # 3. Start Pose Search
            # ---------------------------------------------------
            q_start_1, q_start_2 = None, None
            start_obj_pos, start_obj_quat = None, None
            
            for _ in range(10): # 베이스가 정해진 상태에서 물체 위치 탐색
                # 물체는 두 로봇 베이스의 중간 쯤에 위치하도록 범위 설정
                center_x = (r1_x + r2_x) / 2.0
                center_y = (r1_y + r2_y) / 2.0
                
                pos = np.random.uniform([center_x-0.1, center_y-0.1, 0.4], 
                                        [center_x+0.1, center_y+0.1, 0.8])
                yaw = np.random.uniform(-np.pi/4, np.pi/4)
                quat = self._get_quat_from_euler(0.0, 0.0, yaw)

                # EE World Pose
                ee1_pos, ee1_quat = self._get_ee_pose_from_object(pos, quat, off_pos_1, off_quat_1)
                ee2_pos, ee2_quat = self._get_ee_pose_from_object(pos, quat, off_pos_2, off_quat_2)

                # [핵심] World -> Local 변환 (이제 Base Rotation도 고려됨)
                loc1_pos, loc1_quat = self._to_local(ee1_pos, ee1_quat, base_pos_1, base_quat_1)
                loc2_pos, loc2_quat = self._to_local(ee2_pos, ee2_quat, base_pos_2, base_quat_2)

                q1, succ1 = self.ik_solver.solve(loc1_pos, loc1_quat)
                if not succ1: continue
                q2, succ2 = self.ik_solver.solve(loc2_pos, loc2_quat)
                if not succ2: continue

                q_start_1, q_start_2 = q1, q2
                start_obj_pos, start_obj_quat = pos, quat
                break
            
            if q_start_1 is None: continue 

            # ---------------------------------------------------
            # 4. Goal Pose Search
            # ---------------------------------------------------
            q_goal_1, q_goal_2 = None, None
            goal_obj_pos, goal_obj_quat = None, None
            g_ee1_pos, g_ee1_quat = None, None
            g_ee2_pos, g_ee2_quat = None, None
            
            # [수정 1] 탐색 횟수를 10 -> 50 정도로 늘려줍니다 (조건이 까다로워졌으므로)
            for _ in range(50):
                center_x = (r1_x + r2_x) / 2.0
                center_y = (r1_y + r2_y) / 2.0
                
                # [수정 2] Goal 탐색 범위를 기존 +/- 0.2에서 +/- 0.3~0.4로 약간 넓힙니다.
                # 범위가 너무 좁으면 0.5m 떨어진 점을 찾기 어렵습니다.
                # 단, 로봇 팔 길이를 고려해 너무 멀면 IK에서 걸러집니다.
                pos = np.random.uniform([center_x-0.35, center_y-0.35, 0.3], 
                                        [center_x+0.35, center_y+0.35, 0.85])
                
                # [NEW] 최소 거리 조건 체크 (50cm 이상)
                # 시작 위치(start_obj_pos)와 현재 샘플링된 목표 위치(pos) 사이 거리 계산
                dist = np.linalg.norm(pos - start_obj_pos)
                if dist < 0.1:
                    continue  # 거리가 너무 가까우면 다시 뽑기

                yaw = np.random.uniform(-np.pi/4, np.pi/4)
                quat = self._get_quat_from_euler(0.0, 0.0, yaw)

                ee1_pos, ee1_quat = self._get_ee_pose_from_object(pos, quat, off_pos_1, off_quat_1)
                ee2_pos, ee2_quat = self._get_ee_pose_from_object(pos, quat, off_pos_2, off_quat_2)

                loc1_pos, loc1_quat = self._to_local(ee1_pos, ee1_quat, base_pos_1, base_quat_1)
                loc2_pos, loc2_quat = self._to_local(ee2_pos, ee2_quat, base_pos_2, base_quat_2)

                q1, succ1 = self.ik_solver.solve(loc1_pos, loc1_quat)
                if not succ1: continue
                q2, succ2 = self.ik_solver.solve(loc2_pos, loc2_quat)
                if not succ2: continue

                q_goal_1, q_goal_2 = q1, q2
                goal_obj_pos, goal_obj_quat = pos, quat
                g_ee1_pos, g_ee1_quat = ee1_pos, ee1_quat
                g_ee2_pos, g_ee2_quat = ee2_pos, ee2_quat
                break
            
            # 50번 시도해도 조건(거리+IK)을 만족 못하면 에피소드 전체 다시 생성 (continue)
            if q_goal_1 is None: continue

            # ---------------------------------------------------
            # 5. 결과 반환 (Base Pose 포함)
            # ---------------------------------------------------
            return {
                # Base Poses (World) - 중요! Env가 로봇을 이 위치로 옮겨야 함
                "base_pose_1": np.concatenate([base_pos_1, base_quat_1]),
                "base_pose_2": np.concatenate([base_pos_2, base_quat_2]),
                
                # Joint Angles
                "q_start_1": q_start_1,
                "q_start_2": q_start_2,
                
                # Object Poses
                "start_obj_pose": np.concatenate([start_obj_pos, start_obj_quat]),
                "goal_obj_pose": np.concatenate([goal_obj_pos, goal_obj_quat]),
                
                # EE Poses (for Relative Pose calculation)
                "goal_ee1_pose": np.concatenate([g_ee1_pos, g_ee1_quat]),
                "goal_ee2_pose": np.concatenate([g_ee2_pos, g_ee2_quat]),

                "obj_width": width
            }
            
        raise RuntimeError("Failed to sample valid episode with random bases")