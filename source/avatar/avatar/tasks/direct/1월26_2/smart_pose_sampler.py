import numpy as np
import pinocchio as pin
from ik_solver import FrankaIKSolver

class SmartPoseSampler:
    def __init__(self):
        self.ik_solver = FrankaIKSolver()
        
        # 로봇 팔의 안전 작업 반경 (최대 0.855m지만 여유를 둠)
        self.max_reach = 0.75 
        self.min_reach = 0.30 # 너무 가까우면 충돌 위험

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
        """World Frame -> Base Frame 변환"""
        w, x, y, z = base_quat
        b_rot = pin.Quaternion(np.array([x, y, z, w])).matrix()
        T_world_base = pin.SE3(b_rot, base_pos)
        
        w, x, y, z = world_quat
        t_rot = pin.Quaternion(np.array([x, y, z, w])).matrix()
        T_world_target = pin.SE3(t_rot, world_pos)
        
        T_base_target = T_world_base.actInv(T_world_target)
        
        loc_pos = T_base_target.translation
        loc_rot = pin.Quaternion(T_base_target.rotation).coeffs() # [x, y, z, w]
        return loc_pos, np.array([loc_rot[3], loc_rot[0], loc_rot[1], loc_rot[2]])

    def _get_ee_pose_from_object(self, obj_pos, obj_quat, offset_pos, offset_quat):
        w, x, y, z = obj_quat
        obj_rot = pin.Quaternion(np.array([x, y, z, w])).matrix()
        T_world_obj = pin.SE3(obj_rot, obj_pos)
        
        w, x, y, z = offset_quat
        offset_rot = pin.Quaternion(np.array([x, y, z, w])).matrix()
        T_obj_ee = pin.SE3(offset_rot, offset_pos)
        
        T_world_ee = T_world_obj * T_obj_ee
        return T_world_ee.translation, pin.Quaternion(T_world_ee.rotation).coeffs()

    def _get_random_grasp_quat(self, base_roll, base_pitch, base_yaw):
        noise_range = 0.05 
        return self._get_quat_from_euler(
            base_roll + np.random.uniform(-noise_range, noise_range), 
            base_pitch + np.random.uniform(-noise_range, noise_range), 
            base_yaw + np.random.uniform(-noise_range, noise_range)
        )

    def sample_valid_episode(self, required_dist=0.7, max_retries=100):
        """
        [Inverse Strategy]
        1. 경로(Start -> Goal)를 먼저 생성
        2. 해당 경로를 커버할 수 있는 위치에 로봇 베이스를 배치
        """
        for _ in range(max_retries):
            # ---------------------------------------------------
            # 1. 경로 생성 (Trajectory Generation)
            # ---------------------------------------------------
            # 월드 중심 근처에서 시작 위치 선정
            start_x = np.random.uniform(-0.1, 0.1)
            start_y = np.random.uniform(-0.1, 0.1)
            start_z = np.random.uniform(0.4, 0.6) # 높이 적당히
            start_obj_pos = np.array([start_x, start_y, start_z])

            # 이동 방향 랜덤 선정 (XY 평면 위주)
            theta = np.random.uniform(0, 2*np.pi)
            direction = np.array([np.cos(theta), np.sin(theta), 0]) # Z축 이동은 최소화 (난이도 조절)
            
            # 목표 위치 계산 (Start에서 required_dist만큼 떨어진 곳)
            goal_dist = required_dist + np.random.uniform(0.0, 0.1) # 0.7 ~ 0.8m
            goal_obj_pos = start_obj_pos + direction * goal_dist

            # 물체 회전 (Yaw만 랜덤)
            obj_yaw = np.random.uniform(-np.pi/4, np.pi/4)
            obj_quat = self._get_quat_from_euler(0.0, 0.0, obj_yaw)

            # ---------------------------------------------------
            # 2. 로봇 베이스 배치 (Heuristic Placement)
            # ---------------------------------------------------
            # 논리: Start와 Goal의 '중점(Midpoint)'을 기준으로 로봇을 양옆에 배치하면
            #       Start와 Goal 모두에 닿을 확률이 가장 높음.
            
            mid_point = (start_obj_pos + goal_obj_pos) / 2.0
            
            # 경로의 수직 벡터 계산 (로봇을 경로의 좌우에 배치하기 위함)
            # direction = (x, y, 0) -> perp = (-y, x, 0)
            perp_vec = np.array([-direction[1], direction[0], 0])
            
            # 로봇이 경로에서 떨어질 거리 (Offset)
            # 너무 가까우면(0.3) 몸통 충돌, 너무 멀면(0.7) 닿지 않음. 0.5m가 적당 (피타고라스 정리: sqrt(0.35^2 + 0.5^2) = 0.61m < Max Reach)
            base_offset_dist = np.random.uniform(0.45, 0.60) 
            
            # Robot 1 Base (경로 왼쪽)
            base_pos_1 = mid_point + perp_vec * base_offset_dist
            base_pos_1[2] = 0.0 # 바닥에 고정
            
            # Robot 2 Base (경로 오른쪽)
            base_pos_2 = mid_point - perp_vec * base_offset_dist
            base_pos_2[2] = 0.0 

            # 베이스 회전 (항상 중점을 바라보도록 설정 + 약간의 노이즈)
            # Robot 1 -> Midpoint 바라보기
            vec_to_mid_1 = mid_point - base_pos_1
            yaw_1 = np.arctan2(vec_to_mid_1[1], vec_to_mid_1[0]) + np.random.uniform(-0.2, 0.2)
            base_quat_1 = self._get_quat_from_euler(0.0, 0.0, yaw_1)

            # Robot 2 -> Midpoint 바라보기
            vec_to_mid_2 = mid_point - base_pos_2
            yaw_2 = np.arctan2(vec_to_mid_2[1], vec_to_mid_2[0]) + np.random.uniform(-0.2, 0.2)
            base_quat_2 = self._get_quat_from_euler(0.0, 0.0, yaw_2)

            # ---------------------------------------------------
            # 3. 파지 오프셋 (Grasp Setup)
            # ---------------------------------------------------
            width = 0.8
            half_width = width / 2.0
            
            # 로봇 1은 물체의 왼쪽(-X or Perp Left), 로봇 2는 오른쪽 잡기
            # 간단하게 물체 기준 로컬 좌표계 사용
            off_pos_1 = np.array([-half_width, 0.0, 0.0])
            off_quat_1 = self._get_random_grasp_quat(0.0, np.pi/4, 0.0)

            off_pos_2 = np.array([half_width, 0.0, 0.0])
            off_quat_2 = self._get_random_grasp_quat(0.0, -np.pi/4, 0.0)

            # ---------------------------------------------------
            # 4. IK Verification (Start & Goal 모두 가능한지 확인)
            # ---------------------------------------------------
            
            # Check Start Pose
            ee1_start, q1_start = self._get_ee_pose_from_object(start_obj_pos, obj_quat, off_pos_1, off_quat_1)
            ee2_start, q2_start = self._get_ee_pose_from_object(start_obj_pos, obj_quat, off_pos_2, off_quat_2)
            
            # World EE -> Local EE -> IK
            loc1_s, rot1_s = self._to_local(ee1_start, np.array([q1_start[3], q1_start[0], q1_start[1], q1_start[2]]), base_pos_1, base_quat_1)
            loc2_s, rot2_s = self._to_local(ee2_start, np.array([q2_start[3], q2_start[0], q2_start[1], q2_start[2]]), base_pos_2, base_quat_2)

            q_start_1, succ1_s = self.ik_solver.solve(loc1_s, rot1_s)
            q_start_2, succ2_s = self.ik_solver.solve(loc2_s, rot2_s)
            
            if not (succ1_s and succ2_s):
                continue # 시작 자세 불가

            # Check Goal Pose
            ee1_goal, q1_goal = self._get_ee_pose_from_object(goal_obj_pos, obj_quat, off_pos_1, off_quat_1)
            ee2_goal, q2_goal = self._get_ee_pose_from_object(goal_obj_pos, obj_quat, off_pos_2, off_quat_2)

            loc1_g, rot1_g = self._to_local(ee1_goal, np.array([q1_goal[3], q1_goal[0], q1_goal[1], q1_goal[2]]), base_pos_1, base_quat_1)
            loc2_g, rot2_g = self._to_local(ee2_goal, np.array([q2_goal[3], q2_goal[0], q2_goal[1], q2_goal[2]]), base_pos_2, base_quat_2)

            q_goal_1_dummy, succ1_g = self.ik_solver.solve(loc1_g, rot1_g)
            q_goal_2_dummy, succ2_g = self.ik_solver.solve(loc2_g, rot2_g)

            if not (succ1_g and succ2_g):
                continue # 목표 자세 불가

            # 모두 통과!
            return {
                "base_pose_1": np.concatenate([base_pos_1, base_quat_1]),
                "base_pose_2": np.concatenate([base_pos_2, base_quat_2]),
                "q_start_1": q_start_1,
                "q_start_2": q_start_2,
                "start_obj_pose": np.concatenate([start_obj_pos, obj_quat]),
                "goal_obj_pose": np.concatenate([goal_obj_pos, obj_quat]),
                "goal_ee1_pose": np.concatenate([ee1_goal, np.array([q1_goal[3], q1_goal[0], q1_goal[1], q1_goal[2]])]),
                "goal_ee2_pose": np.concatenate([ee2_goal, np.array([q2_goal[3], q2_goal[0], q2_goal[1], q2_goal[2]])]),
                "obj_width": width
            }

        raise RuntimeError("Failed to sample smart episode")
