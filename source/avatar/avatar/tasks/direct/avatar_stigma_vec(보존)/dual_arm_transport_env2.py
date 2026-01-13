
from __future__ import annotations
#----------------------얘넨 항상 최상단------------------------
from isaaclab.app import AppLauncher
import argparse
# 이 파일이 직접 실행될 때(__name__ == "__main__")는 여기서 앱을 먼저 켭니다.
# train.py에서 임포트될 때는 이 부분이 건너뛰어지고, train.py가 이미 앱을 켠 상태이므로 OK입니다.
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Env Test Script")
#     parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
#     AppLauncher.add_app_launcher_args(parser)
#     args_cli = parser.parse_args()
    
#     # 앱 실행
#     app_launcher = AppLauncher(args_cli)
#     simulation_app = app_launcher.app


#-----------------------------------------------------------
import math
import torch
import numpy as np

from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.scene import InteractiveScene

import isaaclab.utils.math as math_utils





from dual_arm_transport_cfg import DualrobotCfg
from pose_sampler import DualArmPoseSampler

class DualrobotEnv(DirectRLEnv):
    """
    Dofbot 2대를 스폰하는 환경 클래스입니다.
    """
    cfg: DualrobotCfg # Cfg 클래스 타입 힌트

    def __init__(self, cfg: DualrobotCfg, render_mode: str | None = None, **kwargs):
        # 원본 Cfg를 부모 클래스에 전달
        super().__init__(cfg, render_mode, **kwargs)

        self.pose_sampler = DualArmPoseSampler()
        
        # (관절 인덱스 등은 나중에 필요시 여기에 추가)
        self.robot_1_joint_ids = self.robot_1.actuators["all_joints"].joint_indices
        self.robot_2_joint_ids = self.robot_2.actuators["all_joints"].joint_indices

        self.vel_limit_1 = self.cfg.robot_1.actuators["all_joints"].velocity_limit_sim
        self.vel_limit_2 = self.cfg.robot_2.actuators["all_joints"].velocity_limit_sim
        #
        try:
            self.ee_body_idx_1 = self.robot_1.body_names.index("panda_hand")
            self.ee_body_idx_2 = self.robot_2.body_names.index("panda_hand")
        except ValueError as e:
            raise ValueError(
                "Could not find 'panda_hand' in the robot's body_names."
                "Check the USD file or link name."
            ) from e
        
        #각 에피소드의 목표 상대 포즈
        self.target_ee_rel_poses = torch.zeros(self.num_envs, 7, device=self.device)
        self.violation_occurred = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _setup_scene(self):
        """씬에 모든 에셋을 로드하고 등록합니다."""
        
        # 1. 로봇  로드
        self.robot_1 = Articulation(self.cfg.robot_1)
        self.robot_2 = Articulation(self.cfg.robot_2)
        # 3. 바닥 평면 로드
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # (수정) 목표 마커를 RigidObject로 스폰
        self.goal_marker_ee1 = RigidObject(self.cfg.goal_1)
        self.goal_marker_ee2 = RigidObject(self.cfg.goal_2)
        
        # 4. 씬 복제 (num_envs 개수만큼)
        # (이 시점에 로봇 2대와 바닥이 모두 복제됨)
        self.scene.clone_environments(copy_from_source=False)

        # 5. 씬(Scene)에 로봇들 등록 (필수)
        self.scene.articulations["robot_1"] = self.robot_1
        self.scene.articulations["robot_2"] = self.robot_2

        # Scene 등록
        self.scene.rigid_objects["goal_ee1"] = self.goal_marker_ee1
        self.scene.rigid_objects["goal_ee2"] = self.goal_marker_ee2
        
        # 6. 조명 추가
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """액션을 받아서 저장합니다."""
        # 8차원 액션을 저장
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # 1. 액션 텐서 가져오기 (self.actions는 _pre_physics_step에서 저장됨)
        # Shape: [Num_Envs, 14]
        actions = self.actions.clone()
        
        # 2. 로봇별 액션 분리
        # 앞 7개: Robot 1, 뒤 7개: Robot 2
        actions_1 = actions[:, :7]
        actions_2 = actions[:, 7:]
        # 3. 스케일링 (Normalized -> Physical Value)
        # vel_limit는 cfg에서 정의됨 (보통 1.5 rad/s 등) 학습 초기에는 안전을 위해 0.5 정도로 스케일링 팩터를 곱해주기도 함
        scaling_factor = 1.0 
        
        targets_1 = actions_1 * self.vel_limit_1 * scaling_factor
        targets_2 = actions_2 * self.vel_limit_2 * scaling_factor

        self.robot_1.set_joint_velocity_target(targets_1, joint_ids=self.robot_1_joint_ids)
        self.robot_2.set_joint_velocity_target(targets_2, joint_ids=self.robot_2_joint_ids)

    def _get_observations(self) -> dict:
        """GNN 파이프라인에 필요한 모든 원본 텐서를 수집합니다."""
        
        # ---------------------------------------------------------
        # 1. 데이터 계산
        # ---------------------------------------------------------
        ee_state_1 = self.robot_1.data.body_state_w[:, self.ee_body_idx_1, :] 
        ee_state_2 = self.robot_2.data.body_state_w[:, self.ee_body_idx_2, :] 
        
        # 로봇 노드: [Joint Pos(7), Joint Vel(7)]
        robot_1_data = torch.cat([
            self.robot_1.data.joint_pos[:, self.robot_1_joint_ids], 
            self.robot_1.data.joint_vel[:, self.robot_1_joint_ids]
        ], dim=-1)
        robot_2_data = torch.cat([
            self.robot_2.data.joint_pos[:, self.robot_2_joint_ids],
            self.robot_2.data.joint_vel[:, self.robot_2_joint_ids]
        ], dim=-1)
        
        # Stack Robots [B, 2, 14]
        robot_state = torch.stack([robot_1_data, robot_2_data], dim=1)

        # 목표 포즈 [B, 2, 7]
        goal_pose_1 = self.goal_marker_ee1.data.root_state_w[:, :7] 
        goal_pose_2 = self.goal_marker_ee2.data.root_state_w[:, :7]
        task_state = torch.stack([goal_pose_1, goal_pose_2], dim=1) 
        
        # 현재 EE 포즈 (Edge 계산용) [B, 2, 7]
        current_ee_poses = torch.stack([ee_state_1[:, :7], ee_state_2[:, :7]], dim=1)

        # 베이스 포즈 (Edge 계산용) [B, 2, 7]
        base_pose_1 = self.robot_1.data.root_state_w[:, :7]
        base_pose_2 = self.robot_2.data.root_state_w[:, :7]
        base_poses = torch.stack([base_pose_1, base_pose_2], dim=1)
        
        # 글로벌 (Relative Pose)
        # 1. ee_1의 역회전(inverse rotation) 계산
        ee_state_1_inv_quat = math_utils.quat_conjugate(ee_state_1[:, 3:7])
        # 2. rel_pos 변환
        pos_diff = ee_state_2[:, :3] - ee_state_1[:, :3]
        current_rel_pos = math_utils.quat_apply(ee_state_1_inv_quat, pos_diff)
        # 3. rel_rot 변환
        current_rel_rot = math_utils.quat_mul(ee_state_1_inv_quat, ee_state_2[:, 3:7])
        
        current_rel_pose = torch.cat([current_rel_pos, current_rel_rot], dim=-1)
        target_rel_poses_batch = self.target_ee_rel_poses 

        normalized_time = self.episode_length_buf / self.max_episode_length
        normalized_time = normalized_time.unsqueeze(-1) # [B, 1]

        # global_state = torch.cat([
        #     current_rel_pose,      # [B, 7]
        #     target_rel_poses_batch # [B, 7]
        # ], dim=-1) # [B, 14]
        global_state = torch.cat([
        current_rel_pose,       # [B, 7] (제약조건 정보 - 유지!)
        target_rel_poses_batch, # [B, 7] (제약조건 목표 - 유지!)
        normalized_time         # [B, 1] (시간 정보 - 추가!)
        ], dim=-1) # 최종 [B, 15]
        
        # ---------------------------------------------------------
        # 2. 딕셔너리 조립 (키 이름 중요!)
        # ---------------------------------------------------------
        # graph_converter.py가 이 키 이름들을 찾습니다.
        raw_state_dict = {
            "robot_nodes": robot_state,           # [KeyError 원인 해결]
            "current_ee_poses": current_ee_poses, # Edge 계산용
            "goal_poses": task_state,             # Task Node용
            "base_poses": base_poses,             # Edge 계산용
            "target_rel_pose": self.target_ee_rel_poses, # Global용
            "globals": global_state               # (Legacy, 참고용)
        }

        return {"policy": raw_state_dict}

    #offset 추가 버전
    def _get_rewards(self) -> torch.Tensor:
        # ---------------------------------------------------------
        # 1. Threshold Definitions
        # ---------------------------------------------------------
        curr_thresh = 0.3
        
        # ---------------------------------------------------------
        # 2. 데이터 수집 & 오차 계산 (기존 코드 유지)
        # ---------------------------------------------------------
        ee1_pos = self.robot_1.data.body_state_w[:, self.ee_body_idx_1, 0:3]
        ee2_pos = self.robot_2.data.body_state_w[:, self.ee_body_idx_2, 0:3]
        ee1_quat = self.robot_1.data.body_state_w[:, self.ee_body_idx_1, 3:7]
        ee2_quat = self.robot_2.data.body_state_w[:, self.ee_body_idx_2, 3:7]
        
        goal1_pos = self.goal_marker_ee1.data.root_state_w[:, 0:3]
        goal2_pos = self.goal_marker_ee2.data.root_state_w[:, 0:3]
        target_rel_pos = self.target_ee_rel_poses[:, 0:3]
        target_rel_rot = self.target_ee_rel_poses[:, 3:7]

        # (A) Position Error
        ee1_inv_quat = math_utils.quat_conjugate(ee1_quat)
        curr_rel_pos_local = math_utils.quat_apply(ee1_inv_quat, ee2_pos - ee1_pos)
        pos_error = torch.norm(curr_rel_pos_local - target_rel_pos, dim=-1)

        # (B) Rotation Error
        curr_rel_rot = math_utils.quat_mul(ee1_inv_quat, ee2_quat)
        target_rel_rot_inv = math_utils.quat_conjugate(target_rel_rot)
        q_diff = math_utils.quat_mul(curr_rel_rot, target_rel_rot_inv)
        q_diff_v = q_diff[:, 1:4]
        q_diff_w = q_diff[:, 0]
        rot_error = 2.0 * torch.atan2(torch.norm(q_diff_v, dim=-1), torch.abs(q_diff_w))

        # ---------------------------------------------------------
        # 3. 리워드 계산 (Offset + Linear 적용)
        # ---------------------------------------------------------
        
        # (1) Task Reward (Distance)
        dist_1 = torch.norm(ee1_pos - goal1_pos, dim=-1)
        dist_2 = torch.norm(ee2_pos - goal2_pos, dim=-1)
        r_dist = -(dist_1 + dist_2)

        # (2) Constraint Reward: [Offset + Slope]
        # 위반량(Deadzone) 계산: 0.3 이내면 0, 넘으면 양수
        pos_violation = torch.clamp(pos_error - curr_thresh, min=0.0)
        rot_violation = torch.clamp(rot_error - curr_thresh, min=0.0)
        is_currently_violated = (pos_violation > 1e-4) | (rot_violation > 1e-4)
        
        self.extras["log/is_currently_violated"] = is_currently_violated
        
        # "한 번이라도 위반하면 True로 영구 고정" (OR 연산 누적)
        self.violation_occurred = self.violation_occurred | is_currently_violated

        # 위반 여부 확인 (아주 작은 오차 1e-4 이상이면 위반으로 간주)
        is_violated = (pos_violation > 1e-4) | (rot_violation > 1e-4)

        # A. Offset (Step Penalty): "위반하면 일단 -5점 먹고 시작해라"
        # 경계를 명확히 구분해주는 역할 (Step Function의 효과)
        ct_offset_val = 1.0 #1.0 vs 5.0
        r_step = torch.where(is_violated, -ct_offset_val, 0.0)

        # B. Slope (Linear Penalty): "그래도 줄이면 봐준다"
        # 2차(Square) 대신 1차(Linear)를 사용하여 경계 근처에서도 확실한 기울기 제공
        w_slope = 2.0
        r_slope = -1.0 * w_slope * (pos_violation + rot_violation)

        # 최종 제약 보상 합산
        r_constraint = r_step + r_slope

        # (3) Action & Time Penalty
        action_norm = torch.norm(self.actions, p=2, dim=-1)
        w_action = 0.1 
        r_action = -1.0 * w_action * action_norm

        w_time = 0.1 
        r_time = -1.0 * w_time

        # ---------------------------------------------------------
        # 4. 가중 합산 & 성공 보너스
        # ---------------------------------------------------------
        w_d = 5.0
        r_dist_slope = -1.0 * w_d * (dist_1 + dist_2)

        # 성공 판정
        is_reached = (dist_1 < 0.1) & (dist_2 < 0.1)

        dist_offset_val = 0.5
        r_dist_offset = torch.where(is_reached, 0.0, -dist_offset_val)
        r_dist = r_dist_slope + r_dist_offset

        # "진정한 성공": 현재 도달함 AND 과거에도 깨끗함 AND 현재도 안전함
        is_truly_success = is_reached & (~self.violation_occurred)

        reward_weighted = r_constraint + r_dist + r_action + r_time

        r_success = torch.where(is_truly_success, 20.0, 0.0)

        total_reward = reward_weighted + r_success

        # ---------------------------------------------------------
        # 5. Logging (로그에 r_step 비중 확인용 추가)
        # ---------------------------------------------------------
        self.extras["log/curr_threshold"] = torch.tensor(curr_thresh, device=self.device)
        self.extras["log/r_action"] = torch.mean(r_action)
        self.extras["log/r_dist"] = torch.mean(r_dist)
        self.extras["log/r_constraint"] = torch.mean(r_constraint)
        self.extras["log/r_success"] = torch.mean(r_success)
        self.extras["log/total_reward"] = torch.mean(total_reward)
        self.extras["log/err_pos"] = torch.mean(pos_error) 
        self.extras["log/err_rot"] = torch.mean(rot_error) 
        
        return total_reward


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # ---------------------------------------------------------
        # 2. 데이터 수집 & 오차 계산
        # ---------------------------------------------------------
        ee1_pos = self.robot_1.data.body_state_w[:, self.ee_body_idx_1, 0:3]
        ee2_pos = self.robot_2.data.body_state_w[:, self.ee_body_idx_2, 0:3]
        
        goal1_pos = self.goal_marker_ee1.data.root_state_w[:, 0:3]
        goal2_pos = self.goal_marker_ee2.data.root_state_w[:, 0:3]

        # ---------------------------------------------------------
        # 3. Done Conditions
        # ---------------------------------------------------------
        dist_1 = torch.norm(ee1_pos - goal1_pos, dim=-1)
        dist_2 = torch.norm(ee2_pos - goal2_pos, dim=-1)
        
        # [조건 1] 단순히 거리가 가까워졌는지 (도달 여부)
        is_reached = (dist_1 < 0.1) & (dist_2 < 0.1)
        
        # [조건 2] 진정한 성공 여부 (도달함 AND 위반 기록 없음)
        # 이 값은 로그 기록 및 학습 성공률 판단에 사용됩니다.
        is_success = is_reached & (~self.violation_occurred)
        
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        terminated = is_success #is_reached

        # 로그에는 "깨끗한 성공"만 1.0으로 기록됨
        self.extras["log/success"] = is_success.float()

        # [NEW] 도달 여부와 위반 여부를 extras에 추가 (test.py에서 확인용)
        self.extras["log/is_reached"] = is_reached.float()
        self.extras["log/violation"] = self.violation_occurred.float()
        
        return terminated, time_out


    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot_1._ALL_INDICES
        
        super()._reset_idx(env_ids)
        num_resets = len(env_ids)

        # ---------------------------------------------------------
        # 1. Sampling Loop (CPU)
        # ---------------------------------------------------------
        # 샘플러로부터 Base, Joint, Goal EE Pose를 모두 받아옵니다.
        base_pose_1_list = []
        base_pose_2_list = []
        q_start_1_list = []
        q_start_2_list = []
        # (참고: goal_obj_pose는 보상 계산용으로 받아두지만, 시각화는 EE로 합니다)
        goal_ee1_pose_list = []
        goal_ee2_pose_list = []

        for _ in range(num_resets):
            sample = self.pose_sampler.sample_valid_episode()
            
            base_pose_1_list.append(sample["base_pose_1"])
            base_pose_2_list.append(sample["base_pose_2"])
            q_start_1_list.append(sample["q_start_1"])
            q_start_2_list.append(sample["q_start_2"])
            goal_ee1_pose_list.append(sample["goal_ee1_pose"])
            goal_ee2_pose_list.append(sample["goal_ee2_pose"])

        # ---------------------------------------------------------
        # 2. Tensor Conversion
        # ---------------------------------------------------------
        # Robot Base Poses [N, 7]
        base_pose_1 = torch.tensor(np.array(base_pose_1_list), dtype=torch.float32, device=self.device)
        base_pose_2 = torch.tensor(np.array(base_pose_2_list), dtype=torch.float32, device=self.device)
        
        # Joint Angles [N, 7]
        q_start_1 = torch.tensor(np.array(q_start_1_list), dtype=torch.float32, device=self.device)
        q_start_2 = torch.tensor(np.array(q_start_2_list), dtype=torch.float32, device=self.device)
        
        # Goal EE Poses [N, 7]
        goal_ee1_pose = torch.tensor(np.array(goal_ee1_pose_list), dtype=torch.float32, device=self.device)
        goal_ee2_pose = torch.tensor(np.array(goal_ee2_pose_list), dtype=torch.float32, device=self.device)

        # ---------------------------------------------------------
        # 3. Applying to Sim (Robot)
        # ---------------------------------------------------------
        env_origins = self.scene.env_origins[env_ids]
        zeros_vel = torch.zeros(num_resets, 6, device=self.device)
        zeros_joint_vel = torch.zeros_like(q_start_1)

        # A. Robot 1 (Base + Joint)
        # 샘플러 좌표(0,0,0 기준)에 환경 원점(env_origins)을 더해줍니다.
        r1_world_pose = base_pose_1.clone()
        r1_world_pose[:, :3] += env_origins
        
        self.robot_1.write_root_pose_to_sim(r1_world_pose, env_ids)
        self.robot_1.write_root_velocity_to_sim(zeros_vel, env_ids)
        self.robot_1.write_joint_state_to_sim(q_start_1, zeros_joint_vel, None, env_ids)

        # B. Robot 2 (Base + Joint)
        r2_world_pose = base_pose_2.clone()
        r2_world_pose[:, :3] += env_origins
        
        self.robot_2.write_root_pose_to_sim(r2_world_pose, env_ids)
        self.robot_2.write_root_velocity_to_sim(zeros_vel, env_ids)
        self.robot_2.write_joint_state_to_sim(q_start_2, zeros_joint_vel, None, env_ids)

        # ---------------------------------------------------------
        # 4. Applying to Sim (Markers) - 여기가 시각화 핵심
        # ---------------------------------------------------------
        # 각 로봇 손이 가야 할 목표 위치에 5cm 박스를 배치합니다.
        
        # Goal Marker 1 (EE1 목표 - 빨간 박스 예상)
        marker_pose_1 = goal_ee1_pose.clone()
        marker_pose_1[:, :3] += env_origins
        self.goal_marker_ee1.write_root_pose_to_sim(marker_pose_1, env_ids)
        
        # Goal Marker 2 (EE2 목표 - 파란 박스 예상)
        marker_pose_2 = goal_ee2_pose.clone()
        marker_pose_2[:, :3] += env_origins
        self.goal_marker_ee2.write_root_pose_to_sim(marker_pose_2, env_ids)

        # ---------------------------------------------------------
        # 5. Relative Pose Update (Target Calculation)
        # ---------------------------------------------------------
        # EE1을 기준으로 EE2가 어디에 있어야 하는지(상대 포즈) 계산하여 버퍼에 저장
        
        p1 = goal_ee1_pose[:, 0:3]
        q1 = goal_ee1_pose[:, 3:7] # [w, x, y, z]
        p2 = goal_ee2_pose[:, 0:3]
        q2 = goal_ee2_pose[:, 3:7]

        # Relative Position: q1_inv * (p2 - p1)
        q1_inv = math_utils.quat_conjugate(q1)
        p_diff = p2 - p1
        rel_pos = math_utils.quat_apply(q1_inv, p_diff)

        # Relative Rotation: q1_inv * q2
        rel_rot = math_utils.quat_mul(q1_inv, q2)

        self.target_ee_rel_poses[env_ids] = torch.cat([rel_pos, rel_rot], dim=-1)
        self.violation_occurred[env_ids] = False

if __name__ == "__main__":

    # add argparse arguments
    parser = argparse.ArgumentParser(
        description="This script demonstrates adding a custom robot to an Isaac Lab environment."
    )
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
    # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)
    # parse the arguments
    args_cli = parser.parse_args()
    # Config 및 환경 생성
    env_cfg = DualrobotCfg()
    # args_cli는 위에서 이미 파싱됨
    env_cfg.scene.num_envs = args_cli.num_envs
    
    env = DualrobotEnv(cfg=env_cfg, render_mode="human")
    env.reset()
    
    # 시뮬레이션 루프
    while simulation_app.is_running():
        # 랜덤 액션 테스트
        actions = 2 * torch.rand(env.num_envs, env.cfg.action_space, device=env.device) - 1
        obs, rew, terminated, truncated, info = env.step(actions)
        
        if truncated.any() or terminated.any():
            print(f"[Info] Reset triggered")
            
    env.close()
    simulation_app.close()

# def main():
    
#     """
#     현재 환경(DualDofbotEnv)을 로드하고 시각적으로 테스트하기 위한 
#     메인 함수입니다.
#     """
    
#     # 1. Cfg 인스턴스 생성 (기본값으로)
#     env_cfg = DualrobotCfg()

#     # 2. (수정) Cfg의 'num_envs' 속성을 CLI 인수로 덮어쓰기
#     # num_envs는 'scene' 객체 내부에 있습니다.
#     env_cfg.scene.num_envs = args_cli.num_envs
    
#     # 3. 환경 생성
#     # (시각적 테스트를 위해 render_mode="human"을 명시적으로 설정합니다)
#     env = DualrobotEnv(cfg=env_cfg, render_mode="human")

#     # 4. 환경 리셋
#     env.reset()
    
#     # 4. 시뮬레이션 루프
#     while simulation_app.is_running():
#         # 환경이 시뮬레이션 중일 때만
#         #if env.is_running():
#             # (1) 무작위 액션 생성 [num_envs, 14]
#             #     (-1 ~ 1 사이의 값으로 정규화된 타겟 위치)
#         actions = 2 * torch.rand(env.num_envs, env.cfg.action_space, device=env.device) - 1
            
#             # (2) 환경 스텝 실행
#         obs, rew, terminated, truncated, info = env.step(actions)
            
#             # (3) (선택 사항) 일정 스텝마다 리셋(새 에피소드)
#         # time_outs(truncated)이나 terminated가 True인 환경이 하나라도 있으면 리셋된 것임
#         if truncated.any() or terminated.any():
#             print(f"[Info] Reset triggered at step {env.common_step_counter}")
#     # 5. 환경 종료
#     env.close()


# if __name__ == "__main__":
#     app_launcher = AppLauncher()
#     simulation_app = app_launcher.app
#     # AppLauncher가 CLI 인수를 파싱하고 시뮬레이션을 시작합니다.
#     # (파일 상단에 이미 app_launcher.app이 정의되어 있음)
#     try:
#         # 메인 함수 실행
#         main()
#     except Exception as e:
#         print(f"FATAL ERROR: {e}")
#     finally:
#         # 시뮬레이션 앱 종료
#         simulation_app.close()