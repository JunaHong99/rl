from __future__ import annotations
#----------------------얘넨 항상 최상단------------------------
from isaaclab.app import AppLauncher
import argparse
#-----------------------------------------------------------
import math
import torch
import numpy as np

from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.scene import InteractiveScene

import isaaclab.utils.math as math_utils

from dual_arm_transport_cfg import DualrobotCfg
from vectorized_pose_sampler import VectorizedPoseSampler, CachedPoseSampler
from safety_filter import SafetyFilter
from franka_jacobian_ik import FrankaJacobianIK
from cooperative_impedance_controller import CooperativeImpedanceController
from per_arm_impedance_controller import PerArmImpedanceController
from per_arm_velocity_controller import PerArmVelocityController

class DualrobotEnv(DirectRLEnv):
    """
    Dofbot 2대를 스폰하는 환경 클래스입니다.
    Potential-based Reward (PBR) 적용 버전.
    """
    cfg: DualrobotCfg # Cfg 클래스 타입 힌트

    # ── Grasp 결합 강성 (2026-06-15) ── None=rigid fixed joint(옛), 값이면 compliant D6 spring.
    # compliant → arm K가 grasp 힘/내력 제어 (가변 임피던스 연구 전제). Step A/B에서 튜닝.
    # _setup_scene(super().__init__ 중 실행)에서 읽으므로 반드시 클래스 속성이어야 함.
    # 2026-06-16 revert: compliant D6는 단순 Jᵀ 컨트롤러를 발산시킴(모델기반 제어 필요=future work).
    # → rigid fixed joint 복귀(None). 중력 ON+보상은 유지. 연구는 rigid 폐루프 + GNN 형태일반화로.
    GRASP_K_COUPLE = None      # None=rigid fixed joint (값이면 compliant D6)
    GRASP_D_COUPLE = 100.0
    GRASP_K_ROT = 1.0
    GRASP_D_ROT = 5.0

    def __init__(self, cfg: DualrobotCfg, render_mode: str | None = None, **kwargs):
        # 원본 Cfg를 부모 클래스에 전달
        super().__init__(cfg, render_mode, **kwargs)

        # Phase A (2026-05-26): CachedPoseSampler 사용. 학습 시작 시 100k samples 사전 생성 +
        # 파일 저장. 매 reset은 cache에서 O(1) random pick → PoseSampler 호출 폭주 회피.
        self.pose_sampler = CachedPoseSampler(
            device=self.device, cache_size=100_000, fixed_grasp_roll=True
        )
        self.external_samples = None # 외부 샘플 저장용 (테스트용)

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
        
        self.safety_filter = SafetyFilter(self)

        #각 에피소드의 목표 상대 포즈
        self.target_ee_rel_poses = torch.zeros(self.num_envs, 7, device=self.device)
        self.violation_occurred = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # [NEW] 에피소드 내 최대 오차 추적용 버퍼
        self.episode_max_pos_error = torch.zeros(self.num_envs, device=self.device)
        self.episode_max_rot_error = torch.zeros(self.num_envs, device=self.device)

        # [NEW] Joint Space Reward용 버퍼
        self.target_joint_pos = torch.zeros(self.num_envs, 14, device=self.device)
        self.prev_joint_dist = torch.full((self.num_envs,), float('inf'), device=self.device)
        self.prev_dist = torch.full((self.num_envs,), float('inf'), device=self.device)

        # [Phase 3.3] One-shot success bonus용 — 도달 후 termination 안 하고 episode 계속.
        # Episode 내 첫 도달 시에만 +200 bonus. 이후는 r_goal (dense, 거리 0이면 ≈0)만.
        self.reached_once = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # [Phase 3.3] Episode 내 최소 오차 추적 (timeout 시 평균 측정용 metric)
        # 평균 pos_err은 stuck 케이스에 흐려짐. min은 "가장 가까이 갔던 거리"라
        # episode_success_rate보다 부드럽고 학습 진척 보기 좋음.
        self.episode_min_pos_err = torch.full((self.num_envs,), float("inf"), device=self.device)
        self.episode_min_rot_err = torch.full((self.num_envs,), float("inf"), device=self.device)

        # ── Phase 3: Cooperative impedance controller ──
        # action(3-dim positional delta)을 누적해 target_obj_pos를 만듦
        # target_obj_quat은 reset 시점 rod 자세로 고정 (회전은 v1에서 미사용)
        self.target_obj_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_obj_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.target_obj_quat[:, 0] = 1.0  # identity quat as default

        # Phase 2 (Cooperative refactor):
        # target_x_rel — 두 EE 사이 desired 분리 벡터 (world frame).
        # reset 시 sampler의 start_obj_pose로부터 해석적으로 계산해 저장.
        # Fixed-joint setup이라 episode 내내 일정 값으로 유지.
        self.target_x_rel = torch.zeros(self.num_envs, 3, device=self.device)

        # 장애물 (cluttered transport): 활성 마스크 + 이전 action(부드러움 페널티용).
        self.obstacle_active = torch.zeros(
            self.num_envs, self.cfg.n_obstacles, dtype=torch.bool, device=self.device
        )
        self._prev_action = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        # 장애물 curriculum: 0(운반만)→1(전체) frac. train script가 step에 따라 설정.
        # 운반을 먼저 학습(Stage 0처럼) 후 장애물 점진 추가 → "안 움직이고 회피만" 국소최적 회피.
        self._obstacle_curr_frac = 1.0

        # 2026-05-29: 원복 — per-arm impedance (torque mode). velocity 전환 추후 재시도.
        # 2026-06-05: cooperative 교체 시도했으나 hold/goal test에서 동일하거나 더 나쁨.
        #   원인은 협동 부재 아니라 (1) test_tracking.py hold mode가 episode auto-reset
        #   (6s마다) 무시한 것 (2) 그래도 첫 episode에서 5cm→16mm 안에 들어옴 (per-arm).
        self.controller = PerArmImpedanceController(self)
        # Stage 1 per-arm K: action[6:8] ∈ [-1,1] → K_base·exp(ln4·a), K ∈ [K_base/4, K_base*4].
        self._K_LOG_RANGE = math.log(4.0)
        self.K_arm1 = None
        self.K_arm2 = None
        self.K_arm_prev = None  # (K1,K2) 직전 step — K smoothness 보상용
        # Stage 1 architecture C: 내력 보상 가중치 (train_stage1_K.py가 설정; 0=off=Stage0 무영향)
        self._internal_w = 0.0
        self._ksmooth_w = 0.0
        self._f_safe = 0.0
        self._k_reward_mode = False   # True면 K 정책용 dense 보상(track+내력), sparse success 미사용
        self._w_track = 0.0           # track_err 페널티 가중치
        self._last_ctrl_info = {}  # 진단용 (외부 스크립트에서 읽기 위함)

        # ── Post-reset settle (2026-06-08): reset 직후 PhysX fixed-joint snap이
        # 시스템에 큰 운동에너지 주입 → 일부 envs (~11%)에서 controller가 그 transient를 못 잡아
        # drift 폭주. settle 30 step (zero action, target=rod_init 유지)으로 transient 흡수.
        # diagnose_unstable.py 검증: settle 10→30이 unstable% 11% → 4% 감소.
        # episode_length_buf는 settle 중 0으로 freeze → RL episode 길이 정상 유지.
        self.SETTLE_STEPS_AT_RESET = 30
        self._settle_remaining = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Per-env tracking: 현재 episode의 cache index (실패 episode 재현용)
        # external_samples 사용 중일 땐 -1 유지.
        self.current_sample_idxs = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)


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

        # 공유 강체 (rod) + 목표 위치 표시용 marker
        self.rod = RigidObject(self.cfg.rod)
        self.goal_rod_marker = RigidObject(self.cfg.goal_rod)

        # 장애물 (cluttered transport): n_obstacles개 kinematic 구. 충돌은 해석적 clearance로
        # 보상 처리하므로 collision_enabled=False (물리 접촉 X, 시각+상태만).
        self.obstacles = []
        for i in range(self.cfg.n_obstacles):
            obs_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/obstacle_{i}",
                spawn=sim_utils.SphereCfg(
                    radius=self.cfg.obstacle_radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True, disable_gravity=True,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                    physics_material=None,
                ),
            )
            self.obstacles.append(RigidObject(obs_cfg))

        self._add_rod_fixed_joints_template()

        # 4. 씬 복제 (num_envs 개수만큼)
        # (이 시점에 로봇 2대와 바닥이 모두 복제됨)
        self.scene.clone_environments(copy_from_source=False)

        # 5. 씬(Scene)에 로봇들 등록 (필수)
        self.scene.articulations["robot_1"] = self.robot_1
        self.scene.articulations["robot_2"] = self.robot_2

        # Scene 등록
        self.scene.rigid_objects["goal_ee1"] = self.goal_marker_ee1
        self.scene.rigid_objects["goal_ee2"] = self.goal_marker_ee2
        self.scene.rigid_objects["rod"] = self.rod
        self.scene.rigid_objects["goal_rod"] = self.goal_rod_marker
        for i, obs in enumerate(self.obstacles):
            self.scene.rigid_objects[f"obstacle_{i}"] = obs
        
        # 6. 조명 추가
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _add_rod_fixed_joints_template(self):
        """
        Phase 2 (TCP 정합): env_0 템플릿에 fixed joint 2개를 추가.
        - joint_1: Robot_1.panda_hand <-> rod (rod 왼쪽 끝)
        - joint_2: Robot_2.panda_hand <-> rod (rod 오른쪽 끝)

        anchor 정의 (sampler가 grasp_roll=π로 고정 → panda_hand가 rod에 대해 Rx(π) 회전):
        - body0(panda_hand) local: pos=(0, 0, +TCP_OFFSET), rot=identity
            → anchor가 TCP(fingers 사이) 위치
        - body1(rod) local:        pos=(±0.4, 0, 0), rot=Rx(π)=(0,1,0,0)
            → anchor가 rod 양 끝, frame은 panda_hand 방향에 맞춤

        검증: panda_hand_world = rod_end + (0,0,+TCP_OFFSET), R_panda_hand = R_rod × Rx(π)
              ⇒ TCP world via body0 = rod_end ✓
              ⇒ 회전 정합 R_panda_hand × identity = R_rod × Rx(π) ✓
        """
        import omni.usd
        from pxr import UsdPhysics, Gf, Sdf
        from vectorized_pose_sampler import VectorizedPoseSampler

        stage = omni.usd.get_context().get_stage()
        base = "/World/envs/env_0"
        half_width = 0.4                       # rod 길이 0.8m / 2
        tcp_z = VectorizedPoseSampler.TCP_OFFSET  # 0.1034m
        # Rx(π) quaternion in wxyz: (cos(π/2), sin(π/2), 0, 0) = (0, 1, 0, 0)
        rx_pi = Gf.Quatf(0.0, 1.0, 0.0, 0.0)
        identity = Gf.Quatf(1.0, 0.0, 0.0, 0.0)

        # ── Grasp 결합: K_couple=None이면 rigid fixed(옛), 값이면 compliant D6 spring(2026-06-15) ──
        # compliant로 하면 arm K가 grasp 힘/내력을 제어 → 가변 임피던스 연구 전제 성립.
        K_couple = getattr(self, "GRASP_K_COUPLE", None)   # 선형 N/m (None=rigid)
        D_couple = getattr(self, "GRASP_D_COUPLE", 50.0)
        K_rot = getattr(self, "GRASP_K_ROT", 200.0)        # 각 N·m/deg
        D_rot = getattr(self, "GRASP_D_ROT", 10.0)

        def _make_grasp_joint(path, body0, lp1):
            if K_couple is None:
                j = UsdPhysics.FixedJoint.Define(stage, path)
            else:
                j = UsdPhysics.Joint.Define(stage, path)   # generic D6
            j.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
            j.CreateBody1Rel().SetTargets([Sdf.Path(f"{base}/rod")])
            j.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, tcp_z))
            j.CreateLocalRot0Attr().Set(identity)
            j.CreateLocalPos1Attr().Set(lp1)
            j.CreateLocalRot1Attr().Set(rx_pi)
            j.CreateExcludeFromArticulationAttr().Set(True)
            if K_couple is not None:
                prim = j.GetPrim()
                for axis, k, d in [("transX", K_couple, D_couple), ("transY", K_couple, D_couple),
                                   ("transZ", K_couple, D_couple), ("rotX", K_rot, D_rot),
                                   ("rotY", K_rot, D_rot), ("rotZ", K_rot, D_rot)]:
                    drv = UsdPhysics.DriveAPI.Apply(prim, axis)
                    drv.CreateTypeAttr().Set("force")
                    drv.CreateStiffnessAttr().Set(float(k))
                    drv.CreateDampingAttr().Set(float(d))
                    drv.CreateTargetPositionAttr().Set(0.0)
            return j

        _make_grasp_joint(f"{base}/rod_joint_1", f"{base}/Robot_1/panda_hand", Gf.Vec3f(-half_width, 0.0, 0.0))
        _make_grasp_joint(f"{base}/rod_joint_2", f"{base}/Robot_2/panda_hand", Gf.Vec3f(+half_width, 0.0, 0.0))

        mode = "rigid fixed" if K_couple is None else f"compliant D6 (K_couple={K_couple} N/m)"
        print(f"[grasp] Added joints ({mode}, TCP offset={tcp_z}m, grasp_roll=π)")

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Accumulating action + bounds. NaN 방지."""
        # ── Post-reset settle: 일부 envs는 reset 후 N step zero-action 유지 (target 동결) ──
        # 효과: PhysX joint snap transient 흡수, 정책에 unstable 초기 데이터 학습 안 시킴
        in_settle = self._settle_remaining > 0
        # ★ 외부 모듈(train script, eval)이 transition buffer mask로 사용할 수 있도록 노출.
        # _pre_physics_step 시점 mask (즉 이번 env.step()이 settle이었는지) 의미.
        self._is_settle_step = in_settle.clone()
        if in_settle.any():
            actions = actions.clone()
            actions[in_settle] = 0.0
            # episode_length_buf freeze (settle step이 episode timeout 카운트하지 않도록)
            self.episode_length_buf[in_settle] = 0
            self._settle_remaining = (self._settle_remaining - 1).clamp_min(0)

        self.actions = actions.clone()
        pos_disp = self.actions[:, 0:3]
        rot_aa = self.actions[:, 3:6]
        self.target_obj_pos = self.target_obj_pos + pos_disp

        # 팔-장애물 회피는 컨트롤러(모델기반 nullspace, control A)가 담당 → RL은 rod(6-DoF)만.
        # K_arm 폐기(고정 K).
        self.K_arm1 = None
        self.K_arm2 = None

        # ★ target_obj_pos를 env-local ±1m 내로 clamp (graph feature 폭주 방지)
        env_origins = self.scene.env_origins
        local_pos = self.target_obj_pos - env_origins
        local_pos = torch.clamp(local_pos, min=-1.0, max=1.0)
        self.target_obj_pos = local_pos + env_origins

        angle = torch.norm(rot_aa, dim=-1, keepdim=True)
        eps = 1e-8
        axis = rot_aa / (angle + eps)
        half = angle * 0.5
        q_delta = torch.cat([torch.cos(half), axis * torch.sin(half)], dim=-1)
        self.target_obj_quat = math_utils.quat_mul(q_delta, self.target_obj_quat)
        # ★ Quat 정규화 (NaN 방지) — norm이 0이면 identity로
        q_norm = torch.norm(self.target_obj_quat, dim=-1, keepdim=True)
        q_norm_safe = torch.where(q_norm < 1e-6, torch.ones_like(q_norm), q_norm)
        self.target_obj_quat = self.target_obj_quat / q_norm_safe
        # NaN/Inf 안전망: target이 invalid이면 identity quat로 복원
        invalid = torch.isnan(self.target_obj_quat).any(dim=-1) | torch.isinf(self.target_obj_quat).any(dim=-1)
        if invalid.any():
            identity_quat = torch.zeros_like(self.target_obj_quat)
            identity_quat[:, 0] = 1.0
            self.target_obj_quat = torch.where(invalid.unsqueeze(-1), identity_quat, self.target_obj_quat)

        # ── Rod safety filter (RoboBallet식 hard 충돌방지의 rod 버전) ──
        # RL 목표를 적용하기 직전에, 그 목표 자세의 rod 선분이 활성 장애물을 침범하지 않도록 projection.
        if getattr(self.cfg, "use_rod_safety_filter", False) and self.cfg.n_obstacles > 0:
            self._apply_rod_safety_filter()

    def _apply_rod_safety_filter(self) -> None:
        """target_obj_pos(world)를 활성 장애물 밖으로 projection. 목표 rod 선분 clearance ≥ d_safe 유지.
        선분 근사: 중심 target_obj_pos, 축 = quat_apply(target_obj_quat, [HALF_W,0,0]).
        활성 장애물 없으면 no-op (push=0) → base transport 보존."""
        HALF_W, ROD_R = 0.4, 0.02
        d_safe = ROD_R + self.cfg.obstacle_radius + self.cfg.rod_safety_margin  # 중심선분↔장애물중심 최소거리
        B = self.num_envs
        # 장애물 중심 (world) + 활성 마스크
        obs_c = torch.stack([o.data.root_pos_w for o in self.obstacles], dim=1)   # (B,N,3)
        active = self.obstacle_active.unsqueeze(-1)                                # (B,N,1)
        # 목표 자세 rod 축 (world)
        axis = math_utils.quat_apply(
            self.target_obj_quat,
            torch.tensor([HALF_W, 0.0, 0.0], device=self.device).expand(B, 3),
        )                                                                         # (B,3)
        seg = 2.0 * axis                                                          # end1→end2
        seg_len2 = (seg * seg).sum(-1, keepdim=True).clamp_min(1e-8)              # (B,1)
        p0 = self.target_obj_pos.clone()                                          # 개입량 측정용 (projection 전)
        p = self.target_obj_pos
        for _ in range(int(self.cfg.rod_safety_iters)):
            end1 = p - axis                                                       # (B,3)
            AP = obs_c - end1.unsqueeze(1)                                        # (B,N,3)
            u = ((AP * seg.unsqueeze(1)).sum(-1) / seg_len2).clamp(0.0, 1.0)      # (B,N)
            closest = end1.unsqueeze(1) + u.unsqueeze(-1) * seg.unsqueeze(1)      # (B,N,3)
            v = obs_c - closest                                                  # closest→장애물중심
            dist = torch.norm(v, dim=-1, keepdim=True)                           # (B,N,1)
            n = v / dist.clamp_min(1e-6)                                         # 단위벡터 (장애물 방향)
            pen = (d_safe - dist)                                                # (B,N,1) >0이면 너무 가까움
            pen = torch.where(active & (pen > 0), pen, torch.zeros_like(pen))
            # 목표 중심을 장애물 반대방향(-n)으로 침투량만큼 이동 (모든 장애물 합산)
            push = (pen * (-n)).sum(dim=1)                                        # (B,3)
            p = p + push
        # env-local ±1m 재clamp (projection이 경계 밖으로 밀 수 있음)
        env_origins = self.scene.env_origins
        self.target_obj_pos = torch.clamp(p - env_origins, min=-1.0, max=1.0) + env_origins
        # 개입량(=필터가 target을 민 거리). reward에서 페널티로 사용 (RL이 rod 회피 학습).
        self._rod_filter_push = torch.norm(self.target_obj_pos - p0, dim=-1)      # (B,)

    def _apply_action(self) -> None:
        """target_obj_pos/quat를 controller로 joint torque에 매핑해 양 arm에 인가."""
        tau_1, tau_2, info = self.controller.compute_torques(
            self.target_obj_pos, self.target_obj_quat, self.target_x_rel,
            K_arm1=getattr(self, "K_arm1", None), K_arm2=getattr(self, "K_arm2", None),
        )
        self.robot_1.set_joint_effort_target(tau_1, joint_ids=self.robot_1_joint_ids)
        self.robot_2.set_joint_effort_target(tau_2, joint_ids=self.robot_2_joint_ids)
        self._last_ctrl_info = info

    def _get_grasp_wrenches(self):
        """
        각 panda_hand body가 받는 6D wrench 측정 (world frame).

        Isaac Lab API: get_link_incoming_joint_force() — 해당 link이 articulation 내부
        부모 joint(wrist joint)로부터 받는 reaction wrench. quasi-static 상태에서
        F_wrist_on_hand ≈ −F_rod_on_hand (Newton 3rd law via Newton 2nd:
        m·a_hand = F_wrist + F_rod ≈ 0 → F_rod ≈ −F_wrist).

        따라서 이 값은 부호 반전된 rod-side wrench로 해석.
        내력 squeeze는 두 panda_hand wrench의 difference로 산출 (부호 일관성 유지됨).

        Returns:
            wrench_1, wrench_2: (B, 6) [Fx, Fy, Fz, Mx, My, Mz] world frame
        """
        # Shape: (B, num_links, 6)
        wrench_all_1 = self.robot_1.root_physx_view.get_link_incoming_joint_force()
        wrench_all_2 = self.robot_2.root_physx_view.get_link_incoming_joint_force()

        wrench_1 = wrench_all_1[:, self.ee_body_idx_1, :]  # (B, 6)
        wrench_2 = wrench_all_2[:, self.ee_body_idx_2, :]
        return wrench_1, wrench_2

    def _get_observations(self) -> dict:
        """GNN 파이프라인에 필요한 모든 원본 텐서를 수집합니다.

        ★ Phase 3.3 fix: 모든 position을 env-local 좌표로 변환 (world coord 제거).
            env_spacing=4m로 env마다 좌표 수십 m 차이 → policy가 env index를 feature로
            학습하던 비효율 해결.
        """
        # ---------------------------------------------------------
        # 0. env_origins 가져오기 (모든 world position에서 빼서 env-local로 변환)
        # ---------------------------------------------------------
        env_origins = self.scene.env_origins  # (num_envs, 3)

        # ---------------------------------------------------------
        # 1. 데이터 계산
        # ---------------------------------------------------------
        ee_state_1_raw = self.robot_1.data.body_state_w[:, self.ee_body_idx_1, :]
        ee_state_2_raw = self.robot_2.data.body_state_w[:, self.ee_body_idx_2, :]
        # env-local 변환 (position만 빼기; quat/vel은 frame-invariant)
        ee_state_1 = ee_state_1_raw.clone()
        ee_state_1[:, :3] = ee_state_1_raw[:, :3] - env_origins
        ee_state_2 = ee_state_2_raw.clone()
        ee_state_2[:, :3] = ee_state_2_raw[:, :3] - env_origins

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

        # 목표 포즈 [B, 2, 7] — env-local 변환
        goal_pose_1_raw = self.goal_marker_ee1.data.root_state_w[:, :7]
        goal_pose_2_raw = self.goal_marker_ee2.data.root_state_w[:, :7]
        goal_pose_1 = goal_pose_1_raw.clone()
        goal_pose_1[:, :3] = goal_pose_1_raw[:, :3] - env_origins
        goal_pose_2 = goal_pose_2_raw.clone()
        goal_pose_2[:, :3] = goal_pose_2_raw[:, :3] - env_origins
        task_state = torch.stack([goal_pose_1, goal_pose_2], dim=1)

        # 현재 EE 포즈 (Edge 계산용) [B, 2, 7] — 이미 env-local
        current_ee_poses = torch.stack([ee_state_1[:, :7], ee_state_2[:, :7]], dim=1)

        # 베이스 포즈 (Edge 계산용) [B, 2, 7] — env-local 변환
        base_pose_1_raw = self.robot_1.data.root_state_w[:, :7]
        base_pose_2_raw = self.robot_2.data.root_state_w[:, :7]
        base_pose_1 = base_pose_1_raw.clone()
        base_pose_1[:, :3] = base_pose_1_raw[:, :3] - env_origins
        base_pose_2 = base_pose_2_raw.clone()
        base_pose_2[:, :3] = base_pose_2_raw[:, :3] - env_origins
        base_poses = torch.stack([base_pose_1, base_pose_2], dim=1)

        # 글로벌 (Relative Pose) — frame-invariant (좌표 차이라 env_origins 무관)
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
        # 2. Grasp wrench 측정 (Phase 3.5: internal/external force)
        # ---------------------------------------------------------
        # panda_hand body wrench from wrist joint (≈ −rod_joint reaction in quasi-static)
        wrench_panda_1, wrench_panda_2 = self._get_grasp_wrenches()  # (B, 6) each

        # 6D wrench: [F (3); M (3)]
        # External wrench: 두 panda_hand wrench의 합 → 객체에 작용하는 총 wrench (proxy)
        # Internal wrench: 두 panda_hand wrench의 (1/2)×차이 → squeeze 강도
        external_wrench = wrench_panda_1 + wrench_panda_2          # (B, 6)
        internal_wrench = 0.5 * (wrench_panda_1 - wrench_panda_2)  # (B, 6)

        # Diagnostic — Phase 4 r_safety 가공 재료 + 학습/평가 모니터링
        # 단일 지표로 통합 (max/ang은 sim-to-real Phase에서 필요 시 재추가)
        f_int_lin_norm = torch.norm(internal_wrench[:, :3], dim=-1)  # (B,) N
        self.extras["diag/f_int_lin_mean"] = f_int_lin_norm.mean()

        # ---------------------------------------------------------
        # 3. Rod state + EE velocities (Phase 3.1 graph_converter 입력용)
        #     position은 env-local 변환, quat/velocity는 그대로 (frame-invariant)
        # ---------------------------------------------------------
        rod_pos = self.rod.data.root_pos_w - env_origins  # (B, 3) env-local
        rod_quat = self.rod.data.root_quat_w             # (B, 4) wxyz
        rod_lin_vel = self.rod.data.root_lin_vel_w       # (B, 3)
        rod_ang_vel = self.rod.data.root_ang_vel_w       # (B, 3)

        # EE velocities from body_state (linear/angular)
        ee_lin_1 = ee_state_1[:, 7:10]   # (B, 3)
        ee_ang_1 = ee_state_1[:, 10:13]
        ee_lin_2 = ee_state_2[:, 7:10]
        ee_ang_2 = ee_state_2[:, 10:13]
        ee_lin_vel = torch.stack([ee_lin_1, ee_lin_2], dim=1)   # (B, 2, 3)
        ee_ang_vel = torch.stack([ee_ang_1, ee_ang_2], dim=1)   # (B, 2, 3)

        # ---------------------------------------------------------
        # 4. 딕셔너리 조립 (키 이름 중요!)
        # ---------------------------------------------------------
        # graph_converter.py가 이 키 이름들을 찾습니다.
        raw_state_dict = {
            "robot_nodes": robot_state,           # [KeyError 원인 해결]
            "current_ee_poses": current_ee_poses, # Edge 계산용
            "goal_poses": task_state,             # Task Node용
            "base_poses": base_poses,             # Edge 계산용
            "target_rel_pose": self.target_ee_rel_poses, # Global용
            "target_joint_pos": self.target_joint_pos,   # [추가] Joint Space Target
            "globals": global_state,              # (Legacy, 참고용)
            # ── Phase 3.5: Grasp wrenches (RL observation + safety) ──
            "wrench_panda_1": wrench_panda_1,     # (B, 6) panda_hand_1 wrench
            "wrench_panda_2": wrench_panda_2,     # (B, 6) panda_hand_2 wrench
            "internal_wrench": internal_wrench,   # (B, 6) squeeze
            "external_wrench": external_wrench,   # (B, 6) total
            # ── Phase 3.1: graph_converter 입력 ──
            "rod_pos": rod_pos,                   # (B, 3)
            "rod_quat": rod_quat,                 # (B, 4)
            "rod_lin_vel": rod_lin_vel,           # (B, 3)
            "rod_ang_vel": rod_ang_vel,           # (B, 3)
            "ee_lin_vel": ee_lin_vel,             # (B, 2, 3)
            "ee_ang_vel": ee_ang_vel,             # (B, 2, 3)
        }

        return {"policy": raw_state_dict}

    def _calc_rot_error(self, current_quat, target_quat):
        """
        Calculate rotation error (angle difference) between two quaternions.
        Returns angle in radians.
        """
        current_inv = math_utils.quat_conjugate(current_quat)
        q_diff = math_utils.quat_mul(current_inv, target_quat)
        q_diff_v = q_diff[:, 1:4]
        q_diff_w = q_diff[:, 0]
        # Angle = 2 * atan2(norm(v), abs(w))
        rot_error = 2.0 * torch.atan2(torch.norm(q_diff_v, dim=-1), torch.abs(q_diff_w))
        return rot_error

    def _get_rewards(self) -> torch.Tensor:
        """
        Current reward (2026-05-22):
        object progress PBR + joint-space auxiliary PBR + sparse first-reach bonus.

        의도:
        - object progress: 최종 task 신호
        - joint PBR: policy→object 경로가 길어 생기는 credit assignment 완화용 보조 신호
        - first reach bonus: threshold 도달을 명시적으로 강화

        Weight:
        - object progress × 10
        - joint progress × 1
        - success bonus +50 (first-time only)

            r_progress_obj  = (prev_obj_dist   - curr_obj_dist)   * 10
            r_progress_joint = (prev_joint_dist - curr_joint_dist) * 1
            r_success       = +50 if first-time reach
        """
        # ── Rod 현재 자세 ──
        rod_pos = self.rod.data.root_pos_w               # (B, 3)
        rod_quat = self.rod.data.root_quat_w             # (B, 4)

        # ── 목표 자세 ──
        goal_rod_pos = self.goal_rod_marker.data.root_pos_w
        goal_rod_quat = self.goal_rod_marker.data.root_quat_w

        # 위치/회전 오차
        pos_err = torch.norm(goal_rod_pos - rod_pos, dim=-1)                   # (B,) m
        rod_inv = math_utils.quat_conjugate(rod_quat)
        q_diff = math_utils.quat_mul(goal_rod_quat, rod_inv)
        rot_err = 2.0 * torch.atan2(
            torch.norm(q_diff[:, 1:4], dim=-1), torch.abs(q_diff[:, 0])
        )                                                                       # (B,) rad

        # ── 1. Dense progress 제거 (Roboballet식 sparse-only + HER). ──
        # 정지 정책에 대한 음수 reward도 없애 HER virtual exploit 안 키움.
        # is_first는 success threshold gate(첫 step bug 패치)에 여전히 사용.
        current_dist = pos_err + 0.1 * rot_err
        is_first = torch.isinf(self.prev_dist)
        self.prev_dist = current_dist  # update only — used by is_first detection only
        r_progress = torch.zeros_like(pos_err)
        r_joint_pbr = torch.zeros_like(r_progress)

        # ── Tracking diagnostic: controller가 target_obj_pos를 얼마나 잘 추종하는지 ──
        # 큰 값 → controller 미추종 (RL 학습에 noise) / 작은 값 → 신뢰 가능
        track_err_pos = torch.norm(rod_pos - self.target_obj_pos, dim=-1)            # (B,) m
        rod_inv = math_utils.quat_conjugate(rod_quat)
        target_q_diff = math_utils.quat_mul(self.target_obj_quat, rod_inv)
        track_err_rot = 2.0 * torch.atan2(
            torch.norm(target_q_diff[:, 1:4], dim=-1), torch.abs(target_q_diff[:, 0])
        )                                                                              # (B,) rad
        self.extras["diag/track_err_pos_mm"] = track_err_pos.mean() * 1000
        self.extras["diag/track_err_rot_deg"] = track_err_rot.mean() * 180.0 / math.pi

        # ── 2. r_safety ── Stage 1에서는 비활성. 진단용 측정만 유지.
        wrench_panda_1, wrench_panda_2 = self._get_grasp_wrenches()
        f_int_lin_norm = 0.5 * torch.norm(wrench_panda_1[:, :3] - wrench_panda_2[:, :3], dim=-1)
        r_safety = torch.zeros_like(pos_err)             # 진단용 측정만

        # ── 3. r_success ── sparse one-shot (episode 내 첫 도달 시 한 번만)
        # 현재 threshold는 10cm / 17°.
        POS_THRESH = 0.02    # 2cm. curriculum 3-5cm 대비 tight + HER virtual trivial-reach 차단
        ROT_THRESH = 0.1745  # ≈ 10°. K_abs_rot=20으로 rotation drift 큰 점 감안한 완화
        # 첫 step (prev_dist=inf reset 직후) is_reached 강제 False — bug 패치.
        # curriculum 시작 거리 ≥ 12cm > threshold 10cm라 첫 step 도달 물리적으로 불가.
        # Isaac Lab reset timing 이슈로 spurious SUCCESS 발생 방지.
        is_reached = (pos_err < POS_THRESH) & (rot_err < ROT_THRESH) & ~is_first
        first_reach = is_reached & ~self.reached_once
        self.reached_once = self.reached_once | is_reached
        r_success = torch.where(first_reach, 100.0, 0.0)

        # ── Time penalty (2026-06-10) ──
        # 정책의 "가만히 기다리기" 패턴 해결용. 실패 case의 min_pos가 초기 거리와 비슷한
        # 것이 정책 게으름 신호였음. State-independent라 HER virtual goal과 동일하게 적용,
        # exploit 위험 낮음. -0.2/step × 30 step = -6 → r_success 100 대비 더 작음.
        # (2026-06-10 v2) -0.5 → -0.2 약화: 초기 exploration 압력 완화 (Stage 2 0% 위험 처치).
        r_time = torch.full_like(pos_err, -0.2)

        # ── Stage 1 (architecture C): 내력 페널티 + K smoothness (flag-gated) ──
        # _internal_w=0이면 0 → Stage 0 학습/eval 무영향. train_stage1_K.py가 weight 설정.
        r_internal = torch.zeros_like(pos_err)
        r_K_smooth = torch.zeros_like(pos_err)
        if getattr(self, "_internal_w", 0.0) > 0.0:
            r_internal = -self._internal_w * torch.clamp(f_int_lin_norm - self._f_safe, min=0.0)
            if getattr(self, "K_arm1", None) is not None and getattr(self, "K_arm_prev", None) is not None:
                dK = (self.K_arm1 - self.K_arm_prev[0]).abs() + (self.K_arm2 - self.K_arm_prev[1]).abs()
                r_K_smooth = -self._ksmooth_w * dK / self.controller.K_pos
            self.extras["reward/r_internal_mean"] = r_internal.mean()
            self.extras["diag/f_int_reward_N"] = f_int_lin_norm.mean()
        # K_arm_prev 갱신 (smoothness용; settle 중 K=K_base)
        if getattr(self, "K_arm1", None) is not None:
            self.K_arm_prev = (self.K_arm1.detach().clone(), self.K_arm2.detach().clone())

        # ── 장애물 회피 (cluttered transport, 2026-06-17) ── RoboBallet식 충돌 cost.
        r_clearance = torch.zeros_like(pos_err)
        r_collision = torch.zeros_like(pos_err)
        r_smooth = torch.zeros_like(pos_err)
        if self.cfg.n_obstacles > 0:
            ost = self._get_obstacle_state()
            rod_clr = ost["min_clearance"]                      # (B,) rod↔obstacle
            arm_clr = ost["arm_min_clearance"]                  # (B,) arm↔obstacle
            margin = self.cfg.obstacle_collision_margin
            # 2026-06-19: RL 보상 = rod·팔 둘 다 (min). nullspace 컨트롤러 OFF라 RL이 거시 routing으로
            # 팔 충돌도 회피하도록 incentive 부여 (obstacle→arm 엣지로 GNN이 인지).
            min_clr = torch.minimum(rod_clr, arm_clr)
            r_clearance = -self.cfg.w_clearance * torch.clamp(margin - min_clr, min=0.0)
            in_collision = min_clr < 0.0
            r_collision = torch.where(in_collision, -float(self.cfg.w_collision),
                                      torch.zeros_like(pos_err))
            self._in_collision = in_collision
            self.extras["diag/rod_clearance_mm"] = rod_clr.mean() * 1000
            self.extras["diag/arm_clearance_mm"] = arm_clr.mean() * 1000
            self.extras["diag/rod_collision_rate"] = (rod_clr < 0.0).float().mean()
            self.extras["diag/arm_collision_rate"] = (arm_clr < 0.0).float().mean()  # 컨트롤러 성능 지표
            self.extras["reward/r_clearance_mean"] = r_clearance.mean()
            self.extras["reward/r_collision_mean"] = r_collision.mean()
        # object-pose 명령 부드러움 (RoboBallet accel 페널티 참고)
        r_smooth = -self.cfg.w_smooth * torch.sum((self.actions - self._prev_action) ** 2, dim=-1)
        self._prev_action = self.actions.detach().clone()

        # nullspace 사용 페널티: ‖action[6:8]‖² → 멀 땐 0(정밀도 보존), 팔회피 이득>페널티일 때만 사용.
        r_null = torch.zeros_like(pos_err)
        if self.actions.shape[1] >= 8:
            r_null = -self.cfg.w_null * torch.sum(self.actions[:, 6:8] ** 2, dim=-1)

        # ── 필터 개입 페널티 (방향 b): 필터가 target을 민 양에 비례 → RL이 rod 회피 학습 ──
        r_filter = torch.zeros_like(pos_err)
        if getattr(self.cfg, "use_rod_safety_filter", False) and self.cfg.n_obstacles > 0:
            push = getattr(self, "_rod_filter_push", None)
            if push is not None:
                r_filter = -self.cfg.w_filter_intervene * push
                self.extras["reward/r_filter_intervene_mean"] = r_filter.mean()
                self.extras["diag/filter_push_mm"] = push.mean() * 1000

        # goal-무관 보상 합 (HER가 virtual goal에도 보존하도록 train에서 buffer에 전달).
        # time+smooth+clearance+collision+null+filter는 goal에 무관 (충돌·nullspace·필터는 state/action 기반).
        self._goal_indep_reward = r_time + r_smooth + r_clearance + r_collision + r_null + r_filter

        # ── Total ──
        if getattr(self, "_k_reward_mode", False):
            # Stage 1 K 정책 전용 dense 보상: 추종 유지(track_err) + 내력 최소화.
            # sparse +100 success/time penalty 제거 → q_loss 안정 + K가 진짜 트레이드오프 학습.
            # (frozen pose가 task 구조 담당; track_err↓ = rod가 target 잘 따라감 = success.)
            r_track = -self._w_track * track_err_pos     # track_err_pos: m (rod↔target)
            total_reward = r_track + r_internal + r_K_smooth
            self.extras["reward/r_track_mean"] = r_track.mean()
        else:
            total_reward = (r_progress + r_joint_pbr + r_safety + r_success + r_time
                            + r_internal + r_K_smooth + r_clearance + r_collision + r_smooth + r_null
                            + r_filter)

        # ── Episode 최소 오차 추적 (timeout 시 _get_dones에서 평균 측정) ──
        self.episode_min_pos_err = torch.min(self.episode_min_pos_err, pos_err)
        self.episode_min_rot_err = torch.min(self.episode_min_rot_err, rot_err)

        # ── Logging ──
        # task/* episode metric은 _get_dones에서 done 시점에만 기록한다.
        # 여기서는 reward decomposition만 남긴다.
        self.extras["reward/r_progress_mean"] = r_progress.mean()
        self.extras["reward/r_joint_pbr_mean"] = r_joint_pbr.mean()
        self.extras["reward/r_first_reach_count"] = first_reach.float().sum()
        # Stage 1: per-arm K 관찰 (정책이 K를 움직이나, 양팔 다르게 쓰나)
        if getattr(self, "K_arm1", None) is not None:
            self.extras["diag/K_arm1_mean"] = self.K_arm1.mean()
            self.extras["diag/K_arm2_mean"] = self.K_arm2.mean()
            self.extras["diag/K_arm_absdiff_mean"] = (self.K_arm1 - self.K_arm2).abs().mean()
        # Internal flag (자동 logging 안 됨 — 0-dim 아니라). _get_dones에서 사용.
        self.extras["log/is_reached"] = is_reached

        return total_reward


    # ──────────────────────────────────────────────────────────────────
    # Phase 3.3 helper — PPO rollout 용 PyG Batch 직접 생성
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _push_point_from_segment(p, a, b, min_clear):
        """점 p(R,3)를 선분 a-b(R,3)에서 min_clear 이상 떨어지도록 수직으로 밀어냄."""
        ab = b - a
        denom = (ab * ab).sum(-1, keepdim=True).clamp_min(1e-8)
        t = (((p - a) * ab).sum(-1, keepdim=True) / denom).clamp(0.0, 1.0)
        closest = a + t * ab
        diff = p - closest
        d = diff.norm(dim=-1, keepdim=True)
        direction = diff / d.clamp_min(1e-6)
        return p + direction * (min_clear - d).clamp_min(0.0)

    def _get_obstacle_state(self):
        """장애물 상태 + rod 선분까지 거리/clearance. (cluttered transport)
        rod를 길이축 선분(양 끝 ±0.4m)으로 근사 → 점-선분 거리로 든 물체의 extent 반영.
        Returns dict: pos_local(B,N,3), radius(B,N), vel(B,N,3), dist_to_rod(B,N),
            active(B,N), min_clearance(B,), clearance_all(B,N)."""
        B, n_obs = self.num_envs, self.cfg.n_obstacles
        env_origins = self.scene.env_origins
        HALF_W, ROD_R = 0.4, 0.02
        rod_pos = self.rod.data.root_pos_w                              # (B,3)
        rod_quat = self.rod.data.root_quat_w
        axis = math_utils.quat_apply(
            rod_quat, torch.tensor([HALF_W, 0.0, 0.0], device=self.device).expand(B, 3)
        )
        end1, end2 = rod_pos - axis, rod_pos + axis
        seg = end2 - end1
        seg_len2 = (seg * seg).sum(-1, keepdim=True).clamp_min(1e-8)    # (B,1)
        obs_pos_w = torch.stack([o.data.root_pos_w for o in self.obstacles], dim=1)  # (B,N,3)
        AP = obs_pos_w - end1.unsqueeze(1)
        u = ((AP * seg.unsqueeze(1)).sum(-1) / seg_len2).clamp(0.0, 1.0).unsqueeze(-1)  # (B,N,1)
        closest = end1.unsqueeze(1) + u * seg.unsqueeze(1)
        dist = torch.norm(obs_pos_w - closest, dim=-1)                  # (B,N)
        radius = torch.full((B, n_obs), self.cfg.obstacle_radius, device=self.device)
        clearance = dist - ROD_R - radius
        clr_masked = torch.where(self.obstacle_active, clearance, torch.full_like(clearance, 1e3))

        # ── 팔 링크↔장애물 clearance (nullspace로 제어 가능 → 보상으로 학습) ──
        # feature 아님(정책엔 관절각+상대pose 엣지만). 여기선 보상 신호용 실제 충돌 검사.
        if not hasattr(self, "_arm_link_ids"):
            names = ["panda_link4", "panda_link5", "panda_link6"]   # 팔꿈치/팔뚝/손목
            self._arm_link_ids = [self.robot_1.body_names.index(n) for n in names]
        ARM_R = 0.06
        l1 = self.robot_1.data.body_pos_w[:, self._arm_link_ids]    # (B,L,3)
        l2 = self.robot_2.data.body_pos_w[:, self._arm_link_ids]
        arm_pts = torch.cat([l1, l2], dim=1)                        # (B,2L,3)
        d_am = torch.cdist(obs_pos_w, arm_pts)                      # (B,n_obs,2L)
        arm_clr = d_am.min(dim=2).values - ARM_R - radius           # (B,n_obs)
        arm_clr_masked = torch.where(self.obstacle_active, arm_clr, torch.full_like(arm_clr, 1e3))

        return {
            "pos_local": obs_pos_w - env_origins.unsqueeze(1),
            "radius": radius,
            "vel": torch.zeros(B, n_obs, 3, device=self.device),
            "dist_to_rod": dist,
            "active": self.obstacle_active,
            "min_clearance": clr_masked.min(dim=1).values,          # rod↔obstacle
            "arm_min_clearance": arm_clr_masked.min(dim=1).values,  # arm↔obstacle
            "clearance_all": clr_masked,
        }

    def _build_policy_batch(self):
        """
        현재 env 상태를 PyG Batch로 변환 (graph_converter 호출 wrapper).
        joint_limits는 한 번만 cache.

        ★ goal_rod_marker의 실제 목표 자세를 graph에 노출 (이전엔 target_obj_pos만 전달해
          RL이 진짜 목표를 못 봤음 — 학습 정체 핵심 원인).
        """
        # Lazy init joint limits
        if not hasattr(self, "_joint_limits_low"):
            limits_1 = self.robot_1.data.soft_joint_pos_limits[0, self.robot_1_joint_ids]  # (7, 2)
            limits_2 = self.robot_2.data.soft_joint_pos_limits[0, self.robot_2_joint_ids]
            self._joint_limits_low = torch.stack([limits_1[:, 0], limits_2[:, 0]], dim=0)   # (2, 7)
            self._joint_limits_high = torch.stack([limits_1[:, 1], limits_2[:, 1]], dim=0)

        import graph_converter as gc
        raw_state = self._get_observations()["policy"]
        normalized_time = (self.episode_length_buf.float() / self.max_episode_length).clamp(0.0, 1.0)

        # 진짜 goal — reward 계산에 쓰는 그 위치/자세 (env-local)
        env_origins = self.scene.env_origins
        goal_pos = self.goal_rod_marker.data.root_pos_w - env_origins  # (B, 3) env-local
        goal_quat = self.goal_rod_marker.data.root_quat_w              # (B, 4)

        # 장애물 상태 (cluttered transport)
        obs_kw = {}
        if self.cfg.n_obstacles > 0:
            ost = self._get_obstacle_state()
            obs_kw = dict(
                obstacle_pos=ost["pos_local"],
                obstacle_radius=ost["radius"],
                obstacle_vel=ost["vel"],
                obstacle_dist=ost["dist_to_rod"],
            )

        return gc.convert_batch_state_to_graph(
            raw_state=raw_state,
            num_envs=self.num_envs,
            goal_pos=goal_pos,
            goal_quat=goal_quat,
            target_x_rel=self.target_x_rel,
            normalized_time=normalized_time,
            joint_limits_low=self._joint_limits_low,
            joint_limits_high=self._joint_limits_high,
            joint_torque=None,
            **obs_kw,
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Current done logic:
        - success termination ON
        - timeout도 유지
        - episode metric은 done 시점에 계산
        """
        is_reached = self.extras.get("log/is_reached", None)
        if is_reached is None:
            rod_pos = self.rod.data.root_pos_w
            goal_pos = self.goal_rod_marker.data.root_pos_w
            pos_err = torch.norm(goal_pos - rod_pos, dim=-1)
            is_reached = pos_err < 0.02

        is_success = is_reached
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # success 도달 시 episode를 끊어 새로운 goal을 더 자주 보게 한다.
        # (2026-06-11) settle 중 종료 금지: _get_dones는 _get_rewards보다 먼저 호출돼
        # log/is_reached가 직전 step(=직전 episode 성공) 값으로 stale. 성공 직후 reset된
        # 새 episode의 첫 settle step에서 이 stale True가 terminated를 즉시 발화 → RL 0-step
        # phantom episode(실패로 오집계)가 발생했음. settle 중엔 정책이 작동 안 하므로
        # 애초에 success 불가 → ~_is_settle_step gate로 stale 누수와 phantom 동시 차단.
        terminated = is_success & ~self._is_settle_step

        # Episode-level metric은 terminated 또는 timeout 시점에만 갱신한다.
        done = terminated | time_out
        if done.any():
            self.extras["task/episode_success_rate"] = (
                self.reached_once[done].float().mean()
            )
            # inf filter: reset 직후 step 0에 done 가능성 (희소) → inf 제외
            min_pos_done = self.episode_min_pos_err[done]
            min_rot_done = self.episode_min_rot_err[done]
            valid_pos = min_pos_done[~torch.isinf(min_pos_done)]
            valid_rot = min_rot_done[~torch.isinf(min_rot_done)]
            if valid_pos.numel() > 0:
                self.extras["task/min_pos_err_mean_mm"] = valid_pos.mean() * 1000
            if valid_rot.numel() > 0:
                self.extras["task/min_rot_err_mean_deg"] = valid_rot.mean() * 180.0 / math.pi

        # Internal flag (0-dim 아니라 자동 logging 안 됨). 다른 모듈이 참조할 수 있음.
        self.extras["log/is_reached"] = is_reached.float()

        # HER 등 외부 모듈용: auto-reset 직전 rod 상태 (post-step, pre-reset) 스냅샷.
        # _get_dones는 reset보다 항상 먼저 호출되므로 done env의 terminal state 보존 가능.
        self._last_rod_pos_w = self.rod.data.root_pos_w.clone()
        self._last_rod_quat_w = self.rod.data.root_quat_w.clone()

        return terminated, time_out


    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot_1._ALL_INDICES
        
        super()._reset_idx(env_ids)
        num_resets = len(env_ids)

        # ---------------------------------------------------------
        # 1. Sampling Loop (GPU)
        # ---------------------------------------------------------
        if self.external_samples is not None:
            # [Fixed] 외부 주입 샘플 사용 시 해당 env_ids에 맞는 데이터만 슬라이싱
             samples = {}
             for k, v in self.external_samples.items():
                 samples[k] = v[env_ids]
             self.current_sample_idxs[env_ids] = -1  # external은 cache idx 의미 없음
        else:
             # 기존 방식: 샘플러로부터 Base, Joint, Goal EE Pose를 모두 받아옵니다.
             samples = self.pose_sampler.sample_episodes(num_resets)
             # 마지막 sample_episodes의 cache idx를 env별로 저장 (실패 episode 재현용)
             if hasattr(self.pose_sampler, "last_idxs"):
                 self.current_sample_idxs[env_ids] = self.pose_sampler.last_idxs
        
        base_pose_1 = samples["base_pose_1"]
        base_pose_2 = samples["base_pose_2"]
        q_start_1 = samples["q_start_1"]
        q_start_2 = samples["q_start_2"]
        q_goal_1 = samples["q_goal_1"]
        q_goal_2 = samples["q_goal_2"]
        goal_ee1_pose = samples["goal_ee1_pose"]
        goal_ee2_pose = samples["goal_ee2_pose"]

        # Store target joint positions for reward calculation
        self.target_joint_pos[env_ids] = torch.cat([q_goal_1, q_goal_2], dim=1)
        self.prev_joint_dist[env_ids] = float('inf')

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
        self.robot_1.write_joint_state_to_sim(q_start_1, zeros_joint_vel, self.robot_1_joint_ids, env_ids)

        # B. Robot 2 (Base + Joint)
        r2_world_pose = base_pose_2.clone()
        r2_world_pose[:, :3] += env_origins
        
        self.robot_2.write_root_pose_to_sim(r2_world_pose, env_ids)
        self.robot_2.write_root_velocity_to_sim(zeros_vel, env_ids)
        self.robot_2.write_joint_state_to_sim(q_start_2, zeros_joint_vel, self.robot_2_joint_ids, env_ids)
        
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

        # 공유 강체 (rod) - sampler가 반환하는 start_obj_pose 위치에 배치
        # Phase 2: rod는 dynamic body이므로 velocity도 reset (residual motion 제거)
        # IK 정합 + grasp_roll=0이라 fixed joint 제약은 자동 만족됨
        if "start_obj_pose" in samples:
            rod_pose = samples["start_obj_pose"].clone()
            rod_pose[:, :3] += env_origins
            self.rod.write_root_pose_to_sim(rod_pose, env_ids)
            self.rod.write_root_velocity_to_sim(zeros_vel, env_ids)

            # ── Phase 3: controller target을 rod 시작 자세로 초기화 ──
            # 이후 action(delta)이 누적되어 target이 움직이게 됨
            self.target_obj_pos[env_ids] = rod_pose[:, :3]
            self.target_obj_quat[env_ids] = rod_pose[:, 3:7]

            # ── Phase 2 (Cooperative refactor): target_x_rel 해석적 계산 ──
            # 기하학: ee1 - ee2 = R_rod · (-2·HALF_W, 0, 0)
            # (TCP_OFFSET은 Z 방향이라 X 방향 차에 영향 X → x_rel은 R_rod·(-0.8,0,0))
            HALF_W = 0.4
            delta_x_rel_in_obj = torch.tensor(
                [-2.0 * HALF_W, 0.0, 0.0], device=self.device
            ).expand(num_resets, 3).contiguous()
            # rod_pose는 world 좌표계로 이미 env_origins가 더해져 있음. quat은 변환 무관.
            start_obj_quat = samples["start_obj_pose"][:, 3:7]
            x_rel_at_start = math_utils.quat_apply(start_obj_quat, delta_x_rel_in_obj)
            self.target_x_rel[env_ids] = x_rel_at_start

        # goal_rod_marker는 kinematic 유지 (시각 표시용) — pose만 write
        if "goal_obj_pose" in samples:
            goal_rod_pose = samples["goal_obj_pose"].clone()
            goal_rod_pose[:, :3] += env_origins
            self.goal_rod_marker.write_root_pose_to_sim(goal_rod_pose, env_ids)

        # ── 장애물 배치 (cluttered transport) ── RoboBallet 참고: 에피소드마다 랜덤화.
        # start↔goal 경로 위 t∈[0.3,0.7] 지점 + 수직 랜덤 오프셋(경로 차단). 활성 K개, 나머지는 멀리.
        if "start_obj_pose" in samples and "goal_obj_pose" in samples and self.cfg.n_obstacles > 0:
            start_p = samples["start_obj_pose"][:, :3]   # (R,3) env-local
            goal_p = samples["goal_obj_pose"][:, :3]
            n_obs = self.cfg.n_obstacles
            # 장애물 curriculum: frac 0이면 0개(운반만), 1이면 최대 n_obs개.
            frac = getattr(self, "_obstacle_curr_frac", 1.0)
            max_active = int(round(frac * n_obs))
            if max_active <= 0:
                K = torch.zeros(num_resets, dtype=torch.long, device=self.device)
            else:
                K = torch.randint(0, max_active + 1, (num_resets,), device=self.device)
            active = torch.arange(n_obs, device=self.device).unsqueeze(0) < K.unsqueeze(1)  # (R,n_obs)
            self.obstacle_active[env_ids] = active
            ident_q = torch.zeros(num_resets, 4, device=self.device); ident_q[:, 0] = 1.0
            zeros6 = torch.zeros(num_resets, 6, device=self.device)
            # 시작/도착 rod 선분 (충돌-free 보장용 push 대상)
            HALF_W = 0.4
            sq = samples["start_obj_pose"][:, 3:7]; gq = samples["goal_obj_pose"][:, 3:7]
            x_axis = torch.tensor([HALF_W, 0.0, 0.0], device=self.device).expand(num_resets, 3)
            s_ax = math_utils.quat_apply(sq, x_axis); g_ax = math_utils.quat_apply(gq, x_axis)
            s1, s2 = start_p - s_ax, start_p + s_ax
            g1, g2 = goal_p - g_ax, goal_p + g_ax
            min_clear = 0.02 + self.cfg.obstacle_radius + 0.04   # ROD_R + obs_r + safety
            for i in range(n_obs):
                t = 0.3 + 0.4 * torch.rand(num_resets, 1, device=self.device)
                on_path = start_p + t * (goal_p - start_p)
                offset = torch.randn(num_resets, 3, device=self.device) * self.cfg.obstacle_path_offset_std
                obs_local = on_path + offset
                # 시작/도착 rod에서 떨어뜨려 reset 충돌 방지 (mid-path 차단은 유지)
                obs_local = self._push_point_from_segment(obs_local, s1, s2, min_clear)
                obs_local = self._push_point_from_segment(obs_local, g1, g2, min_clear)
                far = torch.full_like(obs_local, self.cfg.obstacle_far_away)
                obs_local = torch.where(active[:, i:i + 1], obs_local, far)
                pose = torch.cat([obs_local + env_origins, ident_q], dim=-1).contiguous()
                self.obstacles[i].write_root_pose_to_sim(pose, env_ids)
                self.obstacles[i].write_root_velocity_to_sim(zeros6, env_ids)
        self._prev_action[env_ids] = 0.0

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
        
        # [NEW] 리셋 시 최대 오차 초기화
        self.episode_max_pos_error[env_ids] = 0.0
        self.episode_max_rot_error[env_ids] = 0.0

        # [NEW] 리셋 시 이전 거리도 초기화 (무한대로)
        self.prev_dist[env_ids] = float('inf')
        self.reached_once[env_ids] = False
        self.episode_min_pos_err[env_ids] = float('inf')
        self.episode_min_rot_err[env_ids] = float('inf')

        # [2026-06-08] Post-reset settle counter — N step zero-action 강제 적용
        self._settle_remaining[env_ids] = self.SETTLE_STEPS_AT_RESET

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
