#simulation 설정, robot, scene에 대한 설정 -> cfg. MDP는 env코드에서 정의하는 것.


import math
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

ROBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # Phase 3.2 시도 시 발산 — get_generalized_gravity_forces() 부호/frame convention
            # 미해결. Phase 3.1 검증 통과 상태(gravity OFF) 유지 + 추후 별도 디버깅.
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            # 회전 명령 작아져서 (action_scale_rot 0.025) closed-chain stress 감소.
            # 64+8로 fps 4× ↑. RL 학습 4시간 내.
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=8,
        ),
    ),
    #초기 state가 joint limit 벗어나면 오류발생함.
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.785,  
            "panda_joint3": 0.0,
            "panda_joint4": -2.356,  
            "panda_joint5": 0.0,
            "panda_joint6": 1.571, 
            "panda_joint7": 0.785  
        },
        # (베이스 위치는 아래 .replace()에서 덮어쓸 것임)
        pos=(0.0, 0.0, 0.0), 
    ),
    actuators={
        # Phase 3: torque pass-through 모드
        # stiffness=0, damping=0 → set_joint_effort_target가 그대로 모터 토크가 됨
        # cooperative impedance controller가 외부에서 토크를 계산해 주입
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-7]"],
            effort_limit_sim=50.0,
            velocity_limit_sim=1.5,
            stiffness=0.0,
            damping=0.0,
        ),
    }
)

GOAL_MARKER_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/goal", # 나중에 goal_1, goal_2로 대체됨

    # 5cm x 5cm x 5cm 작은 박스 (그리퍼 중심 표시용)
    spawn=sim_utils.CuboidCfg(
        size=(0.05, 0.05, 0.05),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),

        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True
        ),
        physics_material=None
    )
)

# 공유 강체 (rod): 두 그리퍼가 함께 들고 있다고 가정하는 막대
# 길이 = sampler의 obj_width(0.8m), X축이 길이 방향 (sampler offset과 일치)
# Phase 2: dynamic body로 전환 + 양 panda_hand에 fixed joint로 결합
ROD_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/rod",
    spawn=sim_utils.CuboidCfg(
        size=(0.8, 0.04, 0.04),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),  # 갈색
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,           # 본 task: dynamic rod. fixed joint로 양 EE에 결합.
            disable_gravity=True,              # 중력은 일단 OFF (안정성 우선)
            max_depenetration_velocity=5.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.1),  # 가벼운 막대
        # Phase 3.3: collision 활성화 — rod가 robot 몸체 통과 방지 (realistic trajectory).
        # GPU 메모리 약간 ↑ (contact pair) but 학습 정확도/sim-to-real 우선.
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        physics_material=None,
    ),
)

# 공유 강체의 목표 자세 시각화용 마커 (반투명 녹색)
GOAL_ROD_MARKER_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/goal_rod",
    spawn=sim_utils.CuboidCfg(
        size=(0.8, 0.04, 0.04),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.9, 0.2)),  # 녹색
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        physics_material=None,
    ),
)

@configclass
class DualrobotCfg(DirectRLEnvCfg):
    """
    Dofbot 2대를 스폰하는 환경의 설정 클래스입니다.
    """
    
    # === 1. 환경 기본 설정 ===
    decimation = 48 # 5Hz RL. 이전 per-arm impedance 검증 cfg와 일치.
    episode_length_s = 6.0   # 30 step (5Hz).
    
    # === 2. Spaces 정의 ===
    # Phase 3.0: action = 6-dim object pose delta
    #   action[0:3] = positional delta (world frame, m/step)
    #   action[3:6] = rotation delta as axis-angle vector (rad/step)
    # 누적되어 target_obj_pos, target_obj_quat 갱신.
    action_space = 6
    observation_space = 0 # (자리 채우기, 나중에 _get_obs 수정 시 함께 변경)
    state_space = 0

    # === 3. 시뮬레이션 설정 ===
    # dt는 시뮬레이션 주기: 물리적 계산을 1초에 240번 진행
    sim: SimulationCfg = SimulationCfg(dt=1 / 240, render_interval=decimation)

    # === 4. 씬(Scene) 설정 ===
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=100, env_spacing=4.0, replicate_physics=True
    )

    # === 5. 에셋(Assets) 정의 ===
    # (여기서 로봇 2대를 정의합니다)

    # 5.1. Dofbot 1 (왼쪽). 본 task: 양 로봇 지면(z=0)에서 rod와 fixed joint로 결합.
    robot_1: ArticulationCfg = ROBOT_CONFIG.replace(
        prim_path="/World/envs/env_.*/Robot_1",
        init_state=ROBOT_CONFIG.init_state.replace(pos=(-0.5, 0.0, 0.0))
    )
    robot_2: ArticulationCfg = ROBOT_CONFIG.replace(
        prim_path="/World/envs/env_.*/Robot_2",
        init_state=ROBOT_CONFIG.init_state.replace(pos=(0.5, 0.0, 0.0))
    )

    goal_1: RigidObjectCfg = GOAL_MARKER_CFG.replace(
        prim_path="/World/envs/env_.*/goal_ee1", # 이름 변경: goal_ee1
        spawn=GOAL_MARKER_CFG.spawn.replace(
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)) # 왼쪽: 빨간색
        )
    )
    goal_2: RigidObjectCfg = GOAL_MARKER_CFG.replace(
        prim_path="/World/envs/env_.*/goal_ee2", # 이름 변경: goal_ee2
        spawn=GOAL_MARKER_CFG.spawn.replace(
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)) # 오른쪽: 파란색
        )
    )

    # 공유 강체 (rod) - 양 그리퍼 사이에 위치하는 막대
    rod: RigidObjectCfg = ROD_CFG

    # 공유 강체의 목표 자세를 표시하는 녹색 막대
    goal_rod: RigidObjectCfg = GOAL_ROD_MARKER_CFG
