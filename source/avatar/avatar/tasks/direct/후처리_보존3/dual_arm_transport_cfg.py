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
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
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
        # (RL을 위해 4개 관절을 하나의 그룹으로 단순화)
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-7]"],
            effort_limit_sim=50.0,
            velocity_limit_sim=1.5,
            stiffness=0.0,
            damping=100.0,
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

@configclass
class DualrobotCfg(DirectRLEnvCfg):
    """
    Dofbot 2대를 스폰하는 환경의 설정 클래스입니다.
    """
    
    # === 1. 환경 기본 설정 ===
    decimation = 12 #물리연산 decimation당 1번 제어.
    episode_length_s = 30.0 
    
    # === 2. Spaces 정의 ===
    action_space = 14  # (DoF) * 2대
    observation_space = 0 # (자리 채우기, 나중에 _get_obs 수정 시 함께 변경)
    state_space = 0

    # === 3. 시뮬레이션 설정 ===
    #dt는 시뮬레이션 주기: 물리적 계산을 1초에 120번 진행, render_interval은 물리계산 몇번당 화면 업데이트할건지
    #따라서 물리 계산 120번 중 12번당 1번 할거면 1초에 10번 화면 업데이트하는거임 -> 초당 10번 정책이 행동 결정
    #그럼 만약 에피소드가 20초면 정책은 총 200번의 행동을 결정. 하나의 에피소드는 200스텝이 됨.
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # === 4. 씬(Scene) 설정 ===
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=100, env_spacing=4.0, replicate_physics=True
    )

    # === 5. 에셋(Assets) 정의 ===
    # (여기서 로봇 2대를 정의합니다)

    # 5.1. Dofbot 1 (왼쪽)
    robot_1: ArticulationCfg = ROBOT_CONFIG.replace(
        prim_path="/World/envs/env_.*/Robot_1",
        # init_state의 'pos'만 덮어써서 왼쪽(-0.5m)에 배치
        init_state=ROBOT_CONFIG.init_state.replace(pos=(-0.5, 0.0, 0.0))
    )

    # 5.2. Dofbot 2 (오른쪽)
    robot_2: ArticulationCfg = ROBOT_CONFIG.replace(
        prim_path="/World/envs/env_.*/Robot_2",
        # init_state의 'pos'만 덮어써서 오른쪽(+0.5m)에 배치
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
