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
            # 2026-06-15: 중력 ON. 적응형 K 연구의 전제(질량/자세→K 트레이드오프)는 중력 필요.
            # 컨트롤러에 gravity_comp(τ+=G(q)) 추가 — sign은 gravity_test.py로 검증.
            disable_gravity=False,
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
            disable_gravity=False,             # 2026-06-15: 중력 ON (적응형 K 연구 전제)
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
    # Cluttered transport (2026-06-19): RL=6-DoF rod pose만. 팔-장애물 회피는 컨트롤러(모델기반
    #   nullspace potential, control A)가 담당 → RL은 깨끗한 운반+rod 라우팅에 집중.
    #   action[0:3] = rod positional delta (누적), action[3:6] = rod rotation delta (axis-angle)
    #   action[6:8] = 팔당 nullspace α∈[-1,1] (팔꿈치 swing → 팔-장애물 회피). 2026-06-24 arm 회피 단계.
    #   (구 6-DoF 모델 평가 시엔 action_space=6으로 override 필요.)
    action_space = 8
    null_gain = 5.0   # nullspace α→토크 gain (τ_null = N·(α·gain·e − D_null·qd)). 튜닝 대상.
    # ── ψ_des swivel nullspace handle (2026-06-30): action[6:8]을 α(토크계수)→*목표 swivel각*으로 ──
    # 7-DoF 팔이 6-DoF EE 고정 시 여분 1-DoF = swivel각(어깨-손목 축 둘레 팔꿈치 원 위 위치). 네트워크가
    #   ψ_des 출력 → 컨트롤러가 팔꿈치를 그 각으로 servo (τ_null=N·Jₑᵀ·K·(E_des−E_cur)). α(e=ones, 1방향
    #   blunt push)와 달리 *전 원 도달* + position setpoint라 rod 끌림에 self-correct. 4번의 페널티-실패가
    #   "병목=회피 *능력*"을 가리켜 도입. use False면 기존 α 경로.
    use_swivel_nullspace = False
    swivel_gain = 60.0   # E_des servo 게인 (K_sw). 튜닝 대상(스모크).
    # Hard 안전 필터 (RoboBallet velocity-zeroing 근사, 2026-06-25): 팔 링크가 장애물 임박 시
    #   접근속도 제동 + 강한 barrier (nullspace 투영 X = 안전>task). 충돌 0 지향. 배포/학습 시 켬.
    use_hard_safety = False
    hard_d0 = 0.08      # 안전 작동 거리 [m] (링크-장애물 표면거리 < d0면 제동/반발)
    hard_krep = 30.0    # barrier 반발 게인
    hard_kbrake = 40.0  # 접근속도 제동 게인
    # Velocity-zeroing (RoboBallet식, 2026-06-26): 충돌 임박+접근 시 sim 속도를 *직접 0*으로 write.
    #   제동 토크(effort 한계·관성에 막힘)와 달리 즉시 정지. 방향성=접근중일 때만(물러남은 허용)→escapable.
    use_collision_stop = False
    stop_margin = 0.01   # 표면거리 < margin & 접근중이면 그 env 속도 0 [m]
    # ── Action-level CBF stop (2026-06-28): RoboBallet stop의 *외과적* 버전 ──
    # rod 변위 명령(action[:,0:3])을 장애물 barrier 반평면 밖으로 *최소* 투영(법선 성분만 제거,
    #   접선=미끄러짐 보존). 이산 CBF: n̂·Δp ≥ −α·h (h=standoff clearance). h→0(벽)에서 법선 침투
    #   차단=stop, h<0(추종 lag로 안쪽)이면 밀어냄. 상태 overwrite·제동토크 아님 → 명령만 수정(물리적).
    #   ★ train-WITH(교육 scaffold + 안전탐험), eval은 @off로 *네트워크가 배웠나* 판정. (velocity-zeroing/
    #   hard brake가 reach 박살낸 원인=과잉 제거; CBF는 법선만 → 협응 보존.) cbf on 시 soft filter는 off.
    use_cbf_stop = False
    cbf_alpha = 0.7      # barrier rate (per step). 접근 허용량 = α·h. ↑수록 벽 근처만 개입(=stop에 가까움).
    cbf_margin = 0.03    # standoff [m] (실제 rod의 추종 lag 흡수 — 목표를 접촉 전 standoff에서 멈춤).
    cbf_vmargin = 0.06   # 속도 인지 standoff [s]: 접근속도 v_app↑일수록 standoff += vmargin·v_app (정지거리).
                         #   target-제어는 rod 관성 overshoot를 못 막음 → 빠르면 일찍 개입해 미리 감속.
    cbf_iters = 3        # 다중 장애물 동시 해소용 투영 반복.
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

    # === 6. 장애물 (cluttered cooperative transport, 2026-06-17) ===
    # RoboBallet 참고: 에피소드마다 장애물 위치/개수 랜덤화 → 일반화 원천.
    # v1: 고정 반경 kinematic 구. 충돌은 해석적 clearance(rod 선분↔구)로 보상 처리(물리접촉 X).
    #   비활성 슬롯은 멀리(far_away) 보냄. 형태/크기 변이·가변노드는 일반화 단계에서.
    n_obstacles: int = 4              # N_OBS_MAX 슬롯 수 (그래프 obstacle 노드 수와 일치시킬 것)
    # 일반화 실험용: 학습 시 활성 장애물 최대치 cap (기본 = n_obstacles, 즉 무제한).
    #   train에 2로 주면 활성 ∈ {0,1,2} → eval(기본 4)에서 3-4는 unseen. GNN vs MLP 일반화 비교.
    max_active_obstacles: int = 4
    obstacle_radius: float = 0.06     # 구 반경 [m] (고정, v1)
    obstacle_collision_margin: float = 0.05  # clearance < margin이면 graded 페널티 시작 [m] (안전 margin↑)
    obstacle_min_active: int = 1      # 에피소드당 활성 장애물 최소
    obstacle_path_offset_std: float = 0.15   # start↔goal 경로 수직 오프셋 std [m] (분산↑로 완전차단 완화)
    obstacle_far_away: float = 50.0   # 비활성 장애물을 보낼 거리 [m]
    # 보상 가중치 (2026-06-22 Hong2025 교훈: 충돌 페널티 작게 — 크면 정책이 hesitate/freeze.
    #   RL이 회피를 *배우되* 안 굳도록. rod는 필터 backstop+개입페널티, 팔은 이 작은 페널티로 학습.)
    w_clearance: float = 1.0          # graded clearance 페널티 (5→1 축소)
    w_collision: float = 7.0          # hard 충돌(관통) 페널티 (25→7 축소, Hong2025 −7 스케일) [per-step, 비종료 모드]
    # ── Terminate-on-collision (2026-06-28): 필터/stop 다 폐기 → 충돌 허용하되 *에피소드를 끊어* RL이 회피 학습 ──
    #   sim과 안 싸우는 표준 RL 교육신호(낙상→종료 식). 충돌 step에서 종료(terminated) + 일회성 페널티.
    #   ★ 스케일 근거: 일회 페널티 > 잔여 time penalty 최대치(0.2×30=6) 여야 '시간페널티 피하려 자살충돌' 방지.
    #     +실패 시 foregone success(+100)·progress가 암묵 억제 → 과대페널티(38% 천장) 불필요. 20이 안전대.
    terminate_on_collision: bool = False
    w_collision_term: float = 20.0    # 충돌 종료 시 일회성 페널티 (>6 자살방지, <과대 천장).
    w_smooth: float = 0.01            # object-pose 명령 부드러움(가속) 페널티
    w_null: float = 0.2               # nullspace 사용 페널티 (멀 땐 0 → 정밀도 보존, 가까울 땐 팔회피)
    # ── Rod safety filter (RoboBallet식 hard 충돌방지의 rod 버전, 2026-06-22) ──
    # RL이 누적한 rod 목표(target_obj_pos)를, 그 자세의 rod 선분이 활성 장애물로부터
    # clearance ≥ d_safe를 유지하도록 projection. 실제 rod는 임피던스로 목표 추종 →
    # 목표에 lag 버퍼(margin)를 주면 실제 rod도 충돌 X. RL에서 rod 회피 부담 제거.
    # 활성 장애물 0개면 no-op → base transport 정확히 보존. 팔 충돌은 별도(미처리).
    use_rod_safety_filter: bool = True
    rod_safety_margin: float = 0.04   # contact(=ROD_R+obs_R) 위로 추가 버퍼 [m] (tracking lag 흡수)
    rod_safety_iters: int = 3         # 다중 장애물 동시 해소용 projection 반복 횟수
    # 필터 개입 페널티 (2026-06-22, 방향(b)): 필터가 target을 밀어낸 양(=RL이 장애물로 명령한 정도)에
    # 비례 페널티 → RL이 rod 회피를 *학습*(필터=안전backstop, 페널티=학습신호. RoboBallet 충실판).
    w_filter_intervene: float = 20.0  # × push[m]. push≤~0.02-0.04/step → -0.4~0.8/step (gentle)
