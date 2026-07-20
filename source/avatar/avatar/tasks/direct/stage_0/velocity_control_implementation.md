# Per-arm Velocity Control 구현 및 학습 계획

## 0. 배경 및 전환 이유

### 현재 상황
- Cooperative impedance (torque control) → 회전 채널에서 70° peak 발산
- PhysX fixed joint + Jacobian transpose + cooperative wrench split 조합이 본질적으로 불안정
- Solver iterations 255, force couple, D 증가, angle clamping 등 모든 fix 시도 후에도 해결 불가
- Translation은 완벽 (< 0.1mm 잔차), Rotation만 실패

### 전환 방향
- Torque pass-through → Velocity actuator
- Cooperative wrench split → Per-arm 독립 추종
- Jacobian transpose → Jacobian pseudoinverse (task velocity → joint velocity)

### Contribution 유지
- RL이 물체 목표를 실시간 생성 (사전 궤적 불필요) → 유지
- RL이 팔별 stiffness를 상황에 따라 조절 → 유지
- 장애물 회피 → 유지
- DA-VIL 대비: 팔별 독립 K(12D) vs 양팔 동일 K(6D), 실시간 목표 vs 사전 궤적

---

## 1. 시스템 아키텍처

### 전체 데이터 흐름

```
RL Policy (10Hz, 매 100ms)
  입력: rod 상태, goal, 관절 상태, (장애물)
  출력: desired_rod_vel (6D)
         ↓
기하학적 변환 (매 physics step, 240Hz)
  rod velocity → ee1_vel, ee2_vel
         ↓
Jacobian Pseudoinverse (240Hz)
  ee_vel → joint velocity
         ↓
Isaac Lab Velocity Actuator (240Hz, implicit PD)
  joint velocity target → PhysX가 토크 계산 및 실행
```

### RoboBallet과의 대응

```
RoboBallet:  GNN → joint velocity (56D) → velocity clamping → 적분
우리:         MLP → rod velocity (6D) → 기하 변환 → J_pinv → joint velocity → velocity actuator
```

RoboBallet보다 action이 task-space(6D)라서 차원이 낮고, 기하 변환이 물리적 의미를 보존.

---

## 2. 구현 상세

### 2.1 Actuator 변경

```python
# 기존 (torque pass-through)
ActuatorCfg(
    joint_names_expr=["panda_joint.*"],
    effort_limit=50.0,
    velocity_limit=1.5,
    stiffness=0.0,    # torque pass-through
    damping=0.0,
)

# 변경 (velocity control)
ActuatorCfg(
    joint_names_expr=["panda_joint.*"],
    effort_limit=50.0,
    velocity_limit=1.5,
    stiffness=0.0,    # velocity mode
    damping=40.0,     # velocity tracking gain (튜닝 필요)
)
```

주의:
- `stiffness=0, damping>0`이면 velocity control mode
- `damping`이 velocity tracking의 P gain 역할. 너무 낮으면 느리고, 너무 높으면 진동.
- 초기값 40.0으로 시작, 검증 시퀀스에서 튜닝.

### 2.2 Per-arm Velocity 계산

```python
def compute_joint_velocities(
    robot1, robot2, rod,
    desired_rod_lin_vel,    # (num_envs, 3) rod 선속도
    desired_rod_ang_vel,    # (num_envs, 3) rod 각속도
    ee1_idx, ee2_idx,
    grasp_offset_local,     # (3,) rod 중심에서 ee까지 offset (rod body frame)
):
    """
    Rod의 task-space velocity를 각 팔의 joint velocity로 변환.
    
    Returns:
        dq1, dq2: (num_envs, 7) 각 팔의 목표 관절 속도
    """
    # ---- 1. Grasp point offset을 world frame으로 변환 ----
    rod_quat = rod.data.root_quat_w                    # (num_envs, 4)
    
    # Rod body frame에서의 grasp offset
    # ee1: rod 중심에서 -x방향 (왼쪽)
    # ee2: rod 중심에서 +x방향 (오른쪽)
    r1_local = -grasp_offset_local  # e.g., (-0.4, 0, 0.1034)
    r2_local = +grasp_offset_local  # e.g., (+0.4, 0, 0.1034)
    
    r1_world = quat_rotate(rod_quat, r1_local)  # (num_envs, 3)
    r2_world = quat_rotate(rod_quat, r2_local)  # (num_envs, 3)
    
    # ---- 2. Rod velocity → EE velocity (rigid body 관계) ----
    # v_ee = v_rod + ω_rod × r_ee
    ee1_lin_vel = desired_rod_lin_vel + torch.cross(desired_rod_ang_vel, r1_world, dim=-1)
    ee2_lin_vel = desired_rod_lin_vel + torch.cross(desired_rod_ang_vel, r2_world, dim=-1)
    
    # Angular velocity는 rigid body이므로 동일
    ee1_ang_vel = desired_rod_ang_vel
    ee2_ang_vel = desired_rod_ang_vel
    
    # 6D EE velocity
    ee1_vel = torch.cat([ee1_lin_vel, ee1_ang_vel], dim=-1)  # (num_envs, 6)
    ee2_vel = torch.cat([ee2_lin_vel, ee2_ang_vel], dim=-1)  # (num_envs, 6)
    
    # ---- 3. Jacobian pseudoinverse ----
    J1 = get_jacobian(robot1, ee1_idx)  # (num_envs, 6, 7)
    J2 = get_jacobian(robot2, ee2_idx)  # (num_envs, 6, 7)
    
    # Damped pseudoinverse (singularity robustness)
    J1_pinv = damped_pseudoinverse(J1, damping=0.05)  # (num_envs, 7, 6)
    J2_pinv = damped_pseudoinverse(J2, damping=0.05)  # (num_envs, 7, 6)
    
    # ---- 4. Joint velocity = J_pinv × ee_vel ----
    dq1 = torch.bmm(J1_pinv, ee1_vel.unsqueeze(-1)).squeeze(-1)  # (num_envs, 7)
    dq2 = torch.bmm(J2_pinv, ee2_vel.unsqueeze(-1)).squeeze(-1)  # (num_envs, 7)
    
    # ---- 5. Joint velocity clamping ----
    dq1 = torch.clamp(dq1, -velocity_limit, velocity_limit)
    dq2 = torch.clamp(dq2, -velocity_limit, velocity_limit)
    
    return dq1, dq2
```

### 2.3 Damped Pseudoinverse (특이점 방지)

```python
def damped_pseudoinverse(J, damping=0.05):
    """
    Damped least squares pseudoinverse.
    J_pinv = J^T (J J^T + λ²I)^{-1}
    
    특이점 근처에서 J^T (J J^T)^{-1}이 발산하는 것을 방지.
    
    Args:
        J: (num_envs, 6, 7) Jacobian
        damping: float, damping factor (λ)
    
    Returns:
        J_pinv: (num_envs, 7, 6) damped pseudoinverse
    """
    JJT = torch.bmm(J, J.transpose(1, 2))           # (num_envs, 6, 6)
    eye = torch.eye(6, device=J.device).unsqueeze(0) # (1, 6, 6)
    JJT_damped = JJT + damping ** 2 * eye            # (num_envs, 6, 6)
    JJT_inv = torch.linalg.solve(JJT_damped, 
                                  torch.eye(6, device=J.device).unsqueeze(0).expand(J.shape[0], -1, -1))
    J_pinv = torch.bmm(J.transpose(1, 2), JJT_inv)  # (num_envs, 7, 6)
    return J_pinv
```

### 2.4 RL Action → Rod Velocity

```python
def apply_action(self, action):
    """
    RL의 action을 rod velocity로 변환.
    10Hz (decimation=24)로 호출.
    
    action: (num_envs, 6) — tanh 범위 [-1, 1]
    """
    # Action → rod velocity
    desired_rod_lin_vel = action[:, :3] * self.max_lin_vel    # max_lin_vel = 0.1 m/s
    desired_rod_ang_vel = action[:, 3:6] * self.max_ang_vel   # max_ang_vel = 0.5 rad/s
    
    # Joint velocity 계산
    dq1, dq2 = compute_joint_velocities(
        self.robot1, self.robot2, self.rod,
        desired_rod_lin_vel, desired_rod_ang_vel,
        self.ee1_idx, self.ee2_idx,
        self.grasp_offset_local,
    )
    
    # Velocity actuator에 명령
    self.robot1.set_joint_velocity_target(dq1)
    self.robot2.set_joint_velocity_target(dq2)
```

### 2.5 환경 설정 변경

```python
# decimation 변경: 60Hz → 10Hz
decimation = 24   # physics 240Hz / 24 = 10Hz RL

# episode length 유지 (step 수 변경)
episode_length_s = 12.5   # 초
# 에피소드당 RL step: 12.5 * 10 = 125 steps (기존 750에서 감소)

# max_lin_vel, max_ang_vel (튜닝 필요)
max_lin_vel = 0.1    # m/s. 100ms에 최대 1cm 이동
max_ang_vel = 0.5    # rad/s. 100ms에 최대 ~3° 회전
```

---

## 3. 검증 시퀀스 (RL 학습 전 필수)

RL 코드 건드리기 전에, velocity controller가 정상 작동하는지 확인.

### 검증 코드

```python
def velocity_controller_verification(env, test_stage):
    """RL 없이 velocity controller만 테스트."""
    env.reset()
    rod_init_pos = env.rod.data.root_pos_w.clone()
    rod_init_quat = env.rod.data.root_quat_w.clone()
    
    if test_stage == 0:
        # 정지: velocity = 0
        desired_lin_vel = torch.zeros(num_envs, 3)
        desired_ang_vel = torch.zeros(num_envs, 3)
    
    elif test_stage == 1:
        # 느린 x방향 이동: 0.02 m/s (1초에 2cm)
        desired_lin_vel = torch.tensor([[0.02, 0, 0]]).expand(num_envs, -1)
        desired_ang_vel = torch.zeros(num_envs, 3)
    
    elif test_stage == 2:
        # 빠른 x방향 이동: 0.1 m/s (1초에 10cm)
        desired_lin_vel = torch.tensor([[0.1, 0, 0]]).expand(num_envs, -1)
        desired_ang_vel = torch.zeros(num_envs, 3)
    
    elif test_stage == 3:
        # 순수 회전: z축 0.1 rad/s
        desired_lin_vel = torch.zeros(num_envs, 3)
        desired_ang_vel = torch.tensor([[0, 0, 0.1]]).expand(num_envs, -1)
    
    elif test_stage == 4:
        # 순수 회전: z축 0.5 rad/s (빠름)
        desired_lin_vel = torch.zeros(num_envs, 3)
        desired_ang_vel = torch.tensor([[0, 0, 0.5]]).expand(num_envs, -1)
    
    elif test_stage == 5:
        # 병진 + 회전 동시
        desired_lin_vel = torch.tensor([[0.05, 0, 0]]).expand(num_envs, -1)
        desired_ang_vel = torch.tensor([[0, 0, 0.2]]).expand(num_envs, -1)
    
    # 시뮬레이션 루프 (5초)
    history = {'time': [], 'rod_pos': [], 'rod_quat': [], 'rod_vel': [], 'dq1': [], 'dq2': []}
    
    for step in range(1200):  # 240Hz × 5초
        t = step / 240.0
        
        dq1, dq2 = compute_joint_velocities(
            env.robot1, env.robot2, env.rod,
            desired_lin_vel, desired_ang_vel,
            env.ee1_idx, env.ee2_idx,
            env.grasp_offset_local,
        )
        
        env.robot1.set_joint_velocity_target(dq1)
        env.robot2.set_joint_velocity_target(dq2)
        env.sim.step()
        
        # 기록
        rod_pos = env.rod.data.root_pos_w.clone()
        rod_vel = env.rod.data.root_lin_vel_w.clone()
        history['time'].append(t)
        history['rod_pos'].append(rod_pos[0].cpu().numpy())
        history['rod_vel'].append(rod_vel[0].cpu().numpy())
    
    return history
```

### 통과 기준

| 단계 | 테스트 | 통과 기준 |
|---|---|---|
| 0 | velocity = 0 | drift < 1mm/sec |
| 1 | 0.02 m/s x방향 | 실제 속도가 명령의 80% 이상, 직진 |
| 2 | 0.1 m/s x방향 | 실제 속도가 명령의 70% 이상, 직진 |
| 3 | 0.1 rad/s z회전 | 실제 각속도가 명령의 80% 이상, 순수 회전 |
| 4 | 0.5 rad/s z회전 | 실제 각속도가 명령의 60% 이상, 발산 없음 |
| 5 | 병진 + 회전 동시 | 두 채널 모두 추종, cross-coupling 작음 |

핵심: **단계 3~4가 통과하면 torque control 대비 회전이 해결된 것**. 70° peak 대신 안정적 회전 확인.

### 결과 해석

- 단계 0 실패: actuator 설정 오류. damping 값 확인.
- 단계 1~2 실패: Jacobian pseudoinverse 또는 기하학적 변환 오류.
- 단계 3~4 실패: velocity actuator의 damping 부족 또는 closed chain 문제 잔존. damping 올리기.
- 단계 0~5 전부 통과: RL 학습 진행.

---

## 4. RL 학습 설정

### 4.1 알고리즘: TD3 (RoboBallet과 동일)

```python
# TD3 hyperparameters
batch_size = 512
buffer_size = 1_000_000
gamma = 0.99
tau_polyak = 0.005
lr_actor = 3e-4
lr_critic = 3e-4
exploration_noise_std = 0.3      # 초기 탐색 노이즈
exploration_noise_decay = 0.999  # 에피소드마다 decay
exploration_noise_min = 0.05     # 최소 노이즈
policy_delay = 2                 # critic 2번 업데이트마다 actor 1번
target_noise_std = 0.2           # target policy smoothing
target_noise_clip = 0.5
warmup_steps = 10_000
```

SAC 대신 TD3를 쓰는 이유:
- RoboBallet이 TD3로 성공 (검증된 조합)
- α collapse 문제 구조적으로 불가능
- Deterministic policy가 이 태스크에 적합 (명확한 최적 행동 존재)

### 4.2 Action Space (6D)

```python
# RL 출력: tanh [-1, 1] → rod velocity
action_dim = 6

# Action 해석
desired_rod_lin_vel = action[:, :3] * max_lin_vel    # (3D)
desired_rod_ang_vel = action[:, 3:6] * max_ang_vel   # (3D)
```

Non-accumulating: 매 스텝 현재 rod 상태에서 velocity를 계산. 누적 없음.

### 4.3 Observation (MLP 우선)

```python
# 모든 관측을 flat vector로 (MLP용)
obs = torch.cat([
    # Rod 상태 (19D)
    rod_pos_local,          # (3) env-local 위치
    rod_quat,               # (4) 자세
    rod_lin_vel,            # (3) 선속도
    rod_ang_vel,            # (3) 각속도
    
    # Goal (7D)
    goal_pos_local,         # (3) env-local 목표 위치
    goal_quat,              # (4) 목표 자세
    
    # 오차 (6D)
    pos_error,              # (3) goal - rod_pos
    rot_error_axis_angle,   # (3) 자세 오차
    
    # 팔1 관절 (14D)
    joint_q_1,              # (7) 관절 각도
    joint_dq_1,             # (7) 관절 속도
    
    # 팔2 관절 (14D)
    joint_q_2,              # (7) 관절 각도
    joint_dq_2,             # (7) 관절 속도
    
    # EE 위치 (6D)
    ee1_pos_local,          # (3) EE1 위치
    ee2_pos_local,          # (3) EE2 위치
    
    # 시간 (1D)
    normalized_time,        # (1) 에피소드 진행도
], dim=-1)  # total: ~67D
```

### 4.4 Network (MLP 우선)

```python
# Actor: deterministic (TD3)
actor = MLP([67, 256, 256, 6])        # ~200K params
actor_output = torch.tanh(actor(obs))  # [-1, 1]

# Twin Critics
critic1 = MLP([67 + 6, 256, 256, 1])
critic2 = MLP([67 + 6, 256, 256, 1])
```

### 4.5 Reward

```python
# 1. Progress reward (핵심, dense)
prev_dist = torch.norm(prev_rod_pos - goal_pos, dim=-1)
curr_dist = torch.norm(curr_rod_pos - goal_pos, dim=-1)
r_progress_pos = (prev_dist - curr_dist) * 30.0

# 회전도 포함
prev_rot_dist = compute_rot_distance(prev_rod_quat, goal_quat)
curr_rot_dist = compute_rot_distance(curr_rod_quat, goal_quat)
r_progress_rot = (prev_rot_dist - curr_rot_dist) * 10.0

r_progress = r_progress_pos + r_progress_rot

# 2. Success reward (sparse)
pos_err = torch.norm(rod_pos - goal_pos, dim=-1)
rot_err = compute_rot_distance(rod_quat, goal_quat)
is_success = (pos_err < 0.05) & (rot_err < 0.15)  # 5cm, ~8.6°
r_success = 50.0 * is_success.float()

# 3. Action smoothness (선택)
r_smooth = -0.01 * torch.norm(action - prev_action, dim=-1)

# Total
reward = r_progress + r_success + r_smooth
```

### 4.6 HER (Hindsight Experience Replay)

```python
# Future strategy, k=4
# 에피소드 종료 시:
for t in range(episode_length):
    for _ in range(4):
        future_t = random.randint(t, episode_length - 1)
        virtual_goal_pos = trajectory[future_t].rod_pos
        virtual_goal_quat = trajectory[future_t].rod_quat
        
        # Observation에서 goal 부분 교체
        virtual_obs = replace_goal(obs[t], virtual_goal_pos, virtual_goal_quat)
        virtual_next_obs = replace_goal(next_obs[t], virtual_goal_pos, virtual_goal_quat)
        
        # Reward 재계산
        virtual_reward = compute_reward(trajectory[t].rod_pos, virtual_goal_pos)
        
        buffer.add(virtual_obs, action[t], virtual_reward, virtual_next_obs)
```

### 4.7 Curriculum

```python
# 자동 진행
curriculum_stages = [
    {'dist_min': 0.03, 'dist_max': 0.05, 'rot_max': 0.10},  # Stage 0: 3~5cm
    {'dist_min': 0.05, 'dist_max': 0.10, 'rot_max': 0.15},  # Stage 1: 5~10cm
    {'dist_min': 0.08, 'dist_max': 0.15, 'rot_max': 0.20},  # Stage 2: 8~15cm
    {'dist_min': 0.10, 'dist_max': 0.20, 'rot_max': 0.30},  # Stage 3: 10~20cm
    {'dist_min': 0.15, 'dist_max': 0.30, 'rot_max': 0.50},  # Stage 4: 15~30cm
]

# 진행 조건: rolling success rate > 70%
if rolling_success_rate > 0.7 and current_stage < len(curriculum_stages) - 1:
    current_stage += 1
```

### 4.8 Termination

```python
# 성공 시 즉시 종료
terminated = is_success

# 또는 timeout
truncated = (step >= max_steps)
```

---

## 5. 실행 계획

### Phase 1: Velocity Controller 구현 + 검증 (1~2일)

| 작업 | 파일 | 예상 시간 |
|---|---|---|
| Actuator를 velocity mode로 변경 | env config | 30분 |
| compute_joint_velocities 구현 | controller 파일 | 2시간 |
| damped_pseudoinverse 구현 | 유틸 파일 | 30분 |
| 검증 시퀀스 Stage 0~5 실행 | 검증 스크립트 | 2시간 |
| damping 튜닝 (필요 시) | env config | 1시간 |

성공 기준: Stage 0~5 전부 통과. 특히 Stage 3~4(회전)에서 발산 없음.

### Phase 2: RL 학습 환경 구축 (1일)

| 작업 | 파일 | 예상 시간 |
|---|---|---|
| decimation=24 (10Hz) 설정 | env config | 10분 |
| apply_action 수정 (velocity command) | env 파일 | 1시간 |
| Observation 구성 (flat vector) | env 파일 | 1시간 |
| Reward 함수 구현 | env 파일 | 1시간 |
| MLP policy 구현 | policy 파일 | 1시간 |
| TD3 trainer 구현 (또는 기존 SAC에서 수정) | trainer 파일 | 2시간 |

### Phase 3: 기본 학습 (2~3일)

| 실험 | 설정 | 기대 결과 |
|---|---|---|
| Exp 1 | MLP + 10Hz + 3~5cm + HER 없이 | success > 50% baseline |
| Exp 2 | Exp 1 + HER | success > 80% |
| Exp 3 | Exp 2 + curriculum 확장 (5→10→20cm) | success > 70% at 20cm |

각 실험 5~10M RL steps, 약 2~4시간.

### Phase 4: Variable Stiffness 추가 (1주)

```python
# Action 확장: 6D → 18D
# rod_velocity (6D) + K_arm1 (6D) + K_arm2 (6D)

desired_rod_vel = action[:, :6] * max_vel

K_arm1 = softplus(action[:, 6:12]) * K_scale   # 항상 양수
K_arm2 = softplus(action[:, 12:18]) * K_scale

# K를 velocity 크기에 반영
# K가 크면 빨리 추종, 작으면 천천히 추종 (compliance)
ee1_vel = (K_arm1 / D_nominal) * (ee1_target_vel)
ee2_vel = (K_arm2 / D_nominal) * (ee2_target_vel)
```

비교: Fixed-K vs RL-K. 내력, 토크 피크, 성공률 측정.

### Phase 5: 장애물 추가 (1주)

장애물 관측 추가. Curriculum으로 장애물 도입.

---

## 6. 코드 파일 매핑

| 변경 사항 | 대상 파일 | 수정 범위 |
|---|---|---|
| Actuator mode 변경 | env config | ~5줄 |
| compute_joint_velocities | 신규 또는 controller 파일 | ~80줄 |
| damped_pseudoinverse | 유틸 파일 | ~20줄 |
| apply_action 수정 | env 파일 | ~20줄 |
| decimation 변경 | env config | 1줄 |
| Observation 구성 | env 파일 | ~30줄 |
| Reward 함수 | env 파일 | ~20줄 |
| MLP policy | 신규 | ~50줄 |
| TD3 trainer | 신규 또는 SAC 수정 | ~200줄 |
| HER replay buffer | 신규 | ~100줄 |
| 검증 스크립트 | 신규 | ~100줄 |
| Curriculum 로직 | env 파일 | ~30줄 |

총 신규/수정: ~650줄

---

## 7. 변경하지 말 것

- Rod 물리 (mass, size, fixed joint): 변경 없음
- 로봇 배치 (±0.5m, yaw 0/π): 변경 없음
- 좌표계 (env-local): 변경 없음
- 중력 설정 (OFF): Phase 1~3에서 변경 없음
- Graph 구조 (GNN용): 당장 사용 안 하지만 삭제하지 말 것

---

## 8. 성공 판단 기준

### Phase 1 후
- 검증 Stage 0~5 전부 PASS
- 특히 Stage 3~4(회전)에서 발산 없음 확인
- PASS면 Phase 2로. FAIL이면 damping 튜닝 또는 actuator 설정 재검토.

### Phase 3 후
- 3~5cm에서 success > 80%: 구조 확정. curriculum 확장으로 진행.
- 3~5cm에서 success 30~80%: reward/action scale 튜닝 후 재시도.
- 3~5cm에서 success < 30%: velocity controller 또는 HER 구현 문제. 검증 시퀀스 재확인.

### Phase 4 후
- RL-K가 Fixed-K 대비 내력 감소: Variable stiffness contribution 성립.
- RL-K가 Fixed-K와 동등: K modulation이 이 태스크에서 이점 없음. 실시간 목표 생성만으로 contribution.

---

## 9. 모니터링 지표

### 학습 지표 (TD3)
- Episode return (상승해야 함)
- Critic loss (안정적으로 감소)
- Actor loss (완만하게 변화)
- Exploration noise std (decay 확인)

### 태스크 지표
- episode_success_rate (핵심)
- min_pos_err_mean (최소 도달 거리)
- min_rot_err_mean (최소 자세 오차)
- episode_length_mean (빨리 도달할수록 짧아짐)

### 안전 지표 (Phase 4 이후)
- internal_force_mean, internal_force_max
- joint_torque_peak
- K_arm1_mean, K_arm2_mean (K 통계)

---

## 10. 향후 확장 (Phase 4 이후)

### Variable Stiffness 논문 framing

> "Per-arm velocity control 위에서 RL이 task-space stiffness를 팔별로 독립 조절하여,
>  고정 stiffness 대비 내력 안전성과 에너지 효율을 개선함을 실험적으로 보인다."

### DA-VIL 대비 차별점

| | DA-VIL | 우리 |
|---|---|---|
| K 출력 | 양팔 동일 K (6D) | 팔별 독립 K (12D) |
| 궤적 | 사전 보간 (고정) | 실시간 생성 (RL) |
| 장애물 | 불가 | 가능 (Phase 5) |
| 내력 보고 | 없음 | 있음 |
| 시뮬레이터 | MuJoCo | Isaac Lab (PhysX) |
| Controller | QP + impedance | Per-arm velocity + K modulation |
