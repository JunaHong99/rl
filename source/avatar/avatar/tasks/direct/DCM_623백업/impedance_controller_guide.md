# 임피던스 컨트롤러 구현 가이드

## 0. 개요

임피던스 컨트롤러는 로봇 end-effector가 "가상의 스프링-댐퍼로 목표에 연결된 것처럼" 행동하도록 관절 토크를 계산하는 제어기.

최종 출력은 **관절 토크(τ)**. 이것을 모터에 인가하면 로봇이 움직임.

---

## 1. 단일 팔 임피던스 컨트롤러 (기본형)

### 1.1 전체 흐름

```
목표 pose (x_target, quat_target)
        ↓
현재 pose 읽기 (FK로 계산)
        ↓
오차 계산: pos_error, rot_error, vel_error
        ↓
임피던스 법칙: f = K * error + D * vel_error    (6D wrench)
        ↓
관절 토크 변환: τ = Jᵀ × f
        ↓
중력 보상 추가: τ += τ_gravity
        ↓
토크 클램핑: τ = clamp(τ, -limit, +limit)
        ↓
모터에 인가
```

### 1.2 임피던스 법칙

```
f = K * (x_target - x_current) + D * (ẋ_target - ẋ_current)
```

| 기호 | 의미 | 차원 |
|---|---|---|
| x_target | EE 목표 위치/자세 | 6D (pos 3 + rot 3) |
| x_current | EE 현재 위치/자세 | 6D |
| ẋ_target | EE 목표 속도 (보통 0) | 6D |
| ẋ_current | EE 현재 속도 | 6D |
| K | 강성 행렬 (대각) | 6×6. 단위: N/m (병진), Nm/rad (회전) |
| D | 감쇠 행렬 (대각) | 6×6. 단위: Ns/m, Nms/rad |
| f | 원하는 wrench | 6D (force 3 + torque 3) |

### 1.3 관절 토크 변환

```
τ_impedance = Jᵀ × f
```

| 기호 | 의미 | 차원 |
|---|---|---|
| J | Jacobian | 6×7 (7-DOF 팔) |
| Jᵀ | Jacobian transpose | 7×6 |
| f | wrench | 6×1 |
| τ | 관절 토크 | 7×1 |

### 1.4 최종 토크

```
τ = τ_impedance + τ_gravity
τ = clamp(τ, -effort_limit, +effort_limit)
```

### 1.5 코드

```python
def compute_torque_single_arm(robot, target_pos, target_quat,
                               K_pos, K_rot, D_pos, D_rot,
                               ee_idx, effort_limit):
    """
    단일 팔 임피던스 컨트롤러.
    
    Args:
        robot: Isaac Lab articulation
        target_pos: (num_envs, 3) 목표 EE 위치
        target_quat: (num_envs, 4) 목표 EE 자세 (w,x,y,z)
        K_pos: float or (3,) 병진 강성
        K_rot: float or (3,) 회전 강성
        D_pos: float or (3,) 병진 감쇠
        D_rot: float or (3,) 회전 감쇠
        ee_idx: int, EE body 인덱스
        effort_limit: float, 토크 한계
    
    Returns:
        tau: (num_envs, 7) 관절 토크
    """
    # ---- 1. 현재 EE 상태 읽기 ----
    ee_pos = robot.data.body_pos_w[:, ee_idx, :]         # (num_envs, 3)
    ee_quat = robot.data.body_quat_w[:, ee_idx, :]       # (num_envs, 4)
    ee_lin_vel = robot.data.body_lin_vel_w[:, ee_idx, :]  # (num_envs, 3)
    ee_ang_vel = robot.data.body_ang_vel_w[:, ee_idx, :]  # (num_envs, 3)
    
    # ---- 2. 위치 오차 ----
    pos_error = target_pos - ee_pos                       # (num_envs, 3)
    
    # ---- 3. 자세 오차 (axis-angle) ----
    rot_error = compute_orientation_error(target_quat, ee_quat)  # (num_envs, 3)
    
    # ---- 4. 속도 오차 (목표 속도 = 0 가정) ----
    lin_vel_error = -ee_lin_vel                           # (num_envs, 3)
    ang_vel_error = -ee_ang_vel                           # (num_envs, 3)
    
    # ---- 5. 임피던스 법칙 → wrench ----
    force = K_pos * pos_error + D_pos * lin_vel_error     # (num_envs, 3)
    torque = K_rot * rot_error + D_rot * ang_vel_error    # (num_envs, 3)
    wrench = torch.cat([force, torque], dim=-1)           # (num_envs, 6)
    
    # ---- 6. Jacobian 가져오기 ----
    jacobian_full = robot.root_physx_view.get_jacobians()
    J_ee = jacobian_full[:, ee_idx, :, :]                 # (num_envs, 6, 7)
    
    # ---- 7. 관절 토크 = Jᵀ × wrench ----
    tau = torch.bmm(
        J_ee.transpose(1, 2),           # (num_envs, 7, 6)
        wrench.unsqueeze(-1)            # (num_envs, 6, 1)
    ).squeeze(-1)                        # (num_envs, 7)
    
    # ---- 8. 중력 보상 (gravity ON일 때) ----
    # tau += compute_gravity_compensation(robot)
    
    # ---- 9. 토크 클램핑 ----
    tau = torch.clamp(tau, -effort_limit, effort_limit)
    
    return tau
```

---

## 2. Cooperative 임피던스 컨트롤러 (양팔용)

### 2.1 단일 팔과의 차이

단일 팔 controller를 두 번 독립적으로 돌리면 DA-VIL 방식.
→ K_abs와 K_rel을 독립 조절 불가, 내력 제어 안 됨.

Cooperative 버전은 **물체 수준에서 먼저 힘을 계산하고 양팔에 분배**.
→ K_abs(궤적 추종)와 K_rel(내력 조절) 독립 조절 가능.

### 2.2 전체 흐름

```
양팔 EE 상태 읽기 (ee1, ee2)
        ↓
Cooperative Task Space 변환
  x_abs = (ee1 + ee2) / 2     (물체 중심)
  x_rel = ee1 - ee2            (두 팔 사이 거리)
        ↓
절대 운동 임피던스: f_abs = K_abs * err_abs + D_abs * vel_err_abs
상대 운동 임피던스: f_rel = K_rel * err_rel + D_rel * vel_err_rel
        ↓
양팔 분배:
  f_ee1 = f_abs/2 + f_rel
  f_ee2 = f_abs/2 - f_rel
        ↓
각 팔 관절 토크:
  τ₁ = J₁ᵀ × f_ee1 + gravity_comp₁
  τ₂ = J₂ᵀ × f_ee2 + gravity_comp₂
        ↓
토크 클램핑 → 모터에 인가
```

### 2.3 코드

```python
def compute_cooperative_impedance(
    robot1, robot2,
    target_obj_pos, target_obj_quat,
    x_rel_desired_pos, x_rel_desired_quat,
    K_abs_pos, D_abs_pos, K_abs_rot, D_abs_rot,
    K_rel_pos, D_rel_pos, K_rel_rot, D_rel_rot,
    ee_idx, effort_limit
):
    """
    Cooperative 임피던스 컨트롤러.
    두 팔에 대해 하나의 controller가 존재.
    양팔의 상태를 동시에 읽어서 양팔의 토크를 동시에 출력.
    
    Args:
        robot1, robot2: Isaac Lab articulations
        target_obj_pos: (num_envs, 3) 물체 목표 위치
        target_obj_quat: (num_envs, 4) 물체 목표 자세
        x_rel_desired_pos: (num_envs, 3) 원하는 두 팔 사이 거리 (초기값 기억)
        x_rel_desired_quat: (num_envs, 4) 원하는 두 팔 사이 상대 자세
        K/D 파라미터들: 강성/감쇠
        ee_idx: EE body 인덱스
        effort_limit: 토크 한계
    
    Returns:
        tau_1: (num_envs, 7) 팔1 관절 토크
        tau_2: (num_envs, 7) 팔2 관절 토크
    """
    
    # ==================================================
    # 1. 현재 양팔 EE 상태 읽기
    # ==================================================
    ee1_pos = robot1.data.body_pos_w[:, ee_idx, :]
    ee1_quat = robot1.data.body_quat_w[:, ee_idx, :]
    ee1_lin_vel = robot1.data.body_lin_vel_w[:, ee_idx, :]
    ee1_ang_vel = robot1.data.body_ang_vel_w[:, ee_idx, :]
    
    ee2_pos = robot2.data.body_pos_w[:, ee_idx, :]
    ee2_quat = robot2.data.body_quat_w[:, ee_idx, :]
    ee2_lin_vel = robot2.data.body_lin_vel_w[:, ee_idx, :]
    ee2_ang_vel = robot2.data.body_ang_vel_w[:, ee_idx, :]
    
    # ==================================================
    # 2. Cooperative Task Space 변환
    # ==================================================
    
    # --- 절대 운동 (물체 중심) ---
    x_abs_pos = (ee1_pos + ee2_pos) / 2              # (num_envs, 3)
    v_abs_lin = (ee1_lin_vel + ee2_lin_vel) / 2      # (num_envs, 3)
    v_abs_ang = (ee1_ang_vel + ee2_ang_vel) / 2      # (num_envs, 3)
    
    # 절대 자세: rod의 실제 자세를 직접 사용하는 것이 더 정확
    # 또는 두 EE 자세의 평균을 사용 (근사)
    # 실용적 선택: rod.data.root_quat_w 사용 권장
    x_abs_quat = compute_average_quaternion(ee1_quat, ee2_quat)
    
    # --- 상대 운동 (두 팔 사이) ---
    x_rel_pos = ee1_pos - ee2_pos                    # (num_envs, 3)
    v_rel_lin = ee1_lin_vel - ee2_lin_vel             # (num_envs, 3)
    v_rel_ang = ee1_ang_vel - ee2_ang_vel             # (num_envs, 3)
    
    # 상대 자세
    x_rel_quat = quat_mul(quat_conjugate(ee2_quat), ee1_quat)
    
    # ==================================================
    # 3. 절대 운동 임피던스 (물체를 목표로 끌어당김)
    # ==================================================
    pos_err_abs = target_obj_pos - x_abs_pos
    rot_err_abs = compute_orientation_error(target_obj_quat, x_abs_quat)
    
    f_abs_lin = K_abs_pos * pos_err_abs + D_abs_pos * (0 - v_abs_lin)
    f_abs_ang = K_abs_rot * rot_err_abs + D_abs_rot * (0 - v_abs_ang)
    f_abs = torch.cat([f_abs_lin, f_abs_ang], dim=-1)   # (num_envs, 6)
    
    # ==================================================
    # 4. 상대 운동 임피던스 (내력 조절)
    # ==================================================
    pos_err_rel = x_rel_desired_pos - x_rel_pos
    rot_err_rel = compute_orientation_error(x_rel_desired_quat, x_rel_quat)
    
    f_rel_lin = K_rel_pos * pos_err_rel + D_rel_pos * (0 - v_rel_lin)
    f_rel_ang = K_rel_rot * rot_err_rel + D_rel_rot * (0 - v_rel_ang)
    f_rel = torch.cat([f_rel_lin, f_rel_ang], dim=-1)   # (num_envs, 6)
    
    # ==================================================
    # 5. 양팔에 분배
    # ==================================================
    f_ee1 = f_abs / 2 + f_rel    # (num_envs, 6)
    f_ee2 = f_abs / 2 - f_rel    # (num_envs, 6)
    
    # ==================================================
    # 6. 각 팔의 관절 토크 = Jᵀ × wrench
    # ==================================================
    J1 = get_jacobian(robot1, ee_idx)   # (num_envs, 6, 7)
    J2 = get_jacobian(robot2, ee_idx)   # (num_envs, 6, 7)
    
    tau_1 = torch.bmm(J1.transpose(1, 2), f_ee1.unsqueeze(-1)).squeeze(-1)
    tau_2 = torch.bmm(J2.transpose(1, 2), f_ee2.unsqueeze(-1)).squeeze(-1)
    
    # ==================================================
    # 7. 중력 보상 (gravity ON일 때 활성화)
    # ==================================================
    # tau_1 += compute_gravity_compensation(robot1)
    # tau_2 += compute_gravity_compensation(robot2)
    
    # ==================================================
    # 8. 토크 클램핑
    # ==================================================
    tau_1 = torch.clamp(tau_1, -effort_limit, effort_limit)
    tau_2 = torch.clamp(tau_2, -effort_limit, effort_limit)
    
    return tau_1, tau_2
```

---

## 3. 유틸리티 함수

### 3.1 자세 오차 계산

```python
def compute_orientation_error(q_target, q_current):
    """
    두 quaternion 사이의 orientation error를 axis-angle (3D)로 반환.
    
    Args:
        q_target: (num_envs, 4) 목표 자세 (w, x, y, z)
        q_current: (num_envs, 4) 현재 자세 (w, x, y, z)
    
    Returns:
        error: (num_envs, 3) axis-angle 오차
    """
    # q_error = q_target * q_current^{-1}
    q_error = quat_mul(q_target, quat_conjugate(q_current))
    
    # Double cover 처리: q와 -q는 같은 회전
    # w < 0이면 negate해서 짧은 경로 선택
    q_error = torch.where(
        q_error[:, 0:1] < 0,
        -q_error,
        q_error
    )
    
    # Quaternion → axis-angle
    # 작은 각도 근사: axis-angle ≈ 2 * [x, y, z]
    # 큰 각도: angle = 2 * arccos(w), axis = [x,y,z] / ||[x,y,z]||
    w = q_error[:, 0:1]                           # (num_envs, 1)
    xyz = q_error[:, 1:4]                          # (num_envs, 3)
    
    # 안전한 arccos
    angle = 2.0 * torch.acos(torch.clamp(w, -1.0 + 1e-7, 1.0 - 1e-7))
    
    # axis 정규화
    xyz_norm = torch.norm(xyz, dim=-1, keepdim=True)  # (num_envs, 1)
    axis = xyz / (xyz_norm + 1e-8)
    
    # axis-angle = axis * angle
    orientation_error = axis * angle               # (num_envs, 3)
    
    return orientation_error
```

### 3.2 Quaternion 연산

```python
def quat_mul(q1, q2):
    """
    Quaternion 곱셈. q1 * q2.
    Convention: (w, x, y, z)
    """
    w1, x1, y1, z1 = q1[:, 0:1], q1[:, 1:2], q1[:, 2:3], q1[:, 3:4]
    w2, x2, y2, z2 = q2[:, 0:1], q2[:, 1:2], q2[:, 2:3], q2[:, 3:4]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return torch.cat([w, x, y, z], dim=-1)


def quat_conjugate(q):
    """
    Quaternion 켤레. (w, -x, -y, -z)
    """
    return torch.cat([q[:, 0:1], -q[:, 1:4]], dim=-1)
```

### 3.3 Jacobian 가져오기 (Isaac Lab)

```python
def get_jacobian(robot, ee_body_idx):
    """
    Isaac Lab에서 EE의 geometric Jacobian 가져오기.
    
    Returns:
        J: (num_envs, 6, num_dofs) Jacobian
    """
    # 방법 1: root_physx_view 사용
    jacobian_full = robot.root_physx_view.get_jacobians()
    # shape: (num_envs, num_bodies, 6, num_dofs)
    J = jacobian_full[:, ee_body_idx, :, :]
    # shape: (num_envs, 6, num_dofs)
    
    return J
    
    # 방법 2: Isaac Lab의 differential_inverse_kinematics 유틸 참고
    # from omni.isaac.lab.utils.math import ...
```

### 3.4 중력 보상 (gravity ON 시 필요)

```python
def compute_gravity_compensation(robot):
    """
    현재 관절 배치에서 중력을 상쇄하는 토크.
    Gravity OFF 상태에서는 불필요 (return 0).
    
    Returns:
        tau_gravity: (num_envs, num_dofs)
    """
    # 방법 1: Isaac Lab의 computed_torque 활용
    # (주의: 이것이 순수 중력 보상인지 확인 필요)
    
    # 방법 2: 직접 계산
    # τ_gravity = G(q) where G is the gravity vector
    # Isaac Lab에서 gravity term만 분리하는 방법은 API에 따라 다름
    
    # 방법 3: 속도/가속도 = 0에서의 inverse dynamics
    # τ_gravity = ID(q, 0, 0) = h(q, 0)
    # 여기서 h는 coriolis + gravity term, q̇=0이면 gravity만 남음
    
    pass
```

---

## 4. K, D 파라미터 설정 가이드

### 4.1 Critical Damping 조건

진동 없이 가장 빨리 수렴하는 damping:

```python
# 단위 질량 가정 (간이)
D = 2.0 * torch.sqrt(K)

# Task-space 관성 고려 (정확)
# Λ = (J M^{-1} J^T)^{-1}   (task-space inertia)
# D = √Λ √K + √K √Λ
```

간이 버전(`D = 2√K`)으로 시작하고, 진동이 생기면 D를 높이는 식으로 튜닝.

### 4.2 K 값 가이드라인

| K | 물리적 의미 | 병진 (N/m) | 회전 (Nm/rad) |
|---|---|---|---|
| 너무 낮음 | 느리게 수렴, 외란에 약함 | < 50 | < 5 |
| 적절 | 적당한 속도로 수렴 | 100~500 | 10~50 |
| 너무 높음 | 빠르지만 진동/불안정 가능 | > 1000 | > 100 |

현재 설정: K_abs_pos=200, K_abs_rot=20. 합리적인 시작점.

### 4.3 Closed Chain에서의 K_rel 설정

Fixed joint로 양팔이 연결된 경우:
- K_rel = 0, D_rel = 0 (현재 설정)
- 이유: fixed joint가 기하학적 제약을 보장하므로, 상대 운동 임피던스가 불필요
- K_rel > 0으로 하면 fixed joint와 충돌하여 numerical noise 발생 가능

마찰 그래스프에서는 K_rel > 0이 필요 (내력으로 물체를 쥐어야 함).

---

## 5. 검증 시퀀스

Controller 구현 후 반드시 아래 순서로 검증. RL은 검증 완료 후에만 시작.

### 검증 코드 구조

```python
def controller_verification_test(env, test_stage, test_params):
    """
    RL 없이 controller만 테스트.
    target을 수동으로 설정하고 rod의 응답을 기록.
    """
    env.reset()
    
    # 초기 rod 상태 기록
    rod_init_pos = env.rod.data.root_pos_w.clone()
    rod_init_quat = env.rod.data.root_quat_w.clone()
    
    # 테스트별 target 설정
    if test_stage == 0:
        # 단계 0: target = 현재 위치 (정지 유지)
        target_pos = rod_init_pos.clone()
        target_quat = rod_init_quat.clone()
    
    elif test_stage == 1:
        # 단계 1: +1mm step
        target_pos = rod_init_pos.clone()
        target_pos[:, 0] += 0.001  # x방향 1mm
        target_quat = rod_init_quat.clone()
    
    elif test_stage == 2:
        # 단계 2: +1cm step
        target_pos = rod_init_pos.clone()
        target_pos[:, 0] += 0.01   # x방향 1cm
        target_quat = rod_init_quat.clone()
    
    elif test_stage == 3:
        # 단계 3: +5cm step
        target_pos = rod_init_pos.clone()
        target_pos[:, 0] += 0.05   # x방향 5cm
        target_quat = rod_init_quat.clone()
    
    elif test_stage == 4:
        # 단계 4: 순수 회전 5°
        target_pos = rod_init_pos.clone()
        angle_rad = 5.0 * 3.14159 / 180.0
        delta_quat = axis_angle_to_quat(torch.tensor([[0, 0, angle_rad]]))
        target_quat = quat_mul(delta_quat, rod_init_quat)
    
    # 시뮬레이션 루프 (10초)
    history = {'time': [], 'rod_pos': [], 'rod_quat': [], 'pos_error': [], 'tau_1': [], 'tau_2': []}
    
    for step in range(2400):  # 240Hz × 10초
        t = step / 240.0
        
        # Controller 토크 계산
        tau_1, tau_2 = compute_cooperative_impedance(
            robot1=env.robot1, robot2=env.robot2,
            target_obj_pos=target_pos,
            target_obj_quat=target_quat,
            x_rel_desired_pos=env.x_rel_init_pos,
            x_rel_desired_quat=env.x_rel_init_quat,
            K_abs_pos=200, D_abs_pos=60,
            K_abs_rot=20, D_abs_rot=8,
            K_rel_pos=0, D_rel_pos=0,
            K_rel_rot=0, D_rel_rot=0,
            ee_idx=env.ee_idx,
            effort_limit=50.0
        )
        
        # 토크 인가
        env.robot1.set_joint_effort_target(tau_1)
        env.robot2.set_joint_effort_target(tau_2)
        
        # Physics step
        env.sim.step()
        
        # 기록
        rod_pos = env.rod.data.root_pos_w.clone()
        pos_error = torch.norm(target_pos - rod_pos, dim=-1)
        
        history['time'].append(t)
        history['rod_pos'].append(rod_pos[0].cpu().numpy())
        history['pos_error'].append(pos_error[0].item())
        history['tau_1'].append(tau_1[0].cpu().numpy())
        history['tau_2'].append(tau_2[0].cpu().numpy())
    
    return history
```

### 통과 기준

| 단계 | 테스트 | 통과 기준 | 의미 |
|---|---|---|---|
| 0 | target = rod_init (zero offset) | drift < 1mm/sec | "가만히 있기"가 됨 |
| 1 | target = rod_init + 1mm | monotonic 수렴, overshoot 없음 | 선형 영역 정상 |
| 2 | target = rod_init + 1cm | < 20% overshoot, 5초 내 settling | 작은 step 안정 |
| 3 | target = rod_init + 5cm | settling (overshoot 허용) | task 거리 동작 |
| 4 | 순수 회전 5°, 10°, 20° | 각도별 settling | 회전 channel 동작 |
| 5 | Goal mode 16-env | 56% 이상 | 통계적 검증 |

### 결과 해석

**단계 0에서 실패** (drift > 1mm/sec):
- controller에 버그 있음
- 확인: 좌표 변환 부호, Jacobian 인덱스, wrench 분배 부호, quat convention
- 특히 `f_ee1 = f_abs/2 + f_rel` vs `f_ee1 = f_abs/2 - f_rel` 부호 확인

**단계 1~2에서 진동/발산**:
- D(감쇠)가 부족하거나 K가 너무 높음
- D를 2배로 올려서 재시도
- 또는 K를 절반으로 낮춰서 재시도

**단계 3에서 overshoot > 50%**:
- closed chain의 유효 관성이 커서 단일 팔보다 제동이 어려움
- D를 더 높이거나, K를 낮추거나
- effort_limit에 걸려서 토크가 부족할 수 있음 → effort_limit 확인

**단계 0~3 통과하는데 단계 5에서 56%**:
- 특정 자세/방향 조합에서 실패
- IK reachability, 특이점, workspace 경계 문제
- 실패 케이스의 시작 자세/목표 자세를 분석하여 패턴 찾기

---

## 6. 흔한 구현 실수

### 6.1 Quaternion Convention 불일치
Isaac Lab은 (w, x, y, z), 일부 라이브러리는 (x, y, z, w).
모든 곳에서 같은 convention을 쓰는지 확인.

### 6.2 Jacobian Body Index 오류
`ee_idx`가 정확한 body를 가리키는지 확인.
`robot.body_names`를 출력해서 인덱스 매핑 확인.

### 6.3 좌표계 불일치
target이 world frame인데 controller가 local frame으로 계산하거나 반대.
모든 위치/속도가 같은 frame에 있는지 확인.

### 6.4 Wrench 분배 부호
`f_ee1 = f_abs/2 + f_rel`이 맞는지, 로봇 배치에 따라 부호가 달라질 수 있음.
두 로봇의 base frame orientation이 반대(yaw 0 vs π)이면 주의.

### 6.5 토크 클램핑으로 인한 성능 저하
K가 높으면 계산된 토크가 effort_limit을 초과 → 클램핑 → 실제 적용 토크가 의도보다 작음.
검증 시 클램핑이 일어나는 빈도와 크기를 로깅.

---

## 7. 다음 단계

1. 위 코드를 구현하고 검증 시퀀스 단계 0~4를 통과시킴
2. 통과 후 RL 연결 (10Hz non-accumulating action)
3. RL로 학습이 되면 Variable Impedance (K를 RL 출력으로) 추가
4. 장애물 추가 및 장애물 회피 학습
