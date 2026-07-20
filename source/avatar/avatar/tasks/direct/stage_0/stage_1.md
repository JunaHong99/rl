# 연구 계획 (전체)
# GNN Primitive 분해 기반 다양한 물체의 듀얼암 협조 운반 + Per-arm Adaptive Stiffness

> 2026-06-11 기준. Stage 0 ~ Stage 3 전체 로드맵.

---

## 1. 연구 개요

### 한 줄 요약

두 로봇 팔이 다양한 형태의 물체를 협력하여 운반하는 태스크에서, (1) 물체를 primitive로 분해하고 GNN으로 표현하여 형태 일반화를 달성하고, (2) RL이 팔별 강성(K)을 독립적으로 조절하여 비대칭 물체에서 내력을 관리하는 프레임워크.

### 핵심 아이디어

RoboBallet(Science Robotics, 2025)이 복잡한 장애물을 cuboid primitive로 분해하고 GNN으로 일반화한 것처럼, 우리는 **운반 물체를 primitive로 분해**하고 GNN으로 다양한 형태의 물체에 zero-shot 일반화한다.

```
RoboBallet: 다양한 장애물 → primitive 분해 → GNN → 회피 일반화
우리:       다양한 물체   → primitive 분해 → GNN → 운반 일반화
```

### DA-VIL, RoboBallet 대비 포지셔닝

|                     | DA-VIL           | RoboBallet      | 우리                    |
|---------------------|------------------|-----------------|------------------------|
| 태스크              | 협조 운반         | 독립 reaching    | 협조 운반               |
| 물체 표현           | 물체별 고정       | primitive 분해   | primitive 분해          |
| K 구조              | 양팔 동일 (shared)| 없음             | 팔별 독립 (per-arm)     |
| 궤적                | 사전 생성         | RL 생성          | RL 생성 (reference-free)|
| 정책 구조           | MLP              | GNN             | GNN                    |
| Cross-embodiment    | ❌               | 언급만           | Future work            |
| Closed chain        | ✅               | ❌              | ✅                     |
| 내력 분석           | ❌               | 해당 없음        | ✅                     |
| Zero-shot 물체 일반화| ❌              | 장애물만          | ✅ 운반 물체            |

---

## 2. 연구 단계 전체 로드맵

```
[Stage 0] Per-arm Impedance + MLP로 기본 운반 성공           ← 완료 (98%)
     ↓
[Stage 1] Per-arm K Modulation (RL이 K 출력)                ← 다음
     ↓
[Stage 2] GNN 전환 + Primitive 표현                         ← GNN 도입
     ↓
[Stage 3] 물체 다양화 + Zero-shot 일반화                    ← 핵심 contribution
     ↓
[논문 작성]
```

---

## 3. Stage 0: Per-arm Impedance + MLP 기본 운반 [완료]

### 목표
두 Franka Panda가 fixed joint로 잡은 rod를 목표 pose로 운반. MLP + SAC + HER로 학습.

### 달성 결과
- Success rate: **98%** (pos < 20mm AND rot < 10°)
- 알고리즘: SAC + HER (sparse reward + virtual goal)
- 정책: MLP (385K params)
- Controller: Per-arm task-space impedance (velocity-based)
  - K_pos=200, D_pos=60 고정
  - 각 arm이 자기 EE target을 독립 추종
- Action: 6D (rod target pose delta)
- 환경: Isaac Lab (PhysX), 1024 envs, gravity OFF

### 해결한 핵심 문제들

| 문제 | 해결 |
|---|---|
| Torque control 회전 발산 (70° peak) | Velocity control로 전환 |
| Reset transient (11% unstable) | Post-reset settle 30 step |
| Sampler edge case (joint limit 충돌) | Joint margin filter (≥0.2 rad) |
| Alpha collapse (SAC) | Fixed alpha = 0.1 |
| Delta 누적 local optimum | Non-accumulating action |
| 60Hz credit assignment 어려움 | 10Hz 결정 주기 (decimation=24) |

### 현재 기술 스택

| 컴포넌트 | 선택 | 비고 |
|---|---|---|
| Simulator | Isaac Lab (PhysX) | GPU 병렬 1024 envs |
| RL Algorithm | SAC + HER | sparse reward + virtual goal |
| Policy | MLP (385K params) | 입력: rod state + goal + joint state (~67D) |
| Controller | Per-arm velocity impedance | K_pos=200, D_pos=60 고정 |
| Action space | 6D (rod target pose delta) | pos delta + axis-angle |
| Decision freq | 10Hz (decimation=24) | 에피소드당 125 결정 |
| Pose sampler | 100k cached samples | IK + joint margin filter |

---

## 4. Stage 1: Per-arm K Modulation [다음 착수]

### 목표
RL이 rod target delta(6D)에 더해 **K_arm1, K_arm2를 동시에 출력**. 고정 K 대비 내력 감소, structural stability 향상을 검증.

### 구현

#### Action 확장
```python
# Stage 0: 6D
action[:6]  → rod target pose delta

# Stage 1: 8D
action[:6]  → rod target pose delta (기존 동일)
action[6]   → K_arm1_scale (스칼라)
action[7]   → K_arm2_scale (스칼라)

# Controller에서 K 적용
K_arm1 = K_base * softplus(action[6]) * K_scale
K_arm2 = K_base * softplus(action[7]) * K_scale

# Velocity 계산에 K 반영
ee1_vel = (K_arm1 / D_nominal) * (ee1_target - ee1_current)
ee2_vel = (K_arm2 / D_nominal) * (ee2_target - ee2_current)
```

#### 관측 확장
```python
# 기존 observation에 추가
obs += [
    manipulability_arm1 / manip_max,     # (1,) 팔1 조작성
    manipulability_arm2 / manip_max,     # (1,) 팔2 조작성
    joint_margin_min_arm1,               # (1,) 팔1 관절 여유
    joint_margin_min_arm2,               # (1,) 팔2 관절 여유
    mass / mass_max,                     # (1,) 물체 질량 (Stage 1에서 randomization 시작)
    length / length_max,                 # (1,) 물체 길이
]
```

#### 보상 추가
```python
# 기존 reward에 추가
r_internal = -w_internal * max(0, ||f_internal|| - f_safe)  # 내력 페널티
r_K_smooth = -w_smooth * (||K_t - K_{t-1}||)                # K 급변 방지
```

### 실험 설계

#### Baselines
| # | 방법 | K 구조 | Action |
|---|---|---|---|
| B1 | Fixed-K (Stage 0 모델) | K=200 고정 | 6D |
| B2 | Shared RL-K | K_arm1 = K_arm2 = RL 출력 | 7D |
| **Ours** | **Per-arm RL-K** | **K_arm1 ≠ K_arm2** | **8D** |

#### 평가 시나리오
| 시나리오 | 내용 | 핵심 비교 |
|---|---|---|
| 일반 운반 | 다양한 시작/목표 pose, 1854 에피소드 | 기본 성능 |
| Stress test | 한쪽 팔 특이점/관절한계 근처 | Per-arm K의 이점 |
| 질량 변화 | mass 0.1~2.0kg randomization | K 적응성 |
| 길이 변화 | length 0.5~1.0m randomization | K 적응성 |

#### 측정 지표
- Success rate
- Internal force (mean/max): ContactSensor로 측정
- Torque peak: 최대 관절 토크
- Structural unstable rate: 폭주 에피소드 비율
- K 패턴: K_arm1, K_arm2 시계열 (manipulability와 상관관계 분석)

### 이론적 근거

#### 내력과 K의 관계
```
f_internal ∝ K_arm1 * e1 - K_arm2 * e2

Shared K:   f_internal ∝ K * (e1 - e2)    → e1 ≠ e2이면 내력 발생, 조절 불가
Per-arm K:  K1/K2 = e2/e1로 조절 가능      → f_internal ≈ 0 달성 가능
```

#### Leader-follower 창발
```
K_arm1 > K_arm2 → Arm1이 leader (경로 주도), Arm2가 follower (순응)
상황에 따라 역할 전환 → 비대칭 협력의 창발
```

### 기대 결과
```
| 방법 | 일반 | Stress | 내력 mean | 내력 max |
|---|---|---|---|---|
| Fixed-K | 98% | 80% | 15N | 40N |
| Shared RL-K | 97% | 85% | 10N | 30N |
| Per-arm RL-K | 97% | 92% | 5N | 15N |
```

### 기간: 2주

---

## 5. Stage 2: GNN 전환 + Primitive 표현

### 목표
MLP를 GNN으로 교체하고, 물체를 primitive 노드로 표현. Panda + Rod에서 MLP와 동등한 성능을 확인한 후, Box 물체를 추가하여 혼합 학습.

### GNN 그래프 설계

#### Rod (primitive 1개)
```
[Arm1] ──grasp── [Rod_prim] ──grasp── [Arm2]
                      │
                   [Goal]
노드 4개, 엣지 3개
```

#### Box (primitive 1개)
```
[Arm1] ──grasp── [Box_prim] ──grasp── [Arm2]
                      │
                   [Goal]
노드 4개, 엣지 3개
```

#### L자 (primitive 2개, Stage 3에서 zero-shot 테스트용)
```
[Arm1] ──grasp── [L_prim1] ──rigid── [L_prim2] ──grasp── [Arm2]
                      │                    │
                   [Goal]               [Goal]
노드 5개, 엣지 5개
```

### 노드 Feature

#### Arm 노드 (×2)
```python
arm_features = {
    'ee_pos_local': ee_pos / max_reach,              # (3,)
    'ee_quat': ee_quat,                               # (4,)
    'ee_lin_vel': ee_lin_vel / max_vel,               # (3,)
    'ee_ang_vel': ee_ang_vel / max_ang_vel,           # (3,)
    'manipulability': manipulability / manip_max,      # (1,)
    'joint_margin_min': min_margin / margin_range,     # (1,)
    'current_K': K / K_max,                            # (1,)
}  # ~16D
```

#### Object Primitive 노드 (×N, 가변)
```python
prim_features = {
    'pos_local': prim_pos / max_reach,                # (3,)
    'quat': prim_quat,                                 # (4,)
    'lin_vel': prim_lin_vel / max_vel,                 # (3,)
    'ang_vel': prim_ang_vel / max_ang_vel,             # (3,)
    'bbox_dims': [w, h, d] / max_dim,                  # (3,)
    'mass_fraction': prim_mass / total_mass,           # (1,)
    'is_grasped': 0 or 1,                              # (1,)
    'com_offset': offset_from_obj_com / max_dim,       # (3,)
}  # ~21D
```

#### Goal 노드 (×1)
```python
goal_features = {
    'goal_pos_local': goal_pos / max_reach,            # (3,)
    'goal_quat': goal_quat,                             # (4,)
    'pos_error': pos_error / max_reach,                # (3,)
    'rot_error': axis_angle_error,                      # (3,)
    'distance': distance / max_reach,                   # (1,)
}  # ~14D
```

### 엣지 Feature

#### Grasp 엣지 (Arm ↔ Primitive)
```python
grasp_edge = {
    'edge_type': one_hot('grasp', 4),                  # (4,)
    'grasp_offset': offset_in_prim_frame / prim_size,  # (3,)
    'grasp_normal': contact_normal,                     # (3,)
}  # ~10D
```

#### Rigid 엣지 (Primitive ↔ Primitive, 물체 형태 인코딩의 핵심)
```python
rigid_edge = {
    'edge_type': one_hot('rigid', 4),                  # (4,)
    'relative_pos': relative_position / max_dim,       # (3,)
    'relative_rot': relative_rotation_6d,               # (6,)
}  # ~13D
```

#### Goal 엣지 (Primitive ↔ Goal)
```python
goal_edge = {
    'edge_type': one_hot('goal', 4),                   # (4,)
    'target_pos_error': pos_error / max_reach,         # (3,)
    'target_rot_error': rot_error,                      # (3,)
}  # ~10D
```

#### Cooperative 엣지 (Arm ↔ Arm)
```python
coop_edge = {
    'edge_type': one_hot('cooperative', 4),            # (4,)
    'relative_ee_pos': rel_pos / max_reach,            # (3,)
    'relative_ee_vel': rel_vel / max_vel,              # (3,)
}  # ~10D
```

### GNN Architecture (RoboBallet 스타일)

```python
class CooperativeTransportGNN(nn.Module):
    def __init__(self, hidden_dim=256):
        # Embedding
        self.node_embed = MLP([max_node_dim, hidden_dim])
        self.edge_embed = MLP([max_edge_dim, hidden_dim])
        self.global_embed = MLP([global_dim, hidden_dim])

        # Core GNN (2 rounds message passing)
        self.edge_update = MLP([hidden_dim*4, hidden_dim])
        self.node_update = MLP([hidden_dim*3, hidden_dim])
        self.global_update = MLP([hidden_dim*3, hidden_dim])

        # Readout heads
        self.target_head = MLP([hidden_dim, 64, 6])     # global → target delta
        self.K_head = MLP([hidden_dim, 32, 1])           # arm node → K (weight sharing!)

    def forward(self, graph):
        # Embed → 2 rounds message passing → readout
        target_delta = self.target_head(global_emb)      # (6,)
        K_arm1 = softplus(self.K_head(arm1_emb))          # (1,) weight shared
        K_arm2 = softplus(self.K_head(arm2_emb))          # (1,) weight shared
        return target_delta, K_arm1, K_arm2
```

K_head가 양팔에서 **weight sharing**: "manipulability가 낮으면 K를 낮춰라"를 한 번 학습, 양팔에 적용.

### GNN 일반화가 작동하는 이유

```
Grasp(Arm1→Rod)  ≈ Grasp(Arm2→Rod)        ← 양팔에서 동일한 관계
Grasp(Arm1→Rod)  ≈ Grasp(Arm1→Box)        ← 물체가 달라도 동일한 관계
Rigid(L_p1→L_p2) ≈ Rigid(T_p1→T_p2)       ← 물체 내 결합이 동일한 관계
```

### 기간: 2~3주

---

## 6. Stage 3: 물체 다양화 + Zero-shot 일반화

### 목표
학습 시 본 적 없는 형태의 물체를 zero-shot으로 운반. GNN의 primitive 분해 일반화를 검증.

### 물체 목록

#### 학습용 (2~3종)
| 물체 | Primitive 수 | Grasp | 대칭성 |
|---|---|---|---|
| Rod | 1 | 양 끝, 대칭 | 대칭 |
| Box | 1 | 양 옆면, 대칭 | 대칭 |
| L자 (선택) | 2 | 비대칭 | 비대칭 |

#### 테스트용 — Zero-shot (학습 시 안 본 것)
| 물체 | Primitive 수 | Grasp | 핵심 도전 |
|---|---|---|---|
| L자 | 2 | 비대칭 | 비대칭 CoM, 비대칭 레버 팔 |
| T자 | 3 | 양 끝 | Primitive 수 증가, 분기 구조 |
| Cylinder | 1 | 양 끝 | 새로운 bbox 비율 |

### Domain Randomization (학습 시)
```python
# 매 에피소드 물체를 랜덤 선택 + 파라미터 랜덤화
object_type = random.choice(['rod', 'box'])  # 또는 ['rod', 'box', 'l_shape']
mass = uniform(0.1, 1.5)
# Rod: length = uniform(0.5, 1.0)
# Box: width = uniform(0.3, 0.6), height = uniform(0.1, 0.3)
# L자: long_length = uniform(0.3, 0.5), short_length = uniform(0.15, 0.3)
```

### 실험 설계

#### Baselines (전체)
| # | 정책 | K 구조 | 물체 대응 |
|---|---|---|---|
| B1 | MLP + Fixed-K | K 고정 | 학습 물체만 |
| B2 | MLP + Per-arm RL-K | 팔별 독립 K | 학습 물체만 (OOD는 zero-padding) |
| B3 | GNN + Fixed-K | K 고정 | 가변 primitive |
| **Ours** | **GNN + Per-arm RL-K** | **팔별 독립 K** | **가변 primitive** |

#### 평가 시나리오 (전체)
| 시나리오 | 내용 | 핵심 비교 |
|---|---|---|
| S1: In-distribution 기본 | Rod, Box (학습 시 본 것) | 기본 성능 |
| S2: 질량/크기 OOD | Rod 2.0kg, 1.2m (학습 범위 밖) | 물리 파라미터 일반화 |
| S3: 형태 OOD (zero-shot) | L자, T자, Cylinder (학습 시 안 본 것) | GNN primitive 일반화 |
| S4: 비대칭 Stress test | L자 + 한쪽 팔 특이점 근처 | Per-arm K의 극한 이점 |

#### 핵심 비교 테이블 (논문 Table 1)
```
| 시나리오 | B1 MLP Fixed | B2 MLP RL-K | B3 GNN Fixed | Ours GNN RL-K |
|---|---|---|---|---|
| S1 Rod (ID) | 98% | 97% | 96% | 97% |
| S1 Box (ID) | 95% | 95% | 94% | 95% |
| S2 Heavy (OOD) | 70% | 85% | 72% | 87% |
| S3 L자 (ZS) | 20%* | 25%* | 65% | 75% |
| S3 T자 (ZS) | 0%* | 0%* | 50% | 60% |
| S4 Stress | 75% | 88% | 78% | 92% |

*MLP는 zero-padding으로 primitive 수 변화에 대응, 성능 저하 큼
```

### 기간: 2~3주

---

## 7. Contributions (최종)

### C1: GNN Primitive 분해 기반 다양한 물체의 Zero-shot 운반

RoboBallet의 obstacle primitive 분해 원리를 cooperative manipulation의 운반 물체에 확장. 물체를 cuboid primitive로 분해하고 GNN 노드로 표현하여, 학습 시 본 적 없는 형태(L자, T자)의 물체도 zero-shot으로 운반.

핵심 주장: "Arm ↔ Primitive (grasp)" 관계와 "Primitive ↔ Primitive (rigid)" 관계를 한 번 학습하면, primitive 수와 배치가 달라져도 같은 모델이 작동한다.

### C2: Reference-free Per-arm Adaptive Stiffness

사전 궤적 없이 정책이 경로와 팔별 K를 동시에 출력. 비대칭 물체에서 per-arm K가 shared K 대비 내력 감소와 structural stability 향상에 기여함을 이론 + 실험으로 보임.

이론: f_internal ∝ K1*e1 - K2*e2. Per-arm K로 K1/K2 = e2/e1 조절하면 내력 상쇄 가능.

DA-VIL 대비: reference-free(사전 궤적 불필요) + per-arm K(팔별 독립) + 내력 분석(DA-VIL에 없음).

### C3: Multi-axis Systematic Ablation

정책 구조(MLP vs GNN) × K 전략(Fixed vs Shared vs Per-arm) × 물체 복잡도(대칭 vs 비대칭 vs OOD)의 교차 비교. 각 설계 선택의 독립적 기여를 분리 검증.

---

## 8. 논문 구조

### Title (후보)
"Graph-based Primitive Decomposition for Generalizable Dual-arm Cooperative Transport with Adaptive Per-arm Stiffness"

### Abstract
듀얼암 협조 운반에서 기존 방법은 특정 물체에 맞춰 학습되어 형태가 바뀌면 재학습이 필요하고(DA-VIL), 양팔에 동일한 강성을 적용하여 비대칭 물체에서 내력 제어가 어렵다. 본 연구는 운반 물체를 cuboid primitive로 분해하고 GNN으로 표현하여 다양한 형태에 zero-shot 일반화하는 프레임워크를 제안한다. 또한 RL이 팔별 강성을 독립적으로 출력하여 비대칭 물체에서 내력을 적응적으로 관리한다. 실험에서 학습 시 본 적 없는 L자, T자 물체에 대한 zero-shot 운반과, per-arm K의 내력 감소 효과를 보인다.

### Sections

- **I. Introduction**: 듀얼암 협조 운반의 중요성 → 기존 한계 (DA-VIL: shared K + 사전 궤적 + 물체 고정) → RoboBallet의 primitive 분해 아이디어 → 우리의 확장 → contributions
- **II. Related Work**: Cooperative manipulation (Caccavale, Khatib) / RL + variable impedance (VICES, DA-VIL) / GNN for robotics (RoboBallet, NerveNet) / Object generalization
- **III. Problem Formulation**: 환경 정의, closed chain, 다양한 물체, 목표 정의
- **IV. Method**:
  - A. Primitive Decomposition of Objects (물체를 cuboid로 분해하는 방법)
  - B. Graph Representation (노드/엣지 설계, 왜 이 구조인가)
  - C. GNN Architecture (RoboBallet 스타일 적용, K head weight sharing)
  - D. Per-arm Adaptive Stiffness (내력 이론, K 비율과 내력의 관계)
  - E. Reference-free Policy (action space, reward, HER, 10Hz)
  - F. Training Pipeline (SAC + HER, domain randomization, settle masking)
- **V. Experiments**:
  - Setup (물체, baselines, 시나리오, 지표)
  - Results (4 시나리오 × 4 방법 비교 테이블)
  - Zero-shot 일반화 분석
  - Per-arm K 효과 분석 (내력, 토크, unstable rate)
- **VI. Analysis**:
  - K 패턴 시각화 (leader-follower 전환)
  - 경로 분석 (reference-free가 만드는 경로 특성)
  - 실패 사례 분석
- **VII. Discussion & Limitations**
- **VIII. Conclusion**

---

## 9. 실행 타임라인

| Phase | 기간 | 핵심 작업 | 산출물 | 논문 가능 여부 |
|---|---|---|---|---|
| **Stage 0** | 완료 | Per-arm impedance + MLP, 98% | 기반 모델 | ❌ (기반만) |
| **Stage 1** | 2주 | Per-arm K modulation + 질량/길이 random | Fixed-K vs Per-arm-K 비교 | ✅ 최소 논문 |
| **Stage 2** | 2~3주 | GNN 전환 + Box 추가 + 혼합 학습 | GNN ≥ MLP 확인 | ✅ 중간 논문 |
| **Stage 3** | 2~3주 | L자/T자/Cylinder zero-shot 테스트 | 전체 비교 테이블 | ✅ 강한 논문 |
| **논문 작성** | 2~3주 | 분석 + 시각화 + 작성 | 제출 원고 | - |
| **총** | **~11주** | | | |

### Fallback 전략
```
최강 논문: Stage 0~3 전부 + 상세 분석                (11주)
중간 논문: Stage 0~2 + in-distribution GNN 비교       (7주)
최소 논문: Stage 0~1 + MLP per-arm K + 내력 분석      (4주)
```

어느 시점에서 멈춰도 논문이 되도록 설계.

---

## 10. 리스크 대응

| 리스크 | 발생 시점 | 대응 |
|---|---|---|
| Per-arm K가 Fixed-K 대비 개선 없음 | Stage 1 | 비대칭 물체(L자)에서 집중 비교. 보상에 내력 페널티 추가 |
| GNN이 MLP보다 나쁨 | Stage 2 | Architecture 튜닝 (rounds, hidden size). 안 되면 MLP로 논문 진행 |
| Zero-shot 일반화 실패 | Stage 3 | L자를 학습에 추가, T자만 OOD. 또는 primitive feature 보강 |
| 물체 USD 준비 어려움 | Stage 3 | Isaac Lab 기본 primitive 조합으로 composite body 생성 |
| 시간 부족 | 아무 때나 | Fallback 전략에 따라 현재 시점의 최선 논문 작성 |

---

## 11. 코드 구조 (예상)

```
project/
├── envs/
│   ├── dual_arm_transport_env3.py       # 기본 환경 (Stage 0~1)
│   ├── dual_arm_transport_cfg.py        # 환경 설정
│   └── multi_object_env.py              # 다양한 물체 환경 (Stage 2~3)
├── controllers/
│   └── per_arm_impedance_controller.py  # Per-arm velocity impedance
├── policies/
│   ├── mlp_policy.py                    # MLP (Stage 0~1)
│   └── gnn_policy.py                    # GNN (Stage 2~3)
├── graphs/
│   ├── graph_builder.py                 # 물체 → 그래프 변환
│   └── primitive_decomposer.py          # 물체 → cuboid primitives
├── training/
│   ├── sac_trainer.py                   # SAC + HER
│   └── her_buffer.py                    # HER with valid_mask
├── evaluation/
│   ├── eval_mlp.py                      # MLP 평가
│   ├── eval_gnn.py                      # GNN 평가
│   └── analyze_K_patterns.py            # K 시계열 분석
├── assets/
│   ├── rod.usd
│   ├── box.usd
│   ├── l_shape.usd                      # Stage 3
│   ├── t_shape.usd                      # Stage 3
│   └── cylinder.usd                     # Stage 3
└── configs/
    ├── train_stage1.yaml                # Stage 1 설정
    ├── train_stage2.yaml                # Stage 2 설정
    └── train_stage3.yaml                # Stage 3 설정
```

---

## 12. 향후 연구 (논문 Future Work)

- **Cross-embodiment**: Arm 노드 feature에 로봇 특성을 인코딩하면 다른 로봇 형태에도 transfer 가능 (이론적 근거: GNN weight sharing)
- **동적 장애물 회피**: 장애물 노드를 그래프에 추가하면 자연스러운 확장 (RoboBallet과 동일 원리)
- **3팔 이상 확장**: Arm 노드를 추가하면 GNN이 자연스럽게 확장
- **Sim-to-real**: Domain randomization (질량, 마찰, 센서 노이즈) 기반 transfer
- **Deformable object**: Primitive 분해의 한계, 별도 연구 필요
- **안전 상호작용**: K modulation을 사람 근처에서의 충격 완화에 활용
