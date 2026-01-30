# Constrained Motion Planning 및 RL 학습 전략 요약

이 문서는 Dual-arm Closed-chain 로봇 시스템을 위한 제약 기반 모션 플래닝, 영공간 투영(Null-space Projection), 그리고 강화학습(RL) 시 유효한 목표 생성 전략에 대한 기술적 논의를 정리한 것입니다.

---

## 1. 영공간 투영 vs 프로젝션 방식

두 방식은 구속 조건(Constraint)을 다루는 목적과 메커니즘에서 차이가 있습니다.

### 1.1 영공간 투영 (Null Space Projection)
* **목적:** 주 작업(Primary Task)을 방해하지 않으면서 남는 자유도(Redundancy)를 활용해 보조 작업(Secondary Task)을 수행.
* **수식:**
    $$\dot{q} = J^{\dagger} \dot{x}_{des} + (I - J^{\dagger} J) \dot{q}_{sub}$$
* **특징:** 계층적 제어(Hierarchical Control). 보조 작업 $\dot{q}_{sub}$ 은 주 작업에 수학적으로 영향을 주지 않음.

### 1.2 프로젝션 방식 (Projection Method / Manifold Projection)
* **목적:** 로봇 상태가 구속 조건 다양체(Manifold)를 벗어났을 때, 다시 유효한 영역($C(q)=0$)으로 강제 복귀(Error Correction).
* **수식 (Newton-Raphson):**
    $$q_{new} = q_{curr} - J_C^{\dagger} \cdot \lambda \cdot C(q_{curr})$$
* **특징:** 물리적 구속 조건(Closed-chain) 유지에 필수적. 반복 연산(Iterative)을 통해 수렴시킴.

---

## 2. CLIK와 오차 보정 (Drift Correction)

Closed-chain 시스템(예: 두 팔로 물체 파지)에서 **투영(Projection)** 과 **보정(Correction)** 은 함께 사용되어야 합니다.

### 2.1 알고리즘 구조
$$\dot{q}_{safe} = \underbrace{(I - J_c^\dagger J_c) \dot{q}_{actor}}_{\text{1. 예방 (Projection)}} + \underbrace{J_c^\dagger ( -K_p \cdot e )}_{\text{2. 치료 (Correction)}}$$

1.  **예방 (Projection):** $I - J_c^\dagger J_c$ 행렬을 통해 Actor의 행동 중 제약을 위반하는 성분(법선 방향 성분)을 제거합니다.
2.  **치료 (Correction):** 이미 발생한 오차($e$)를 줄이기 위해 제약 위반의 반대 방향($-e$)으로 피드백 속도를 더해줍니다.

### 2.2 제약 자코비안 ($J_c$)의 정의
두 매니퓰레이터를 하나의 시스템으로 보았을 때의 자코비안입니다.
$$J_c = \begin{bmatrix} J_{left} & -J_{right} \end{bmatrix}$$
* 의미: 왼팔과 오른팔의 상대 속도가 0이 되도록 하는 관계식.

---

## 3. 매니폴드의 불연속성 (The "Island" Problem)

### 3.1 문제 정의
구속 조건($C(q)=0$)을 만족하는 형상 공간(C-Space)상의 매니폴드는 하나로 연결되어 있지 않고, 여러 개의 **'섬(Disjoint Components)'** 으로 분리될 수 있습니다.
* **원인:** 역기구학 해의 다중성(Elbow-up vs Elbow-down), 관절 제한(Joint Limits), 장애물 등.

### 3.2 경로 존재 여부
* **결론:** 시작점($q_{start}$)과 목표점($q_{goal}$)이 서로 다른 섬에 존재한다면, **구속 조건을 위반하지 않고 이동할 수 있는 연속적인 경로는 존재하지 않습니다.**
* **해결책:** 만약 이동해야 한다면 물체를 놓았다가 다시 잡는(Regrasping) 과정이 필요합니다.

---

## 4. 강화학습을 위한 시작/목표 생성 전략

RL 에이전트가 "갈 수 없는 곳"을 목표로 삼는 문제를 방지하기 위해, 시작점과 목표점은 반드시 **같은 섬(Connected Component)** 내에 있어야 합니다.

### 4.1 잘못된 방법 (Avoid)
* 단순히 작업 공간(Task Space)에서 랜덤한 6D Pose를 목표로 설정하는 것.
* 이 경우 IK 해가 다른 모드(다른 섬)로 잡힐 확률이 높아 학습 실패의 원인이 됨.

### 4.2 추천 전략: Object Pose Sampling + Seeded IK
**[참고 문헌: Jang et al., 2022]**
1.  **물체 포즈 샘플링:** 작업 공간 내에서 목표 물체 포즈($x_{goal}$)를 생성.
2.  **Seeded IK:** $x_{goal}$ 에 대한 IK를 풀 때, **현재 시작 자세($q_{start}$)를 초기값(Seed)으로 설정** 하여 수치적 IK를 수행.
3.  **효과:** 수치적 IK는 초기값 근처의 해로 수렴하므로, $q_{start}$ 와 위상적으로 연결된(같은 섬에 있는) $q_{goal}$ 을 찾을 확률이 비약적으로 상승함.

### 4.3 대안 전략: Valid Set Pre-computation
**[참고 문헌: Park et al., 2024]**
1.  사전에 유효한(On-manifold) 관절 자세 데이터를 대량(예: 10,000개)으로 생성 및 저장.
2.  학습 시 이 데이터셋 내에서 서로 가까운(또는 연결성이 확인된) 두 점을 샘플링하여 사용.

---

## 5. 플래너/에이전트 입출력 (I/O) 표준

대부분의 관련 연구 및 논문은 플래너의 입출력을 **Joint Position ($q$)** 으로 정의합니다.

### 5.1 이유
1.  **유일성 (Uniqueness):** EE Pose는 다수의 IK 해를 가지지만, $q$ 는 로봇의 상태를 유일하게 정의함.
2.  **연속성 (Continuity):** 작업 공간에서의 직선 경로는 관절 공간에서 불연속일 수 있음. $q$ 공간에서의 경로는 물리적 실행 가능성을 담보함.

### 5.2 권장 파이프라인
1.  **목표 생성 (외부):** Task Goal(6D Pose) $\rightarrow$ **Seeded IK** $\rightarrow$ Joint Goal ($q_{goal}$)
2.  **플래너 입력:** Current Joint ($q_{curr}$), Goal Joint ($q_{goal}$)
3.  **플래너 출력:** Joint Velocity ($\dot{q}$) 또는 Torque ($\tau$)
