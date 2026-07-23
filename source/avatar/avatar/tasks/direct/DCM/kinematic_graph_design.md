# Kinematic Graph 설계 (2026-07-23 확정)

리더-팔로워 관절 액션 하에서 **파지(폭·각도) + base 간격 변이에 대응**하는 GNN 관측.
lean 3-노드(EE만)로는 관절 상태를 표현 못 함(정책이 리더 관절을 직접 소유). → 관절을 노드로.

관련 파일: `graph_converter_kin.py`(빌더), `gnn_policy_kin.py`(정책), `dual_arm_transport_env3.py:_build_kin_raw`(env), `her_buffer.py`(kin relabel), `sac_trainer.py`(kin 버퍼).

---

## 0. 배경 / 동기

- object-centric은 파지 일반화 공짜(94% zero-shot)였으나 **네트워크가 기구학을 안 배움** → morphology 일반화 토대 없음.
- → 리더-팔로워(리더 관절 Δq 학습 + 팔로워 IK) 전환, 고정 태스크 **운반 85% 성공**(MLP baseline) 확인.
- **협응은 학습 대상 아님**(팔로워 IK가 구조적 보장). 학습 = 리더 관절로 rod를 목표 pose에 운반.
- kinematic graph의 값어치 = 파지/base가 바뀌면 **관절 상태·feasibility가 달라지는 것**을 정책이 봐야 함. 이건 관절 노드가 있어야 표현됨(lean EE-only 불가).

---

## 1. 그래프 구조 (17노드)

**로봇 = 같은 종류(Franka×2)** → link 노드 불필요(링크길이 상수). base+joint+object만.

### 노드 순서 (고정 인덱스)
```
[0]      base_L        type base     리더 base
[1..7]   joint_L 1~7   type joint    리더 관절   ← ★ 출력 head (리더 Δq 7)
[8]      rod           type object   운반 대상
[9]      base_F        type base     팔로워 base
[10..16] joint_F 1~7   type joint    팔로워 관절  ← backbone 참여, 출력 X(IK 결정)
= 17 nodes/env
```
리더 joint = 고정 인덱스 1~7 → 출력 head가 이 7개만 gather (단순·빠름). DoF 변이는 추후 role 마스크로.

### 노드 feature (타입별 raw + type one-hot 3, 패딩 max=9 → NODE_FEATURE_DIM=12)
| 타입 | raw feature | dim | 비고 |
|---|---|---|---|
| **base** | identity (없음) | 0 | 위치는 엣지로만 |
| **joint** | 축(3) + sin/cos(q)(2) + margin(min·max)(2) | 7 | 아래 §3 |
| **object(rod)** | goal오차 pos(3, 리더 base frame) + rot 6D(6) | 9 | 태스크 신호 |

### 엣지 (4 타입, 전부 양방향, receiver-frame 상대 pose = trans3+rot6d=9; EDGE_FEATURE_DIM = 4+9=13)
| 타입 | 연결 | 담는 변이 |
|---|---|---|
| **base→first** | base ↔ joint1 (양팔) | mount |
| **joint-chain** | joint_i ↔ joint_{i+1} (양팔) | 로봇 구조(DH 변환) |
| **grasp** | joint7(EE) ↔ rod (양팔) | **★ 파지 폭 d + 각도 θ** |
| **base-rel** | base_L ↔ base_F | **★ base 간격** |
= 34 edges/env

### Global
- normalized_time(1) + f_int(1) = **GLOBAL_FEATURE_DIM = 2**
- (관절각은 이제 joint 노드에 있으니 global서 제거)

---

## 2. 좌표계 (핵심 — 요소별 다른 frame, 각자 역할에 맞게)

| 요소 | 좌표계 | 이유 |
|---|---|---|
| **관절축** | **부모링크 국소(상수)** | NerveNet식. 노드=국소 구조, 배치 불변 → 전이 최강 (§3) |
| **joint 값·margin** | 관절공간(각도) | frame 무관 |
| **object goal오차** | **리더 base frame** | 액션(리더 관절)과 정렬 → 사상 학습 쉬움 |
| **엣지 상대 pose** | receiver-frame 상대 | 프레임 무관량 (차·상대회전) |
| **액션 (리더 Δq)** | 리더 base 관절공간 | — |

→ **task 신호(obj_goal)와 액션은 리더 base frame으로 통일**, 노드 축은 국소, 엣지는 상대 → 각자 역할.
**검증**: env 관측 obj_goal == HER relabel obj_goal, 회전 base에서도 diff 0.0 (일관성 확인).

---

## 3. Joint 노드 feature 상세 (설계 논쟁 반영)

### 축 = 부모링크 frame 상수 (방식 B, 확정)
- Modified DH 도출: 부모frame 기준 joint_i 축 = Rx(α_i)·[0,0,1].
  - j1=[0,0,1], j2=[0,1,0], j3=[0,-1,0], j4=[0,-1,0], j5=[0,1,0], j6=[0,-1,0], j7=[0,-1,0].
- **왜 world/base frame(방식 A) 아닌가**:
  - A(공통 base frame): 노드가 공간 정보까지 이중으로 담아 엣지와 중복 → GNN이 여러 홉 재구성 비효율. base 배치 의존.
  - **B(부모링크 국소)**: 노드=국소 구조만, 공간관계는 엣지 → **message passing 친화적**(국소 전파), base/자세/배치 **완전 불변** → morphology 전이. NerveNet 정석.
- 축이 상수라 정보량 적어 보이나, **관절 순서/구조는 엣지가 담음** + 관절마다 축 다름(구조 정보 유지).

### sin/cos(q) = 현재 관절 상태
- 관절 명령 = 현재각 기준 증분(Δq) → 현재 위치 필요. sin/cos = 각도 연속성(±π 튐 방지).

### margin = feasibility (연구 동기 직결)
- `(q−q_min)/range`, `(q_max−q)/range` ∈[0,1]. 0 근접 = 관절한계/특이점 근접.
- 미해결 6~7%(경로 중 특이점/도달불가) → object-centric엔 관절정보 0이라 못 봄. **margin이 정책에 feasibility 노출** = kinematic graph 값어치.

### 역할 분리 (깔끔)
- 축 = 구조 상수 / sin·cos = 상태 / margin = 여유. 서로 안 섞임.
- 의도적 제외: 관절속도(lean 교훈), 링크길이(같은 로봇→link 노드 자체 제외; 크기 변이 때 추가).

---

## 4. 정책 (KinGNN, NerveNet식 타입공유)

- **Backbone**: NodeType/Edge/Global embedder(kin 차원) + RoboBallet GN block × T. **T=8**(사슬 깊이 base→EE + rod 건너 반대팔).
- **Actor 출력**: 리더 joint 노드 7개 gather → **공유 per-joint head**(raw skip + MP embedding → 스칼라 Δq) → (B,7). tanh·dq_scale.
  - 공유 head = 관절 수 무관 = **DoF 일반화의 원천**.
  - 팔로워 joint 노드: backbone 정보전파엔 참여, 출력 head엔 미포함(IK 결정).
- **Critic/Q**: global mean-pool(노드 수 무관) + action → scalar.
- 6.9M params (T=8). action_dim=7, dq_scale=0.30.

---

## 5. 리워드 / 학습 (리더-팔로워와 동일)

- **task**: 도달 +100(one-shot) + 시간페널티 −0.2.
- **dense progress**(`lf_dense_progress`, w=10): Cartesian rod 접근량 `w·(prev−curr)dist`. sparse+HER는 리더관절 액션엔 약함(relabel trivial) → dense 필수(85% 성공의 주역). potential-based+도달종료라 제자리 꼼수 없음.
- **내력 페널티 없음**(리더-팔로워는 협응 구조보장 → freeze 위험만). 내력 제어는 추후(운반 성공 후).
- **HER**(kin 대응, 방식 B): object 노드의 base-frame goal오차를 가상 goal로 relabel. 그 시점 리더 base quat 저장(`ep_base_quat`)해 재변환. env 계산과 수식 동일(검증됨).
- 레시피: 8192env / updates24 / batch2048 / warmup700k / buffer3M / fixed_alpha0.1.

---

## 6. 단계 (게이트)

1. **고정 태스크 재현** (현재): 고정 로봇·고정 파지·장애물X. **게이트: KinGNN이 MLP 85% 재현** (~13-16M). → 그래프 표현이 학습 안 깨나 + GNN이 MLP만큼 하나.
2. **파지 변이**(`--vary_grasp`): grasp 엣지가 폭 d+각도 θ 인코딩. **게이트: held-out 파지 zero-shot**.
   - ⚠️ 파지 변이 시 팔로워 IK offset(`_lf_off_pos2`, `_lf_sep`)을 per-env d로 바꿔야 함(현재 고정 0.4).
3. **base 간격 변이**: base-rel 엣지가 간격 인코딩. **게이트: held-out 간격 zero-shot**.
4. (이후) 로봇 크기(link 노드 추가) → 종류/DoF(role 마스크). → morphology 일반화 완성.

---

## 7. 리스크 / 미해결

| 항목 | 상태 |
|---|---|
| fps ~1400 (T=8, 6.9M params) | 30M ~6h. 필요시 num_rounds↓ |
| GNN 학습이 MLP보다 느릴 수 | ~13M 전 조기판정 금지 |
| 파지 변이 시 팔로워 IK per-env d | 단계2에서 수정 필요 |
| DoF 변이(고정 인덱스 1~7) | role 마스크로 확장 (단계4) |
| 관절축 상수라 정보량 | 엣지가 구조 담당, 문제없다고 판단(재확인 필요) |
