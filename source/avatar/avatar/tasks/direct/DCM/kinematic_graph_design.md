# Kinematic Graph 설계 (2026-07-23)

리더-팔로워 관절 액션 하에서 **파지(폭 d·각도 θ) + 로봇 base 간격 변이에 일반화**하는 GNN 관측.
lean 3-노드(EE만)로는 관절 상태를 표현 못 함(정책이 리더 관절을 직접 명령). → 관절을 노드로.

**구조 그림**: `kin_graph_structure.png`
**관련 파일**: `graph_converter_kin.py`(빌더), `gnn_policy_kin.py`(정책), `dual_arm_transport_env3.py:_build_kin_raw`(env raw), `her_buffer.py`(kin relabel), `sac_trainer.py`(kin 버퍼).

---

## 0. 동기

- object-centric은 파지 일반화 공짜(94% zero-shot)였으나 **네트워크가 기구학을 안 배움** → morphology 일반화 토대 없음.
- → 리더-팔로워(리더 관절 Δq 학습 + 팔로워 IK) 전환, 고정 태스크 **운반 85% 성공**(MLP baseline).
- **협응은 학습 대상 아님**(팔로워 IK가 구조적 보장). 학습 = **리더 관절로 rod를 목표 pose에 운반**.
- 파지/base가 바뀌면 **목표 rod pose ↔ 필요한 관절 config**의 사상이 달라짐 → 정책이 그 변이를 *보려면* 관절 노드 필요. (MLP 85%는 단일 고정 기구학을 통째로 외운 것이라, 축·구조 정보 없이도 됐음. 일반화엔 그래프 필요.)

---

## 1. 그래프 구조 (17노드, 34엣지)

로봇 = 같은 종류(Franka×2). 링크길이 상수 → **link 노드 불필요**(크기/종류 변이 때 추가). base+joint+object만.

### 노드 순서 (고정 인덱스)
```
[0]      base_L        base     리더 base
[1..7]   jL1 .. jL7    joint    리더 관절    ← ★ 출력 head (리더 Δq 7)
[8]      rod           object   운반 대상
[9]      base_F        base     팔로워 base
[10..16] jF1 .. jF7    joint    팔로워 관절   ← backbone 정보전파만, 출력 X(IK 결정)
```
리더 joint = 고정 인덱스 1~7 → 출력 head가 이 7개만 gather (DoF 변이는 추후 role 마스크).
jL7 / jF7 = 각 팔의 EE(end-effector, panda_hand 위치 근사).

### 노드 feature (NODE_FEATURE_DIM = 12 = raw 패딩 9 + type one-hot 3)
| 타입 | raw | dim | 내용 |
|---|---|---|---|
| **base** | 없음(identity) | 0 | 공간 위치는 엣지로만 |
| **joint** | 축(3) + sin/cos(q)(2) + margin(2) | 7 | §3 |
| **object(rod)** | goal오차 pos(3, 리더 base frame) + rot 6D(6) | 9 | 태스크 신호 |

### 엣지 (4 타입, 전부 양방향, EDGE_FEATURE_DIM = 13 = type one-hot 4 + 상대pose 9)
| 타입 | 연결 | 개수 | 담는 것 |
|---|---|---|---|
| **base→first** | base ↔ joint1 (양팔) | 4 | 로봇 mount(뿌리→첫관절) |
| **joint-chain** | joint_i ↔ joint_{i+1} (양팔 6쌍씩) | 24 | 이웃 관절 상대pose = 링크 기하 + 현재 꺾임 (§4) |
| **grasp** | joint7(EE) ↔ rod (양팔) | 4 | **★ 파지 폭 d + 각도 θ** (EE↔rod 상대pose) |
| **base-rel** | base_L ↔ base_F | 2 | **★ base 간격** (두 base 잇는 "가상 링크") |
= 34 edges/env

**닫힌사슬 루프**: base_L→(사슬)→jL7→(grasp)→rod→(grasp)→jF7→(사슬)→base_F, 그리고 base_L↔base_F(base-rel)로 닫힘. GNN이 message passing으로 이 loop를 인지.

### Global
- normalized_time(1) + f_int(1) = **GLOBAL_FEATURE_DIM = 2**
- (관절각은 이제 joint 노드에 있으니 global서 제거)

---

## 2. 좌표계 (요소별 역할에 맞게 — 핵심)

| 요소 | 좌표계 | 이유 |
|---|---|---|
| **관절축** | 부모링크 국소(상수, DH) | NerveNet식. 노드=국소 구조, 배치 불변 → 전이 (§3) |
| **joint 값·margin** | 관절공간(각도) | frame 무관 |
| **object goal오차** | **리더 base frame** | 액션(리더 관절)과 정렬 → 사상 학습 쉬움 |
| **엣지 상대 pose** | **receiver(도착 노드) frame** | 프레임 무관량; message passing에서 "받는 노드 기준 이웃" (§4) |
| **액션(리더 Δq)** | 리더 base 관절공간 | — |

- task 신호(obj_goal)와 액션은 **리더 base frame**으로 통일, 노드 축은 국소, 엣지는 상대 → 각자 역할.
- **검증**: env 관측 obj_goal == HER relabel obj_goal, 회전 base에서도 diff 0.0 (수식 일치 확인). quat 규약(wxyz) math_utils와 her_buffer 동일.

---

## 3. Joint 노드 feature 상세

### 축 (3) = 부모링크 frame 상수
- Modified DH 도출: 부모frame 기준 joint_i 축 = Rx(α_i)·[0,0,1].
  - jL1=[0,0,1], jL2=[0,1,0], jL3=[0,-1,0], jL4=[0,-1,0], jL5=[0,1,0], jL6=[0,-1,0], jL7=[0,-1,0]. (양팔 동일.)
- **왜 world/base frame 아닌 국소 상수**:
  - 공통 base frame이면 노드가 공간정보까지 이중으로 담아 엣지와 중복 → GNN이 여러 홉 재구성 비효율, base 배치 의존.
  - **부모링크 국소**: 노드=국소 구조만, 공간관계는 엣지 → message passing 친화적, base/자세/배치 **완전 불변** → morphology 전이. NerveNet 정석.
- **왜 상수인데 필요**(잉여 아님): 파지/base마다 다른 관절 config를 정책이 풀려면 "각 관절이 어느 축으로 도나"가 **기구학 추론의 고정 재료**. 곱셈표처럼 상수여도 계산에 씀. (일반화 변수는 아니지만 계산 재료.)

### sin/cos(q) (2) = 현재 관절 상태
- 관절 명령 = 현재각 기준 증분(Δq) → 현재 위치 필요.
- **왜 q 대신 sin·cos 둘 다**: 각도는 순환값(−π=+π)이라 raw q는 경계 불연속. (cos q, sin q)는 단위원 위 점 → 연속·유일. 하나만이면 각도 모호(sin(30°)=sin(150°)). 둘 다라야 사분면 결정.

### margin (2) = feasibility (연구 동기 직결)
- `(q−q_min)/range`, `(q_max−q)/range` ∈[0,1]. 0 근접 = 관절한계/특이점 근접.
- 미해결 6~7%(경로 중 특이점·도달불가) → object-centric엔 관절정보 0이라 못 봄. **margin이 정책에 feasibility 노출** = kinematic graph 값어치.

### 역할 분리
축 = 구조 상수 / sin·cos = 상태 / margin = 여유. 안 섞임.
제외: 관절속도(lean 교훈), 링크길이(같은 로봇 → link 노드 자체 제외).

---

## 4. 엣지 상대 pose가 담는 것 (joint-chain 예시)

엣지 feature = type one-hot(4) + **receiver-frame 상대 pose**:
```
rel_pos = R_dst_inv · (p_src − p_dst)   # 도착(dst) 좌표계 기준 출발(src) 상대 위치 (3)
rel_6d  = 6d(R_dst_inv · R_src)         # 상대 회전 (6, Zhou et al 6D)
```
p, R은 **sim의 실제 link body pose**(robot.data.body_pos_w). DH 숫자를 넣는 게 아니라, sim이 관절각대로 배치한 실제 위치를 읽으면 구조·상태가 자동 반영됨.

**joint-chain(jL_i ↔ jL_{i+1})에서:**
- **rel_pos(위치차)** = 두 관절 사이 **링크가 어떻게 뻗어있나**(길이·방향). 관절각 변해도 거의 상수 = **로봇 구조**.
- **rel_6d(회전차)** = 그 관절이 **지금 얼마나 꺾여있나**(관절 회전 시 다음 링크 좌표계가 딸려 돎) = **현재 자세**.

**방향**: 양방향이라 j1↔j2는 두 엣지 — dst=j2(j2 기준 j1) + dst=j1(j1 기준 j2). "부모 기준 자식"은 그중 자식→부모 엣지(dst=부모)에 해당. 코드는 부모/자식이 아니라 src/dst의 receiver frame으로 일반화.

**joint 노드 sin/cos와 중복?** 상보적: 노드=그 관절 각도값(스칼라), 엣지=그 각도가 공간에서 두 링크를 어떻게 배치시키나(기하). 정책이 각도+공간효과를 함께 봐서 기구학 추론.

---

## 5. 정책 (KinGNN, NerveNet식 타입공유)

- **Backbone**: NodeType/Edge/Global embedder(kin 차원) + RoboBallet GN block × T. **T=8**(사슬 깊이 base→EE 7 + rod 건너 1).
- **Actor 출력**: 리더 joint 노드 7개 gather → **공유 per-joint head**(각 노드 raw skip + MP embedding → 스칼라 Δq) → (B,7). tanh·dq_scale.
  - 공유 head = 관절 수 무관 = **DoF 일반화 원천**. 팔로워 joint 노드는 전파만, 출력 X.
- **Critic/Q**: global mean-pool(노드 수 무관) + action → scalar. twin Q + target.
- 6.9M params (T=8), action_dim=7, dq_scale=0.30.

---

## 6. 리워드 / 학습 (리더-팔로워와 동일)

- **task**: 도달 +100(one-shot) + 시간페널티 −0.2.
- **dense progress**(`lf_dense_progress`, w=10): Cartesian rod 접근량 `w·(prev−curr)dist`. sparse+HER는 리더관절 액션엔 약함(relabel trivial)→학습 굶음. potential-based+도달종료라 제자리 꼼수 없음(ep_len 감소가 실제도달 증거).
- **내력 페널티 없음**(협응 구조보장 → freeze 위험만 방지). 내력 제어는 운반 성공 후.
- **HER**(kin, 방식 B): object 노드의 base-frame goal오차를 가상 goal로 relabel. 그 시점 리더 base quat 저장(`ep_base_quat`)해 재변환. env 계산과 수식 동일(검증됨).
- 레시피: 8192env / updates24 / batch2048 / warmup700k / buffer3M / fixed_alpha0.1.

---

## 7. 단계 (게이트)

1. **고정 태스크 재현** (현재 학습 중): 고정 로봇·파지·장애물X. **게이트: KinGNN이 MLP 85% 재현**(~13-16M). = 그래프 표현이 학습 안 깨나 + GNN이 MLP만큼 하나.
2. **파지 변이**(`--vary_grasp`): grasp 엣지가 폭 d+각도 θ 인코딩. **게이트: held-out 파지 zero-shot**.
   - ⚠️ 파지 변이 시 팔로워 IK offset(`_lf_off_pos2`, `_lf_sep`)을 per-env d로 바꿔야 함(현재 고정 0.4).
3. **base 간격 변이**: base-rel 엣지가 간격 인코딩. **게이트: held-out 간격 zero-shot**.
4. (이후, 목표 밖) 로봇 크기(link 노드 추가) → 종류/DoF(role 마스크).

---

## 8. 리스크 / 미해결

| 항목 | 상태 |
|---|---|
| fps ~1240 (T=8, 6.9M params) | 30M ~6.7h. 재현 확인 후 num_rounds 8→6 검토(성능 영향 작음) |
| GNN 학습이 MLP보다 느릴 수 | ~13M 전 조기판정 금지 |
| 파지 변이 시 팔로워 IK per-env d | 단계2에서 수정 |
| DoF 변이(고정 인덱스 1~7) | role 마스크로 확장 (단계4) |
| 관절축 상수 정보량 | 엣지가 구조 담당, 기구학 재료로 유효 판단 |
