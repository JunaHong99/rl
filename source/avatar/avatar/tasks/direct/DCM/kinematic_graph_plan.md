# Kinematic-Graph 형태 일반화 계획 (2026-07-20)

## 0. 목표 & 기여
**하나의 GNN 정책**이 물체를 목표로 운반하되, **재학습 없이(zero-shot)** 아래 변이에 일반화:
- **grasp pose 변이** (양 매니퓰레이터 상대-pose 제약 변경)
- **로봇 크기 / 종류(DoF) / 두 base 간격 변이**

동시에 **IK 해 존재 / 특이점 회피 / 장애물 충돌 회피**를 만족.

**기여**: model-free RL + kinematic-graph GNN으로, 닫힌사슬 양팔 협동 운반에서 **로봇/파지/배치 형태 일반화**. (NerveNet=자유팔 morphology 일반화, RoboBallet=자유팔 다중 계획 — 둘 다 **닫힌사슬·든 물체 없음**. 우리 차별점.)

---

## 1. 설계 (합의된 것)

### 1.1 그래프 (NerveNet식, 타입별 weight 공유)
**노드 타입:**
| 타입 | feature (intrinsic·상대만, 절대 pose X) | 역할 |
|---|---|---|
| **base** (팔당) | identity (위치는 엣지로) | 팔 뿌리 |
| **link** (팔당 링크수) | 링크 길이 + 기하(반경/형상) | 기구학 + **충돌 몸체** |
| **joint** (팔당 DoF) | 관절 축(3) + 값 + 한계(min/max) + 관절타입 | **IK/특이점 feasibility** |
| **object** (rod) | goal 오차(pos_err3 + rot_err3) | 태스크 신호 |
| **obstacle** (N개) | 반경/형상 (feature 없어도 됨, RoboBallet식) | 회피 대상 |

**엣지 타입 (전부 receiver 기준 상대 pose = trans3 + rot6d):**
| 엣지 | feature | 담는 변이 |
|---|---|---|
| link-chain (joint↔link↔joint) | 고정 변환(DH/링크길이) | **로봇 크기/종류** |
| base→first | mount 변환 | |
| **grasp (EE_link ↔ object, 양팔)** | EE의 object기준 상대pose | **★ grasp 변이** |
| base-rel (base↔base) | 두 base 상대pose | **★ base 간격** |
| proximity (obstacle ↔ link) | 최근접 상대pose | **충돌 회피** |

→ object가 양 EE-link와 연결 = **닫힌사슬 루프**. NerveNet은 일반 directed 그래프 지원.

### 1.2 정책 (NerveNet 4모듈, 타입 공유)
- **Input** Fin_type(x): 노드 관측 → 상태. 타입별.
- **Message** M_edgetype(h): 엣지 타입별.
- **Aggregate** A: 합/평균/max.
- **Update** U_nodetype(h, m̄): 노드 타입별. T번 propagation (T ≥ 최대 DoF, base→EE 전파).
- **Output (하이브리드)**:
  - **object 노드 → 물체 목표 pose delta(6D)** = 운반.
  - **arm(joint/link) 노드 → nullspace 명령** = 팔-장애물 회피 (여유 DoF).

### 1.3 리워드 (일반화 위해 고정임계 IK/특이점 페널티 **금지**)
- 태스크 리워드(도달) + HER = 주 신호 (feasibility는 실패로 암묵 벌).
- track_err = 도달불가 명령의 dense 신호 (팔이 못 따라감).
- 충돌: 작은 페널티 (기존 노선).
- 특이점/IK: **명시 리워드 X**, 그래프 관측으로 정책이 추론하게. (필요시 *상대* manip만 보조.)

### 1.4 컨트롤러
- action = object pose(로봇무관) + nullspace. 컨트롤러가 **로봇별 Jacobian을 sim에서 조회** → τ=Jᵀw. (Phase 5-6에서 다로봇 대응.)

---

## 2. 단계별 계획 (한 번에 한 변이, 각 단계 검증 게이트)

### Phase 0 — 기반 (완료 ✅)
- lean 3노드 그래프 93% obstacle-free. 검증된 레시피(8192/updates24/batch2048). 진단(IK/특이점/충돌 성공·실패 분해). 태그 보존.

### Phase 1 — kinematic 그래프 표현 (고정 로봇, 변이 없음)
목적: 3노드 → **full kinematic 그래프**로 교체하되 **현재 고정 세팅**(Franka×2, 고정 grasp/base)에서.
- **1a** 노드/엣지 스키마 확정 (타입·feature·차원). ← §3 결정포인트 먼저.
- **1b** 그래프 빌더 구현 (env obs → kinematic 그래프). **Isaac-free 테스트**.
- **1c** NerveNet식 GNN 정책 구현 (타입별 input/message/update, per-node output).
- **1d** 하이브리드 출력 배선 (object→pose, arm→nullspace).
- **1e** 학습. **게이트: obstacle-free ≥ 90%** (풍부한 표현이 학습 안 깨는지).
- 리스크: 그래프 커져 학습 난이도↑ / propagation step T 튜닝 / 정책 재작성 버그.

### Phase 2 — 장애물 복귀 (링크노드 충돌 회피)
목적: obstacle 노드 + obstacle→link 엣지 + nullspace 출력으로 **학습된 팔-장애물 회피**.
- **2a** obstacle 노드/proximity 엣지 추가.
- **2b** nullspace 출력이 팔을 밀어내는지 (진단: arm_collision_rate).
- **게이트: 장애물 success ↑ + 팔충돌률 ↓** (기존 filter 대비). = 링크노드+nullspace가 실제로 회피 학습하는지 입증.

### Phase 3 — grasp pose 일반화
목적: per-env grasp offset 변이 → grasp 엣지로 적응.
- **3a** sim에서 per-env grasp 기하 변이 (용접 앵커). **3b** grasp 엣지 feature 반영. **3c** 변이 학습(커리큘럼: 좁게→넓게).
- **게이트: held-out grasp에 zero-shot success** (변이 안 준 grasp).

### Phase 4 — base 간격 일반화
목적: 두 base 거리 변이 → base-rel 엣지로 적응.
- **4a** per-env base 간격 변이. **4b** base-rel 엣지. **4c** 학습.
- **게이트: held-out 간격 zero-shot.**

### Phase 5 — 로봇 크기 일반화
목적: 링크 길이 스케일 변이 → link 노드/엣지로 적응.
- **5a** per-env 링크 스케일. **5b** 컨트롤러가 스케일된 Jacobian 조회. **5c** 학습.
- **게이트: held-out 크기 zero-shot.**

### Phase 6 — 로봇 종류/DoF 일반화 (최난도)
목적: 다른 로봇(다른 DoF/구조) → 노드 수 변이를 GNN이 흡수.
- **6a** 여러 로봇 모델 확보(6/7-DoF 등). **6b** 컨트롤러 로봇별 J 일반화. **6c** 다로봇 학습.
- **게이트: unseen 로봇 종류 zero-shot.** ← 핵심 기여 실험.

---

## 3. 먼저 정할 설계 결정 (Phase 1a 전제)
1. **관절값 표현**: raw(1) vs sin/cos(2)? (연속성 위해 sin/cos 권장)
2. **관절 축 프레임**: 부모링크 고정(일반화 유리) vs 현재 world?
3. **link 엣지 feature**: DH 4param vs 상대변환(trans3+rot6d=9)?
4. **update 함수**: MLP vs GRU(NerveNet recurrent)? propagation step T=?
5. **닫힌사슬 루프**: grasp 엣지 2개로 암묵 표현 vs 명시 loop-closure 엣지?
6. **출력 노드**: nullspace를 어느 노드에서? (팔당 대표 노드 1개 vs joint 노드들)

---

## 4. 교차 관심사
- **진단 재사용**: 매 phase에서 IK/특이점/충돌 성공·실패 분해 로깅으로 뭐가 깨지는지.
- **MLP 베이스라인**: 각 일반화 phase에서 GNN vs flatten-MLP → **GNN이 일반화의 원천**임을 입증(make-or-break).
- **커리큘럼**: 각 변이 phase 내에서 좁게→넓게.
- **레시피**: 8192/updates24/batch2048 유지 (그래프 커지면 재튜닝 가능).
- **명령/args 저장**: 이미 args.json 자동저장 (재현).

---

## 5. 리스크 & 대비
| 리스크 | 대비 |
|---|---|
| kinematic 그래프가 고정로봇서 93% 못 냄 (Phase1 게이트 실패) | 표현/T/정책 디버그. 그래도 안 되면 표현 단순화. |
| 다변이 동시 → 학습 붕괴 | 한 번에 하나(Phase 순서 엄수). |
| 컨트롤러 다로봇 일반화 (Phase5-6) | sim Jacobian 조회. 안 되면 학습 토크로 전환 검토. |
| feasibility 학습이 근사에 그침 | 진단으로 잔여 infeasible율 추적, 훈련 변이 확대. |
| 조기 판정 (과거 전례) | **≥13M + tfevents success로 판정, 1~2M 금지.** |

---

## 다음 액션
**§3 결정포인트 6개 확정 → Phase 1a 스키마 fix → 1b 그래프빌더(Isaac-free 테스트) → 1c 정책.**
Phase 1 게이트(고정로봇 kinematic 그래프 ≥90%) 통과가 전체의 관문.
