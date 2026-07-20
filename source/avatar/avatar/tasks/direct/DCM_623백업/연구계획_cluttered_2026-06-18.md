# 연구 계획 — Cluttered 닫힌사슬 협응 운반 + Zero-shot 일반화 (작성 2026-06-18, 최종갱신 2026-06-23)

> 이전 계획(자유공간 GNN 형태일반화)을 대체/갱신.
> 핵심 전환: 자유공간 형태일반화는 **공허**(rigid+object-pose action이면 형태 무관, 컨트롤러가 흡수)임을
> 규명 → **장애물 회피**를 추가해 형태/배치가 정책에 *내재적으로* 중요해지게 만듦.

---

## 1. Thesis / Contribution

> **Cluttered 환경에서 닫힌 운동학 사슬(closed kinematic chain)을 이루는 dual-arm 협응 운반 —
> 든 물체와 양팔이 장애물을 회피하며, 로봇 배치·방향 / 장애물 config / 물체 형태에 zero-shot
> 일반화하는 단일 GNN 정책 (reference-free).**

차별점:
- **RoboBallet** (Lai et al. 2025, Science Robotics): GNN+TD3로 다중 *자유팔* 장애물 회피 reaching.
  **든 물체·닫힌사슬 없음.** 우리는 **든 물체의 형태까지 충돌 회피 + 닫힌사슬**.
- **DA-VIL**: 물체별 학습 + pre-planned trajectory + compliant grasp. 우리는 **reference-free +
  형태/배치/장애물 일반화 + rigid 닫힌사슬**.
- reference-free: 사전 궤적 없이 RL이 실시간 object-pose(+nullspace) delta 생성.

---

## 2. 시스템 아키텍처

### 2.1 제어 — 닫힌사슬 DoF
- 양팔(14-DoF)이 rigid grasp 2개로 rod(6-DoF)에 묶임 → **제어 가능 DoF = 6(rod pose) + 2(팔당
  nullspace) = 8.**
- **현재 구현 = 6-DoF (rod pose delta만, `action_space=6`).** nullspace(2)는 **arm-장애물 회피용으로 GNN
  단계에 추가 예정 — 아직 미구현.** (rod-pose+nullspace 8-DoF는 full-joint 14-DoF와 회피능력 동등하고
  일반화 우위임을 재검토로 확인 — 2026-06-23, 메모리 참조.)
- (계획) RL action = **rod pose delta(6) + 팔당 nullspace 명령(2)**.
  - rod pose: 임피던스 컨트롤러가 grasp 변환으로 EE 추종 (`per_arm_impedance_controller`, K_pos=200).
  - **nullspace**: `τ += N·τ_sec` (N = I−Jᵀ(JJᵀ+λI)⁻¹J). EE(=rod) 안 흔들며 팔꿈치를 RL 명령 방향
    으로 → **팔-장애물 회피** 가능. (6-DoF로는 팔꿈치 redundant DoF 제어 불가 → 팔 회피 불가였음.)
  - 검증: 팔꿈치 ±12cm 이동, rod drift 3cm, jvel 안정.
- 중력 ON (gravity comp sign=+1 검증됨).

### 2.2 관측 = 그래프 (GNN)
- **노드**: Robot1, EE1, Rod, EE2, Robot2, [obstacle×N]. (NODES_PER_ENV = 5+N)
  - Robot 노드: q(7)+dq(7)+joint margin(14). **관절각 → GNN이 FK 암묵 학습.**
  - Obstacle 노드: pos(3)+반경(1)+**lin_vel(3)**+dist_to_rod(1). vel 슬롯 → **동적 장애물 준비됨.**
- **엣지 (RoboBallet식 상대 pose, EDGE_FEATURE_DIM=13)**: 타입 one-hot(4) + **sender의
  receiver-frame 상대 위치(3)+상대 회전 6D(6)**. → 관절각+상대pose로 팔-장애물 추론 + **배치/방향
  일반화**(모든 게 상대라 절대 위치 불변). 팔 링크 좌표는 *안* 줌 (RoboBallet 방식).
- **정책 입력 = 관절각(노드) + 상대-pose(엣지)**. 팔 링크 위치 명시 X.

### 2.3 보상 (RoboBallet식 cost + Hong2025 교훈)
- sparse success(+100 one-shot, threshold 2cm/10°) + HER (random_task relabel).
- **장애물 cost (2026-06-22 Hong2025 교훈으로 대폭 축소)**: r_clearance(graded, w=1) + r_collision(관통
  시 w=7) + r_smooth. **충돌 페널티 크면 정책이 hesitate/freeze**(Hong2025 명시, 우리 38% 천장 원인) →
  작게. rod·팔(panda_link4/5/6) 둘 다 clearance (env-side 실제 검사).
- **rod safety filter (2026-06-22)**: RL의 rod 목표를 활성 장애물 밖으로 projection(`_apply_rod_safety_filter`).
  안전 backstop + 개입 시 페널티(`w_filter_intervene`)로 RL이 rod 회피 학습(RoboBallet velocity-zeroing
  충실판). cfg `use_rod_safety_filter`로 on/off.
- **HER goal-무관 보존**: 충돌/clearance/time/smooth/filter는 goal과 무관 → HER relabel해도 보존
  (`her_buffer.ep_goal_indep`). RoboBallet R_collision 원리.

### 2.4 장애물 환경
- 정적 구 N=4, kinematic, collision_enabled=False (해석적 clearance로 보상 처리).
- 에피소드마다 위치/개수 랜덤(경로 차단 t∈[0.3,0.7] + offset 0.15), start/goal push로 충돌-free.
- **장애물 curriculum**: 0개(운반만)→점진 4개. 운반 먼저 학습 후 회피 추가 ("안 움직이고 회피만"
  국소최적 회피).

---

## 3. 현재 상태 (2026-06-23) — MLP baseline 정립 완료, GNN 단계 진입 직전

진단·실험 체인(Phase A→E)으로 cluttered MLP baseline을 세웠다. 모두 deterministic eval(`eval_cluttered.py`,
full 장애물, 충돌없는-success 기준).

### 핵심 발견 (메모리 `project_dual_arm_rod_transport`에 상세)
1. **범인 = 장애물 관측이 sparse-reward 발견을 방해.** Stage0(rod+global 36dim) base transport는 현재 env에서
   91.7%인데, 장애물 obs를 입력에 더하면 SAC+HER의 갑작스러운 발견(~1.9M)이 안 일어나 무장애물 운반조차 0%대.
   → **2-phase warm-start**(Stage0 base → 장애물 입력 확장, zero-init으로 base 보존, `convert_phaseA_to_phaseB.py`)로
   발견 문제 우회.
2. **충돌 페널티가 38% 천장의 원인** (Hong2025 명시: 페널티 크면 hesitate). w_collision 25→7, w_clearance 5→1 축소.
3. **rod safety filter** = rod 충돌 24.6%→8.7% (구조적 backstop). 잔차 9%는 tracking lag(제어), RL 무관.
4. **arm 충돌 ~8%는 구조적 한계** = 6-DoF rod-pose로는 팔꿈치 nullspace 제어 불가 → 모든 phase에서 7-8.5% 불변
   (필터·페널티 무반응). 풀려면 awareness(obstacle→arm obs) + control(8-DoF nullspace) + reward 3개 동시 = **GNN 단계**.

### Phase별 결과 (충돌없는-success, best ckpt)
| Phase | 구성 | 충돌없는 success | rod충돌 | arm충돌 | 0-장애물 |
|---|---|---|---|---|---|
| B | 필터✗, 큰페널티 | 38.3% | 24.6% | 7.9% | 41.9% |
| C | 필터✓, 큰페널티 | 54.8% | 8.7% | 7.5% | 53%(불안정) |
| D | 필터 backstop + 개입페널티 + **작은페널티** | **62.8%** | 9.2% | 7.7% | 70% |
| E | **필터 OFF** + 작은페널티 (learned avoidance 검증) | 🔄 진행중 | — | — | — |

- D-final: 장애물 0개 70% / 2개 69% / 4개 58% — 완만한 degradation.
- **Phase E 진행중**(filter OFF로 RL이 rod 회피 스스로 학습하나 검증). 완료 후 결정 트리로 필터 거취 결정.

### 미확정
- Phase E 결과(learned rod 회피 가능 여부), arm 충돌(GNN 단계), solvability 천장, 일반화 전부.

---

## 4. 로드맵 (싼 것 → 비싼 것)

> ★ 핵심 원칙 (2026-06-18 정리): **일반화는 GNN을 RL로 *직접 학습*해야만 나온다.** 증류는
> 비일반화 teacher 모방이라 일반화를 못 만든다 → **일반화 경로에서 제외.** 증류는 과거 "GNN이
> 학습 자체가 안 되던" 문제의 *진단*(GNN net 멀쩡 + SAC가 원인임을 밝힘)으로 역할을 다했고 은퇴.
> trainability(우리 SAC+HER로 처음부터 풀리나)가 B/C의 진짜 관문 — teacher가 없으므로.

```
[Gate] cluttered MLP 작동? (eval) ──低면 처치(ramp↓/장애물↓/solvability)
   │작동 (env/보상/8-DoF 건전성 확인 + 표현가능성은 과거 증류 91%로 이미 확인)
   ▼
[Phase 1] ★ GNN을 cluttered에서 SCRATCH RL로 학습 (SAC+HER, dense 보상) = trainability 게이트
   - sparse-rod(6번 실패한 최악 케이스)가 아니라 **dense-cluttered**(RoboBallet가 GNN+TD3로 성공한
     regime)에서 테스트 → B/C와 같은 환경 + 성공 가능성↑.
   - 통과 = "우리 파이프라인으로 GNN 처음부터 학습됨" → B/C(teacher 없음) 리스크 직접 해소.
   - 고전 시 fallback: 증류 warm-start + RL 하이브리드 (단 rod 과적합이 탐험 좁히는지 ablation).
   │
   ▼  (이후 일반화는 GNN scratch RL로. 각 축: 학습분포 변형 + unseen 테스트 + GNN vs 고정-MLP baseline)
[Phase 2a] 베이스 위치·방향 일반화  ← 가장 쌈 (상대-pose 엣지 인프라 이미 있음, 컨트롤러 임의베이스 OK)
            작업: 학습 때 베이스 pose 랜덤화 + cache 재생성 → unseen 배치 zero-shot
[Phase 2b] 장애물 config 일반화      ← 학습 ≤K → 테스트 >K. GNN(가변노드) vs padded-MLP(K슬롯 한계)
[Phase 2c] 물체 형태 일반화 (C1 헤드라인) ← 가장 큼. 물체→primitive 노드+rigid 엣지 신규 인프라.
            학습 {rod,L,box} → unseen {T,cylinder} zero-shot
[Phase 3] 동적 장애물 (확장)         ← observation 그대로(vel 포함), 움직이는 장애물로 학습/미세조정
```

### 우선순위·스코프
- **전부 = 한 논문엔 과함.** 강한 조합 선택. 추천: 2a(싸고 인프라O) → 2b → 2c(헤드라인).
- 싼 것부터 → **일찍 win 확보**, 충분하면 마무리.
- **모든 축 baseline(고정-MLP) 비교 필수** — GNN이 못 이기면 그 축은 contribution 아님.
- 각 축 metric = **unseen 분포에서 충돌없는-success (GNN vs baseline)**.

---

## 5. 핵심 설계 결정 + 근거

- **8-DoF (nullspace 포함)**: 팔-장애물 회피엔 팔꿈치 제어 필수. 6-DoF object-pose는 팔꿈치(redundant)
  제어 불가 → 회피 불가. 8 = 닫힌사슬의 *완전* DoF.
- **증류(distillation) — 과거 진단, 일반화엔 무력, 은퇴**:
  - 썼던 이유: GNN+SAC가 sparse-rod에서 6번 plateau → "GNN net이 *표현*조차 못 하나 vs SAC 학습만
    안 되나"를 가르는 *진단*. 증류 91% → **net 멀쩡(표현가능), 원인은 SAC 학습**임을 확정. 그게 전부.
  - **일반화엔 기여 못 함**: 비일반화 teacher(rod-MLP) 모방 → student ceiling = teacher, unseen 형태
    배울 게 없음. **representability ≠ trainability**: 증류는 표현만, B/C는 trainability 필요(teacher 없음).
  - → **일반화 경로에서 제외.** 역할(진단) 끝나 은퇴. 기껏해야 Stage-A 웜스타트 옵션(그마저 일반화 X,
    rod 과적합 risk → ablation).
- **trainability가 B/C의 진짜 관문**: 형태 일반화는 teacher가 없으니 결국 GNN을 RL로 직접 학습해야 함.
  GNN+SAC 6번 실패는 **sparse-rod 한정**(보상신호 0 죽은루프). dense-cluttered/multi-shape는 RoboBallet
  성공 regime이라 trainability 가능성↑ → **거기서 검증**(sparse-rod 비관 테스트 X).
- **상대-pose 엣지**: 배치/방향 일반화 + 팔-장애물 추론을 *하나의 메커니즘*으로 (RoboBallet식).
- **장애물 curriculum**: 처음부터 장애물 주면 "안 움직이고 회피만" 국소최적(success 0%) — 실측됨.
  운반 먼저(0개) 학습 후 추가.

## 6. 정직한 리스크 / Open

- cluttered MLP success 천장 — solvability(불가 배치)가 깎을 수 있음. eval에서 장애물 수별로 확인.
- **GNN scratch RL trainability가 최대 리스크** (B/C는 teacher 없어 RL 필수). 완화: dense-cluttered에서
  먼저 검증(sparse-rod 아님) + RoboBallet 선례. **증류는 일반화 fallback이 못 됨**(비일반화 teacher) —
  기껏 웜스타트(일반화 기여 X). 즉 trainability가 안 되면 진짜 막힘 → 그땐 PPO/보상설계/모폴로지 등 재검토.
- 물체 형태 일반화(C1)가 진짜 GNN을 정당화하나 — baseline(padded-MLP)이 못 이겨야 성립. 미검증.
- 로봇 *종류*(모폴로지) 일반화는 scope out (RoboBallet도 future work).
- 모바일 매니퓰레이터 전환은 보류 — 고정베이스 cluttered가 근본적으로 안 될 때만 의도적 pivot.

## 7. 실험 설계 (각 일반화 축 공통)
- **학습 분포**: 해당 축을 랜덤화 (배치/장애물/형태).
- **테스트**: 학습 안 한 값 (unseen 배치 / >K 장애물 / unseen 형태) zero-shot.
- **비교**: GNN vs 고정-입력 MLP(padded). 동일 학습 방식(증류 or 직접).
- **지표**: 충돌없는-success rate (+ 장애물 수/형태/배치별 버킷).
- **판정**: GNN ≫ MLP → 그 축 contribution 성립. GNN ≈ MLP → graph 불필요(정직한 negative).

## 8. 주요 파일
- `dual_arm_transport_cfg.py` — **action_space=6**(현재), 장애물 파라미터, curriculum, 보상가중치
  (w_collision=7/w_clearance=1), rod safety filter(`use_rod_safety_filter`/margin/iters/`w_filter_intervene`).
- `dual_arm_transport_env3.py` — 장애물 스폰/reset 랜덤·curriculum, `_get_obstacle_state`(rod+팔 clearance),
  `_apply_rod_safety_filter`(rod 목표 projection), 보상, `_goal_indep_reward`. (nullspace action은 GNN 단계 추가 예정.)
- `per_arm_impedance_controller.py` — 임피던스 `τ=Jᵀ·wrench`, `_nullspace_obstacle_torque`(모델기반, 현재 OFF).
- `graph_converter.py` — N_OBSTACLES, `_obstacle_features`, 상대-pose 엣지(13d), `_quat_apply`/`_quat_to_6d`.
- `mlp_policy.py` — MLP SAC (rod+global / lean obstacle / full_state 입력 모드).
- `her_buffer.py` — `ep_goal_indep` (goal-무관 보상 보존). `sac_trainer.py` — SAC 학습.
- `train_phase3_sac.py` — `--mlp_lean_obstacle --init_weights_path --no_rod_filter --obstacle_curriculum`.
- `convert_phaseA_to_phaseB.py` — 2-phase zero-init warm-start (rod+global 36 → lean 56, 장애물 컬럼=0).
- `eval_cluttered.py` — deterministic eval (`--model_paths "p@on/@off,..."` 단일프로세스 연속평가).
- `viz_cluttered.py` — GUI 시각화. `distill_gnn.py` — MLP→GNN 증류(진단용, 일반화엔 은퇴).
- 컨트롤러 일반 설명: `impedance_controller_guide.md`.

## 9. (TODO 나중) Task node + RoboBallet식 그래프 설계 — GNN 단계용
**현황**: goal이 global `target_x_rel(3)` + rod node의 `goal_pos/goal_quat`에 분산. task node 없음.

**RoboBallet 정의 (Sci.Robot.2025, Methods)**:
- node: Robot=joint(7)+jvel(7)+dwell(1); **Task=status 1개(pose는 노드 아님)**; Obstacle=feature 없음.
- edge(모두 3d translation + 6d rotation, sender→receiver, receiver=robot tip 프레임):
  robot↔robot(양방향)=sender base pose; **task→robot=task EE pose**; obstacle→robot=중심 translation
  + rotation matrix인데 **각 basis를 primitive span(크기)으로 스케일**(장애물 형태/크기 인코딩).
- 핵심 inductive bias: **pose를 노드 아닌 상대 엣지(robot tip 프레임)** → 절대 배치 불변 = 배치 일반화 근거.

**우리 매핑 (transport는 rod가 도달 주체, robot 아님)**:
- **task node = rod goal pose**. 도달 주체 = rod.
- **task→rod 엣지(핵심): goal pose를 현재 rod 프레임 기준 상대(rel_pos 3 + rel_rot 6d)** = 정책이 0으로
  줄일 운반 에러. (선택) task→robot(EE): goal을 robot tip 기준 → 양팔 협응.
- task node feature: status(1) [+ normalized_time 후보].
- 효과: goal을 엣지로 일원화 → placement 불변 + multi-goal/순차 task 확장 자연스러움.
- **형태 일반화 시**: obstacle을 cuboid primitive 분해 + obstacle→(rod/arm) 엣지에 span-scaled
  rotation matrix(현 sphere radius 대체). rod 형태도 primitive 분해 동일 적용.
