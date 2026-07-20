# Stage 0 최종 정리 — Dual-arm Cooperative Rod Transport (2026-06-12)

## 0. 한 줄 요약

base RL(SAC + per-arm impedance)이 **두 Franka가 도달 가능한 workspace 전체(최대 601mm 운반)에서
협응 막대 운반을 성공률 90.1%로 수행** — Stage 0(base RL이 task를 푼다) 사실상 완성.

---

## 1. 시스템 구성

- **로봇**: Franka 2대 (지면 z=0, base ±0.5m). rod(0.8m, dynamic)가 양 panda_hand에 **fixed joint**로 결합.
- **제어 계층**:
  ```
  SAC 정책 ──6-dim object pose delta(누적)──▶ rod target pose
           ──▶ PerArmImpedanceController: 각 arm이 자기 grasp점 target 독립 추종
               f = K_pos·Δpos + D_pos·(−v),  τ = Jᵀ·[f,m] (clamp ±50)
           ──▶ 양 arm joint torque (effort mode)
  ```
- **gains**: K_pos=200, D_pos=60, K_rot=20, D_rot=8 (고정). **중력 OFF**(안정성 우선, 아래 §6).
- **성공 기준**: 에피소드 중 한 번이라도 `min_pos<20mm AND min_rot<10°`.

---

## 2. 최종 결과

**deterministic eval (model_final.pt, 945 ep, 전체 10–601mm 분포):**

| 거리 구간 | n | success |
|---|---|---|
| 가까움 100–250mm | 139 | **95.0%** |
| 중간 250–450mm | 592 | **90.9%** |
| 먼 450–601mm | 210 | **85.7%** |
| **전체** | 945 | **90.1%** |

- **최대 운반 reach = 유클리드 601mm** (sampler 필터가 EE 0.5–0.85m + IK + joint margin으로 물리 한계까지 cap).
- 거리에 따라 단조 감소(정상). 실패는 대부분 near-miss (min_pos median 26mm).

---

## 3. 여기까지 해결한 문제들 (학습이 아예 안 되던 상태 → 90%)

### A. 물리/리셋 불안정 (학습 자체가 안 되던 단계)
- **settle 30 step 빌트인**: reset 시 PhysX fixed-joint snap이 ~10% env를 폭주시키던 것 해결
  (unstable 11%→4%). reset 직후 action=0 강제 + episode_length_buf freeze.
- **HER buffer settle mask**: settle transition을 학습 buffer에서 제외 (정책이 'action=0 valid' 안 배움).
- **sampler joint margin filter** (≥0.2 rad): limit 근처 불안정 자세 제거.

### B. 학습 신호·평가 정정
- **평가 기준 정정**: 옛 `terminated`(마지막 step pos만, 회전 무시) → `min_pos<20mm AND min_rot<10°`.
- **action_scale_rot 0.025→0.05**: 회전 학습 약점 해결 (단일 변경 중 가장 큰 기여).

### C. Task 정의 — "풀던 게 가짜였다"
- **Stage 1 sampler 발견**: goal이 3–5cm뿐 → "가만히 있어도 절반 도달"하는 가짜 task (58% plateau의 정체).
- **Stage 2 전환**: goal 10–30cm, dz±15cm, yaw±30° (진짜 운반).

### D. 진짜 task에서 0% + GPU crash → 3처치
- Stage 2 첫 시도 700k step 0% + PhysX CUDA 719 crash → 재부팅.
- **3처치**: ① action_scale_pos 0.02→0.05, ② time penalty -0.5→-0.2, ③ **curriculum 자동 확대**
  (cache 거리순 정렬해 가까운 것부터 점진). → 0% 탈출, 30M 완주.

### E. **가장 큰 반전 — 57%는 측정 버그였다 (phantom)**
- eval 57%로 "plateau"로 오해 → 시각화로 정체 발견: 실패가 전부 **SUCCESS 직후 RL=0 step, rod 안 움직임**.
- 원인: IsaacLab이 `_get_dones`를 `_get_rewards`보다 먼저 호출 → stale `log/is_reached`(직전 성공값)를
  읽어 reset 직후 settle step에서 terminated 즉시 발화 → 1-step phantom 에피소드(실패로 오집계).
- 수정: `terminated = is_success & ~self._is_settle_step` (한 줄).
- → **phantom 0%, 보고 success 57% → 97.9%.** 정책은 처음부터 task를 풀고 있었음.

### F. 최대 reach 확장 (resume)
- goal 범위 10–55cm / ±25cm / ±60°로 확대 → 최대 reach 601mm. 97.9% 모델에서 **resume**
  (curriculum를 숙련범위 frac 0.7에서 시작, ramp를 resume 지점에 anchor, buffer refill로 콜드스타트 방지).
- 60M 완주 → 전체 분포 90.1%, **근거리 95% 유지(catastrophic forgetting 없음)**.

> **핵심 통찰**: 90%로 가는 길의 절반은 "정책 개선"이 아니라 **"정책이 실제로 얼마나 잘하는지 제대로 측정"**
> 하는 문제였다 (평가 기준 정정, 가짜 task 발견, phantom 버그). 특히 phantom은 57%→97.9%의 최대 단일 반전.

---

## 4. 남은 한계 — near-miss (정책측, 컨트롤러 아님)

- genuine 실패의 min_pos median 26mm → **정책이 target을 goal 앞 ~34mm에 park**(park 시 track_err 5mm =
  rod는 target에 도달, target이 goal에 안 닿음). 컨트롤러는 제 역할 함.
- track_err 29mm(평균)는 **운동 중 속도 지연**(lag≈D_pos·v/K_pos≈75mm)이고, 멈추면 0으로 수렴 → 정밀도 천장 아님.
- **정밀도 향상 레버**(미적용): 정책측(threshold 근처 reward shaping / 추가 학습) 또는 K_pos↑(운동 지연↓).

---

## 5. 부수적으로 확인하고 "문제 아님"으로 종결한 것

- **멈춤(freeze) 구간** → 모델/컨트롤러 충돌 아님. 에피소드 사이 settle/reset 정지(설계대로). (`diagnose_freeze.py`)
- **time penalty(-0.2)** → 부재도 약함도 아님. 정책은 모든 transport step에서 적극 명령.
- **reset 충격** → joint frame 정상. IK 6mm 오차가 원천이고 settle이 흡수(배포 무관). best-fit으로 2.5mm
  까지 줄일 수 있음(`make_bestfit_cache.py`, optional). ⚠️ settle은 sim reset 아티팩트라 배포 땐 빠짐.

---

## 6. 알려진 갭 / 다음 단계

- **중력 OFF (sim-to-real 갭)**: 현재 sim은 중력 비활성 (안정성). Phase 3.2에서 gravity ON +
  `get_generalized_gravity_forces()` 보상 시도했으나 부호/frame convention으로 **발산** → OFF 유지.
  배포 현실성 위해선 이걸 디버깅해 gravity ON으로 가야 함 (성공률엔 영향 없음).
- **Stage 1**: RL이 per-arm K(K_arm1, K_arm2) 동시 출력 (DA-VIL의 shared K 대비 contribution).
- **Stage 2**: DA-VIL 비교. **Stage 3**: real-world.

---

## 7. 주요 산출물 / 도구

- `dual_arm_transport_env3.py` — settle, phantom-fix(`terminated = is_success & ~_is_settle_step`).
- `vectorized_pose_sampler.py` — maxreach goal 범위 + curriculum 정렬.
- `train_phase3_sac.py` — curriculum(resume anchor) + `--resume_refill_steps`.
- `eval_mlp.py` — 거리 구간별 success 분해 + phantom 진단 + os._exit(Isaac shutdown hang/GPU 누수 차단).
- `diagnose_freeze.py` — 멈춤 원인(모델 vs 컨트롤러) 분류.
- `make_bestfit_cache.py` — reset-shock 완화 cache (optional).
- 모델: `logs/phase3_sac_20260610-154019/model_final.pt` (60M, 90.1%).
- cache: `pose_cache_100000.pt` (maxreach, 최대 601mm). 백업: `*.OLD_stage2_10-30.pt` 등.

**eval 명령** (action_scale 0.05/0.05 필수):
```bash
python -u eval_mlp.py --num_envs 128 --num_steps 360 \
    --action_scale_pos 0.05 --action_scale_rot 0.05 \
    --model_path logs/phase3_sac_20260610-154019/model_final.pt --headless
```
