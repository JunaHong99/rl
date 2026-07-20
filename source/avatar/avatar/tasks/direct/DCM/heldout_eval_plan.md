# S2 파지 일반화 eval 프로토콜 (2026-07-20)

## 결정 (이번 런)
- 현재 S2 = **8-버킷**(고정 seed 12345로 전 범위 [0.25,0.40]×[±45°]에 흩뿌린 8종 (d,θ)) + same_side ON.
- **보간(interpolation) eval**로 진행 (재구성 없이 첫 신호). 강한 외삽 주장은 추후 재구성 런에서.
- 판정: 학습 8종 success vs **held-out 랜덤 파지** success → **generalization gap**.

## 실행 (학습이 ≥13M 도달 후, 체크포인트로)
```
# (1) 학습 파지 success — 8버킷 재현
python -u eval_grasp.py --model_path logs/<run>/model_final.pt --same_side --num_envs 1024 --headless

# (2) held-out 파지 success — 안 준 랜덤 파지
python gen_grasp.py --n 32 --seed 777 --out heldout.pt
python -u eval_grasp.py --model_path logs/<run>/model_final.pt --same_side --grasp_preset heldout.pt --num_envs 1024 --headless
```
⚠️ **--same_side 필수** (학습과 파지 분포 일치). action_scale은 학습과 같게(기본 0.05/0.05, 다르면 인자로).

## 읽는 법
- `▶ 전체 success` : 그 파지셋 평균.
- **per-버킷 표**(d, θ, success) + **|θ| 구간별**(0~15/15~30/30~46°) : **어디서 무너지나**.
- 기대 시나리오: 위치·저각도 OK, **고각도(|θ|>30°)에서 success 급락** → lean 그래프가 기구학 못 배움 = **kinematic 그래프(S4) 동기의 정량 근거**.
- gap = (1)−(2). 작으면 일반화 성공, 크면(특히 고θ) 실패 → S4.

## 원인 분해 (gap이 크면)
`ik_feasibility_diag.py`(rollout success gate 먼저) + reach/특이점/oor 성공·실패 분해.
"held-out 실패 = 도달불가(IK infeasible)↑"면 kinematic 동기 정확히 입증.

## 이후
- 보간 신호 좋으면 → 재구성 런(촘촘 버킷 + 영역 hold-out)으로 **강한 외삽 주장**.
- 무너지면 → S4 kinematic 그래프 착수 (플랜 §3 결정포인트부터).
