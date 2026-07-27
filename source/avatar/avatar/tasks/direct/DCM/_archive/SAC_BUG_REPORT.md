# SAC Bug Report

## Summary

현재 SAC 구현에서 alpha auto-tuning이 탐색을 유지하지 못하고 `alpha_min`까지 떨어지는 현상이 있었다. 점검 결과, 핵심 원인은 SAC temperature loss의 부호/타깃 해석 문제이며, 부가적으로 replay buffer가 GNN 입력에 필요한 `edge_type`을 복원하지 않고 기본 action dimension을 6으로 고정하는 문제가 있었다.

수정 후 alpha 업데이트는 추정 엔트로피 `H_est = -log_pi`가 target보다 낮으면 alpha를 증가시키고, target보다 높으면 alpha를 감소시키는 방향으로 동작한다.

## Findings

### 1. Alpha Loss Direction

파일: `sac_trainer.py`

기존 구현:

```python
alpha_loss = -(self.log_alpha * (log_pi_new_flat.detach() + cfg.target_entropy)).mean()
```

현재 설정은 `target_entropy=-6.0`처럼 음수 관례를 사용한다. 이 식에서는 `log_pi + target_entropy`가 대부분 음수가 되기 쉬워 alpha가 계속 감소하는 방향으로 업데이트될 수 있다. 그 결과 auto-alpha를 켜면 entropy bonus가 사라지고, 정책 탐색이 급격히 죽는다.

수정 구현:

```python
target_entropy_abs = abs(float(cfg.target_entropy))
entropy_error = (-log_pi_new_flat.detach()) - target_entropy_abs
alpha_loss = (self.log_alpha * entropy_error).mean()
```

의도한 동작:

- `H_est < target`: entropy가 부족하므로 alpha 증가
- `H_est > target`: entropy가 충분하므로 alpha 감소

### 2. Replay Buffer Drops `edge_type`

파일: `sac_trainer.py`

GNN의 `NodeModel`은 edge type별 aggregation을 사용한다.

```python
aggrs = [
    scatter_mean(edge_attr[edge_type == t], col[edge_type == t], dim=0, dim_size=N)
    for t in range(self.n_edge_types)
]
```

하지만 non-HER `ReplayBuffer.sample()`은 PyG `Batch`를 재구성할 때 `edge_type`을 넣지 않았다. 이 경로를 타면 GNN forward에서 `batch.edge_type`이 없거나 잘못된 입력이 되어야 정상이다. HER buffer에는 이미 `edge_type` 복원이 구현되어 있었다.

수정 내용:

- `ReplayBuffer.__init__()`에서 static `_EDGE_TYPE` 저장
- `sample()`에서 batch 크기만큼 `edge_type` 복원
- current/next batch 모두에 `edge_type` 전달

### 3. Replay Buffer Action Dimension Hardcoded to 6

파일: `sac_trainer.py`

환경 기본 설정은 `dual_arm_transport_cfg.py` 기준 `action_space=8`이다.

```python
action_space = 8
```

하지만 `SACTrainer` 기본 replay buffer는 `action_dim=6`으로 고정되어 있었다. HER를 켜면 `train_phase3_sac.py`에서 buffer를 `env.cfg.action_space`로 교체하기 때문에 숨겨질 수 있지만, non-HER 경로에서는 action 저장 shape mismatch 또는 잘못된 학습 데이터가 발생할 수 있다.

수정 내용:

```python
action_dim = getattr(getattr(agent, "actor", None), "action_dim", None)
```

이제 replay buffer는 agent actor의 실제 action dimension을 사용한다.

## Network Information Flow Review

### 보존되는 정보

`gnn_policy.py`의 actor/Q head는 message-passing 결과만 쓰지 않고 raw skip을 포함한다.

- Actor input: `raw_rod`, `raw_global`, `MP_rod`, `MP_global`
- Q input: `raw_rod`, `raw_global`, `MP_rod`, `MP_global`, `action`

따라서 rod pose, goal, position error, rotation error 등 핵심 task 정보가 GNN message passing에서 손상되더라도 head로 직접 들어간다. 이 부분은 정보소실 방어 장치로 타당하다.

### 남은 리스크

`NodeModel`은 edge type별로 `scatter_mean`을 사용한다. 이 방식은 obstacle/proximity edge 개수에 대해 scale-invariant라 안정적이지만, 가장 가까운 장애물이나 가장 위험한 링크 같은 worst-case 정보를 평균 과정에서 희석할 수 있다.

특히 장애물 회피에서는 평균보다 minimum distance, max risk, nearest obstacle feature가 중요할 수 있다. 이 문제는 alpha collapse의 직접 원인은 아니지만, 회피 정책이 둔해지거나 장애물 수 변화에 민감해지는 원인이 될 수 있다.

추가로 actor의 `log_std`가 state-independent parameter 하나라서 상태별 탐색량 조절이 불가능하다. 장애물 근처, 목표 근처, 초기 탐색 구간에서 다른 uncertainty를 표현하려면 state-dependent `log_std` head가 더 적합하다.

## Changes Applied

### `sac_trainer.py`

- alpha loss를 entropy error 기반으로 수정
- `target_entropy=-action_dim` 음수 관례를 유지하기 위해 `abs(cfg.target_entropy)` 사용
- non-HER replay sample에 `edge_type` 복원 추가
- replay buffer action dimension 하드코딩 제거
- trainer stats에 `entropy_est` 추가

### `train_phase3_sac.py`

- 콘솔 로그에 `H` 추가
- TensorBoard SAC 로그에 `entropy_est` 추가

## Validation

실행한 검사:

```bash
python3 -m py_compile sac_trainer.py train_phase3_sac.py
```

결과: 통과.

제한 사항:

- 현재 shell의 `python3` 환경에는 `torch`가 없어 tensor-level update smoke test는 실행하지 못했다.

## Recommended Follow-ups

1. 짧은 학습 run에서 `alpha`, `log_pi_mean`, `entropy_est`를 같이 확인한다.
2. `target_entropy`는 현재 `action_dim=8`이면 기본적으로 `-8`도 비교할 가치가 있다. 현재 CLI 기본값은 `-6`이다.
3. alpha가 정상화된 뒤에도 회피가 둔하면 GNN aggregation에 `scatter_min` 또는 nearest-risk summary를 추가한다.
4. actor에 state-dependent `log_std` head를 추가해 상태별 exploration을 허용한다.

