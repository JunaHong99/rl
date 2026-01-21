# 프로젝트 컨텍스트: Isaac Lab 기반 대형 물체 운반 강화학습 최적화

## 1. 프로젝트 개요
- **목표**: Isaac Lab 환경에서 두 대의 로봇 팔(Panda)이 폐쇄 루프 구속 조건(Closed-loop Constraint)을 유지하며 대형 물체를 목표 지점까지 운반하는 정책 학습.
- **핵심 알고리즘**: TD3 (Twin Delayed DDPG) + GNN (Graph Neural Network).
- **상태 표현**: 로봇(노드), 목표(노드), 상대 포즈(에지), 전역 구속 조건 및 시간(글로벌)을 포함하는 그래프 구조.

## 2. 주요 파일 및 역할
- `train.py`: 메인 학습 루프 및 하이퍼파라미터 관리.
- `dual_arm_transport_env2.py`: Isaac Lab 기반 환경 정의, 보상 함수(구속 조건 페널티 포함) 구현.
- `agent.py`: GNN 기반 Actor-Critic 및 TD3 로직.
- `gnn_core.py`: GraphNet 스타일의 메인 GNN 블록 (Edge/Node/Global 모델).
- `graph_converter.py`: 환경 관측값(Tensor)을 GNN 입력용 그래프(Batch)로 변환.
- `replay_buffer.py`: 경험 저장소.
- `pose_sampler.py`: 에피소드 리셋 시 시작/목표 포즈 샘플링 및 IK 수행.

## 3. 핵심 최적화 및 변경 사항 (병렬화 중심)
### A. 벡터화된 그래프 변환 (Vectorized Graph Conversion)
- **변경 전**: `train.py`에서 `num_envs`만큼 Python 루프를 돌며 개별 그래프를 생성 (CPU 병목 심각).
- **변경 후**: `graph_converter.py`에 `convert_batch_state_to_graph` 함수 구현. GPU 텐서 연산만으로 배치 전체의 그래프를 한 번에 생성하여 `num_envs=4096` 이상에서도 오버헤드 거의 없음.

### B. 벡터화된 리플레이 버퍼 (Vectorized Replay Buffer)
- **변경 전**: Python 리스트 및 객체 기반 버퍼 (`deque`). 저장/샘플링 시 CPU-GPU 이동 및 객체화 오버헤드 발생.
- **변경 후**: `VectorizedGraphReplayBuffer` 구현. GPU 메모리에 텐서를 사전 할당하고 배치 단위로 직접 저장. 고정된 토폴로지를 활용하여 샘플링 시 `Batch` 객체를 즉석에서 재조립.

### C. 학습 루프 스케일링
- **변경 사항**: `num_envs`가 커짐에 따라 한 스텝당 유입되는 데이터 양이 폭증하므로, `gradient_steps`를 `num_envs // 128`로 설정하여 데이터 수집과 학습 속도의 균형을 맞춤.
