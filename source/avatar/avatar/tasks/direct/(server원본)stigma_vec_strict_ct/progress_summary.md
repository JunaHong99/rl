# RoboBallet 프로젝트 진행 상황 요약 (2026-01-07)

## 1. 제약 위반 로깅 시스템 개선
*   **목적**: 제약 조건 패널티 강화 후, 평균 에러 지표에서 드러나지 않는 순간적인 위반 상황을 정밀하게 분석하기 위함.
*   **변경 사항**:
    *   `dual_arm_transport_env2.py`: `self.extras`에 환경별 raw 데이터인 `log/raw_err_pos` 및 `log/raw_err_rot` 추가.
    *   `test.py`: 제약 위반 발생 시(`is_currently_violated`), 해당 구간 동안 매 스텝마다 실시간 Position Error와 Rotation Error를 터미널에 출력하도록 로직 수정.
*   **기대 효과**: 위반이 발생하는 특정 궤적 구간에서의 에러 수치를 직접 확인하여 패널티 설계의 적절성 판단 가능.

## 2. Pose Sampler 분석 및 환경 파악
*   **분석 대상**: `pose_sampler.py`
*   **주요 로직**:
    *   두 로봇의 베이스 위치와 회전을 랜덤하게 결정.
    *   물체 파지(Grasp) 시 약 ±15도의 랜덤 노이즈 추가.
    *   시작점과 목표점 사이의 거리가 최소 0.1m 이상이 되도록 샘플링 (IK 유효성 검사 포함).
*   **확인 사항**: 현재 목표 지점 설정 시 End-Effector(EE)의 Orientation도 함께 생성되고 있으나, 기존 환경 코드에서는 이를 성공 판정에 충분히 활용하지 않고 있었음.

## 3. 향후 계획: 성공 조건 및 보상 함수 강화
*   **목표**: 단순 위치 도달을 넘어, 목표 지점의 Orientation(자세)까지 정확히 맞추도록 유도.
*   **강화된 성공 조건**:
    *   **Position Error**: 5cm (0.05m) 이내
    *   **Rotation Error**: 15도 (약 0.26 rad) 이내
*   **구현 예정 사항**:
    *   `_get_rewards`: Task Reward(`r_dist`) 계산 시 위치 오차뿐만 아니라 각 팔의 회전 오차(`rot_dist`)를 포함하도록 수정.
    *   `_get_dones`: 강화된 거리/각도 조건을 모두 만족해야 `is_reached` 및 `is_success`가 True가 되도록 변경.
    *   **추가 지표 로깅**: `max_pos_err`, `max_rot_err`, `violation_rate` 등을 로그에 추가하여 학습 안정성 모니터링.
