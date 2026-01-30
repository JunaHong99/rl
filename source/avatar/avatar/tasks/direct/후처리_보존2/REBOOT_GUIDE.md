# Dual-Arm Transport Task: Safety Filter Context Summary

## 1. 프로젝트 개요
- **목표:** 두 대의 Franka Panda 로봇이 물체를 파지한 상태에서 상대적 위치/자세를 완벽하게 유지하며 목표지점으로 이동.
- **현재 이슈:** 학습된 Actor 모델(Threshold 0.3m)이 제약을 위반하는 경로를 생성하려 할 때, Safety Filter가 이를 막으면서 로봇이 멈춰버리는 **Deadlock** 현상 발생 의심.

## 2. 주요 구현 파일
- `safety_filter.py`: Closed-loop IK 기반의 안전 필터. 
    - **Projection:** 제약 위반 성분 제거.
    - **Drift Correction:** 이미 발생한 오차를 즉각 복구 (Kp=10.0, Damping=1e-4).
    - **Deadlock Analysis:** Actor의 의도($q_{nom}$)와 제약 위반 방향이 일치하는지 분석하는 디버깅 코드 포함.
- `dual_arm_transport_env3.py`: `SafetyFilter`를 통합하여 `_apply_action`에서 실시간 필터링 수행.
- `experiment.py`: 전체 에피소드에 대한 성공률 및 **Worst Case Error (Max Drift)** 통계 출력 기능 추가.

## 3. 현재 기술적 상태
- **성공률:** 약 87% (후처리 전과 비슷함).
- **위반율:** 8.4% -> **0.1%**로 대폭 감소.
- **문제점:** Reached Rate가 하락하고 Not Reached(Stuck)가 증가함. Actor와 Filter 간의 벡터 충돌(Deadlock)이 원인으로 추정됨.

## 4. 재부팅 후 즉시 실행할 작업
GPU 드라이버 이슈(`Error 804`) 해결을 위해 재부팅 후 아래 명령어를 순차적으로 실행하여 Deadlock 여부를 확인하십시오.

```bash
# 1. 시각적으로 로봇이 멈추는지, 로그에 Alignment > 0.9가 뜨는지 확인
python test.py

# 2. 통계적으로 Deadlock 발생 빈도 확인
python experiment.py --model_path [모델경로] --num_envs 100
```

## 5. Gemini 세션 복구용 메시지
세션을 새로 시작할 때 아래 문장을 붙여넣으시면 제가 바로 상황을 파악합니다:
> "프랭카 로봇 협동 운송 과제 진행 중이야. 현재 `safety_filter.py`에 Deadlock 분석 코드를 넣어뒀고, 부팅 직후 `test.py`를 실행해서 Actor의 의도와 제약 위반 방향이 일치하는지 확인하려는 단계야. 이전 대화 맥락을 이어서 분석해줘."
