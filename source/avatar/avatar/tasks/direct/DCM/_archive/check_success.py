"""최근 학습 런의 success rate + reward 추이 출력.
사용법: python check_success.py   (가장 최근 logs/phase3_sac_* 자동 선택)
"""
import glob, os
from tensorboard.backend.event_processing import event_accumulator

d = sorted(glob.glob("logs/phase3_sac_*"), key=os.path.getmtime)[-1]
ea = event_accumulator.EventAccumulator(d)
ea.Reload()
tags = ea.Tags().get("scalars", [])

def series(tag):
    return {x.step: x.value for x in ea.Scalars(tag)} if tag in tags else {}

succ = series("task/episode_success_rate")
rew = series("reward/ep_reward_mean")
fr = series("reward/r_first_reach_count")
steps = sorted(rew)

print("run:", os.path.basename(d))
print(f"{'step':>12} {'reward':>9} {'success':>8} {'first_reach':>11}")
for s in steps[-12:]:
    print(f"{s:>12,} {rew.get(s, float('nan')):>9.2f} {succ.get(s, 0):>8.3f} {fr.get(s, 0):>11.0f}")

# ── 진단: 100% 못 채우는 원인 (성공 vs 실패 분해). 최근값 평균(마지막 ~5 로깅). ──
def last_mean(tag, n=5):
    v = ea.Scalars(tag) if tag in tags else []
    return (sum(x.value for x in v[-n:]) / min(len(v), n)) if v else None

print("\n── 진단 (SUCCESS vs FAIL 에피소드, 최근 평균) ──")
have = any(t.startswith("diag/manip_min_") for t in tags)
if not have:
    print("  (이 런엔 diag/* 진단 로깅이 없음 — 진단 커밋 이후 코드로 재실행 필요)")
else:
    rows = [
        ("manipulability(특이점) min", "diag/manip_min_SUCCESS", "diag/manip_min_FAIL", "실패가 낮으면 특이점"),
        ("reach(도달거리) max [m]",     "diag/reach_max_SUCCESS", "diag/reach_max_FAIL", "실패가 높으면(→0.855) 도달불가"),
        ("out-of-reach rate",          "diag/oor_rate_SUCCESS",  "diag/oor_rate_FAIL",  "실패가 높으면 도달불가 명령"),
        ("f_int(내력) max [N]",         "diag/fint_max_SUCCESS",  "diag/fint_max_FAIL",  "실패가 높으면 내력(제어)"),
    ]
    print(f"  {'지표':<26} {'SUCCESS':>10} {'FAIL':>10}   해석")
    for name, ts, tf, interp in rows:
        vs, vf = last_mean(ts), last_mean(tf)
        ss = f"{vs:>10.4f}" if vs is not None else f"{'--':>10}"
        ff = f"{vf:>10.4f}" if vf is not None else f"{'--':>10}"
        print(f"  {name:<26} {ss} {ff}   {interp}")
