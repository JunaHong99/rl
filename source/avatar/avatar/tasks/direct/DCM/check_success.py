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
