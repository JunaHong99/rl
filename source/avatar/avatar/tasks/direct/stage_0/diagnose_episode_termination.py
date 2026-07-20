"""
Episode 조기 종료 원인 진단 (Phase 3.3 debug)

학습 중 ep_length_mean = 99인데 우리 코드는 success/timeout(1200)만 종료시킴.
어떤 조건이 99 step에서 episode를 끝내는지 추적.

방법:
  - num_envs=16 환경 생성
  - 무작위 action으로 600 step 실행
  - 매 step env._get_dones() 결과 확인
  - terminated/truncated/도중에 reset이 일어났는지 episode_length_buf 추적
  - reset 직전 상태 (rod_pos, joint_pos, joint limits) 출력
"""

import argparse
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Episode termination diagnostic")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=600)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from dual_arm_transport_env3 import DualrobotEnv, DualrobotCfg


def main():
    env_cfg = DualrobotCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = DualrobotEnv(cfg=env_cfg, render_mode=None)

    print("=" * 80)
    print(f"Diagnostic: {args.num_envs} envs × {args.steps} steps with random action")
    print(f"  max_episode_length (cfg) = {env.max_episode_length}")
    print(f"  episode_length_s (cfg) = {env.cfg.episode_length_s}")
    print(f"  decimation = {env.cfg.decimation}")
    print(f"  control freq ≈ {1.0 / (env.cfg.sim.dt * env.cfg.decimation):.1f} Hz")
    print("=" * 80)

    # 이전 episode 길이 추적
    prev_len = torch.zeros(args.num_envs, dtype=torch.long, device=env.device)
    ep_lens = []
    ep_term_reasons = []  # ('success', 'timeout', 'mystery')

    obs, _ = env.reset()

    for step in range(args.steps):
        # Small random action (학습 초기 noise scale과 비슷)
        action = 0.005 * (2 * torch.rand(args.num_envs, env.cfg.action_space, device=env.device) - 1)

        cur_len_before = env.episode_length_buf.clone()
        obs, rew, terminated, truncated, info = env.step(action)
        done = terminated | truncated

        # 어떤 env가 reset됐는지 (env.episode_length_buf가 줄어들었으면 reset됨)
        cur_len_after = env.episode_length_buf
        # 정상 step: len 1 증가. Reset: len이 0으로 돌아감 (혹은 1)
        is_reset = (cur_len_after < cur_len_before) | done

        if done.any() or is_reset.any():
            for i in range(args.num_envs):
                if done[i] or is_reset[i]:
                    actual_len = cur_len_before[i].item() + 1
                    reason = None
                    if terminated[i].item():
                        reason = "success"
                    elif truncated[i].item():
                        reason = "timeout"
                    else:
                        reason = "mystery"  # done이 아닌데 reset?
                    ep_lens.append(actual_len)
                    ep_term_reasons.append(reason)

        if step % 100 == 0 and step > 0:
            print(f"  step {step}: total finished episodes = {len(ep_lens)}")

    print()
    print("=" * 80)
    print("Termination 분석")
    print("=" * 80)
    if not ep_lens:
        print("종료된 episode 없음 (예상치 못함)")
    else:
        import numpy as np
        lens = np.array(ep_lens)
        print(f"총 종료 episode: {len(ep_lens)}")
        print(f"길이 분포: mean={lens.mean():.1f}  min={lens.min()}  max={lens.max()}  median={int(np.median(lens))}")
        print()

        from collections import Counter
        reason_counts = Counter(ep_term_reasons)
        print("Termination 이유:")
        for r, c in reason_counts.items():
            print(f"  {r:<10s}: {c} ({100*c/len(ep_lens):.1f}%)")
        print()

        # Histogram by 50-step bins
        bins = [0, 50, 100, 150, 200, 300, 500, 1000, 1500]
        hist, _ = np.histogram(lens, bins=bins)
        print("길이 히스토그램:")
        for b, h in zip(bins[:-1], hist):
            bar = "█" * min(40, h)
            print(f"  [{b:>4}-{b+50 if b<200 else 'next':<4}]  {h:>4d}  {bar}")

    # 최종 상태 진단
    print()
    print("=" * 80)
    print("마지막 step 시점 상태")
    print("=" * 80)
    rod_pos = env.rod.data.root_pos_w
    print(f"Rod position (world):  min={rod_pos.min(dim=0)[0].cpu().tolist()}")
    print(f"                       max={rod_pos.max(dim=0)[0].cpu().tolist()}")

    q1 = env.robot_1.data.joint_pos[:, env.robot_1_joint_ids]
    q2 = env.robot_2.data.joint_pos[:, env.robot_2_joint_ids]
    print(f"Robot_1 joint q:       min={q1.min(dim=0)[0].cpu().tolist()}")
    print(f"                       max={q1.max(dim=0)[0].cpu().tolist()}")

    # 자세한 한 env 분석
    print()
    print("Env 0 상태:")
    print(f"  episode_length_buf = {env.episode_length_buf[0].item()}")
    print(f"  rod_pos = {rod_pos[0].cpu().tolist()}")
    print(f"  rod_vel_lin = {env.rod.data.root_lin_vel_w[0].cpu().tolist()}")
    print(f"  Robot_1 q = {q1[0].cpu().tolist()}")
    print(f"  Robot_1 limits low  = {env.robot_1.data.soft_joint_pos_limits[0, env.robot_1_joint_ids, 0].cpu().tolist()}")
    print(f"  Robot_1 limits high = {env.robot_1.data.soft_joint_pos_limits[0, env.robot_1_joint_ids, 1].cpu().tolist()}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
