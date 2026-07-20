"""
Rotation channel energy injection 진단.

매 RL step마다 다음을 로깅:
  - rod_quat, target_quat
  - q_err (특히 w 성분 부호)
  - rot_err_aa (axis-angle 3D)
  - m_abs_ang (desired torque on rod)
  - rod_ang_vel
  - Power = m_abs_ang · rod_ang_vel
    > 0 → 에너지 주입 (BUG)
    < 0 → 에너지 발산 (정상 damping)

이걸로 어느 시점에 에너지가 주입되는지 정확히 시각 가능.

사용:
    python test_rotation_energy.py --num_envs 4 --num_steps 100 --yaw_deg 10
"""
import argparse
import math
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_steps", type=int, default=100)
parser.add_argument("--yaw_deg", type=float, default=10.0)
parser.add_argument("--target_env", type=int, default=0, help="자세히 볼 env index")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.utils.math as math_utils
from dual_arm_transport_env3 import DualrobotEnv, DualrobotCfg


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = DualrobotCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.episode_length_s = args.num_steps * cfg.sim.dt * cfg.decimation + 5.0
    env = DualrobotEnv(cfg=cfg, render_mode=None)

    dt_rl = cfg.sim.dt * cfg.decimation
    print(f"\n== Rotation Energy Diagnostic ==")
    print(f"  num_envs={args.num_envs}  num_steps={args.num_steps}  yaw={args.yaw_deg}°")
    print(f"  decimation={cfg.decimation}  RL dt={dt_rl*1000:.1f} ms\n")

    obs, _ = env.reset()
    rod_init_pos = env.rod.data.root_pos_w.clone()
    rod_init_quat = env.rod.data.root_quat_w.clone()
    # target rotation
    yaw_rad = args.yaw_deg * math.pi / 180.0
    half = yaw_rad * 0.5
    cw = math.cos(half); sw = math.sin(half)
    q_delta = torch.tensor([cw, 0.0, 0.0, sw], device=device).unsqueeze(0).expand(args.num_envs, -1)
    target_quat = math_utils.quat_mul(q_delta, rod_init_quat)
    target_quat = target_quat / torch.norm(target_quat, dim=-1, keepdim=True)

    with torch.no_grad():
        env.target_obj_pos.copy_(rod_init_pos)
        env.target_obj_quat.copy_(target_quat)

    print(f"{'step':>4s} {'t_ms':>6s} | "
          f"{'rod_qw':>7s} {'rod_qx':>7s} {'rod_qy':>7s} {'rod_qz':>7s} | "
          f"{'aa_x':>7s} {'aa_y':>7s} {'aa_z':>7s} | "
          f"{'ω_x':>7s} {'ω_y':>7s} {'ω_z':>7s} | "
          f"{'P':>9s}")
    print("-" * 130)

    e = args.target_env
    power_history = []
    a_zero = torch.zeros(args.num_envs, 6, device=device)
    for step in range(args.num_steps):
        env.step(a_zero)
        # Override target every step (accumulating mode가 target 0 action으론 안 바꾸지만 안전)
        with torch.no_grad():
            env.target_obj_pos.copy_(rod_init_pos)
            env.target_obj_quat.copy_(target_quat)

        # 데이터 추출
        rod_q = env.rod.data.root_quat_w[e]                # (4,)
        rod_w = env.rod.data.root_ang_vel_w[e]             # (3,)
        # q_err = target * conj(rod)
        rod_q_b = rod_q.unsqueeze(0)
        target_q_b = target_quat[e].unsqueeze(0)
        rod_conj = math_utils.quat_conjugate(rod_q_b)
        q_err = math_utils.quat_mul(target_q_b, rod_conj)[0]
        # axis-angle (controller와 동일 로직 — double cover)
        sign = torch.sign(q_err[0:1])
        if sign.item() == 0: sign = torch.tensor([1.0], device=device)
        q_signed = q_err * sign
        v = q_signed[1:4]
        w_pos = q_signed[0].clamp(min=-1.0, max=1.0)
        v_norm = torch.norm(v)
        angle = 2.0 * torch.atan2(v_norm, w_pos)
        axis = v / (v_norm + 1e-8)
        rot_aa = axis * angle
        # m_abs_ang (controller 같은 공식)
        m_abs_ang = 20.0 * rot_aa - 8.0 * rod_w
        # Power
        P = torch.dot(m_abs_ang, rod_w).item()
        sign_str = "+" if P > 1e-6 else ("-" if P < -1e-6 else "0")
        power_history.append(P)

        if step % 3 == 0 or step < 15:
            print(f"{step:>4d} {step*dt_rl*1000:>6.0f} | "
                  f"{rod_q[0].item():>+.4f} {rod_q[1].item():>+.4f} {rod_q[2].item():>+.4f} {rod_q[3].item():>+.4f} | "
                  f"{rot_aa[0].item():>+.4f} {rot_aa[1].item():>+.4f} {rot_aa[2].item():>+.4f} | "
                  f"{rod_w[0].item():>+.4f} {rod_w[1].item():>+.4f} {rod_w[2].item():>+.4f} | "
                  f"{P:>+.4f}")

    # 통계
    P_arr = torch.tensor(power_history)
    pos_count = (P_arr > 1e-6).sum().item()
    neg_count = (P_arr < -1e-6).sum().item()
    print(f"\n== Power statistics ==")
    print(f"  Positive (energy injection): {pos_count}/{len(P_arr)} ({100*pos_count/len(P_arr):.1f}%)")
    print(f"  Negative (energy dissipation): {neg_count}/{len(P_arr)} ({100*neg_count/len(P_arr):.1f}%)")
    print(f"  Net work: {P_arr.sum().item() * dt_rl:.4f} J")
    print(f"  Max positive: {P_arr.max().item():.4f}, Max negative: {P_arr.min().item():.4f}")
    if P_arr.sum() > 0:
        print(f"  → 시스템에 NET 에너지 주입 — controller bug 확정")
    else:
        print(f"  → 시스템에서 NET 에너지 소산 — controller 정상이나 oscillation 큼")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
