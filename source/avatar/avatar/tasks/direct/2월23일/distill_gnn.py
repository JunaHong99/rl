"""
BC 증류 (DAgger): 작동하는 MLP expert → GNN student.

배경:
  - GNN은 supervised로는 완벽 학습 가능(프로브 확인)하나 online SAC에선 부트스트랩 죽은 루프로 실패.
  - 따라서 SAC가 학습한 MLP expert를 GNN에 supervised 증류해 task 수행을 이전한다.
  - de-risk 목표: 단일 rod에서 GNN student가 expert급(~90%) success 재현.

DAgger:
  - seed 라운드: expert를 굴려 on-distribution 상태 수집.
  - 이후 라운드: student를 (확률적으로) 굴려 student가 실제 방문하는 상태 수집 → distribution shift 보정.
  - 매 방문 상태를 expert의 mean_raw(pre-squash)로 라벨링 → MSE 회귀.

사용:
  python -u distill_gnn.py --num_envs 1024 --rounds 60 --headless \
      --expert_path logs/phase3_sac_20260610-154019/model_final.pt
"""

import argparse
import os
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="BC distillation MLP->GNN (DAgger)")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--expert_path", type=str,
                    default="logs/phase3_sac_20260610-154019/model_final.pt")
parser.add_argument("--rounds", type=int, default=60, help="DAgger 라운드 수")
parser.add_argument("--seed_rounds", type=int, default=3,
                    help="처음 N 라운드는 expert를 굴려 on-distribution 시드")
parser.add_argument("--collect_steps", type=int, default=25,
                    help="라운드당 env step 수 (× num_envs transition 수집)")
parser.add_argument("--grad_steps", type=int, default=400, help="라운드당 gradient update")
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--buffer_cap", type=int, default=1_500_000)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--num_rounds", type=int, default=2, help="GNN message-passing rounds")
parser.add_argument("--eval_every", type=int, default=5, help="N 라운드마다 success eval")
parser.add_argument("--eval_steps", type=int, default=400)
parser.add_argument("--target_clip", type=float, default=5.0, help="expert mean_raw 타깃 clip")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import math
from dual_arm_transport_env3 import DualrobotEnv, DualrobotCfg
import gnn_policy
import mlp_policy
import graph_converter as gc
import isaaclab.utils.math as math_utils


# ──────────────────────────────────────────────────────────────────────
# Graph BC buffer — x, u, target_mean만 저장 (edge_attr는 static 템플릿이라 재구성)
# ──────────────────────────────────────────────────────────────────────
class BCGraphBuffer:
    def __init__(self, capacity, action_dim, device):
        self.capacity = capacity
        self.device = device
        N, Fn = gc.NODES_PER_ENV, gc.NODE_FEATURE_DIM
        self.x = torch.zeros(capacity, N, Fn, device=device)
        self.u = torch.zeros(capacity, gc.GLOBAL_FEATURE_DIM, device=device)
        self.tgt = torch.zeros(capacity, action_dim, device=device)
        self.ptr = 0
        self.size = 0
        # static edge templates
        self._src = torch.tensor(gc._EDGE_SRC, device=device, dtype=torch.long)
        self._dst = torch.tensor(gc._EDGE_DST, device=device, dtype=torch.long)
        etype = torch.tensor(gc._EDGE_TYPE, device=device, dtype=torch.long)
        self._edge_attr_tmpl = F.one_hot(etype, gc.EDGE_FEATURE_DIM).float()   # (E, F_edge)

    def add(self, batch, target_mean):
        """batch: PyG Batch (B graphs). target_mean: (B, action_dim) expert mean_raw."""
        B = batch.num_graphs
        N = gc.NODES_PER_ENV
        x_bnf = batch.x.view(B, N, -1)
        idxs = (self.ptr + torch.arange(B, device=self.device)) % self.capacity
        self.x[idxs] = x_bnf
        self.u[idxs] = batch.u
        self.tgt[idxs] = target_mean
        self.ptr = (self.ptr + B) % self.capacity
        self.size = min(self.size + B, self.capacity)

    def _make_batch(self, x_BNF, u_BG):
        from torch_geometric.data import Batch
        M, N = x_BNF.shape[0], gc.NODES_PER_ENV
        E = gc.N_EDGES_PER_ENV
        x_flat = x_BNF.reshape(M * N, -1)
        offsets = (torch.arange(M, device=self.device) * N).unsqueeze(-1)
        edge_index = torch.stack([
            (self._src.unsqueeze(0).expand(M, -1) + offsets).reshape(-1),
            (self._dst.unsqueeze(0).expand(M, -1) + offsets).reshape(-1),
        ], dim=0)
        edge_attr = self._edge_attr_tmpl.unsqueeze(0).expand(M, -1, -1).reshape(M * E, -1)
        batch_idx = torch.arange(M, device=self.device).repeat_interleave(N)
        b = Batch(x=x_flat, edge_index=edge_index, edge_attr=edge_attr, u=u_BG, batch=batch_idx)
        b.num_graphs = M
        return b

    def sample(self, batch_size):
        idxs = torch.randint(0, self.size, (batch_size,), device=self.device)
        b = self._make_batch(self.x[idxs], self.u[idxs])
        return b, self.tgt[idxs]


# ──────────────────────────────────────────────────────────────────────
# Expert 로드 — checkpoint shape에서 use_full_state / action_dim / scale 추론
# ──────────────────────────────────────────────────────────────────────
def load_expert(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt["model"]
    action_dim = sd["actor.log_std"].shape[0]
    in_dim = sd["actor.mean_head.0.weight"].shape[1]
    full_in = gc.NODES_PER_ENV * gc.NODE_FEATURE_DIM + gc.GLOBAL_FEATURE_DIM
    rod_in = gc.NODE_FEATURE_DIM + gc.GLOBAL_FEATURE_DIM
    if in_dim == full_in:
        use_full = True
    elif in_dim == rod_in:
        use_full = False
    else:
        raise ValueError(f"expert mean_head in_dim {in_dim} != rod_in {rod_in} or full_in {full_in}")
    action_scale = sd["actor.action_scale"].clone().to(device)
    # hidden_dim 추론
    hidden_dim = sd["actor.mean_head.0.weight"].shape[0]
    agent = mlp_policy.MLPSACAgent(
        action_dim=action_dim, action_scale=action_scale,
        hidden_dim=hidden_dim, num_hidden_layers=2, use_full_state=use_full,
    ).to(device)
    agent.load_state_dict(sd)
    agent.eval()
    for p in agent.parameters():
        p.requires_grad = False
    print(f"✅ Expert loaded: action_dim={action_dim} use_full_state={use_full} "
          f"hidden={hidden_dim} action_scale={action_scale.tolist()}")
    return agent, action_dim, action_scale, ckpt.get("env_steps", -1)


# ──────────────────────────────────────────────────────────────────────
# Eval — eval_stage1_K 방식 success (min pos<2cm & min rot<10°, settle 제외)
# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(env, actor, num_envs, num_steps, device):
    POS_T, ROT_T = 0.02, math.radians(10)
    env.reset()
    batch = env._build_policy_batch()
    rr_min_pos = torch.full((num_envs,), float("inf"), device=device)
    rr_min_rot = torch.full((num_envs,), float("inf"), device=device)
    rr_len = torch.zeros((num_envs,), device=device, dtype=torch.long)
    settle = getattr(env, "SETTLE_STEPS_AT_RESET", 0)
    ep_succ = []
    for _ in range(num_steps):
        action, _, _ = actor.get_action_and_log_prob(batch, deterministic=True)
        _, _, terminated, truncated, _ = env.step(action)
        rod_pos = env.rod.data.root_pos_w
        goal_pos = env.goal_rod_marker.data.root_pos_w
        pos_err = torch.norm(goal_pos - rod_pos, dim=-1)
        rod_inv = math_utils.quat_conjugate(env.rod.data.root_quat_w)
        q_diff = math_utils.quat_mul(env.goal_rod_marker.data.root_quat_w, rod_inv)
        rot_err = 2.0 * torch.atan2(torch.norm(q_diff[:, 1:4], dim=-1), torch.abs(q_diff[:, 0]))
        rr_min_pos = torch.min(rr_min_pos, pos_err)
        rr_min_rot = torch.min(rr_min_rot, rot_err)
        rr_len += 1
        done = terminated | truncated
        if done.any():
            for i in done.nonzero(as_tuple=True)[0].tolist():
                if rr_len[i].item() > settle:
                    ep_succ.append((rr_min_pos[i].item() < POS_T) and (rr_min_rot[i].item() < ROT_T))
                rr_min_pos[i] = float("inf"); rr_min_rot[i] = float("inf"); rr_len[i] = 0
        batch = env._build_policy_batch()
    n = len(ep_succ)
    sr = 100.0 * sum(ep_succ) / n if n else 0.0
    return sr, n


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    expert_path = args.expert_path if os.path.isabs(args.expert_path) \
        else os.path.join(script_dir, args.expert_path)

    run_name = f"distill_gnn_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir = os.path.join(script_dir, "logs", run_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"🚀 BC distillation (DAgger) MLP→GNN  device={device}")
    print(f"📂 log_dir: {log_dir}")

    # Env
    env_cfg = DualrobotCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = DualrobotEnv(cfg=env_cfg, render_mode=None)

    # Expert
    expert, action_dim, action_scale, expert_steps = load_expert(expert_path, device)
    assert action_dim == env.cfg.action_space, \
        f"expert action_dim {action_dim} != env action_space {env.cfg.action_space}"

    # Student (GNN actor만; Q는 BC에 불필요)
    backbone = gnn_policy.GNNBackbone(num_rounds=args.num_rounds)
    student = gnn_policy.GNNActor(backbone, action_dim=action_dim,
                                  action_scale=action_scale.clone()).to(device)
    n_params = sum(p.numel() for p in student.parameters())
    print(f"🧠 GNN student actor: {n_params:,} params  action_dim={action_dim}  "
          f"num_rounds={args.num_rounds}")

    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    buf = BCGraphBuffer(args.buffer_cap, action_dim, device)

    obs, _ = env.reset()
    batch = env._build_policy_batch()
    t0 = time.time()
    total_env_steps = 0

    print("=" * 80)
    for rnd in range(args.rounds):
        use_expert_rollout = rnd < args.seed_rounds
        # ── Collect ──
        for _ in range(args.collect_steps):
            with torch.no_grad():
                expert_mean, _ = expert.actor.forward(batch)          # (B, A) label
                target = expert_mean.clamp(-args.target_clip, args.target_clip)
                buf.add(batch, target)
                if use_expert_rollout:
                    act = action_scale * torch.tanh(expert_mean)      # expert deterministic
                else:
                    # student stochastic rollout (state coverage), labeled by expert
                    act, _, _ = student.get_action_and_log_prob(batch, deterministic=False)
            env.step(act)
            batch = env._build_policy_batch()
            total_env_steps += args.num_envs

        # ── Train (BC: MSE on mean_raw) ──
        loss_sum = 0.0
        for _ in range(args.grad_steps):
            bs, tgt = buf.sample(args.batch_size)
            s_mean, _ = student.forward(bs)
            loss = F.mse_loss(s_mean, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item()
        avg_loss = loss_sum / args.grad_steps

        elapsed = (time.time() - t0) / 60
        mode = "EXPERT-roll" if use_expert_rollout else "STUDENT-roll"
        print(f"[round {rnd:>3}/{args.rounds}] {mode}  bc_loss {avg_loss:.5f}  "
              f"buf {buf.size:>8,}  env_steps {total_env_steps:>10,}  {elapsed:.1f}m", flush=True)
        writer.add_scalar("bc/loss", avg_loss, rnd)
        writer.add_scalar("bc/buffer_size", buf.size, rnd)

        # ── Eval ──
        if rnd % args.eval_every == 0 or rnd == args.rounds - 1:
            sr, n = evaluate(env, student, args.num_envs, args.eval_steps, device)
            print(f"    🎯 EVAL round {rnd}: success {sr:.1f}%  (n={n} genuine episodes)", flush=True)
            writer.add_scalar("eval/success_rate", sr, rnd)
            # re-sync rollout batch after eval reset
            obs, _ = env.reset()
            batch = env._build_policy_batch()
            torch.save({"model_actor": student.state_dict(),
                        "action_scale": action_scale.cpu(),
                        "round": rnd, "success": sr},
                       os.path.join(log_dir, f"student_round_{rnd:03d}.pt"))

    # Final save
    final_path = os.path.join(log_dir, "student_final.pt")
    torch.save({"model_actor": student.state_dict(),
                "action_scale": action_scale.cpu()}, final_path)
    sr, n = evaluate(env, student, args.num_envs, args.eval_steps, device)
    print("=" * 80)
    print(f"✅ Distillation complete. final success {sr:.1f}% (n={n})  saved {final_path}")
    print(f"   total {total_env_steps:,} env steps in {(time.time()-t0)/60:.1f}m")
    writer.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
