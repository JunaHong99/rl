"""
Smoke test for graph_converter.

Isaac Lab을 띄우지 않고 mock raw_state로 graph 변환만 검증.
import → 함수 호출 → shape 확인 → edge index 무결성.

사용법: python test_graph_converter.py
"""

import torch
import sys

import graph_converter as gc


def make_mock_state(B: int, device: str = "cpu"):
    """env3._get_observations()와 동일한 키들을 가진 mock dict."""
    state = {
        "robot_nodes": torch.randn(B, gc.N_ARMS, 14, device=device),         # q+dq
        "current_ee_poses": torch.cat([
            torch.randn(B, gc.N_ARMS, 3, device=device),                     # pos
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(B, gc.N_ARMS, 4),  # identity quat
        ], dim=-1),
        "ee_lin_vel": torch.zeros(B, gc.N_ARMS, 3, device=device),
        "ee_ang_vel": torch.zeros(B, gc.N_ARMS, 3, device=device),
        "wrench_panda_1": torch.zeros(B, 6, device=device),
        "wrench_panda_2": torch.zeros(B, 6, device=device),
        "rod_pos": torch.randn(B, 3, device=device),
        "rod_quat": torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(B, 4),
        "rod_lin_vel": torch.zeros(B, 3, device=device),
        "rod_ang_vel": torch.zeros(B, 3, device=device),
    }
    return state


def main():
    B = 4
    device = "cpu"
    state = make_mock_state(B, device)

    goal_pos = torch.randn(B, 3, device=device)
    goal_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(B, 4)
    target_x_rel = torch.tensor([-0.8, 0.0, 0.0], device=device).expand(B, 3)
    normalized_time = torch.linspace(0, 1, B, device=device)

    # Franka panda joint limits (대략)
    j_low = torch.tensor([[-2.9, -1.7, -2.9, -3.0, -2.9, -0.0, -2.9],
                          [-2.9, -1.7, -2.9, -3.0, -2.9, -0.0, -2.9]], device=device)
    j_high = torch.tensor([[2.9, 1.7, 2.9, 0.0, 2.9, 3.7, 2.9],
                           [2.9, 1.7, 2.9, 0.0, 2.9, 3.7, 2.9]], device=device)

    batch = gc.convert_batch_state_to_graph(
        raw_state=state,
        num_envs=B,
        goal_pos=goal_pos,
        goal_quat=goal_quat,
        target_x_rel=target_x_rel,
        normalized_time=normalized_time,
        joint_limits_low=j_low,
        joint_limits_high=j_high,
        joint_torque=None,
    )

    print("=" * 60)
    print("Graph Converter Smoke Test")
    print("=" * 60)
    print(f"  N_ARM_JOINTS={gc.N_ARM_JOINTS}, N_ARMS={gc.N_ARMS}")
    print(f"  NODES_PER_ENV={gc.NODES_PER_ENV}, N_EDGES_PER_ENV={gc.N_EDGES_PER_ENV}")
    print(f"  NODE_FEATURE_DIM={gc.NODE_FEATURE_DIM}, EDGE_FEATURE_DIM={gc.EDGE_FEATURE_DIM}")
    print(f"  GLOBAL_FEATURE_DIM={gc.GLOBAL_FEATURE_DIM}")
    print()
    print(f"Output (B={B}):")
    print(f"  x.shape:          {tuple(batch.x.shape)}  expected ({B * gc.NODES_PER_ENV}, {gc.NODE_FEATURE_DIM})")
    print(f"  edge_index.shape: {tuple(batch.edge_index.shape)}  expected (2, {B * gc.N_EDGES_PER_ENV})")
    print(f"  edge_attr.shape:  {tuple(batch.edge_attr.shape)}  expected ({B * gc.N_EDGES_PER_ENV}, {gc.EDGE_FEATURE_DIM})")
    print(f"  u.shape:          {tuple(batch.u.shape)}  expected ({B}, {gc.GLOBAL_FEATURE_DIM})")
    print(f"  batch.shape:      {tuple(batch.batch.shape)}  expected ({B * gc.NODES_PER_ENV},)")

    # 무결성 체크
    expected_x = (B * gc.NODES_PER_ENV, gc.NODE_FEATURE_DIM)
    expected_e = (2, B * gc.N_EDGES_PER_ENV)
    expected_u = (B, gc.GLOBAL_FEATURE_DIM)
    assert tuple(batch.x.shape) == expected_x, f"x shape mismatch: {batch.x.shape}"
    assert tuple(batch.edge_index.shape) == expected_e, f"edge_index shape mismatch"
    assert tuple(batch.edge_attr.shape) == (B * gc.N_EDGES_PER_ENV, gc.EDGE_FEATURE_DIM)
    assert tuple(batch.u.shape) == expected_u

    # edge_index 범위 체크
    assert batch.edge_index.min() >= 0
    assert batch.edge_index.max() < B * gc.NODES_PER_ENV

    # batch index 분포
    counts = torch.bincount(batch.batch)
    assert counts.shape[0] == B and (counts == gc.NODES_PER_ENV).all()

    # NaN 체크
    assert not torch.isnan(batch.x).any(), "NaN in x"
    assert not torch.isnan(batch.edge_attr).any(), "NaN in edge_attr"
    assert not torch.isnan(batch.u).any(), "NaN in u"

    # Edge type one-hot 분포 (5-node: kinematic, grasp, cooperative, proximity)
    edge_types = batch.edge_attr.argmax(dim=-1)
    kinematic = (edge_types == 0).sum().item()
    grasp = (edge_types == 1).sum().item()
    cooperative = (edge_types == 2).sum().item()
    proximity = (edge_types == 3).sum().item()
    print()
    print(f"Edge type distribution (B={B}):")
    print(f"  kinematic:   {kinematic}  ({kinematic // B}/env)   expected 4/env (Robot↔EE ×2 dir × 2 arms)")
    print(f"  grasp:       {grasp}  ({grasp // B}/env)   expected 4/env (EE↔Rod ×2 dir × 2 EE)")
    print(f"  cooperative: {cooperative}  ({cooperative // B}/env)   expected 2/env (EE1↔EE2)")
    print(f"  proximity:   {proximity}  ({proximity // B}/env)   expected 0 (Phase 4)")

    print()
    print("✅ Graph converter checks passed")

    # ──────────────────────────────────────────────────────────────
    # GNN policy smoke test
    # ──────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("GNN Policy Smoke Test")
    print("=" * 60)

    import gnn_policy

    ac = gnn_policy.GNNActorCritic(action_dim=6, num_rounds=2, action_scale=0.01).to(device)
    n_params = sum(p.numel() for p in ac.parameters())
    print(f"Total params: {n_params:,}")

    # Forward
    action, log_prob, entropy, value = ac.get_action_and_value(batch, deterministic=False)
    print(f"  action.shape:    {tuple(action.shape)}  expected ({B}, 6)")
    print(f"  log_prob.shape:  {tuple(log_prob.shape)}  expected ({B}, 1)")
    print(f"  entropy:         {'None (squashed Gaussian)' if entropy is None else tuple(entropy.shape)}")
    print(f"  value.shape:     {tuple(value.shape)}  expected ({B}, 1)")
    print(f"  action range:    [{action.min().item():.4f}, {action.max().item():.4f}]  "
          f"(squashed: |a| ≤ action_scale={0.01})")
    print(f"  log_prob mean:   {log_prob.mean().item():.3f}  (squashed: 보통 음수 -2~-10)")

    assert tuple(action.shape) == (B, 6)
    assert tuple(log_prob.shape) == (B, 1)
    assert tuple(value.shape) == (B, 1)
    # Squashed Gaussian: action ∈ (-action_scale, action_scale) 엄격히 보장.
    assert action.abs().max().item() <= 0.01 + 1e-5, "action exceeds action_scale (squashed should bound)"
    # log_prob은 squashed Gaussian에서 정상 범위 (-Inf ~ 양의 작은 수)
    assert torch.isfinite(log_prob).all(), "log_prob has NaN/Inf"

    # Evaluate (PPO update path)
    log_prob_eval, entropy_eval, value_eval = ac.evaluate(batch, action)
    assert tuple(log_prob_eval.shape) == (B, 1)
    print()
    print(f"  evaluate() log_prob mean abs diff: {(log_prob_eval - log_prob).abs().mean().item():.2e}")
    # 같은 input + 같은 action이면 같은 log_prob (deterministic re-forward)
    # 단 actor가 다시 forward 통과해서 약간의 numerical noise 가능

    # Backprop check
    loss = (-log_prob_eval).mean() + value_eval.pow(2).mean()
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in ac.parameters() if p.grad is not None)
    print(f"  total grad norm after backward: {grad_norm:.2f}")

    print()
    print("✅ GNN policy checks passed")


if __name__ == "__main__":
    main()
