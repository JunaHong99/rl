"""
Phase A (rod+global, 36-dim) MLP 체크포인트 → Phase B (lean obstacle, 56-dim) 초기 가중치 변환.

2-phase 전략의 핵심:
  Phase B 망의 1층에서 rod·global(·action) 가중치는 Phase A에서 복사하고,
  장애물 입력 컬럼만 0으로 초기화한다 → Phase B 시작 순간 출력이 Phase A와 정확히 동일.
  (=발견된 base transport 보존. 장애물 정보는 학습되며 점진 유입. zero-init adapter trick.)

증류 아님: 자기 가중치를 이어서 새 능력(장애물 회피)을 얹는 warm-start.

사용:
  python convert_phaseA_to_phaseB.py --src <phaseA model_final.pt> --dst <phaseB_init.pt>
"""
import argparse, torch
import mlp_policy
import graph_converter as gc

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="Phase A 체크포인트 (rod+global, 36-dim)")
ap.add_argument("--dst", required=True, help="출력 Phase B 초기 체크포인트 (lean, 56-dim)")
ap.add_argument("--action_dim", type=int, default=6)
args = ap.parse_args()

ROD = gc.NODE_FEATURE_DIM            # 32
OBS = 5 * gc.N_OBSTACLES            # 20  (rel_pos3 + dist1 + radius1) × N_OBS
GLB = gc.GLOBAL_FEATURE_DIM         # 4
A = args.action_dim

dimA = mlp_policy._state_dim(False, False)        # 36 = ROD+GLB
dimB = mlp_policy._state_dim(False, True)          # 56 = ROD+OBS+GLB
assert dimA == ROD + GLB and dimB == ROD + OBS + GLB, (dimA, dimB, ROD, OBS, GLB)

ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
sdA = ckpt["model"]

# 입력차원 확인 (Phase A는 반드시 rod+global)
in_actor = sdA["actor.mean_head.0.weight"].shape[1]
assert in_actor == dimA, f"src actor in_dim={in_actor} != {dimA}. Phase A(rod+global)가 아님."

agentB = mlp_policy.MLPSACAgent(action_dim=A, use_lean_obstacle=True)
sdB = agentB.state_dict()

def map_state_cols(wA, wB, has_action):
    """1층 weight (out, in)을 phaseA→phaseB 컬럼 매핑. obs 컬럼은 0(=sdB 초기값 유지하지 않고 명시 0)."""
    wB = wB.clone()
    # rod: 0:ROD → 0:ROD
    wB[:, 0:ROD] = wA[:, 0:ROD]
    # obs: ROD:ROD+OBS = 0
    wB[:, ROD:ROD+OBS] = 0.0
    # global: A는 ROD:ROD+GLB,  B는 ROD+OBS:ROD+OBS+GLB
    wB[:, ROD+OBS:ROD+OBS+GLB] = wA[:, ROD:ROD+GLB]
    if has_action:
        # action: A는 ROD+GLB:ROD+GLB+act, B는 ROD+OBS+GLB:...+act
        wB[:, ROD+OBS+GLB:ROD+OBS+GLB+A] = wA[:, ROD+GLB:ROD+GLB+A]
    return wB

FIRST_ACTOR = ["actor.mean_head.0.weight"]
FIRST_Q = ["q.q1.q_head.0.weight", "q.q2.q_head.0.weight",
           "q_target.q1.q_head.0.weight", "q_target.q2.q_head.0.weight"]

n_copied, n_surgery = 0, 0
for k in sdB.keys():
    if k in FIRST_ACTOR:
        sdB[k] = map_state_cols(sdA[k], sdB[k], has_action=False); n_surgery += 1
    elif k in FIRST_Q:
        sdB[k] = map_state_cols(sdA[k], sdB[k], has_action=True); n_surgery += 1
    elif k in sdA and sdA[k].shape == sdB[k].shape:
        sdB[k] = sdA[k].clone(); n_copied += 1
    else:
        # shape 불일치인데 first-layer가 아닌 경우 = 예상 못한 키. 경고.
        print(f"  ⚠️  미매칭 키 (sdB 초기값 유지): {k}  A={sdA.get(k,'-') if k not in sdA else tuple(sdA[k].shape)} B={tuple(sdB[k].shape)}")

agentB.load_state_dict(sdB)
torch.save({"model": agentB.state_dict(), "env_steps": 0,
            "note": f"phaseB zero-init warm-start from {args.src}"}, args.dst)
print(f"✅ surgery {n_surgery}개 (1층 actor/Q), 직접복사 {n_copied}개")
print(f"✅ saved Phase B init → {args.dst}  (actor_in {dimA}→{dimB})")
