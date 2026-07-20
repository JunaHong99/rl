# train.py

import torch
import numpy as np
import os
import argparse
from datetime import datetime
from collections import deque
from torch_geometric.data import Batch
from torch.utils.tensorboard import SummaryWriter

# Isaac Lab Imports
from isaaclab.app import AppLauncher

# [중요] argparse 설정을 AppLauncher보다 먼저 해야 함
parser = argparse.ArgumentParser(description="Train RoboBallet Agent")
# 학습 속도와 안정성을 위해 환경 수와 반복 횟수를 늘렸습니다.
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--max_iterations", type=int, default=500000, help="Total training iterations")
parser.add_argument("--resume_path", type=str, default=None, help="Path to checkpoint prefix (e.g. logs/.../model_step_50000)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 앱 실행
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 나머지 모듈 임포트 (App 실행 후)
from dual_arm_transport_env3 import DualrobotEnv, DualrobotCfg
from graph_converter import (
    convert_batch_state_to_graph,
    NODE_FEATURE_DIM, EDGE_FEATURE_DIM, GLOBAL_FEATURE_DIM
)
from replay_buffer import VectorizedGraphReplayBuffer
from agent import TD3

# ---------------------------------------------------------
# 2. Main Training Loop
# ---------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training Start on {device}")

    # --- A. 환경 및 로깅 초기화 ---
    run_name = f"roboballet_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    script_dir = os.path.dirname(os.path.abspath(__file__))  # train.py가 있는 폴더
    #log_dir = os.path.join(script_dir, "logs", run_name)     # 그 안에 logs 생성
    if args_cli.resume_path:
        # 예: args_cli.resume_path = ".../logs/roboballet_2025.../model_step_50000"
        # os.path.dirname을 하면 ".../logs/roboballet_2025..." 폴더 경로가 나옵니다.
        log_dir = os.path.dirname(args_cli.resume_path)
        print(f"📂 Resuming logging into EXISTING directory: {log_dir}")
    else:
        # 기존 로직 (새 폴더 생성)
        run_name = f"roboballet_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        log_dir = os.path.join(script_dir, "logs", run_name)
        print(f"📂 Creating NEW log directory: {log_dir}")
    writer = SummaryWriter(log_dir)
    
    # 성공률 계산을 위한 이동 평균 버퍼 -> 에피소드가 끝날 때만 이 버퍼에 저장됨. 총 stats_buffer_size 개의 에피소드를 저장
    # 최소 2000개, 혹은 환경 수의 5배 중 큰 값으로 설정
    stats_buffer_size = max(2000, args_cli.num_envs * 5)
    success_buffer = deque(maxlen=stats_buffer_size)

    env_cfg = DualrobotCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = DualrobotEnv(cfg=env_cfg, render_mode=None) # 학습용이라 렌더링 끔, "human"

    # --- B. 에이전트 및 버퍼 초기화 ---
    gnn_params = {
        'node_dim': NODE_FEATURE_DIM,     # 14
        'edge_dim': EDGE_FEATURE_DIM,     # 9
        'global_dim': GLOBAL_FEATURE_DIM, # 19
        'action_dim': 7                   # 로봇 1대당 액션
    }
    
    # 학습률 등 하이퍼파라미터 설정
    agent = TD3(gnn_params=gnn_params, max_action=1.0, lr=5e-5) #5e-5면 로보발레와 동일한 수치, batch size 1024로 두 배 키웠으므로 lr도 2배
    
    # --- C. 학습 루프 시작 ---
    obs_dict, _ = env.reset()
    
    # [Vectorized] 초기 그래프 생성 (GPU 유지)
    current_batch_graph = convert_batch_state_to_graph(obs_dict['policy'], args_cli.num_envs)
    
    # [NEW] Vectorized Buffer 초기화 (템플릿 그래프 필요하므로 여기서 초기화)
    # 메모리f 사용량 주의: (1M * Nodes * Dim * 4bytes) -> 1M * 4 * 14 * 4 ~= 224MB (매우 작음)
    # GPU 메모리가 넉넉하다면 device="cuda" 권장
    buffer = VectorizedGraphReplayBuffer(
        capacity= 60000000, #4096일 때 4천만
        num_envs=args_cli.num_envs,
        node_dim=NODE_FEATURE_DIM,
        edge_dim=EDGE_FEATURE_DIM,
        global_dim=GLOBAL_FEATURE_DIM,
        action_dim=14, # 2 Robots * 7
        template_graph=current_batch_graph,
        device="cuda" # GPU 저장
    )

    start_step = 0  # 기본 시작 스텝

    if args_cli.resume_path is not None:
        if os.path.exists(args_cli.resume_path + "_actor"): # 파일 존재 확인
            print(f"🔄 Resuming training from: {args_cli.resume_path}")
            
            # 1. 모델 및 옵티마이저 불러오기
            agent.load(args_cli.resume_path)
            
            # 2. 파일명에서 스텝 수 추출 (예: "model_step_50000" -> 50000)
            try:
                # 경로의 맨 뒤 파일명만 가져옴 -> '_'로 분리 -> 마지막 숫자 파싱
                filename = os.path.basename(args_cli.resume_path) 
                start_step = int(filename.split('_')[-1])
                print(f"⏩ Start Step updated to: {start_step}")
            except Exception as e:
                print(f"⚠️ Could not parse step from filename. Starting from 0. Error: {e}")
        else:
            print(f"❌ Checkpoint not found at {args_cli.resume_path}. Starting from scratch.")

    print(f"🔄 Start Interaction Loop ({args_cli.max_iterations} steps)...")
    print(f"📂 Logs will be saved to: {log_dir}")

    MAX_EPISODE_STEPS = 1200
    WARMUP_STEPS = MAX_EPISODE_STEPS *2

    for step in range(start_step, args_cli.max_iterations):
        
        # -------------------------------------------------
        # 1. Action Selection (GNN Inference)
        # -------------------------------------------------
        # [Vectorized] 이미 Batch 객체이므로 바로 사용 가능
        if step < WARMUP_STEPS:
            # -0.5 ~ 0.5 사이의 균등 분포 랜덤 액션 (계산 비용 0)
            env_actions_tensor = 1 * torch.rand(args_cli.num_envs, 14, device=device) - 0.5
        else:   
            agent.actor.eval()
            with torch.no_grad():
                # GNN 출력: [Total_Nodes, 7] 
                full_actions = agent.actor(current_batch_graph)
                
                # [수정 제안: 동적 계산] -----------------------------------------
                # 1. 전체 노드 수와 환경 수로 '그래프당 노드 수'를 역산합니다.
                #    이렇게 하면 장애물이나 태스크가 늘어나서 노드 수가 4개가 아니게 되어도 코드가 작동합니다.
                total_nodes = full_actions.shape[0]
                num_envs = args_cli.num_envs
                
                # 산술 검증 (Total Nodes는 반드시 Num Envs의 배수여야 함)
                assert total_nodes % num_envs == 0, f"Node mismatch: {total_nodes} nodes for {num_envs} envs"
                
                num_nodes_per_env = total_nodes // num_envs  # 예: 4, 5, 6... 등으로 자동 계산됨

                # 2. [Num_Envs, Node_Per_Env, Action_Dim] 형태로 변환
                reshaped_actions = full_actions.view(num_envs, num_nodes_per_env, -1)
                
                # 3. 로봇 노드만 슬라이싱
                # (주의: DualArm 환경이므로 로봇은 항상 2대라고 가정하거나, 
                #  env 설정에서 가져오는 변수(env.num_robots 등)를 사용하는 것이 좋습니다)
                num_robots = 2 
                
                # graph_converter에서 로봇 노드를 0, 1번 인덱스에 넣었으므로 앞부분을 가져옵니다.
                robot_actions = reshaped_actions[:, :num_robots, :] # [Num_Envs, 2, 7]
                
                # 4. 환경 입력용 플래튼 [Num_Envs, 14]
                env_actions_tensor = robot_actions.reshape(num_envs, -1)
                # -------------------------------------------------------------
                
                # Exploration Noise 추가
                noise = torch.randn_like(env_actions_tensor) * 0.1
                env_actions_tensor = (env_actions_tensor + noise).clamp(-1.0, 1.0)

            agent.actor.train()

        # -------------------------------------------------
        # 2. Environment Step
        # -------------------------------------------------
        next_obs_dict, rewards, terminated, truncated, extras = env.step(env_actions_tensor)
        
        # 리플레이 버퍼 저장을 위해 terminated와 truncated를 합쳐서 하나의 done으로 만듭니다.
        dones = terminated | truncated
        
        # -------------------------------------------------
        # 3. Data Handling (Convert & Buffer)
        # -------------------------------------------------
        # [Vectorized] Next State 변환 (GPU 유지)
        next_batch_graph = convert_batch_state_to_graph(next_obs_dict['policy'], args_cli.num_envs)
        
        # [Buffer 저장] Vectorized Buffer에 GPU Tensor 그대로 투입 (매우 빠름)
        buffer.add_batch(
            state_batch=current_batch_graph, 
            action=env_actions_tensor,
            next_state_batch=next_batch_graph,
            reward=rewards,
            done=dones
        )
        
        # 상태 업데이트 (GPU 객체 그대로 넘김)
        current_batch_graph = next_batch_graph
        
        # -------------------------------------------------
        # 4. Train Agent
        # -------------------------------------------------
        # 버퍼가 어느 정도 차면 학습 시작 (배치 사이즈 256 권장)
        # [수정] 4096 환경 등 대규모 병렬 처리 시 데이터가 빨리 차므로 워밍업을 줄이고, 업데이트 횟수를 늘림
        TARGET_SIR = 8   
        BATCH_SIZE = 1024 
        if step >= WARMUP_STEPS:
            gradient_steps = max(1, int((args_cli.num_envs * TARGET_SIR) / BATCH_SIZE ))
            for _ in range(gradient_steps): #버퍼에서 256개의 데이터를 뽑아 업데이트, 이걸 그라디언트 스텝만큼 반복.
                agent.train(buffer, batch_size=BATCH_SIZE)

        # -------------------------------------------------
        # 5. Logging (TensorBoard)
        # -------------------------------------------------
        # 성공률 집계 (끝난 에피소드가 있는 경우만)
        if dones.any():
            # extras에서 성공 여부 가져오기 ('log/success' 키가 있다고 가정)
            done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
            
            if "log/success" in extras:
                success_vals = extras["log/success"][done_indices].cpu().numpy()
                success_buffer.extend(success_vals)

        # 주기적 기록 (매 100 스텝)
        if step % 100 == 0:
            mean_reward = torch.mean(rewards).item()

            # (2) 리워드 성분별 기록 (수정됨)
            # extras에 키가 존재하는지 확인 후 기록
            if "log/total_reward" in extras:
                writer.add_scalar("Reward/Constraint", extras["log/r_constraint"].item(), step)
                writer.add_scalar("Reward/Action", extras["log/r_action"].item(), step)

            # [CHANGED] Log potential reward instead of absolute distance reward
            if "log/r_potential" in extras:
                writer.add_scalar("Reward/Potential", extras["log/r_potential"].item(), step)
            elif "log/r_dist" in extras: # Fallback if using old env
                writer.add_scalar("Reward/Distance", extras["log/r_dist"].item(), step)


            # (3) 디버깅용 에러 기록
            if "log/err_pos" in extras:
                writer.add_scalar("Error/position", extras["log/err_pos"].item(), step)
                writer.add_scalar("Error/rotation", extras["log/err_rot"].item(), step)
            
            # [NEW] 추가된 성능 지표 로깅
            if "log/max_err_pos" in extras:
                # Note: This logs the MEAN of the "Episode Max Errors" across envs.
                # Ideally, to detect if ANY violation occurred, we check if this value > threshold.
                writer.add_scalar("Error/Max_Position", extras["log/max_err_pos"].item(), step)
                writer.add_scalar("Error/Max_Rotation", extras["log/max_err_rot"].item(), step)
            if "log/violation_ratio" in extras:
                # This is the Ratio of envs currently violating. 0.0 means PERFECT safety.
                writer.add_scalar("Rollout/ViolationRatio", extras["log/violation_ratio"].item(), step)

            # (4) 성공률 기록
            if len(success_buffer) > 0:
                success_rate = np.mean(success_buffer)
                writer.add_scalar("Rollout/SuccessRate", success_rate, step)
            else:
                success_rate = 0.0

            # 콘솔 출력
            print(f"[Step {step}/{args_cli.max_iterations}] "
                  f"Rew: {mean_reward:.4f} | "
                  f"Succ: {success_rate:.1%} | "
                  f"Buff: {len(buffer)}")
            
            # 모델 저장 (매 5000 스텝)
            if step % 5000 == 0:
                save_path = os.path.join(log_dir, f"model_step_{step}")
                agent.save(save_path)

    print("✅ Training Finished!")
    writer.close()
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    # 실행 시 인자를 바꿀 수 있음 (예: python train.py --num_envs 512)
    main()