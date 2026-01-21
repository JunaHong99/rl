
import torch
import time
import numpy as np
from vectorized_pose_sampler import VectorizedPoseSampler

def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Running Standalone Sampler Test on {device}...")
    
    # 1. Initialize
    try:
        sampler = VectorizedPoseSampler(device=device)
        print("✅ VectorizedPoseSampler initialized.")
    except Exception as e:
        print(f"❌ Initialization Failed: {e}")
        return

    # 2. Test Sampling with a large batch (simulate actual usage)
    num_envs = 4096
    print(f"Testing sampling for {num_envs} environments...")
    
    # Warmup
    sampler.sample_episodes(10)
    torch.cuda.synchronize()
    
    start_time = time.time()
    try:
        samples = sampler.sample_episodes(num_envs)
        torch.cuda.synchronize()
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"✅ Sampling Successful! Time taken: {duration:.4f} seconds ({duration/num_envs*1000:.4f} ms/env)")
        
    except Exception as e:
        print(f"❌ Sampling Failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Verify Outputs
    print("\nVerifying Output Shapes & Constraints...")
    
    required_keys = [
        "base_pose_1", "base_pose_2", 
        "q_start_1", "q_start_2", 
        "start_obj_pose", "goal_obj_pose", 
        "goal_ee1_pose", "goal_ee2_pose"
    ]
    
    all_passed = True
    for key in required_keys:
        if key not in samples:
            print(f"❌ Missing Key: {key}")
            all_passed = False
            continue
            
        tensor = samples[key]
        if tensor.shape[0] != num_envs:
            print(f"❌ Shape Mismatch [{key}]: Expected {num_envs}, Got {tensor.shape[0]}")
            all_passed = False
        
        if torch.isnan(tensor).any():
            print(f"❌ NaN Detected in [{key}]")
            all_passed = False
            
    # 4. Check Distance Constraints (sanity check)
    # Check if bases are respected
    b1_pos = samples["base_pose_1"][:, :3]
    # Check X range: [-0.5, -0.45]
    x_min, x_max = -0.5, -0.45
    valid_x = (b1_pos[:, 0] >= x_min - 1e-4) & (b1_pos[:, 0] <= x_max + 1e-4)
    if not valid_x.all():
         print(f"❌ Base 1 X-range violation detected! Min: {b1_pos[:, 0].min()}, Max: {b1_pos[:, 0].max()}")
         all_passed = False
    else:
         print(f"✅ Base 1 Constraints respecting range [{x_min}, {x_max}]")

    if all_passed:
        print("\n🎉 All Standalone Tests Passed! The sampler is ready for integration.")
    else:
        print("\n⚠️ Some tests failed. Check logs.")

if __name__ == "__main__":
    main()
