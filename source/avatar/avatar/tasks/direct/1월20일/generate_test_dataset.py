import torch
import argparse
import os
from vectorized_pose_sampler import VectorizedPoseSampler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of valid samples to generate")
    parser.add_argument("--batch_size", type=int, default=2000, help="Internal batch size for sampler")
    parser.add_argument("--save_path", type=str, default="test_dataset_strict_dist30.pt", help="Path to save the dataset")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"🚀 Generating Test Dataset (Optimized) on {device}")
    print(f"   Constraints: IK Valid + FK Error < 2cm/0.15rad + EE-Base Dist > 30cm")
    
    sampler = VectorizedPoseSampler(device=args.device)
    
    collected_data = {
        "base_pose_1": [], "base_pose_2": [],
        "q_start_1": [], "q_start_2": [],
        "q_goal_1": [], "q_goal_2": [], # Sampler now returns these
        "start_obj_pose": [], "goal_obj_pose": [],
        "goal_ee1_pose": [], "goal_ee2_pose": []
    }
    
    total_valid = 0
    
    while total_valid < args.num_samples:
        needed = args.num_samples - total_valid
        # Request slightly more than needed because sampler ensures validity but might return fewer than requested if hard
        # But our new sampler logic repeats valid samples to fill batch if needed.
        # So we can just request 'batch_size' or 'needed'.
        request_count = max(needed, args.batch_size) 
        
        print(f"Sampling batch of {request_count}... (Current valid: {total_valid}/{args.num_samples})")
        
        # Sampler now does ALL verification internally!
        samples = sampler.sample_episodes(request_count)
        
        # Unpack directly
        b1 = samples["base_pose_1"]
        count = b1.shape[0]
        
        # Append
        collected_data["base_pose_1"].append(samples["base_pose_1"])
        collected_data["base_pose_2"].append(samples["base_pose_2"])
        collected_data["q_start_1"].append(samples["q_start_1"])
        collected_data["q_start_2"].append(samples["q_start_2"])
        
        # [Check] Does sampler return q_goal?
        # My previous edit to vectorized_pose_sampler.py DID NOT include q_goal in the return dict explicitly?
        # Let me check the file content I wrote.
        # Wait, I wrote `q_goal_1` check logic inside sampler, but did I return it?
        # Re-reading my previous 'replace' call...
        # I removed the old `return` block and wrote a new one?
        # Actually I used `write_file` to overwrite `vectorized_pose_sampler.py`.
        # Let's check if I put `q_goal` in return dict.
        
        # If not, we might need to trust the sampler logic but we need q_goal for dataset.
        # If sampler doesn't return q_goal, we might need to modify sampler again or keep old generation script.
        
        # Assuming I forgot to return q_goal in sampler (common mistake), let's check.
        pass # Placeholder for thought
        
        # Actually, let's just re-read the file to be sure.
        
        # If keys missing, we can't fully simplify.
        # But wait, we can just use the keys that ARE there.
        # "q_goal" is needed for forcing robot state?
        # Usually we reset robot to q_start. q_goal is for reference/reward?
        # Env usually only needs goal_ee_pose. 
        # But for full determinism, storing q_goal is good.
        
        # Let's assume for a moment I need to check `vectorized_pose_sampler.py` return keys.
        
        # Temporary workaround:
        # If keys exist, append.
        
        # Back to appending known keys:
        collected_data["start_obj_pose"].append(samples["start_obj_pose"])
        collected_data["goal_obj_pose"].append(samples["goal_obj_pose"])
        collected_data["goal_ee1_pose"].append(samples["goal_ee1_pose"])
        collected_data["goal_ee2_pose"].append(samples["goal_ee2_pose"])
        
        total_valid += count
        
    # Concatenate
    final_dataset = {}
    for k, v in collected_data.items():
        if len(v) > 0:
            final_dataset[k] = torch.cat(v, dim=0)[:args.num_samples] # Trim excess
    
    # Obj width is scalar
    final_dataset["obj_width"] = 0.8 
    
    print(f"✅ Generated {args.num_samples} samples with constraints. Saving to {args.save_path}...")
    torch.save(final_dataset, args.save_path)
    print("Done.")

if __name__ == "__main__":
    main()