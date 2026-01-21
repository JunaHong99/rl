
import time
import torch
import numpy as np
from pose_sampler import DualArmPoseSampler
from vectorized_pose_sampler import VectorizedPoseSampler

def benchmark():
    # Settings
    batch_sizes = [10, 100, 1000] # Test with different batch sizes
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    print(f"Running benchmark on device: {device}")
    
    # 1. Initialize Samplers
    print("Initializing CPU Sampler (Pinocchio)...")
    cpu_sampler = DualArmPoseSampler()
    
    print(f"Initializing GPU Sampler (Vectorized)...")
    gpu_sampler = VectorizedPoseSampler(device=device)
    
    # Warmup GPU
    print("Warming up GPU...")
    gpu_sampler.sample_episodes(10)
    torch.cuda.synchronize()
    
    print("-" * 50)
    print(f"{'Batch Size':<15} | {'CPU Time (s)':<15} | {'GPU Time (s)':<15} | {'Speedup':<10}")
    print("-" * 50)
    
    for N in batch_sizes:
        # --- CPU Benchmark ---
        start_time = time.time()
        # CPU sampler generates 1 by 1
        for _ in range(N):
            try:
                cpu_sampler.sample_valid_episode()
            except RuntimeError:
                pass # Ignore failures for timing purposes, though failures hurt CPU time more
        end_time = time.time()
        cpu_duration = end_time - start_time
        
        # --- GPU Benchmark ---
        torch.cuda.synchronize()
        start_time = time.time()
        try:
            gpu_sampler.sample_episodes(N)
        except RuntimeError as e:
            print(f"GPU Sampling failed: {e}")
        torch.cuda.synchronize()
        end_time = time.time()
        gpu_duration = end_time - start_time
        
        speedup = cpu_duration / gpu_duration if gpu_duration > 0 else 0.0
        
        print(f"{N:<15} | {cpu_duration:<15.4f} | {gpu_duration:<15.4f} | {speedup:<10.1f}x")

if __name__ == "__main__":
    benchmark()
