import torch

def check_cuda_availability():
    """
    PyTorch에서 CUDA (GPU) 사용 가능 여부를 확인하고 상세 정보를 출력합니다.
    """
    print("--- PyTorch CUDA Availability Check ---")
    
    # 1. CUDA 사용 가능 여부 확인
    is_available = torch.cuda.is_available()
    
    if is_available:
        print("✅ CUDA is available.")
        
        # 2. 사용 가능한 GPU 개수 확인
        gpu_count = torch.cuda.device_count()
        print(f"   - Number of available GPUs: {gpu_count}")
        
        # 3. 현재 활성화된 GPU 장치 인덱스 확인
        current_device_idx = torch.cuda.current_device()
        print(f"   - Current CUDA device index: {current_device_idx}")
        
        # 4. 현재 GPU 장치 이름 확인
        device_name = torch.cuda.get_device_name(current_device_idx)
        print(f"   - Device Name: {device_name}")
        
        # 5. 간단한 텐서 연산을 통해 GPU 사용 테스트
        try:
            print("\nTesting GPU with a simple tensor operation...")
            # GPU에 텐서 생성
            tensor_gpu = torch.tensor([1.0, 2.0]).cuda()
            print("   - Successfully created a tensor on the GPU.")
            print(f"   - Tensor on GPU: {tensor_gpu}")
            print("   - GPU is working correctly.")
        except Exception as e:
            print(f"❌ An error occurred during the GPU test operation: {e}")

    else:
        print("❌ CUDA is not available.")
        print("   - PyTorch will use the CPU for all computations.")
        print("   - To use a GPU, please check your NVIDIA driver, CUDA Toolkit installation, and PyTorch version.")

if __name__ == "__main__":
    check_cuda_availability()
