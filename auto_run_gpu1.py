import os
import time
import GPUtil
import subprocess

def check_gpu_memory(threshold_gb=10):
    """GPU의 여유 메모리를 체크하는 함수"""
    gpu = GPUtil.getGPUs()[1]
    free_memory_gb = gpu.memoryFree / 1024  # MB를 GB로 변환
    if free_memory_gb >= threshold_gb:
        return True
    return False

while True:
    if check_gpu_memory(threshold_gb=14):
        print("Sufficient GPU memory available. Starting WandB agent...")
        
        # 환경 변수 설정 및 wandb agent 실행
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 사용할 GPU 설정
        command = "wandb agent 'laymond1/Comp. Algo. Hyperparam Tuning on Si-Blurry/36gq68co'"
        
        # 명령어 실행
        subprocess.run(command, shell=True)
        break  # 에이전트 실행 후 종료
    else:
        print("Not enough GPU memory. Checking again in 10 minutes...")
        time.sleep(1200)  # 10분 후에 다시 확인
