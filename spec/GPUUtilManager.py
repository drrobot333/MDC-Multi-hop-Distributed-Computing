import subprocess

class GPUUtilManager:
    def __init__(self):
        # 시스템에 있는 GPU의 총 개수를 확인하여 gpu_count 속성으로 저장
        self.gpu_count = self._get_gpu_count()
        if self.gpu_count == 0:
            print("No GPUs found on this system.")
        
        # 기본적으로 빈 딕셔너리로 GPU 상태 저장용 변수를 초기화
        self.gpu_stats = {}
    
    def _get_gpu_count(self):
        """현재 시스템의 GPU 개수를 반환"""
        command = "nvidia-smi --query-gpu=count --format=csv,noheader,nounits"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("Error in executing nvidia-smi.")
            return 0
        
        return int(result.stdout.strip())
    
    def get_gpu_stats(self, gpu_id):
        # GPU ID의 유효성을 확인
        if gpu_id >= self.gpu_count:
            print(f"Invalid GPU ID: {gpu_id}. This system has {self.gpu_count} GPUs.")
            return None
        
        # nvidia-smi 명령어 실행
        command = f"nvidia-smi --query-gpu=power.draw,utilization.gpu,memory.used --format=csv,noheader,nounits -i {gpu_id}"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("Error in executing nvidia-smi.")
            return None

        # 출력 파싱
        output = result.stdout.strip()
        if output:
            power, utilization, memory = output.split(', ')
            stats = {
                "power_usage": float(power),  # W
                "utilization": float(utilization) / 100,  # %
                "memory_usage": float(memory)  # MB
            }
            self.gpu_stats[gpu_id] = stats  # 조회한 상태를 저장
            return stats
        else:
            print("No data returned from nvidia-smi.")
            return None
    
    def get_all_gpu_stats(self):
        """모든 GPU의 상태를 가져옵니다"""
        all_stats = {}
        for gpu_id in range(self.gpu_count):
            all_stats[gpu_id] = self.get_gpu_stats(gpu_id)
        return all_stats

