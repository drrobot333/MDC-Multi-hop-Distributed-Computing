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
        # Jetson Nano에서는 nvidia-smi 명령어 대신 tegrastats를 사용해야 함
        try:
            result = subprocess.run("tegrastats --version", shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print("Error: tegrastats not found or not supported.")
                return 0
            # Jetson Nano는 GPU 1개로 간주
            return 1
        except FileNotFoundError:
            print("tegrastats not available on this system.")
            return 0

    def get_gpu_stats(self, gpu_id=0):
        # Jetson Nano에서는 gpu_id를 사용하지 않음. 항상 0번째 GPU만 있음.
        if gpu_id != 0:
            print(f"Invalid GPU ID: {gpu_id}. Jetson Nano has only one GPU.")
            return None
        
        # tegrastats 명령어 실행
        try:
            result = subprocess.run("tegrastats --interval 1000 --one-shot", shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print("Error in executing tegrastats.")
                return None
            
            output = result.stdout.strip()
            if output:
                # tegrastats 출력에서 GPU 관련 데이터 파싱
                gpu_util = None
                mem_used = None
                for item in output.split():
                    if 'GR3D_FREQ' in item:
                        gpu_util = float(item.split('%@')[0].split(':')[-1]) / 100  # % to fraction
                    elif 'RAM' in item:
                        mem_info = item.split('/')[0].split(':')[-1]
                        mem_used = float(mem_info[:-1])  # MB (assuming 'M' means MB)
                
                if gpu_util is not None and mem_used is not None:
                    stats = {
                        "power_usage": None,  # Jetson Nano에서 정확한 전력 소모는 별도 방법 필요
                        "utilization": gpu_util,  # %
                        "memory_usage": mem_used  # MB
                    }
                    self.gpu_stats[gpu_id] = stats  # 조회한 상태를 저장
                    return stats
                else:
                    print("Failed to parse GPU data from tegrastats output.")
                    return None
            else:
                print("No data returned from tegrastats.")
                return None
        except FileNotFoundError:
            print("tegrastats command not found.")
            return None

    def get_all_gpu_stats(self):
        """모든 GPU의 상태를 가져옵니다 (Jetson Nano의 경우 하나만 있음)"""
        return {0: self.get_gpu_stats()}