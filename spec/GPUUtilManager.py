import subprocess
import platform
import re
import sys


class GPUUtilManager:
    def __init__(self):
        # 시스템 유형에 따라 GPU 개수를 확인하여 gpu_count 속성으로 저장
        self.is_windows = platform.system() == "Windows"
        self.gpu_count = self._get_gpu_count()
        if self.gpu_count == 0:
            print("No GPUs found on this system.")
        
        # 기본적으로 빈 딕셔너리로 GPU 상태 저장용 변수를 초기화
        self.gpu_stats = {}
    
    def _get_gpu_count(self):
        """현재 시스템의 GPU 개수를 반환"""
        if self.is_windows:
            # Windows에서는 nvidia-smi 사용
            command = "nvidia-smi --query-gpu=count --format=csv,noheader,nounits"
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                print("Error in executing nvidia-smi.")
                return 0
            
            return int(result.stdout.strip())
        else:
            # Jetson에서는 tegrastats 사용
            try:
                if sys.version_info.minor > 6:
                    result = subprocess.run("tegrastats --version", shell=True, capture_output=True, text=True)
                    if result.returncode != 0:
                        print("Error: tegrastats not found or not supported.")
                        return 0
                    # Jetson Nano는 GPU 1개로 간주
                    return 1
                else:
                    result = subprocess.run("tegrastats --version", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                    if result.returncode != 0:
                        print("Error: tegrastats not found or not supported.")
                        return 0
                    # Jetson Nano는 GPU 1개로 간주
                    return 1
            except FileNotFoundError:
                print("tegrastats not available on this system.")
                return 0

    def get_gpu_stats(self, gpu_id=0):
        # Windows인 경우 GPU ID 유효성 확인
        if self.is_windows:
            if gpu_id >= self.gpu_count:
                print(f"Invalid GPU ID: {gpu_id}. This system has {self.gpu_count} GPUs.")
                return None
            
            # nvidia-smi 명령어 실행
            command = f"nvidia-smi --query-gpu=power.draw,utilization.gpu,memory.used --format=csv,noheader,nounits -i {gpu_id}"
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                print("Error in executing nvidia-smi.")
                return None
            
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
        else:
            if sys.version_info.minor > 6:
                # Jetson Nano에서는 gpu_id를 사용하지 않음. 항상 0번째 GPU만 있음.
                if gpu_id != 0:
                    print(f"Invalid GPU ID: {gpu_id}. Jetson Nano has only one GPU.")
                    return None
                
                # tegrastats 명령어 실행
                try:
                    result = subprocess.run("sudo tegrastats | head -n 1", shell=True, capture_output=True, text=True)
                    if result.returncode != 0:
                        print("Error in executing tegrastats.")
                        return None
                    
                    output = result.stdout.strip()
                    if output:
                        # tegrastats 출력에서 GPU 관련 데이터 파싱
                            # Regular expressions to match RAM and GR3D_FREQ values
                        ram_pattern = r'RAM (\d+)/(\d+)MB'
                        gr3d_freq_pattern = r'GR3D_FREQ (\d+)%'
                        power_pattern = r'VDD_IN (\d+)mW/(\d+)mW'

                        # Extract RAM and GR3D_FREQ using regex search
                        ram_match = re.search(ram_pattern, output)
                        gr3d_freq_match = re.search(gr3d_freq_pattern, output)
                        power_match = re.search(power_pattern, output)

                        if ram_match and gr3d_freq_match:
                            mem_used = int(ram_match.group(1))
                            gpu_util = int(gr3d_freq_match.group(1))
                            power = int(power_match.group(1))
                        
                        if gpu_util is not None and mem_used is not None:
                            stats = {
                                "power_usage": power / 1000,  # Jetson Nano에서 정확한 전력 소모는 별도 방법 필요
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

            else:
                # Jetson Nano에서는 gpu_id를 사용하지 않음. 항상 0번째 GPU만 있음.
                if gpu_id != 0:
                    print(f"Invalid GPU ID: {gpu_id}. Jetson Nano has only one GPU.")
                    return None
                
                # tegrastats 명령어 실행
                try:
                    result = subprocess.run("sudo tegrastats | head -n 1", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                    if result.returncode != 0:
                        print("Error in executing tegrastats.")
                        return None
                    
                    output = result.stdout.strip()
                    if output:
                        # tegrastats 출력에서 GPU 관련 데이터 파싱
                            # Regular expressions to match RAM and GR3D_FREQ values
                        ram_pattern = r'RAM (\d+)/(\d+)MB'
                        gr3d_freq_pattern = r'GR3D_FREQ (\d+)%'
                        power_pattern = r'POM_5V_IN (\d+)/(\d+)'

                        # Extract RAM and GR3D_FREQ using regex search
                        ram_match = re.search(ram_pattern, output)
                        gr3d_freq_match = re.search(gr3d_freq_pattern, output)
                        power_match = re.search(power_pattern, output)

                        if ram_match and gr3d_freq_match:
                            mem_used = int(ram_match.group(1))
                            gpu_util = int(gr3d_freq_match.group(1))
                            power = int(power_match.group(1))
                        
                        if gpu_util is not None and mem_used is not None:
                            stats = {
                                "power_usage": power / 1000,  # Jetson Nano에서 정확한 전력 소모는 별도 방법 필요
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
        """모든 GPU의 상태를 가져옵니다"""
        all_stats = {}
        for gpu_id in range(self.gpu_count):
            all_stats[gpu_id] = self.get_gpu_stats(gpu_id)
        return all_stats
