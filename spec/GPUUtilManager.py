import subprocess
import platform
import re
import threading
import time
import sys


class GPUUtilManager:
    def __init__(self):
        self.is_windows = platform.system() == "Windows"
        self.gpu_count = self._get_gpu_count()
        if self.gpu_count == 0:
            print("No GPUs found on this system.")

        # GPU 사용률(EMA) 변수 초기화
        self.ema_gpu_util = 0
        self.alpha = 0.1  # EMA 업데이트에 사용될 알파 값
        self.running = True  # 스레드 종료 플래그
        
        # EMA 업데이트를 위한 스레드 시작
        self.update_thread = threading.Thread(target=self._update_gpu_util)
        self.update_thread.start()

    def _get_gpu_count(self):
        # GPU 개수를 반환하는 함수
        if self.is_windows:
            command = "nvidia-smi --query-gpu=count --format=csv,noheader,nounits"
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print("Error in executing nvidia-smi.")
                return 0
            return int(result.stdout.strip())
        else:
            try:
                if sys.version_info.minor > 6:
                    result = subprocess.run("tegrastats --version", shell=True, capture_output=True, text=True)
                    if result.returncode != 0:
                        print("Error: tegrastats not found or not supported.")
                        return 0
                    return 1  # Jetson Nano는 GPU 1개로 간주
                else:
                    result = subprocess.run("tegrastats --version", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                    if result.returncode != 0:
                        print("Error: tegrastats not found or not supported.")
                        return 0
                    return 1  # Jetson Nano는 GPU 1개로 간주
            except FileNotFoundError:
                print("tegrastats not available on this system.")
                return 0

    def _update_gpu_util(self):
        # GPU 사용률 데이터를 주기적으로 업데이트하고 EMA 적용
        while self.running:
            try:
                if self.is_windows:
                    # Windows에서 nvidia-smi를 사용하여 GPU 사용률 수집
                    command = "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits"
                    result = subprocess.run(command, shell=True, capture_output=True, text=True)
                    output = result.stdout.strip()
                    if output:
                        gpu_util = int(output) / 100  # %로 변환
                        # EMA 업데이트
                        self.ema_gpu_util = (self.alpha * gpu_util) + ((1 - self.alpha) * self.ema_gpu_util)
                else:
                    if sys.version_info.minor > 6:
                        # Jetson에서 tegrastats를 사용하여 GPU 사용률 수집
                        result = subprocess.run("tegrastats | head -n 1", shell=True, capture_output=True, text=True)
                        output = result.stdout.strip()
                        if output:
                            gr3d_freq_pattern = r'GR3D_FREQ (\d+)%'
                            gr3d_freq_match = re.search(gr3d_freq_pattern, output)

                            if gr3d_freq_match:
                                gpu_util = int(gr3d_freq_match.group(1)) / 100  # %로 변환
                                # EMA 업데이트
                                self.ema_gpu_util = (self.alpha * gpu_util) + ((1 - self.alpha) * self.ema_gpu_util)
                        else:
                            print("No data returned from tegrastats.")

                    else:
                        # Jetson에서 tegrastats를 사용하여 GPU 사용률 수집
                        result = subprocess.run("tegrastats | head -n 1", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                        output = result.stdout.strip()
                        if output:
                            gr3d_freq_pattern = r'GR3D_FREQ (\d+)%'
                            gr3d_freq_match = re.search(gr3d_freq_pattern, output)

                            if gr3d_freq_match:
                                gpu_util = int(gr3d_freq_match.group(1)) / 100  # %로 변환
                                # EMA 업데이트
                                self.ema_gpu_util = (self.alpha * gpu_util) + ((1 - self.alpha) * self.ema_gpu_util)
                        else:
                            print("No data returned from tegrastats.")
            except FileNotFoundError:
                print("Command not found.")
            time.sleep(0.5)  # 1초마다 데이터 업데이트

    def get_all_gpu_stats(self):
        # EMA로 저장된 GPU 사용률 반환
        return {"utilization": self.ema_gpu_util}

    def stop(self):
        # 스레드 종료
        self.running = False
        self.update_thread.join()