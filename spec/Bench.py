import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import torch
import time
import threading
from spec.GPUUtilManager import GPUUtilManager
from utils import ensure_path_exists, load_model, split_model
from typing import List

class Bench:
    def __init__(self):
        self.gpu_util_manager = GPUUtilManager()
        self._subtasks: List[torch.nn.Module] = []
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._yolo_computing_ratios = [6.24, 2.02, 1.80, 0.64]
        self._model_name = "model"
        self._stop_event = threading.Event()  # 모델 실행 종료 이벤트

    def load_model(self, model_name, split_points):
        model, flatten_index = load_model(model_name)
        for split_point in split_points:
            subtask: torch.nn.Module = split_model(model, split_point, flatten_index).to(self._device)
            self.append_subtask(subtask)

        self._model_name = model_name

    def append_subtask(self, subtask: torch.nn.Module):
        self._subtasks.append(subtask)

    def start_bench(self, times = 100):
        for _ in range(times):
            input_size = (1, 3, 320, 320)
            model_input = torch.randn(input_size).to(self._device)
            
            for idx, subtask in enumerate(self._subtasks):
                time.sleep(1)
                layer_name = f"layer_{idx}"
                gflops = self._yolo_computing_ratios[idx]
                csv_path = f"spec/{self._model_name}/{layer_name}_{gflops:.2f}GFLOPs.csv"
                # ensure_path_exists(csv_path, is_file=True)

                data = {"latency": [], "watt_hour": [0.0]}
                
                # latency 측정을 위한 이벤트 초기화
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                # 전력 측정 쓰레드 실행
                self._stop_event.clear()
                watt_measure_thread = threading.Thread(target=self._measure_watt, args=(data,))
                watt_measure_thread.start()

                start_event.record()
                
                start = time.time()

                # 모델 실행 (비동기적으로 실행)
                with torch.no_grad():
                    model_input = subtask(model_input)  # 모델을 실행하여 output을 얻음

                end = time.time()
                
                end_event.record()
                torch.cuda.synchronize()

                # 모델 실행 종료 후 이벤트로 쓰레드 종료
                self._stop_event.set()
                watt_measure_thread.join()  # 쓰레드 종료 대기
                
                # 최종 latency 측정
                latency = start_event.elapsed_time(end_event) / 1000  # 초 단위
                data["latency"].append(latency)
                
                # 누적된 와트시(watt-hour)로 변환하여 저장
                data["watt_hour"][0] = data["watt_hour"][0] / 3600.0
            
                # CSV 파일로 저장
                df = pd.DataFrame(data)
                df.to_csv(csv_path, index=False, mode='a', header=not os.path.exists(csv_path))
                print(f"Data saved to {csv_path}")

    def _measure_watt(self, data):
        """주기적으로 전력 소비량을 측정하여 누적"""
        start_time = time.time()
        while not self._stop_event.is_set():
            gpu_stats = self.gpu_util_manager.get_gpu_stats(0)
            watt = gpu_stats["power_usage"] if gpu_stats else 0
            # 경과 시간 계산 및 누적 전력 소비량 계산
            elapsed_time = time.time() - start_time  # 초 단위
            data["watt_hour"][0] += watt * elapsed_time  # 와트-초

    def constantize_csv_data(self):
        total_idle_computing_capacity = 0
        for idx, subtask in enumerate(self._subtasks):
            layer_name = f"layer_{idx}"
            gflops = self._yolo_computing_ratios[idx]
            csv_path = f"spec/{self._model_name}/{layer_name}_{gflops:.2f}GFLOPs.csv"
            ensure_path_exists(csv_path, is_file=True)

            df = pd.read_csv(csv_path)

            total_idle_computing_capacity += 1/df["latency"].mean() * gflops
            stats = {
                "idle_computing_capacity": 1/df["latency"].mean() * gflops,
                "watt_hour": df["watt_hour"].mean(),
                "latency": df["latency"].mean(),
            }
            const_csv_path = csv_path.replace(".csv", "_constant.csv")
            pd.DataFrame([stats]).to_csv(const_csv_path, index=False)
            print(f"Constant data saved to {const_csv_path}")
        

if __name__ == "__main__":
    bench = Bench()
    bench.load_model("yolov5", [[0,1], [1,2], [2,3], [3,4]])
    bench.start_bench(100)
    bench.constantize_csv_data()
