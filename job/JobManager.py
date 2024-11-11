from typing import Dict, List, Tuple
import torch

from job import *
from utils import *
from communication import *
from virtual_queue import VirtualQueue, AheadOutputQueue

import threading
import time

try:
    from time import time_ns
except ImportError:
    from datetime import datetime
    # For compatibility with Python 3.6
    def time_ns():
        now = datetime.now()
        return int(now.timestamp() * 1e9)

class JobManager:
    def __init__(self, address, network_info: NetworkInfo):
        # TODO
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._network_info = network_info

        self._virtual_queue = VirtualQueue()
        self._ahead_of_time_outputs = AheadOutputQueue()
        self._dnn_models = DNNModels(self._network_info, self._device, address)

        self._job_queue = []
        self._job_result_queue = []
        self.mutex = threading.Lock()
        self.job_result_mutext = threading.Lock()
        
        self.init_garbage_subtask_collector()
        self.init_job_queue_manager()

    def is_subtask_exists(self, output: DNNOutput):
        previous_subtask_info = output.get_subtask_info()
        if self._virtual_queue.exist_subtask_info(previous_subtask_info):
            return True
        else:
            return False
        
    def is_dnn_output_exists(self, subtask_info: SubtaskInfo):
        if self._ahead_of_time_outputs.exist_dnn_output(subtask_info):
            return True
        else:
            return False
        
    # add dnn_output if schedule is not arrived yet
    def pop_dnn_output(self, subtask_info: SubtaskInfo):
        dnn_output = self._ahead_of_time_outputs.pop_dnn_output(subtask_info)
        return dnn_output
        
    def init_garbage_subtask_collector(self):
        garbage_subtask_collector_thread = threading.Thread(target=self.garbage_subtask_collector, args=())
        garbage_subtask_collector_thread.start()

        garbage_dnn_output_collector_thread = threading.Thread(target=self.garbage_dnn_output_collector, args=())
        garbage_dnn_output_collector_thread.start()

    def garbage_subtask_collector(self):
        collect_garbage_job_time = self._network_info.get_collect_garbage_job_time()
        while True:
            time.sleep(collect_garbage_job_time)

            self._virtual_queue.garbage_subtask_collector(collect_garbage_job_time)

    def init_job_queue_manager(self):
        job_queue_manager_thread = threading.Thread(target=self.job_queue_manager, args=())
        job_queue_manager_thread.start()

    def job_queue_manager(self):
        while True:
            self.mutex.acquire()
            if len(self._job_queue) == 0:
                self.mutex.release()
                continue
            output, is_compressed = self._job_queue.pop(0)
            print("pop job")
            self.mutex.release()
            dnn_output, computing_capacity = self.run(output, is_compressed)

            self.job_result_mutext.acquire()
            self._job_result_queue.append((dnn_output, computing_capacity))
            self.job_result_mutext.release()
            
    def queue_job(self, output, is_compressed):
        self.mutex.acquire()
        self._job_queue.append((output, is_compressed))
        self.mutex.release()

    def garbage_dnn_output_collector(self):
        collect_garbage_job_time = self._network_info.get_collect_garbage_job_time()
        while True:
            time.sleep(collect_garbage_job_time)

            self._ahead_of_time_outputs.garbage_dnn_output_collector(collect_garbage_job_time)

    def get_backlogs(self):
        return self._virtual_queue.get_backlogs()

    def run(self, output: DNNOutput, is_compressed: bool = False) -> Tuple[DNNOutput, float]:
        if is_compressed:
            job_name = output.get_subtask_info().get_job_name()
            decompressed_shape = tuple(self._network_info.get_jobs()[job_name]["real_input"])
            real_data = torch.rand(decompressed_shape)
            output = DNNOutput(real_data, output.get_subtask_info())

        previous_subtask_info = output.get_subtask_info()
        if previous_subtask_info.get_job_type() == "dnn":
            # get next destination
            subtask: DNNSubtask = self._virtual_queue.pop_subtask_info(previous_subtask_info)

            # get output data == get current subtask's input
            data = output.get_output()

            if isinstance(data, list):
                data = [d.to(self._device) for d in data]
            else:
                data = data.to(self._device)

            start_time = time_ns() / 1_000_000_000 # ns to s

            # run job
            dnn_output = subtask.run(data)

            end_time = time_ns() / 1_000_000_000 # ns to s

            computing_capacity = subtask.get_backlog() / (end_time - start_time + 1e-05) if subtask.get_backlog() > 0 else 0

            return dnn_output, computing_capacity
        
    # add subtask_info based SubtaskInfo
    def add_subtask(self, subtask_info: SubtaskInfo):
        job_name = subtask_info.get_job_name()
        model_index = subtask_info.get_model_index()
        subtask_model = self._dnn_models.get_subtask(job_name, model_index) if subtask_info.is_computing() else None
        computing = self._dnn_models.get_computing(job_name, model_index) * subtask_info.get_input_size()
        transfer = self._dnn_models.get_transfer(job_name, model_index) * subtask_info.get_input_size()

        subtask = DNNSubtask(
            subtask_info = subtask_info,
            dnn_model = subtask_model,
            computing = computing,
            transfer = transfer
        )

        success_add_subtask_info = self._virtual_queue.add_subtask_info(subtask_info, subtask)
        
        if not success_add_subtask_info:
            raise Exception(f"Subtask already exists. : {subtask_info.get_subtask_id()}")
        
    # add dnn_output if schedule is not arrived yet
    def add_dnn_output(self, previous_dnn_output: DNNOutput):
        subtask_info = previous_dnn_output.get_subtask_info()
        success_add_dnn_output = self._ahead_of_time_outputs.add_dnn_output(subtask_info, previous_dnn_output)
        
        if not success_add_dnn_output:
            raise Exception(f"DNNOutput already exists. : {previous_dnn_output.get_subtask_info().get_subtask_id()}")

