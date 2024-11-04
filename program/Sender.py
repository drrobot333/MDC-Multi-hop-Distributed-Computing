import sys, os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import pickle
import time
from threading import Thread
import paho.mqtt.publish as publish
import numpy as np
import posix_ipc
import mmap
import torch
try:
    from time import time_ns
except ImportError:
    from datetime import datetime
    # For compatibility with Python 3.6
    def time_ns():
        now = datetime.now()
        return int(now.timestamp() * 1e9)

from utils.utils import get_ip_address
from program import MDC
from program.Communicator import Communicator
from job import JobInfo, SubtaskInfo, DNNOutput


class Sender(MDC):
    def __init__(self, sub_config, pub_configs, job_name):
        self._address = get_ip_address(["eth0", "wlan0"])

        self._job_name = job_name
        self._job_info = None
        self._frame_list = dict()
        self._arrival_rate = 0

        super().__init__(sub_config, pub_configs)

        self.topic_dispatcher["mdc/arrival_rate"] = self.handle_arrival_rate

    def init_job_info(self, input_size: int):
        source_ip = self._address
        terminal_destination = self._network_info.get_jobs()[self._job_name]["destination"]
        job_type = self._network_info.get_jobs()[self._job_name]["job_type"]
        job_name = self._job_name
        start_time = time_ns()
        input_size = input_size 

        job_info = JobInfo(source_ip, terminal_destination, job_type, job_name, start_time, input_size)

        self._job_info = job_info

    def handle_subtask_info(self, topic, data, publisher): # overriding
        subtask_info: SubtaskInfo = pickle.loads(data)

        self._job_manager.add_subtask(subtask_info)

        subtask_layer_node = subtask_info.get_source()

        if subtask_layer_node.get_ip() == self._address and subtask_layer_node.get_layer() == 0:
            job_id = subtask_info.get_job_id()
            input_frame = DNNOutput(torch.tensor(self._frame_list[job_id]).float(), subtask_info)
            del self._frame_list[job_id]
            dnn_output, computing_capacity = self._job_manager.run(input_frame)
            destination_ip = subtask_info.get_destination().get_ip()

            dnn_output.get_subtask_info().set_next_subtask_id()

            dnn_output_bytes = pickle.dumps(dnn_output)
                
            # send job to next node
            # self._node_publisher[destination_ip].publish(f"job/{subtask_info.get_job_type()}", dnn_output_bytes)
            publish.single(f"job/{subtask_info.get_job_type()}", dnn_output_bytes, hostname=destination_ip)

            self._capacity_manager.update_computing_capacity(computing_capacity)

    def handle_arrival_rate(self, topic, data, publisher):
        arrival_rate = pickle.loads(data)

        self._arrival_rate = arrival_rate

    def start(self):
        self.wait_until_can_send()
        self.init_communicator()

        input("Press any key to start sending.")

        self.run_arrival_rate_getter()
        

        while True:
            self._communicator.send_message("waiting")
            agent_message = self._communicator.get_message()

            if agent_message == "action":
                self.handle_action()

            elif agent_message == "reward":
                self.handle_reward()

            elif agent_message == "finish":
                self.handle_finish_from_agent()

    def set_job_info_time(self):
        if self._network_info == None:
            return False
        
        else:
            if self._job_info == None:
                self.init_job_info()
                return True
            else:
                self._job_info.set_start_time(time_ns())
                return True
            
    def set_job_info_input_size(self, frame: np.array):
        if self._network_info == None:
            return False
        
        else:
            if self._job_info == None:
                self.init_job_info()
                return True
            else:
                input_size = sys.getsizeof(torch.tensor(frame).storage())
                self._job_info.set_input_size(input_size)
                return True
            
    def wait_until_can_send(self):
        print("Waiting for network info.")
        while not (self.check_job_manager_exists() and self.check_network_info_exists()):
            time.sleep(1.0)
            
    def arrival_rate_getter(self):
        node_info_bytes = pickle.dumps(self._node_info)
        while True:
            time.sleep(0.1)
            self._controller_publisher.publish("mdc/arrival_rate", node_info_bytes)

    def run_arrival_rate_getter(self):
        arrival_rate_thread = Thread(target=self.arrival_rate_getter, args=())
        arrival_rate_thread.start()

    def init_communicator(self):
        self._communicator = Communicator(queue_name=self._network_info.get_queue_name(), 
                                          buffer_size=4096, 
                                          is_agent=False,
                                          debug_mode=False)

    def handle_action(self):
        frame = self.get_frame()
        self.send_frame(frame)
        
    def get_frame(self) -> float:
        self._communicator.send_message("ACK")
        frame_shape = eval(self._communicator.get_message())
        frame: np.array = np.zeros(frame_shape)

        return frame
    
    def send_frame(self, frame):
        if self._job_info == None:
            input_size = sys.getsizeof(torch.tensor(frame).storage())
            self.init_job_info(input_size)

        self.set_job_info_time()
        job_info_bytes = pickle.dumps(self._job_info)
        self._frame_list[self._job_info.get_job_id()] = frame

        self._controller_publisher.publish("job/request_scheduling", job_info_bytes)
    
    def handle_reward(self):
        self._communicator.send_message(str(self._arrival_rate))
        self._communicator.get_message()

    def handle_finish_from_agent(self):
        job_info_bytes = pickle.dumps(self._job_info)
        self._controller_publisher.publish("mdc/finish", job_info_bytes)

    
        

if __name__ == '__main__':
    sub_config = {
            "ip": "127.0.0.1", 
            "port": 1883,
            "topics": [
                ("job/dnn", 1),
                ("job/subtask_info", 1),
                ("mdc/network_info", 1),
                ("mdc/node_info", 1),
                ("mdc/arrival_rate", 1),
                ("mdc/finish", 1),
            ],
        }
    
    pub_configs = []

    job_name = "test job 1"

    sender = Sender(sub_config, pub_configs, job_name)
    sender.start()
