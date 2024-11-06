import sys, os
 
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from program import Program
from job import *
from communication import *
from job.JobManager import JobManager
from utils.utils import get_ip_address
from spec.GPUUtilManager import GPUUtilManager

import paho.mqtt.publish as publish
import MQTTclient
import pickle
import time

class MDC(Program):
    def __init__(self, sub_config, pub_configs):
        self.sub_config = sub_config
        self.pub_configs = pub_configs
        self._address = get_ip_address(["eth0", "wlan0"])
        self._node_info = NodeInfo(self._address)
        self._controller_publisher = MQTTclient.Publisher(config={
            "ip" : "192.168.1.2",
            "port" : 1883
        })
        self._node_publisher = {}

        self.topic_dispatcher = {
            "job/dnn": self.handle_dnn,
            "job/subtask_info": self.handle_subtask_info,
            "mdc/network_info" : self.handle_network_info,
            "mdc/node_info": self.handle_request_backlog,
            "mdc/finish": self.handle_finish,
            "mdc/network_performance_info": self.handle_requset_network_performance_info,
        }

        self.topic_dispatcher_checker = {
            "job/dnn": [(self.check_network_info_exists, True)],
            "job/subtask_info": [(self.check_job_manager_exists, True)],
            "mdc/network_info": [(self.check_job_manager_exists, False)],
            "mdc/node_info": [(self.check_job_manager_exists, True)],
        }

        self._network_info = None
        self._job_manager = None
        self._neighbors = None
        self._backlogs_zero_flag = False

        self._capacity_manager = CapacityManager()
        self._gpu_util_manager = GPUUtilManager()

        super().__init__(self.sub_config, self.pub_configs, self.topic_dispatcher, self.topic_dispatcher_checker)

        self.request_network_info()

    # request network information to network controller
    # sending node info.
    def request_network_info(self):
        while self._network_info == None:
            print("Requested network info..")
            node_info_bytes = pickle.dumps(self._node_info)

            # send NetworkInfo byte to source ip (response)
            self._controller_publisher.publish("mdc/network_info", node_info_bytes)

            time.sleep(2)

    def handle_subtask_info(self, topic, data, publisher):
        subtask_info: SubtaskInfo = pickle.loads(data)

        self._job_manager.add_subtask(subtask_info)

        if self._job_manager.is_dnn_output_exists(subtask_info):
            dnn_output = self._job_manager.pop_dnn_output(subtask_info) # make another method
            self.run_dnn(dnn_output)
    
    def handle_network_info(self, topic, data, publisher):
        self._network_info: NetworkInfo = pickle.loads(data)
        self._job_manager = JobManager(self._address, self._network_info)

        self.init_node_publisher()

        print(f"Succesfully get network info.")

    def handle_requset_network_performance_info(self, topic, data, publisher):
        gpu_usage = self._gpu_util_manager.get_all_gpu_stats()["utilization"]
        gpu_capacity = 1 - gpu_usage

        network_performance = NetworkPerformance(ip=self._address, gpu_capacity=gpu_capacity)

        network_performance_bytes = pickle.dumps(network_performance)
        
        # send NetworkPerformance byte to source ip (response)
        self._controller_publisher.publish("mdc/network_performance_info", network_performance_bytes)

    def init_node_publisher(self):
        network = self._network_info.get_network()
        neighbors = network[self._address]

        for neighbor in neighbors:
            publisher = MQTTclient.Publisher(config={
                "ip" : neighbor,
                "port" : 1883
            })
            self._node_publisher[neighbor] = publisher


    def handle_request_backlog(self, topic, data, publisher):
        # transfer capacity check current capacity every sync time.
        self._capacity_manager.update_transfer_capacity()

        links = self._job_manager.get_backlogs()

        computing_capacity = self._capacity_manager.get_computing_capacity()
        transfer_capacity = self._capacity_manager.get_transfer_capacity()

        node_link_info = NodeLinkInfo(
            ip = self._address, 
            links = links, 
            computing_capacity = computing_capacity, 
            transfer_capacity = transfer_capacity
            )
        
        node_link_info_bytes = pickle.dumps(node_link_info)

        # send NodeLinkInfo byte to source ip (response)
        self._controller_publisher.publish("mdc/node_info", node_link_info_bytes)

    def check_network_info_exists(self, data = None):
        if self._network_info == None:
            return False
        
        elif self._network_info != None:
            return True
        
    def check_job_manager_exists(self, data = None):
        if self._job_manager == None:
            return False
        
        elif self._job_manager != None:
            return True

    def handle_dnn(self, topic, data, publisher):
        previous_dnn_output: DNNOutput = pickle.loads(data)
        self.run_dnn(previous_dnn_output)

    def handle_finish(self, topic, data, publisher):
        print("finish!! exit program.")
        time.sleep(5)
        os._exit(1)

    def run_dnn(self, previous_dnn_output: DNNOutput):
        # terminal node
        if previous_dnn_output.is_terminal_destination(self._address) and not self._job_manager.is_subtask_exists(previous_dnn_output): 
            subtask_info = previous_dnn_output.get_subtask_info()
            subtask_info_bytes = pickle.dumps(subtask_info)

            # send subtask info to controller
            self._controller_publisher.publish("job/response", subtask_info_bytes)

        else: 
            if self._job_manager.is_subtask_exists(previous_dnn_output):
                # if cao
                is_compressed = self._address == "192.168.1.8" and self._network_info.get_queue_name() == "cao"

                dnn_output, computing_capacity = self._job_manager.run(output=previous_dnn_output, is_compressed=is_compressed)

                subtask_info = dnn_output.get_subtask_info()
                destination_ip = subtask_info.get_destination().get_ip()

                dnn_output.get_subtask_info().set_next_subtask_id()

                dnn_output_bytes = pickle.dumps(dnn_output)
                    
                # send job to next node
                publish.single(f"job/{subtask_info.get_job_type()}", dnn_output_bytes, hostname=destination_ip)

                self._capacity_manager.update_computing_capacity(computing_capacity)
            else:
                self._job_manager.add_dnn_output(previous_dnn_output)

       
if __name__ == '__main__':
    sub_config = {
            "ip": "127.0.0.1", 
            "port": 1883,
            "topics": [
                ("job/dnn", 1),
                ("job/subtask_info", 1),
                ("mdc/network_info", 1),
                ("mdc/node_info", 1),
                ("mdc/finish", 1),
                ("mdc/network_performance_info", 1),
            ],
        }
    
    pub_configs = [
    ]
    
    mdc = MDC(sub_config=sub_config, pub_configs=pub_configs)
    mdc.start()