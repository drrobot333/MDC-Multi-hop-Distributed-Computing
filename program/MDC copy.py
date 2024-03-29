import sys, os, json
 
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from queue import Queue
from threading import Thread
from program import Program
from datetime import datetime

import argparse
import pickle

from utils.utils import get_ip_address, save_latency
from virtual_queue.VirtualQueue import RoutingTable
import paho.mqtt.publish as publish
from job.JobManager import JobManager

class MDC(Program):
    def __init__(self, sub_config, pub_configs, topic):
        self.sub_config = sub_config
        self.pub_configs = pub_configs
        self.address = get_ip_address("eth0")
        self.routing_table = RoutingTable(self.address)
        self.topic = topic
        self.job_manager = JobManager()

        self.topic_dispatcher = {
            "job/dnn": self.handle_dnn,
            "job/response": self.handle_response
        }

        super().__init__(self.sub_config, self.pub_configs, self.topic_dispatcher)

    def handle_dnn(self, topic, data, publisher):
        dnn_job = pickle.loads(data)

        if dummy_job.is_rtt_destination(self.address):
            file_name = dummy_job.info
            latency = dummy_job.calc_latency()
            save_latency(file_name, latency)
            # TODO change to save results.

        elif dummy_job.is_destination(self.address):
            # response to source
            job_id = dummy_job.get_id()
            if self.routing_table.exist_subtask_info(job_id): # if it is mid dst
                destination = self.routing_table.find_subtask_info(job_id)
                dummy_job.set_source(self.address)
                dummy_job.set_destination(destination)
                dummy_job_bytes = pickle.dumps(dummy_job)
                publish.single(self.topic, dummy_job_bytes, hostname=destination)

            else: # if it is final dst
                dummy_job = self.job_manager.run(dummy_job)
                dummy_job.remove_input()
                dummy_job.set_response()
                dummy_job.set_destination(dummy_job.source)
                dummy_job.set_source(self.address)
                dummy_job_bytes = pickle.dumps(dummy_job)
                publish.single(self.topic, dummy_job_bytes, hostname=dummy_job.destination)

        else:
            dummy_job_bytes = pickle.dumps(dummy_job)
            self.publisher[0].publish(self.topic, dummy_job_bytes)

    def handle_response(self, topic, data, publisher):
        dummy_job = pickle.loads(data)
        print(dummy_job)

        if dummy_job.is_rtt_destination(self.address):
            file_name = dummy_job.info
            latency = dummy_job.calc_latency()
            save_latency(file_name, latency)
            # TODO change to save results.

        elif dummy_job.is_destination(self.address):
            # response to source
            job_id = dummy_job.get_id()
            if self.routing_table.exist_subtask_info(job_id): # if it is mid dst
                destination = self.routing_table.find_subtask_info(job_id)
                dummy_job.set_source(self.address)
                dummy_job.set_destination(destination)
                dummy_job_bytes = pickle.dumps(dummy_job)
                publish.single(self.topic, dummy_job_bytes, hostname=destination)

            else: # if it is final dst
                dummy_job.remove_input()
                dummy_job.set_response()
                dummy_job.set_destination(dummy_job.source)
                dummy_job.set_source(self.address)
                dummy_job_bytes = pickle.dumps(dummy_job)
                publish.single(self.topic, dummy_job_bytes, hostname=dummy_job.destination)

        else:
            dummy_job_bytes = pickle.dumps(dummy_job)
            self.publisher[0].publish(self.topic, dummy_job_bytes)

    def start(self):
        pass
        
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-p', '--peer', type=str, default="")
    argparser.add_argument('-t', '--topic', type=str, default="job/packet")
    args = argparser.parse_args()

    sub_config = {
            "ip": "127.0.0.1", 
            "port": 1883,
            "topics": [
                (args.topic, 1),
            ],
        }
    
    pub_configs = [
        {
            "ip": args.peer, 
            "port": 1883,
        }
    ]
    
    ex = MDC(sub_config=sub_config, pub_configs=pub_configs, topic=args.topic)
    ex.start()