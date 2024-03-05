import sys, os, json
 
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from program import Program
from job.Job import Job
from job.DNNJob import DNNJob

import argparse
import pickle
import time

from utils.utils import get_ip_address

class Sender(Program):
    def __init__(self, sub_config, pub_configs, topic):
        self.sub_config = sub_config
        self.pub_configs = pub_configs
        self.topic = topic
        self.address = get_ip_address("eth0")
        self.a = 0

        if "packet" in self.topic:
            self.job = Job
        elif "dnn_output" in self.topic:
            self.job = DNNJob

        self.topic_dispatcher = {
        }

        super().__init__(self.sub_config, self.pub_configs, self.topic_dispatcher)

    def send_dummy_job(self, size, sleep_time, iterate_time, dst, id, ex_mode):
        info = f"sleeptime_{sleep_time}_size_{size}_it_{iterate_time}_mode_{ex_mode}"
        data = "0" * size
        for _ in range(iterate_time):
            dummy_job = self.job(data, self.address, dst, id, info, self.a)
            self.a += 1
            dummy_job_bytes = pickle.dumps(dummy_job)
            self.publisher[0].publish(self.topic, dummy_job_bytes)
            print("send", self.a)

            time.sleep(sleep_time)
        
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-p', '--peer', type=str, default="")
    argparser.add_argument('-d', '--destination', type=str, default="192.168.1.7")
    argparser.add_argument('-t', '--topic', type=str, default="job/packet")
    argparser.add_argument('-g', '--sleep_gap', type=float, default="0.1")
    argparser.add_argument('-s', '--size', type=int, default="100")
    argparser.add_argument('-i', '--iterate', type=int, default="100")
    argparser.add_argument('-id', '--id', type=str, default="test")
    argparser.add_argument('-m', '--mode', type=str, default="lower")
    args = argparser.parse_args()

    sub_config = None
    pub_configs = [
        {
            "ip": args.destination, 
            "port": 1883,
        }
    ]
    
    sender = Sender(sub_config=sub_config, pub_configs=pub_configs, topic=args.topic)
    sender.send_dummy_job(args.size, args.sleep_gap / 1000, args.iterate, args.destination, args.id, args.mode)