from typing import Dict

from communication import NetworkInfo
from layeredgraph import LayerNode, LayerNodePair
from job import JobInfo
from scheduling import Dijkstra

import importlib
import time

class LayeredGraph:
    def __init__(self, network_info: NetworkInfo):
        self._network_info = network_info
        self._layered_graph = network_info.get_network()
        self._layered_graph_backlog = dict()
        self._layer_nodes = []
        self._layer_node_pairs = []
        self._scheduling_algorithm = None
        self._previous_update_time = time.time()

    def set_graph(self, links):
        self._previous_update_time = time.time()
        for link in links:
            self.set_link(link, links[link])
        
    def update_graph(self):
        current_time = time.time()
        elapsed_time = current_time - self._previous_update_time

        links_job_num = {}

        for link in self._layer_node_pairs:
            link: LayerNodePair
            source_node_ip = link.get_source().get_ip()
            destination_node_ip = link.get_destinatioin().get_ip()

            destinations: Dict = links_job_num.setdefault(source_node_ip, {})
            destinations.setdefault(destination_node_ip, 0)

            if self._layered_graph_backlog[link] > 0:
                destinations[destination_node_ip] += 1

        for source in self._layer_node_pairs:
            link: LayerNodePair
            source_node_ip = link.get_source().get_ip()
            destination_node_ip = link.get_destinatioin().get_ip()
            
            link_job_num = links_job_num[source_node_ip][destination_node_ip]
            capacity = self._network_info.get_capacity()[source_node_ip][destination_node_ip]

            if link_job_num > 0:
                job_computing_delta = elapsed_time * capacity / link_job_num

                self._layered_graph_backlog[link] = max(0, self._layered_graph_backlog[link] - job_computing_delta)

        self._previous_update_time = time.time()

    def set_link(self, link: str, backlog: float):
        self._layered_graph_backlog[link] = backlog

    def init_graph(self):
        max_job_length = max([job["split_points"] for job in self._network_info.get_jobs()])

        for layer in range(max_job_length):
            for source_ip in self._layered_graph:
                source = LayerNode(source_ip, layer)
                self._layer_nodes.append(source)

                for destination_ip in self._layered_graph[source_ip]:
                    destination = LayerNode(destination_ip, layer)

                    link = LayerNodePair(source, destination)

                    self._layer_node_pairs.append(link)
                    self._layered_graph_backlog[link.to_string()] = 0

        for layer in range(max_job_length - 1):
            for source_ip in self._layered_graph:
                source = LayerNode(source_ip, layer)
                destination = LayerNode(source_ip, layer + 1)
                link = LayerNodePair(source, destination)

                self._layer_node_pairs.append(link)
                self._layered_graph_backlog[link.to_string()] = 0

    def init_algorithm(self):
        module_path = self._network_info.get_scheduling_algorithm().replace(".py", "")
        self._scheduling_algorithm: Dijkstra = importlib.import_module(module_path)

    def schedule(self, source_ip: str, job_info: JobInfo):
        source_node = LayerNode(source_ip, 0)
        destination_node = job_info.get_terminal_destination
        path = self._scheduling_algorithm.get_path(source_node, destination_node, self._layered_graph, self._layered_graph_backlog, self._layer_nodes)

        return path
    
