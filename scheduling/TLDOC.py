import heapq
from typing import Dict, List

from layeredgraph import LayerNode, LayerNodePair
import numpy as np
import random, math
from latencymodel.TLDOC import cal_total_latency, cal_total_latency_except_end

class TLDOC:
    def __init__(self):
        pass
    
    def init_parameter(self, time_config, energy_config, idle_power, transfer_ratios, scale=0.5, V=1.0, latency_allowed=25, default_rate=0.1):
        """ config shape 
        time_config = {'end': List[각 레이어의 실행시간], 'edge': List[각 레이어의 실행시간], 'cloud': List[각 레이어의 실행시간]}
        energy_config = {'end': List[각 레이어의 소모에너지], 'end_to_edge': List[각 레이어 output을 end -> edge로 전송할 때 소모에너지]}
        """
        #! check: hyperparameters
        self._scale = scale
        self._V = V
        self._latency_allowed = latency_allowed
        self._default_rate = default_rate
        self._epoch = 10
        self._queue = 0
        self._time_config = time_config
        self._energy_config = energy_config 
        self._idle_power = idle_power       #가능?
        self._transfer_ratios = transfer_ratios

    def get_path(self, source_node: LayerNode, destination_node: LayerNode, layered_graph, arrival_rate, network_info, input_size):
        data_size_list = [ input_size * arrival_rate * ratio for ratio in self._transfer_ratios ] #! check: index 확인 
        max_layer = len(self._transfer_ratios) - 1  # 4개
        off_tensor = self._lp_offloading(max_layer, network_info, data_size_list)
        partition_point_1, partition_point_2 = off_tensor[0], off_tensor[0]+off_tensor[1]
        path = self._make_path(source_node, destination_node, layered_graph, partition_point_1, partition_point_2)
        return path
    
    
    def _lp_offloading(self, max_layer, network_info, data_size_list):
        off_tensor = self._create_init_tensor(max_layer)
        count = 0
        for i in range(self._epoch):
            temp_off_tensor = self._create_new_off_tensor(off_tensor, network_info)
            temp_cost, _ = self._objective(temp_off_tensor, network_info, data_size_list)
            cost, _ = self._objective(off_tensor, network_info, data_size_list)
            delta = temp_cost - cost
            if delta < 0:
                off_tensor = temp_off_tensor
            else:
                count += 1
            
            if count >= 2:
                break
        
        _, actual_rate = self._objective(off_tensor, network_info, data_size_list)
        self._queue = self._cal_queue(actual_rate)
        return off_tensor


    def _create_new_off_tensor(self, off_tensor, network_info):
        edge_resource = network_info[0]['edge']
        cloud_resource = network_info[0]['cloud']
        total_resource = edge_resource + cloud_resource
        
        A_end = off_tensor[0]
        delta = int(A_end * self._scale)
        A_end -= delta
        A_edge = off_tensor[1] + int(delta * (edge_resource / total_resource))
        A_cloud = off_tensor[2] + int(delta * (cloud_resource / total_resource))
        
        new_off_tensor = [A_end, A_edge, A_cloud]
        return new_off_tensor
    
    
    def _objective(self, off_tensor, network_info, data_size_list):
        actual_rate = self._get_violation_rate(off_tensor, network_info, data_size_list)
        total_energy = self._cal_total_energy(off_tensor, network_info, data_size_list)
        temp_queue = self._cal_queue(actual_rate)
        cost = self._V * temp_queue * actual_rate + total_energy
        return cost, actual_rate       
    
    
    def _cal_queue(self, actual_rate):
        new_queue = max(self._queue - self._default_rate, 0) + actual_rate
        return new_queue
    
    
    def _create_init_tensor(self, max_layer):
        A_end, A_edge, A_cloud = max_layer, 0, 0 
        off_tensor = [A_end, A_edge, A_cloud]
        return off_tensor
    
    
    def _get_violation_rate(self, off_tensor, network_info, data_size_list):
        actual_rate = 0
        total_latency = cal_total_latency(off_tensor, self._time_config, network_info, data_size_list)
        if total_latency > self._latency_allowed:  
            actual_rate = ((total_latency - self._latency_allowed)/self._latency_allowed)*100
        return actual_rate
    
    
    def _cal_total_energy(self, off_tensor, network_info, data_size_list):
        E_cal = sum(self._energy_config['end'][:off_tensor[0]])
        E_comm = self._energy_config['end_to_edge'][off_tensor[0]]
        E_idle = cal_total_latency_except_end(off_tensor, self._time_config, network_info, data_size_list) * self._idle_power
        
        total_energy = E_cal + E_comm + E_idle
        return total_energy
    
    
    def _make_path(self, source_node, destination_node, layered_graph, partition_point_1, partition_point_2):
        
        path = []
        path.append(source_node)
        neighbors = layered_graph[source_node][:]
        sorted_neighbors = sorted(neighbors, key=lambda neighbor: neighbor.get_layer())
        
        # end computing
        count = 0
        for node in sorted_neighbors:
            if count >= partition_point_1:
                break 
            
            if source_node.is_same_node(node):
                path.append(node)
                count += 1
        
        end_last_node = node
        neighbors = layered_graph[end_last_node][:]
        sorted_neighbors = sorted(neighbors, key=lambda neighbor: neighbor.get_layer())
        
        # end -> edge
        for node in sorted_neighbors:
            if not end_last_node.is_same_node(node):
                path.append(node)
                break
        
        edge_top_node = node
        neighbors = layered_graph[edge_top_node][:]
        sorted_neighbors = sorted(neighbors, key=lambda neighbor: neighbor.get_layer())
        
        # edge computing
        for node in sorted_neighbors:
            if count >= partition_point_2:
                break 
            
            if edge_top_node.is_same_node(node):
                path.append(node)
                count += 1
        
        edge_last_node = node 
        neighbors = layered_graph[edge_last_node][:]
        sorted_neighbors = sorted(neighbors, key=lambda neighbor: neighbor.get_layer())
        
        # edge -> cloud
        for node in sorted_neighbors:
            if destination_node.is_same_node(node):
                path.append(node)
                break 
        
        cloud_node = node
        cloud_layer = cloud_node.get_layer()
        neighbors = layered_graph[cloud_node][:]
        sorted_neighbors = sorted(neighbors, key=lambda neighbor: neighbor.get_layer())
        
        # cloud computing 
        for node in neighbors:
            if destination_node.is_same_node(node) and cloud_layer < node.get_layer():
                path.append(node)
        
        return path