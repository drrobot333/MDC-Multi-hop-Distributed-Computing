#TODO: 단위 맞추기

def cal_total_latency(arrival_rate, task_requirements, network_info):
    computing_requirements, transmission_requirements = task_requirements
    computing_capacities, transmission_rates = network_info
    computing_latency = _cal_computing_latency(computing_requirements['edge'], computing_requirements['cloud'], computing_capacities['edge'], computing_capacities['cloud'])
    transmission_latency = _cal_transmission_latency(transmission_rates['end'], transmission_rates['edge'], transmission_requirements['end'], transmission_requirements['edge'])
    queuing_latency, stability_info = _cal_queueing_latency(arrival_rate, computing_requirements['edge'], computing_requirements['cloud'], computing_capacities['edge'], computing_capacities['cloud'])
    total_latency = arrival_rate * (computing_latency + transmission_latency + queuing_latency)
    
    return total_latency, stability_info


def _cal_computing_latency(edge_computing, cloud_computing, edge_source, cloud_source):
    edge_latency = edge_computing / edge_source
    cloud_latency = cloud_computing / cloud_source
    
    latency = edge_latency + cloud_latency
    return latency


def _cal_transmission_latency(end_trans_rate, edge_trans_rate, end_data_size, edge_data_size):
    end_latency = end_data_size / end_trans_rate        # end -> edge 
    edge_latency = edge_data_size / edge_trans_rate     # edge -> cloud
    
    latency = end_latency + edge_latency
    return latency


def _cal_queueing_latency(arrival_rate, edge_computing, cloud_computing, edge_source, cloud_source):
    edge_latency = 0
    cloud_latency = 0
    
    if edge_computing != 0:
        edge_service_rate = edge_source / edge_computing
        edge_stability = arrival_rate / edge_service_rate
        edge_latency = (arrival_rate / (2 * ((edge_service_rate)**2) * (1 - edge_stability)))
    else:
        edge_stability = float('inf')

    if cloud_computing != 0:
        cloud_service_rate = cloud_source / cloud_computing
        cloud_stability = arrival_rate / cloud_service_rate
        if edge_computing == 0 or edge_service_rate > cloud_service_rate:
            cloud_latency = (arrival_rate / (2 * ((cloud_service_rate)**2) * (1 - cloud_stability)))
    else:
        cloud_stability = float('inf')

    stability_info = (edge_stability, cloud_stability)
    latency = edge_latency + cloud_latency
    
    return latency, stability_info
