def cal_total_latency(off_tensor, time_config, network_info, data_size_list):
    t_cal = sum(time_config['end'][:off_tensor[0]]) + sum(time_config['edge'][off_tensor[0]:off_tensor[0]+off_tensor[1]]) + sum(time_config['cloud'][off_tensor[0]+off_tensor[1]:])
    end_latency = data_size_list[off_tensor[0]] / network_info[1]['end']                        # end -> edge 
    edge_latency = data_size_list[off_tensor[0]+off_tensor[1]] / network_info[1]['edge']        # edge -> cloud
    t_comm = end_latency + edge_latency
    t_wait = 0                                  #!check: waiting time
    
    total_latency = t_cal + t_comm + t_wait
    return total_latency


def cal_total_latency_except_end(off_tensor, time_config, network_info, data_size_list):
    #%
    t_cal = sum(time_config['edge'][off_tensor[0]:off_tensor[0]+off_tensor[1]]) + sum(time_config['cloud'][off_tensor[0]+off_tensor[1]:])
    edge_latency = data_size_list[off_tensor[0]+off_tensor[1]] / network_info[1]['edge']        # edge -> cloud
    t_comm = edge_latency
    t_wait = 0                                  #!check: waiting time
    
    total_latency = t_cal + t_comm + t_wait
    return total_latency
