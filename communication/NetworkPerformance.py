# info class for response NetworkPerformance
class NetworkPerformance:
    def __init__(self, gpu_capacity: float, ip: str):
        self._gpu_capacity = gpu_capacity
        self._ip = ip

    def get_gpu_capacity(self):
        return self._gpu_capacity
    
    def get_ip(self):
        return self._ip