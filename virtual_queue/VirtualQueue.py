from typing import Dict
from job import DNNSubtask

class VirtualQueue:
    def __init__(self, address):
        self.address = address
        self.subtask_infos: Dict[str, DNNSubtask] = dict()

    def exist_subtask_info(self, id):
        return id in self.subtask_infos

    def add_subtask_info(self, id, action: DNNSubtask):
        # ex) "192.168.1.5", Job
        if self.exist_subtask_info(id):
            return False
        
        else:
            self.subtask_infos[id] = action
            return True

    def del_subtask_info(self, id):
        del self.subtask_infos[id]
    
    def find_subtask_info(self, id):
        if self.exist_subtask_info(id):
            return self.subtask_infos[id]
        else:
            raise Exception("No flow subtask_infos : ", id)
        
    def pop_subtask_info(self, id):
        subtask_info = self.find_subtask_info(id)
        self.del_subtask_info(id)

        return subtask_info
    
    def get_backlogs(self):
        links = {}
        for subtask_info in self.subtask_infos:
            subtask: DNNSubtask = self.subtask_infos[subtask_info]

            link = subtask_info.split("_")[2] + "->" + subtask_info.split("_")[3]

            if subtask_info in links:
                links[link] += subtask.get_backlog()
            else:
                links[link] = subtask.get_backlog()

        return links
        
    def __str__(self):
        return str(self.subtask_infos)