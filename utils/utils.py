import subprocess, socket, re, os

import csv

import torch
from torchvision.models import resnet18, mobilenet_v2
from yolov5.Yolov5 import P1, P2, P3, P4

def get_ip_address(interface_name=["eth0"]):
    # check os
    for interface in interface_name:

        if os.name == "nt":  # windows
            ip = get_ip_address_windows(interface)
        else:  # linux / unix
            ip = get_ip_address_linux(interface)

        if "192.168.1" in ip:
            return ip

def get_ip_address_windows(interface_name='eth0'):
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return ip_address

def get_ip_address_linux(interface_name='eth0'):
    try:
        ip_addr_output = subprocess.check_output(["ip", "addr", "show", interface_name], encoding='utf-8')

        ip_pattern = re.compile(r"inet (\d+\.\d+\.\d+\.\d+)/")
        ip_match = ip_pattern.search(ip_addr_output)
        if ip_match:
            return ip_match.group(1)
        else:
            return "IP address not found"
    except subprocess.CalledProcessError:
        return "Failed to execute ip command or interface not found"
    

def save_latency(file_path, latency):
    # 파일이 존재하는지 확인
    file_exists = os.path.exists(file_path)

    latency = f"{latency / 1_000_000} ms"

    # 파일에 데이터 쓰기
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # 파일이 새로 만들어진 경우 열 이름을 씁니다.
        if not file_exists:
            writer.writerow(["latency"])
        
        # 데이터 행을 파일에 씁니다.
        writer.writerow([latency])

def save_virtual_backlog(file_path, virtual_backlog):
    # 파일이 존재하는지 확인
    file_exists = os.path.exists(file_path)

    sorted_virtual_backlog = sorted(virtual_backlog.items(), key=lambda item: item[0])
    links = [link.to_string() for link, _ in sorted_virtual_backlog]
    backlogs = [backlog for _, backlog in sorted_virtual_backlog]

    backlog_sum = sum(backlogs)
    backlog_avg = backlog_sum / len(backlogs) if backlogs else 0

    headers = ["sum", "avg"] + links
    datas = [backlog_sum, backlog_avg] + backlogs

    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        if not file_exists:
            writer.writerow(headers)

        writer.writerow(datas)

def save_path(file_path, path):
    # 파일이 존재하는지 확인
    file_exists = os.path.exists(file_path)

    path = [node.to_string() for node in path]
    path_string = ','.join(path)

    # 파일에 데이터 쓰기
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # 파일이 새로 만들어진 경우 열 이름을 씁니다.
        if not file_exists:
            writer.writerow(["path"])
        
        # 데이터 행을 파일에 씁니다.
        writer.writerow([path_string])
       
def split_model(model: torch.nn.Module, split_point, flatten_index: int) -> torch.nn.Module:
    start, end = split_point
    layers = list(model.children())
    if flatten_index != None:
        layers.insert(flatten_index, torch.nn.Flatten())
    splited_model = torch.nn.Sequential(*layers[start:end])
    return splited_model

def load_model(model_name) -> torch.nn.Module:

    available_model_list = ["yolov5", "resnet-18", "resnet-50", "mobilenet_v2"]

    assert model_name in available_model_list, f"Model must be in {available_model_list}."

    if model_name == "yolov5":
        models = torch.nn.Sequential(P1(), P2(), P3(), P4())
        return models, None
    
    elif model_name == "resnet-18":
        model = resnet18(pretrained=True)
        model.eval()
        return model, -1 # TODO
    
    elif model_name == "resnet-50":
        return None # TODO
    
    elif model_name == "mobilenet_v2":
        model = mobilenet_v2(pretrained=True, )
        model.eval()
        return model, -1 # TODO
    
def ensure_path_exists(path, is_file=False):
    """
    지정된 경로에 폴더 또는 파일이 있는지 확인하고, 없으면 생성합니다.
    
    Parameters:
    path (str): 확인할 경로
    is_file (bool): 파일 경로인지 여부를 지정 (True로 설정 시 파일이 없을 경우 빈 파일 생성)
    """
    if is_file:
        # 파일의 상위 폴더가 없으면 폴더를 먼저 생성
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # 파일이 없으면 빈 파일 생성
        if not os.path.exists(path):
            with open(path, 'w') as f:
                pass
            print(f"File created at: {path}")
        else:
            print(f"File already exists at: {path}")
    else:
        # 폴더가 없으면 폴더 생성
        os.makedirs(path, exist_ok=True)
        print(f"Directory ensured at: {path}")