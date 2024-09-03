import os
import subprocess
import yaml
import pynvml
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

def aggregate_dicts(dicts, fn):
    # All the dicts must have the same field.
    aggregated = {}
    for k, v in dicts[0]:
        aggregated[k] = []
    
    for d in dicts:
        for k, v in d:
            aggregated[k].append(v)
    for k, v in aggregated:
        if isinstance(v[0], dict):
            aggregated[k] = aggregate_dicts(v[0], fn)
        elif isinstance(v[0], list):
            aggregated[k] = aggregate_lists(v[0], fn)
        else:
            aggregated[k] = fn(v)
    return aggregated

def aggregate_lists(lists, fn):
    aggregated = []
    for v in lists[0]:
        aggregated.append([])
    for li, l in enumerate(lists):
        for vi, v in enumerate(l):
            aggregated[vi].append(v)
    for vi, v in enumerate(aggregated):
        if isinstance(v[0], dict):
            aggregated[vi] = aggregate_dicts(v, fn)
        elif isinstance(v[0], list):
            aggregated[vi] = aggregate_lists(v, fn)
        else:
            aggregated[vi] = fn(v)
    return aggregated

def split_path_into_folders(path):
    folders = []
    while True:
        path, folder = os.path.split(path)
        if folder:
            folders.append(folder)
        else:
            if path:
                folders.append(path)
            break
    folders.reverse()
    return folders

def get_config(args):
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    for key, value in vars(args).items():
        if value is not None:
            config[key] = value

    return config

def get_available_gpus(required_gpus=2):
    username = os.getlogin()
    available_gpus = []
    device_count = pynvml.nvmlDeviceGetCount()

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        compute_procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)

        user_procs = [proc for proc in compute_procs if get_username_from_pid(proc.pid) == username]
        if not user_procs:
            available_gpus.append(i)

        if len(available_gpus) >= required_gpus:
            break

    return available_gpus[:required_gpus]

def get_username_from_pid(pid):
    try:
        proc = subprocess.run(['ps', '-o', 'user=', '-p', str(pid)], capture_output=True, text=True)
        return proc.stdout.strip()
    except subprocess.CalledProcessError:
        return None

colors = torch.tensor([
    [0, 0, 255],
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 0],
])

def map_rgb(onehot, ego=False):
    dense = onehot.permute(1, 2, 0).detach().cpu().numpy().argmax(-1)

    rgb = np.zeros((*dense.shape, 3))
    for label, color in enumerate(colors):
        rgb[dense == label] = color

    if ego:
        rgb[94:106, 98:102] = (0, 255, 255)

    return rgb


def save_unc(u_score, u_true, out_path, score_name, true_name):
    u_score = u_score.detach().cpu().numpy()
    u_true = u_true.numpy()

    cv2.imwrite(
        os.path.join(out_path, true_name),
        u_true[0, 0] * 255
    )

    cv2.imwrite(
        os.path.join(out_path, score_name),
        cv2.cvtColor((plt.cm.inferno(u_score[0, 0]) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    )


def save_pred(pred, label, out_path, ego=False):
    if pred.shape[1] != 2:
        pred = map_rgb(pred[0], ego=ego)
        label = map_rgb(label[0], ego=ego)
        cv2.imwrite(os.path.join(out_path, "pred.png"), pred)
        cv2.imwrite(os.path.join(out_path, "label.png"), label)

        return pred, label
    else:
        cv2.imwrite(os.path.join(out_path, "pred.png"), pred[0, 0].detach().cpu().numpy() * 255)
        cv2.imwrite(os.path.join(out_path, "label.png"), label[0, 0].detach().cpu().numpy() * 255)
