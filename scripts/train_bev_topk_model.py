import torch
import numpy as np
import argparse
import pynvml
import time

from uncertainty_bev_mapping.train_bev_topk_model import train
from uncertainty_bev_mapping.utils import get_config, get_available_gpus


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("config")
    parser.add_argument('-q', '--queue', default=False, action='store_true')
    parser.add_argument('-g', '--gpus', nargs='+', required=False, type=int, default=[0])
    parser.add_argument('-w', '--num_workers', default=32, required=False, type=int)

    parser.add_argument('-l', '--logdir', required=False, type=str, default=None)
    parser.add_argument('-p', '--pretrained', required=False, type=str, default=None)

    parser.add_argument('-s', '--split', type=str, default='trainval', choices=['mini', 'trainval'])
    parser.add_argument('--seed', default=0, required=False, type=int)

    args = parser.parse_args()

    print(f"Using config {args.config}")
    config = get_config(args)
    config['gpus'] = args.gpus
    config['num_worker'] = args.num_workers
    if args.logdir is not None:
        config['logdir'] = args.logdir
    if args.pretrained is not None:
        config['pretrained'] = args.pretrained

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    if config['queue']:
        pynvml.nvmlInit()
        print("Waiting for suitable GPUs...")

        required_gpus = 2
        while True:
            available_gpus = get_available_gpus(required_gpus=required_gpus)
            if len(available_gpus) == required_gpus:
                print(f"Running program on GPUs {available_gpus}...")
                config['gpus'] = available_gpus
                break
            else:
                time.sleep(30)

        pynvml.nvmlShutdown()

    dataroot = f"../../Datasets/{config['dataset']}"

    train(config, dataroot)
