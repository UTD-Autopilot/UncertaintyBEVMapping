import argparse
import os
import rerun as rr
import tqdm
from tqdm.contrib.concurrent import process_map
from uncertainty_bev_mapping.mapping import bev_map_frame
from uncertainty_bev_mapping.bev_map_with_color_priority import bev_map_frame_with_color_priority

def bev_map_frame_with_color_priority_wrapper(arg):
    return bev_map_frame_with_color_priority(*arg[0], **arg[1])

def main():
    parser = argparse.ArgumentParser(
        prog='Depth based BEV mapper',
    )

    parser.add_argument('dataset_path', type=str, help='path to the dataset')
    parser.add_argument('-o', '--output_path', type=str, help='output path, default is output to the dataset path (merge with the dataset)', default=None)
    parser.add_argument('--rerun', type=bool, help='log to rerun', default=False)

    args = parser.parse_args()
    dataset_path = args.dataset_path # '../../Datasets/carla/Town01_1'
    output_path = args.output_path

    log_to_rerun = args.rerun

    agent_folders = []
    for dirpath, dirnames, filenames in os.walk(dataset_path):
        if 'sensors.json' in filenames:
            agent_folders.append(dirpath)

    for agent_path in agent_folders:
        print(f'Mapping {agent_path}')
        if log_to_rerun:
            rerun_output_path = os.path.join(agent_path, f'bev_mapping.rrd')
            rerun_output_filename = os.path.basename(rerun_output_path)
            rerun_output_dir = os.path.dirname(rerun_output_path)
            os.makedirs(rerun_output_dir, exist_ok=True)

            rr.init(rerun_output_filename)

        frames = list(map(lambda s: int(s.split('.')[0]), os.listdir(os.path.join(agent_path, 'front_camera'))))

        os.makedirs('tmp', exist_ok=True)
        # for frame in tqdm.tqdm(frames):
        #     # bev_map_frame(agent_path, frame)
        #     bev_map_frame_with_color_priority(agent_path, frame, save_path=output_path)
        
        # Multiprocessing
        args = []
        for frame in frames:
            args.append(((agent_path, frame), dict(save_path=output_path)))
        process_map(bev_map_frame_with_color_priority_wrapper, args)

        if log_to_rerun:
            rr.save(rerun_output_path)

# python scripts/map_pixels.py ../../Datasets/carla/Town01_1 -o outputs/mapping/Town01_1
if __name__ == '__main__':
    main()
