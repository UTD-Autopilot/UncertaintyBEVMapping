import argparse
import os
import rerun as rr
import tqdm
import traceback
from tqdm.contrib.concurrent import process_map
from uncertainty_bev_mapping.utils import split_path_into_folders
from uncertainty_bev_mapping.mapping import bev_map_frame
from uncertainty_bev_mapping.bev_map_with_color_priority import bev_map_frame_with_color_priority, bev_map_frame_with_color_priority_uncertainty

def bev_map_frame_with_color_priority_wrapper(arg):
    # return bev_map_frame_with_color_priority(*arg[0], **arg[1])
    try:
        retval = bev_map_frame_with_color_priority_uncertainty(*arg[0], **arg[1])
    except Exception as e:
        traceback.print_exc(e)
        retval = None
    return retval

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
    ignore_existing = True

    log_to_rerun = args.rerun

    agent_folders = []
    for dirpath, dirnames, filenames in os.walk(dataset_path):
        if 'sensors.json' in filenames:
            agent_folders.append(dirpath)

    for agent_path in agent_folders:
        print(f'Mapping {agent_path}')
        splitted_path = split_path_into_folders(agent_path)
        agent_id = splitted_path[-1]
        town_name = splitted_path[-3]
        uncertainty_data_path = f'~/data/Datasets/uncertainty-bev-mapping-main-backbones-DeepLabV3Plus/{town_name}/'
        uncertainty_data_path = os.path.expanduser(uncertainty_data_path)

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
        #     if ignore_existing and os.path.exists(os.path.join(agent_path, 'bev_mapping_aleatoric', f'{frame}.npy')):
        #         continue
        #     bev_map_frame_with_color_priority_uncertainty(agent_path, frame, uncertainty_data_path, save_path=output_path)

        # Multiprocessing
        args = []
        for frame in frames:
            # if ignore_existing and os.path.exists(os.path.join(agent_path, 'bev_mapping', f'bev_{frame}.png')):
            if ignore_existing and os.path.exists(os.path.join(agent_path, 'bev_mapping_aleatoric', f'{frame}.npy')):
                continue
            # args.append(((agent_path, frame), dict(save_path=output_path)))
            args.append(((agent_path, frame, uncertainty_data_path), dict(save_path=output_path)))
        process_map(bev_map_frame_with_color_priority_wrapper, args)

        if log_to_rerun:
            rr.save(rerun_output_path)

# python scripts/map_pixels.py ../../Datasets/carla/Town01_1 -o outputs/mapping/Town01_1
if __name__ == '__main__':
    main()
