import os
from uncertainty_bev_mapping.utils import split_path_into_folders

if __name__ == '__main__':
    uncertainty_file_path = '~/data/Datasets/uncertainty-bev-mapping-main-backbones-DeepLabV3Plus'
    dataset_path = '~/data/Datasets/carla'
    cameras = ['front_camera', 'front_left_camera', 'front_right_camera', 'back_camera', 'back_left_camera', 'back_right_camera']
    
    stop = False
    town_to_agent_folders = {}
    agent_folders = []
    for dirpath, dirnames, filenames in os.walk(dataset_path):
        if 'sensors.json' in filenames:
            agent_folders.append(dirpath)

    for agent_path in agent_folders:
        if not os.path.exists(os.path.join(agent_path, 'bev_mapping')):
            continue
        splitted_path = split_path_into_folders(agent_path)
        agent_id = splitted_path[-1]
        town_name = splitted_path[-3]
        if agent_id != '0':
            print(f'More than 1 agents detected at {agent_path}, stopping.')
            stop = True
            break
        town_to_agent_folders[town_name] = agent_path

    if stop:
        exit()

    for town in os.listdir(uncertainty_file_path):
        for camera in cameras:
            frames = set()
            camera_directory = os.path.join(uncertainty_file_path, town, camera)
            for filename in os.listdir(camera_directory):
                if filename.endswith('_aleatoric.png'):
                    frames.append(filename.split('_aleatoric.npy')[0])
            for frame in frames:
                os.makedirs(os.path.join(town_to_agent_folders(town), f'{camera}_uncertainty'), exist_ok=True)
                os.rename(os.path.join(camera_directory, f'{frame}_aleatoric.npy'), os.path.join(town_to_agent_folders(town), f'{camera}_uncertainty', f'{frame}_aleatoric.npy'))
                os.rename(os.path.join(camera_directory, f'{frame}_epistemic.npy'), os.path.join(town_to_agent_folders(town), f'{camera}_uncertainty', f'{frame}_epistemic.npy'))
