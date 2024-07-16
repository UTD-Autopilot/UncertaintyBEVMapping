import argparse
import os
import pickle
import json
import numpy as np
from PIL import Image
import tqdm
from uncertainty_bev_mapping.utils import aggregate_dicts, aggregate_lists, split_path_into_folders

muad_classes_id_color = [
    ('road', 0, (128, 64, 128)),
    ('sidewalk', 1, (244, 35, 232)),
    ('building', 2, (70, 70, 70)),
    ('wall', 3, (102, 102, 156)),
    ('fence', 4, (190, 153, 153)),
    ('pole', 5, (153, 153, 153)),
    ('traffic light', 6, (250, 170, 30)),
    ('traffic sign', 7, (220, 220, 0)),
    ('vegetation', 8, (107, 142, 35)),
    ('terrain', 9, (152, 251, 152)),
    ('sky', 10, (70, 130, 180)),
    ('person', 11, (220, 20, 60)),
    ('rider', 12, (255, 0, 0)),
    ('car', 13, (0, 0, 142)),
    ('truck', 14, (0, 0, 70)),
    ('bus', 15, (0, 60, 100)),
    ('train', 16, (0, 80, 100)),
    ('motorcycle', 17, (0, 0, 230)),
    ('bicycle', 18, (119, 11, 32)),
    ('bear deer cow', 19, (50, 100, 144)),
    ('garbage_bag stand_food trash_can', 20, (130, 130, 130)),
    ('other', 255, (0, 0, 0)),
]

def get_color_mask(image, color):
    image = np.array(image)
    mask = np.all(image == color, axis=-1)
    return mask

def boolean_iou(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = intersection / union if union != 0 else 0
    return iou

def calculate_statics_for_trajectory(agent_path):
    mapped_bev_frames = []
    mapping_folder = os.path.join(agent_path, 'bev_mapping')
    bev_folder = os.path.join(agent_path, 'birds_view_semantic_camera')
    for filename in os.listdir(mapping_folder):
        if filename.endswith('.png') and len(filename.split('.')) == 2:
            frame = filename.split('.')[0]
            if frame.startswith('bev_'):
                frame = frame[4:]
            mapped_bev_frames.append(int(frame))

    num_frames = len(mapped_bev_frames)

    class_iou = {}
    class_intersection = {}
    class_union = {}
    for class_name, class_id, color in muad_classes_id_color:
        # class_iou[class_name] = []
        class_intersection[class_name] = []
        class_union[class_name] = []

    for frame in mapped_bev_frames:
        mapped_image_path = os.path.join(mapping_folder, f'bev_{frame}.png')
        bev_image_path = os.path.join(bev_folder, f'{frame}.png')

        mapped_image = Image.open(mapped_image_path).convert('RGB')
        bev_image = Image.open(bev_image_path).convert('RGB')

        ego_range_x = [230, 270]
        ego_range_y = [230, 270]

        for class_name, class_id, color in muad_classes_id_color:
            pred = get_color_mask(mapped_image, color)
            gt = get_color_mask(bev_image, color)

            # Mask ego
            pred[ego_range_y[0]:ego_range_y[1], ego_range_x[0]:ego_range_x[1]] = False
            gt[ego_range_y[0]:ego_range_y[1], ego_range_x[0]:ego_range_x[1]] = False

            intersection = np.logical_and(pred, gt).sum()
            union = np.logical_or(pred, gt).sum()
            iou = intersection / union if union != 0 else 0

            iou = boolean_iou(pred, gt)
            # class_iou[class_name].append(iou)
            class_intersection[class_name].append(intersection)
            class_union[class_name].append(union)
    
    for class_name, class_id, color in muad_classes_id_color:
        # class_iou[class_name] = np.mean(class_iou[class_name])
        class_intersection[class_name] = int(np.sum(class_intersection[class_name]))
        class_union[class_name] = int(np.sum(class_union[class_name]))
        class_iou[class_name] = float(class_intersection[class_name] / class_union[class_name]) if class_union[class_name] != 0 else 0

    return {
        'iou': class_iou,
        'union': class_union,
        'num_frames': num_frames,
    }

def main():
    parser = argparse.ArgumentParser(
        prog='BEV mapping statics',
    )

    parser.add_argument('dataset_path', type=str, help='path to the dataset')
    args = parser.parse_args()
    dataset_path = args.dataset_path # '../../Datasets/carla/Town01_1'

    output_path = 'outputs/bev_mapping_statics/'
    os.makedirs(output_path, exist_ok=True)

    all_statics = {}

    agent_folders = []
    for dirpath, dirnames, filenames in os.walk(dataset_path):
        if 'sensors.json' in filenames:
            agent_folders.append(dirpath)

    for agent_path in tqdm.tqdm(agent_folders):
        if not os.path.exists(os.path.join(agent_path, 'bev_mapping')):
            continue
        statics = calculate_statics_for_trajectory(agent_path)
        splitted_path = split_path_into_folders(agent_path)
        agent_id = splitted_path[-1]
        town_name = splitted_path[-3]
        k = f'{town_name}_{agent_id}'
        all_statics[k] = statics

        with open(os.path.join(output_path, f'{k}.json'), 'w') as f:
            json.dump(all_statics[k], f, indent=4)
    
    aggregated_statics = aggregate_dicts(all_statics.values(), np.mean)
    with open(os.path.join(output_path, f'aggregated.json'), 'w') as f:
        json.dump(aggregated_statics, f, indent=4)

# python scripts/mapping_statics.py ../../Dataset/carla/
if __name__ == '__main__':
    main()
