import json
import os
import sys

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

import torch
import torch.utils.data as data
from uncertainty_bev_mapping.utils import split_path_into_folders

def map_color_dict(img, d, default=0, dtype=np.int32):
    h, w, _ = img.shape[-3:]
    result = np.full((h, w), default, dtype=dtype)
    for color, value in d.items():
        mask = np.all(img == color, axis=-1)
        result[mask] = value
    return result


class CarlaDataset(data.Dataset):
    """Carla dataset, based on the CityScapes class definition.

    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    carla_classes_id_color = [
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

    training_classes = [
        ('background', 0, ['other', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'train']),
        ('vehicle', 1, ['car', 'bus', 'truck', 'motorcycle', 'bicycle', 'rider']),
        ('drivable_area', 2, ['road']),
        ('pedestrian', 3, ['person']),
    ]

    training_classes_color = np.array([
        [0, 0, 0],
        [0, 0, 142],
        [128, 64, 128],
        [220, 20, 60],
    ], dtype=np.uint8)

    training_class_weights = np.array([1.0, 2.0, 1.0, 10.0])

    ood_classes = [
        ('not_ood', 0, []),
        ('bear deer cow', 1, ['bear deer cow']), 
        ('garbage_bag stand_food trash_can', 2, ['garbage_bag stand_food trash_can']),
    ]

    ood_classes_color = np.array([
        [0, 0, 0],
        [50, 100, 144],
        [130, 130, 130],
    ], dtype=np.uint8)

    def __init__(self, dataset_path, transform=None):
        self.dataset_path = os.path.expanduser(dataset_path)
        self.transform = transform

        self.images = []
        self.labels = []
        self.ids = []

        self.muad_class_to_color = {x[0]: x[2] for x in self.carla_classes_id_color}

        self.color_to_train_id = {}
        for train_cls_name, train_idx, cls_names in self.training_classes:
            for data_cls_name in cls_names:
                self.color_to_train_id[self.muad_class_to_color[data_cls_name]] = train_idx
        
        self.color_to_ood_id = {}
        for train_cls_name, train_idx, cls_names in self.ood_classes:
            for data_cls_name in cls_names:
                self.color_to_ood_id[self.muad_class_to_color[data_cls_name]] = train_idx
        
        self.num_classes = len(self.training_classes)
        self.num_ood_classes = len(self.ood_classes)

        for town_name in os.listdir(self.dataset_path):
            record_folder = os.path.join(self.dataset_path, town_name, 'agents')
            for agent_name in os.listdir(record_folder):
                for frame in os.listdir(os.path.join(record_folder, agent_name, 'front_camera')):
                    frame = int(frame.split('.')[0])
                    data_id = f'{town_name}_{agent_name}_{frame}'
                    self.images.append(os.path.join(record_folder, agent_name, f'front_camera/{frame}.png'))
                    self.labels.append(os.path.join(record_folder, agent_name, f'front_semantic_camera/{frame}.png'))
                    self.ids.append(data_id)

    def encode_target(self, target):
        return self.id_to_train_id[np.array(target)]
    
    def visualize_image(self, target):
        return self.training_classes_color[target]
    
    def visualize_ood(self, ood):
        return self.ood_classes_color[ood]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.labels[index]).convert('RGB')
        
        label = map_color_dict(np.array(target), self.color_to_train_id, default=255, dtype=np.uint8)
        ood = map_color_dict(np.array(target), self.color_to_ood_id, default=255, dtype=np.uint8)

        label = Image.fromarray(label)
        ood = Image.fromarray(ood)

        if self.transform:
            image, label, ood = self.transform(image, label, ood)

        return image, label, ood

    def __len__(self):
        return len(self.images)
