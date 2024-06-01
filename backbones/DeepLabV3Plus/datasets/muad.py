import json
import os
import sys

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

import torch
import torch.utils.data as data


class MUADDataset(data.Dataset):
    """MUAD dataset, based on the CityScapes class definition.

    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # Based on https://github.com/mcordts/cityscapesScripts
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

    def __init__(self, root, transform=None):
        self.root = os.path.expanduser(root)
        self.images_dir = os.path.join(self.root, 'leftImg8bit')
        self.labels_dir = os.path.join(self.root, 'leftLabel')
        self.depth_dir = os.path.join(self.root, 'leftDepth')
        self.annotation_dir = os.path.join(self.root, 'leftAnnotation')
        self.transform = transform

        self.images = []
        self.labels = []

        self.muad_class_to_id = {x[0]: x[1] for x in self.muad_classes_id_color}

        self.id_to_train_id = np.zeros((256), dtype=np.uint8)
        for train_cls_name, train_idx, cls_names in self.training_classes:
            for data_cls_name in cls_names:
                self.id_to_train_id[self.muad_class_to_id[data_cls_name]] = train_idx
        
        self.id_to_ood_id = np.zeros((256), dtype=np.uint8)
        for train_cls_name, train_idx, cls_names in self.ood_classes:
            for data_cls_name in cls_names:
                self.id_to_ood_id[self.muad_class_to_id[data_cls_name]] = train_idx
        
        self.num_classes = len(self.training_classes)
        self.num_ood_classes = len(self.ood_classes)

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.labels_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')
        
        for file_name in os.listdir(self.images_dir):
            if not file_name.endswith('.png'):
                continue
            data_id = file_name.split('_leftImg8bit')[0]
            suffix = file_name.split('_leftImg8bit')[1].split('.png')[0]
            self.images.append(os.path.join(self.images_dir, f'{data_id}_leftImg8bit{suffix}.png'))
            self.labels.append(os.path.join(self.labels_dir, f'{data_id}_leftLabel{suffix}.png'))

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
        target = Image.open(self.labels[index]).convert('L')
        
        label = self.id_to_train_id[np.array(target)]
        ood = self.id_to_ood_id[np.array(target)]

        label = Image.fromarray(label)
        ood = Image.fromarray(ood)

        if self.transform:
            image, label, ood = self.transform(image, label, ood)

        return image, label, ood

    def __len__(self):
        return len(self.images)
