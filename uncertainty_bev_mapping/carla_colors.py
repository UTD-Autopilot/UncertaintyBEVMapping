import numpy as np

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

def map_color_dict(img, d, default=0, dtype=np.int32):
    h, w, _ = img.shape[-3:]
    result = np.full((h, w), default, dtype=dtype)
    for color, value in d.items():
        mask = np.all(img == color, axis=-1)
        result[mask] = value
    return result

carla_class_to_color = {x[0]: x[2] for x in carla_classes_id_color}

color_to_train_id = {}
for train_cls_name, train_idx, cls_names in training_classes:
    for data_cls_name in cls_names:
        color_to_train_id[carla_class_to_color[data_cls_name]] = train_idx

color_to_ood_id = {}
for train_cls_name, train_idx, cls_names in ood_classes:
    for data_cls_name in cls_names:
        color_to_ood_id[carla_class_to_color[data_cls_name]] = train_idx

def carla_image_to_train_id(image):
    return map_color_dict(np.array(image), color_to_train_id, default=0, dtype=np.uint8)

def calra_image_to_ood_id(image):
    return map_color_dict(np.array(image), color_to_ood_id, default=0, dtype=np.uint8)
