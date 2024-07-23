import os
import numpy as np
from PIL import Image
import sklearn.metrics
import torch
import tqdm
import json
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score
from uncertainty_bev_mapping.carla_colors import training_classes, ood_classes, carla_image_to_train_id, calra_image_to_ood_id

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def calc_iou(preds, labels, num_classes, exclude=None):
    iou = np.zeros(num_classes)
    for i in range(num_classes):
        p = (preds == i)
        l = (labels == i)

        if exclude is not None:
            p &= ~exclude
            l &= ~exclude

        intersect = (p & l).sum()
        union = (p | l).sum()
        iou[i] = intersect / union if union > 0 else 0
    return iou

def calc_roc_pr(uncertainty_scores, uncertainty_labels, exclude=None):
    y_true = uncertainty_labels.reshape(-1)
    y_score = uncertainty_scores.reshape(-1)

    if exclude is not None:
        include = ~exclude.reshape(-1)
        y_true = y_true[include]
        y_score = y_score[include]

    pr, rec, tr = precision_recall_curve(y_true, y_score, drop_intermediate=True)
    fpr, tpr, _ = roc_curve(y_true, y_score, drop_intermediate=True)

    auroc = auc(fpr, tpr)
    aupr = average_precision_score(y_true, y_score)

    no_skill = np.sum(y_true) / len(y_true)

    return fpr, tpr, rec, pr, auroc, aupr, no_skill

def evaluate_bev_mapping(dataset_path, save_path):
    agent_paths = []
    for run in os.listdir(dataset_path):
        for agent_id in os.listdir(os.path.join(dataset_path, run, 'agents')):
            agent_paths.append(os.path.join(dataset_path, run, 'agents', agent_id))

    num_classes = len(training_classes)
    num_ood_classes = len(ood_classes)
    all_iou = []
    num_frames = 0
    uncertainty_scores = []
    uncertainty_labels = []
    
    for agent_path in tqdm.tqdm(agent_paths):
        for frame in os.listdir(os.path.join(agent_path, 'birds_view_camera')):
            if not frame.endswith('.png'):
                continue
            frame = frame.split('.png')[0]
            bev_aleatoric_path = os.path.join(agent_path, f'bev_mapping_aleatoric/{frame}.npy')
            bev_epistemic_path = os.path.join(agent_path, f'bev_mapping_epistemic/{frame}.npy')
            bev_semantic_path = os.path.join(agent_path, f'birds_view_semantic_camera/{frame}.png')
            bev_pred_path = os.path.join(agent_path, f'bev_mapping_pred/{frame}.png')

            bev_aleatoric = np.load(bev_aleatoric_path)
            bev_epistemic = np.load(bev_epistemic_path)
            
            bev_semantic_image = np.array(Image.open(bev_semantic_path))
            bev_semantic_pred_image = np.array(Image.open(bev_pred_path))

            bev_semantic_gt = carla_image_to_train_id(bev_semantic_image)
            bev_ood_gt = calra_image_to_ood_id(bev_semantic_image)

            bev_semantic_pred = carla_image_to_train_id(bev_semantic_pred_image)
            iou = calc_iou(bev_semantic_pred, bev_semantic_gt, num_classes)

            uncertainty_scores.append(bev_epistemic)
            uncertainty_labels.append(bev_ood_gt==1) # Just calculate for the first class (animals)

            num_frames += 1
            all_iou.append(iou)
    
    pred_miou = np.mean(all_iou, axis=0)
    uncertainty_scores = np.array(uncertainty_scores)
    uncertainty_labels = np.array(uncertainty_labels)

    fpr, tpr, rec, pr, auroc, aupr, no_skill = calc_roc_pr(uncertainty_scores, uncertainty_labels)

    results = {
        'pred_miou': pred_miou.tolist(),
        'auroc': auroc,
        'aupr': aupr,
    }
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'results.json'), 'w') as f:
        json.dump(results, f)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(left=0.05, right=0.95)

    axs[0].plot(fpr, tpr, '-', label=f'AUROC: {auroc:.3f}')
    axs[1].step(rec, pr, '-', where='post', label=f'AUPR: {aupr:.3f}')

    axs[0].set_xlim([-0.05, 1.05])
    axs[0].set_ylim([-0.05, 1.05])
    axs[0].set_title(f"AUROC")
    axs[0].legend(frameon=True)
    axs[1].set_xlim([-0.05, 1.05])
    axs[1].set_ylim([-0.05, 1.05])
    axs[1].set_title(f"AUPR")
    axs[1].legend(frameon=True)

    fig.savefig(os.path.join(save_path, f"ood_metrics.png"), format='png')
    fig.savefig(os.path.join(save_path, f"ood_metrics.svg"), format='svg')

    return num_frames, pred_miou

if __name__ == '__main__':
    # evaluate_bev_mapping('../../Datasets/carla/train', 'outputs/evaluate_bev_mapping/train')
    evaluate_bev_mapping('../../Datasets/carla/val', 'outputs/evaluate_bev_mapping/val')
    # evaluate_bev_mapping('../../Datasets/carla/test', 'outputs/evaluate_bev_mapping/test')
