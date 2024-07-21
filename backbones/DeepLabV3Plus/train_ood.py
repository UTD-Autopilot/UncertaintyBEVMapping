from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets.muad import MUADDataset
from datasets.carla import CarlaDataset
from utils import ext_transforms as et
from metrics import StreamSegMetrics
from metrics.uncertainty import unc_iou, roc_pr

import torch
import torch.nn as nn

from network.evidential import Evidential
from utils.loss import UCELoss, entropy_reg

from torch.utils.tensorboard import SummaryWriter

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

def validate(model, loader, device, criterion, metrics, save_val_results=True):
    """Do validation and return specified samples"""
    metrics.reset()
    if save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        total_loss = 0
        ood_auroc = {k: [] for k in range(1, loader.dataset.num_ood_classes)}
        ood_aupr = {k: [] for k in range(1, loader.dataset.num_ood_classes)}
        for i, (images, labels, oods) in tqdm(enumerate(loader), total=len(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            oods = oods.to(device, dtype=torch.long)

            outputs = model(images)
            loss = criterion(outputs, labels)
            np_loss = np.sum(loss.detach().cpu().numpy())
            total_loss += np_loss
            preds = model.module.activate(outputs).argmax(dim=1).detach().cpu().numpy()

            aleatoric = model.module.aleatoric(outputs)
            epistemic = model.module.epistemic(outputs)

            uncertainty_scores = epistemic
            for ood_class in range(1, loader.dataset.num_ood_classes):
                uncertainty_labels = (oods == ood_class)
                fpr, tpr, rec, pr, auroc, aupr, _ = roc_pr(uncertainty_scores, uncertainty_labels)
                ood_auroc[ood_class].append(auroc)
                ood_aupr[ood_class].append(aupr)

            targets = labels.detach().cpu().numpy()
            oods = oods.detach().cpu().numpy()
            metrics.update(preds, targets, oods)

            if save_val_results:
                for j in range(len(images)):
                    image = images[j].detach().cpu().numpy()
                    target = targets[j]
                    ood = oods[j]
                    pred = preds[j]
                    alea = aleatoric[j].detach().cpu().numpy()
                    epis = epistemic[j].detach().cpu().numpy()

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.visualize_image(target).astype(np.uint8)
                    ood = loader.dataset.visualize_ood(ood).astype(np.uint8)
                    pred = loader.dataset.visualize_image(pred).astype(np.uint8)
                    alea = ((alea + alea.min()) / (alea.max() - alea.min()) * 255.0).astype(np.uint8)
                    epis = ((epis + epis.min()) / (epis.max() - epis.min()) * 255.0).astype(np.uint8)

                    Image.fromarray(image).save(f'results/{img_id}_image.png')
                    Image.fromarray(target).save(f'results/{img_id}_target.png')
                    Image.fromarray(ood).save(f'results/{img_id}_ood.png')
                    Image.fromarray(pred).save(f'results/{img_id}_pred.png')
                    Image.fromarray(alea).save(f'results/{img_id}_aleatoric.png')
                    Image.fromarray(epis).save(f'results/{img_id}_epistemic.png')

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1
        total_loss /= len(loader)
        score = metrics.get_results()
        score['loss'] = total_loss
        ood_auroc = {k: np.mean(v) for k, v in ood_auroc.items()}
        ood_aupr = {k: np.mean(v) for k, v in ood_aupr.items()}
        score['ood_auroc'] = ood_auroc
        score['ood_aupr'] = ood_aupr
    return score

def train_muad_ood(
        dataset_root,
        dataset, # carla or muad
        model_name='deeplabv3plus_mobilenet',
        ckpt=None, load_ckpt=False,
        batch_size=8, lr=0.1, weight_decay=1e-5,
        max_steps=30000,
        val_interval=200,
        seed=43,
        test_only=False,
    ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    writer = SummaryWriter()

    if dataset == 'muad':
        # 1024x2048 -> resize to 900x1800 -> crop from center to 900x1600 -> resize to 450x800
        train_transform = et.ExtCompose([
            et.ExtResize((450, 900)),
            et.ExtCenterCrop((450, 800)),
            # et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            et.ExtResize((450, 900)),
            et.ExtCenterCrop((450, 800)),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        train_dataset = MUADDataset(os.path.join(dataset_root, 'train'), transform=train_transform)
        val_dataset = MUADDataset(os.path.join(dataset_root, 'test_sets/test_OOD'), transform=val_transform)
        # test_dataset = MUADDataset(os.path.join(dataset_root, 'test_sets/test_level2'), transform=val_transform)
    elif dataset == 'carla':
        # 900x1600 -> resize to 450x800
        train_transform = et.ExtCompose([
            et.ExtResize((450, 800)),
            # et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            et.ExtResize((450, 800)),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dataset = CarlaDataset(os.path.join(dataset_root, 'train'), transform=train_transform)
        val_dataset = CarlaDataset(os.path.join(dataset_root, 'val'), transform=val_transform)
    else:
        raise ValueError(f'Dataset {dataset} not supported, must be one of [carla, muad]')

    num_classes = train_dataset.num_classes

    train_loader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True,
    )
    val_loader = data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
    )
    # test_loader = data.DataLoader(
    #     test_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
    # )

    output_stride = 8
    model = network.modeling.__dict__[model_name](num_classes=train_dataset.num_classes, output_stride=output_stride)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    model = Evidential(model)

    metrics = StreamSegMetrics(num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * lr},
        {'params': model.classifier.parameters(), 'lr': lr},
    ], lr=lr, momentum=0.9, weight_decay=weight_decay)

    scheduler = utils.PolyLR(optimizer, max_steps, power=0.9)
    # criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    class_weights = torch.tensor(train_dataset.training_class_weights, device=device)
    criterion = UCELoss(num_classes, weights=class_weights)

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)
    
    utils.mkdir('checkpoints')

    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if ckpt is not None and os.path.isfile(ckpt):
        checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if load_ckpt:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % ckpt)
        print("Model restored from %s" % ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    # ==========   Train Loop   ==========#
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if test_only:
        model.eval()
        val_score = validate(
            model=model, loader=val_loader, device=device, metrics=metrics
        )
        print(metrics.to_str(val_score))
        return

    interval_loss = 0
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for (images, labels, oods) in train_loader:
            cur_itrs += 1

            # print(images.shape, labels.shape, oods.shape)

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            oods = oods.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            # print(outputs.shape, labels.shape)
            loss = criterion(outputs, labels) + entropy_reg(outputs, beta_reg=0.001).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            writer.add_scalar('loss/train', np_loss, cur_itrs)

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, max_steps, interval_loss))
                interval_loss = 0.0

            if (cur_itrs) % val_interval == 0:
                save_ckpt(f'checkpoints/latest_{model_name}.pth')
                print("validation...")
                model.eval()
                val_score = validate(
                    model=model, loader=val_loader, device=device, criterion=criterion, metrics=metrics
                )
                print(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt(f'checkpoints/best_{model_name}.pth')

                writer.add_scalar("loss/val", val_score['loss'], cur_itrs)
                writer.add_scalar("accuracy/val", val_score['Overall Acc'], cur_itrs)
                writer.add_scalar("IoU/val", val_score['Mean IoU'], cur_itrs)
                for cls, value in val_score['Class IoU'].items():
                    writer.add_scalar(f'iou/val/{cls}', value, cur_itrs)
                for ood_class, value in val_score['ood_auroc'].items():
                    writer.add_scalar(f'ood_auroc/val/{ood_class}', value, cur_itrs)
                for ood_class, value in val_score['ood_aupr'].items():
                    writer.add_scalar(f'ood_aupr/val/{ood_class}', value, cur_itrs)
                model.train()

            scheduler.step()

            if cur_itrs >= max_steps:
                return

if __name__ == '__main__':
    # train_muad_ood('../../Datasets/MUAD/train+val+tests', 'muad')
    train_muad_ood('~/data/Datasets/carla', 'carla')
