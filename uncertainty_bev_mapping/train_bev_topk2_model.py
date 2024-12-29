from time import time, sleep
import os
import torch
from torch import nn
import numpy as np
import tqdm

from tensorboardX import SummaryWriter
from .bev_models.metrics import get_iou, roc_pr
from .utils import save_pred, save_unc
from .datasets import datasets
from .bev_models import models

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
torch.multiprocessing.set_sharing_strategy('file_system')

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

print(torch.__version__)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(config, dataroot, split='trainval'):

    # workaround https://github.com/pytorch/pytorch/issues/90613
    for gpu in config['gpus']:
        torch.inverse(torch.ones((1, 1), device=gpu))

    classes = config['classes']
    n_classes = config['n_classes']
    weights = config['weights']

    train_set = config['train_set']
    val_set = config['val_set']

    if config['backbone'] == 'lss' or config['backbone'] == 'simplebev' or config['backbone'] == 'pointbev':
        yaw = 0
    elif config['backbone'] == 'cvt':
        yaw = 180
    else:
        raise NotImplementedError(f"yaw correction for backbone model {config['backbone']} not defined")

    train_loader = datasets[config['dataset']](
        train_set, split, dataroot, config['pos_class'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        is_train=True,
        seed=config['seed'],
        yaw=yaw,
        map_uncertainty=True,
        map_label_expand_size=config['map_label_expand_size'],
    )

    val_loader = datasets[config['dataset']](
        val_set, split, dataroot, config['pos_class'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        is_train=False,
        seed=config['seed'],
        yaw=yaw,
        map_uncertainty=True,
        map_label_expand_size=config['map_label_expand_size'],
    )

    model = models[config['type']](
        config['gpus'],
        backbone=config['backbone'],
        n_classes=n_classes,
        loss_type=config['loss'],
        weights=weights
    )

    model.opt = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    if 'pretrained' in config:
        model.load(torch.load(config['pretrained']))
        print(f"Loaded pretrained weights: {config['pretrained']}")
        scheduler = None
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            model.opt,
            div_factor=10,
            pct_start=.3,
            final_div_factor=10,
            max_lr=config['learning_rate'],
            epochs=config['num_epochs'],
            steps_per_epoch=len(train_loader.dataset) // config['batch_size']
        )

    top_k = config['top_k']
    top_k_criterion = nn.CrossEntropyLoss()

    print("--------------------------------------------------")
    print(f"Using GPUS: {config['gpus']}")
    print(f"Train loader: {len(train_loader.dataset)}")
    print(f"Val loader: {len(val_loader.dataset)}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Output directory: {config['logdir']} ")
    print(f"Using loss {config['loss']}")
    print("--------------------------------------------------")

    writer = SummaryWriter(logdir=config['logdir'])

    writer.add_text("config", str(config))

    step = 0

    # enable to catch errors in loss function
    # torch.autograd.set_detect_anomaly(True)
    device = model.device

    for epoch in range(config['num_epochs']):
        model.train()

        writer.add_scalar('train/epoch', epoch, step)

        total_aupr = []
        total_auroc = []

        lambda_step_length = 1 / np.sqrt(epoch+1)

        for data in train_loader:
            images, intrinsics, extrinsics, labels, ood, mapped_uncertainty, mapped_labels = data
            labels = labels.to(device)
            mapped_uncertainty = mapped_uncertainty.to(device)
            batch_size = labels.shape[0]
            channels = labels.shape[1]

            t_0 = time()
            ood_loss = None

            if config['ood']:
                mapped_uncertainty = mapped_uncertainty.squeeze(1) # remove channel dim
                outs, preds, loss, ood_loss = model.train_step_ood(images, intrinsics, extrinsics, labels, ood, mapped_uncertainty, mapped_labels, top_k=top_k)
                top_k_loss = ood_loss
            else:
                # only works for class 0 (vehicle)
                model.opt.zero_grad(set_to_none=True)
                lr = get_lr(model.opt)
                outs = model(images, intrinsics, extrinsics)
                ce = model.loss(outs, mapped_labels.to(model.device), reduction='none')

                # Mapped labels will be extended to be larger than true labels in dataloader
                mask = (mapped_labels[:, 0] == 1)

                # apply standard CE loss for none mapped region
                ce_loss = ce[~mask].mean()

                mapped_region_ce = ce[mask]
                k = min(mapped_region_ce.shape[0], top_k * batch_size)
                top_k_ce, top_k_idx = torch.topk(mapped_region_ce, k, largest=True)
                # top_k_idx is an index for (b h w)
                top_k_outs = torch.stack([outs[:, i][mask][top_k_idx] for i in range(channels)], dim=-1)

                # label 0 is vehicle
                top_k_loss = top_k_criterion(top_k_outs, torch.full((top_k_outs.shape[0],), 0, dtype=torch.long, device=top_k_outs.device)).mean()

                loss = ce_loss + top_k_loss

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                model.opt.step()

                preds = model.activate(outs)

            step += 1

            if scheduler is not None:
                scheduler.step()

            if step % 50 == 0:
                print(f"[{epoch}] {step} {loss.item()} {time()-t_0}")

                writer.add_scalar('train/step_time', time() - t_0, step)
                writer.add_scalar('train/loss', loss, step)
                writer.add_scalar('train/top_k_loss', top_k_loss, step)
                # writer.add_scalar('train/ce_loss', ce_loss, step)

                if ood_loss is not None:
                    writer.add_scalar('train/ood_loss', ood_loss, step)
                    writer.add_scalar('train/id_loss', loss-ood_loss, step)

                if config['ood']:
                    epistemic = model.epistemic(outs)
                    save_unc(epistemic / epistemic.max(), ood, config['logdir'], "epistemic.png", "ood.png")
                    uncertainty_scores = epistemic.squeeze(1)
                    uncertainty_labels = ood.bool()

                    if torch.sum(uncertainty_labels) > 0:
                        fpr, tpr, rec, pr, auroc, aupr, _ = roc_pr(uncertainty_scores.detach().cpu(), uncertainty_labels.detach().cpu())

                        writer.add_scalar('train/step_ood_aupr', aupr, step)
                        writer.add_scalar('train/step_ood_auroc', auroc, step)

                        total_aupr.append(aupr.item())
                        total_auroc.append(auroc.item())

                save_pred(preds, labels, config['logdir'])

                iou = get_iou(preds, labels)

                print(f'[{epoch}] {step} IOU: {iou}')

                for i in range(0, n_classes):
                    writer.add_scalar(f'train/{classes[i]}_iou', iou[i], step)

        if config['ood']:
            aupr = np.mean(total_aupr)
            auroc = np.mean(total_auroc)
            writer.add_scalar('train/aupr', aupr, epoch)
            writer.add_scalar('train/auroc', auroc, epoch)

        model.eval()

        with torch.no_grad():
            total_loss = []
            total_ood_loss = []
            total_id_loss = []
            total_top_k_loss = []
            total_aupr = []
            total_auroc = []
            total_ious = [[] for _ in range(n_classes)]

            for data in val_loader:
                images, intrinsics, extrinsics, labels, ood, mapped_uncertainty, mapped_labels = data
                labels = labels.to(device)
                mapped_uncertainty = mapped_uncertainty.to(device)
                t_0 = time()
                ood_loss = None

                if config['ood']:
                    mapped_uncertainty = mapped_uncertainty.squeeze(1)
                    outs = model(images, intrinsics, extrinsics)
                    loss, ood_loss = model.loss_ood(outs, labels, ood, mapped_uncertainty, mapped_labels, top_k=top_k)
                    top_k_loss = ood_loss
                    preds = model.activate(outs)
                else:
                    outs = model(images, intrinsics, extrinsics)
                    preds = model.activate(outs)

                    ce = model.loss(outs, mapped_labels.to(model.device))
                    mask = (mapped_labels[:, 0] == 1)
                    ce_loss = ce[~mask].mean()

                    mapped_region_ce = ce[mask]
                    k = min(mapped_region_ce.shape[0], top_k * batch_size)
                    top_k_ce, top_k_idx = torch.topk(mapped_region_ce, k, largest=True)
                    top_k_outs = torch.stack([outs[:, i][mask][top_k_idx] for i in range(channels)], dim=-1)

                    top_k_loss = top_k_criterion(top_k_outs, torch.full((top_k_outs.shape[0],), 0, dtype=torch.long, device=top_k_outs.device)).mean()

                    loss = ce_loss + top_k_loss

                total_loss.append(loss.item())
                total_top_k_loss.append(top_k_loss.item())

                if ood_loss is not None:
                    total_ood_loss.append(ood_loss.item())
                    total_id_loss.append((loss-ood_loss).item())

                if config['ood']:
                    epistemic = model.epistemic(outs)
                    save_unc(epistemic / epistemic.max(), ood, config['logdir'], "epistemic.png", "ood.png")
                    uncertainty_scores = epistemic.squeeze(1)
                    uncertainty_labels = ood.bool()

                    if torch.sum(uncertainty_labels) > 0:
                        fpr, tpr, rec, pr, auroc, aupr, _ = roc_pr(uncertainty_scores.detach().cpu(), uncertainty_labels.detach().cpu())
                        total_aupr.append(aupr.item())
                        total_auroc.append(auroc.item())

                save_pred(preds, labels, config['logdir'])

                iou = get_iou(preds, labels)

                print(f'[{epoch}] {step} IOU: {iou}')

                for i in range(0, n_classes):
                    total_ious[i].append(iou[i])

            loss = np.mean(total_loss)
            top_k_loss = np.mean(total_top_k_loss)
            writer.add_scalar('val/loss', loss, epoch)
            writer.add_scalar('val/top_k_loss', top_k_loss)

            if config['ood']:
                ood_loss = np.mean(total_ood_loss)
                id_loss = np.mean(total_id_loss)
                writer.add_scalar('val/ood_loss', ood_loss, epoch)
                writer.add_scalar('val/id_loss', id_loss, epoch)

                aupr = np.mean(total_aupr)
                auroc = np.mean(total_auroc)
                writer.add_scalar('val/aupr', aupr, epoch)
                writer.add_scalar('val/auroc', auroc, epoch)

            for i in range(0, n_classes):
                iou = np.mean(total_ious[i])
                writer.add_scalar(f'val/{classes[i]}_iou', iou, epoch)

        model.save(os.path.join(config['logdir'], f'{epoch}.pt'))
