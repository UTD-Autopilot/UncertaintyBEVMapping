from time import time, sleep
import os
import torch
import numpy as np
import tqdm
import json

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

@torch.no_grad()
def run_loader(model, loader, map_uncertainty=False):
    predictions = []
    ground_truth = []
    oods = []
    aleatoric = []
    epistemic = []
    raw = []
    mapped_uncertainties = []

    with torch.no_grad():
        for data in tqdm.tqdm(loader, desc="Running validation"):
            if map_uncertainty:
                images, intrinsics, extrinsics, labels, ood, mapped_uncertainty = data
            else:
                images, intrinsics, extrinsics, labels, ood = data
            outs = model(images, intrinsics, extrinsics).detach().cpu()

            predictions.append(model.activate(outs))
            ground_truth.append(labels)
            oods.append(ood)
            aleatoric.append(model.aleatoric(outs))
            epistemic.append(model.epistemic(outs))
            raw.append(outs)
            if map_uncertainty:
                mapped_uncertainties.append(mapped_uncertainty)

    if map_uncertainty:
        return (torch.cat(predictions, dim=0),
                torch.cat(ground_truth, dim=0),
                torch.cat(oods, dim=0),
                torch.cat(aleatoric, dim=0),
                torch.cat(epistemic, dim=0),
                torch.cat(raw, dim=0),
                torch.cat(mapped_uncertainties, dim=0))
    else:
        return (torch.cat(predictions, dim=0),
                    torch.cat(ground_truth, dim=0),
                    torch.cat(oods, dim=0),
                    torch.cat(aleatoric, dim=0),
                    torch.cat(epistemic, dim=0),
                    torch.cat(raw, dim=0))


def train(config, dataroot, split='trainval'):

    # workaround https://github.com/pytorch/pytorch/issues/90613
    for gpu in config['gpus']:
        torch.inverse(torch.ones((1, 1), device=gpu))

    classes = config['classes']
    n_classes = config['n_classes']
    weights = config['weights']

    train_set = config['train_set']
    val_set = config['val_set']

    if config['backbone'] == 'lss':
        yaw = 0
    elif config['backbone'] == 'cvt':
        yaw = 180

    map_uncertainty = config['type'].endswith('_topk')

    train_loader = datasets[config['dataset']](
        train_set, split, dataroot, config['pos_class'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        is_train=True,
        seed=config['seed'],
        yaw=yaw,
        map_uncertainty=map_uncertainty,
    )

    val_loader = datasets[config['dataset']](
        val_set, split, dataroot, config['pos_class'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        is_train=False,
        seed=config['seed'],
        yaw=yaw,
        map_uncertainty=map_uncertainty,
    )

    model = models[config['type']](
        config['gpus'],
        backbone=config['backbone'],
        n_classes=n_classes,
        loss_type=config['loss'],
        weights=weights,
        **model_args,
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

    if 'gamma' in config:
        model.gamma = config['gamma']
        print(f"GAMMA: {model.gamma}")

    if 'ol' in config:
        model.ood_lambda = config['ol']
        print(f"OOD LAMBDA: {model.ood_lambda}")

    if 'k' in config:
        model.k = config['k']

    if 'beta' in config:
        model.beta_lambda = config['beta']
        print(f"Beta lambda is {model.beta_lambda}")

    if 'm_in' in config:
        model.m_in = config['m_in']
    if 'm_out' in config:
        model.m_out = config['m_out']

    print("--------------------------------------------------")
    print(f"Using GPUS: {config['gpus']}")
    print(f"Train loader: {len(train_loader.dataset)}")
    print(f"Val loader: {len(val_loader.dataset)}")
    print(f"Train set: {train_set} Val set: {val_set}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Output directory: {config['logdir']} ")
    print(f"Using loss {config['loss']}")
    print(f"Use mapped uncertainty as regularization: {map_uncertainty}")
    print("--------------------------------------------------")

    writer = SummaryWriter(logdir=config['logdir'])

    writer.add_text("config", str(config))
    with open((os.path.join(config['logdir'], f'config.json')), 'w') as f:
        json.dump(config, f, indent=4)

    step = 0

    # enable to catch errors in loss function
    # torch.autograd.set_detect_anomaly(True)

    for epoch in range(config['num_epochs']):
        model.train()

        writer.add_scalar('train/epoch', epoch, step)

        total_aupr = []
        total_auroc = []

        for data in train_loader:
            if map_uncertainty:
                images, intrinsics, extrinsics, labels, ood, mapped_uncertainty, mapped_labels = data
            else:
                images, intrinsics, extrinsics, labels, ood = data

            t_0 = time()
            ood_loss = None

            if config['ood']:
                if map_uncertainty:
                    outs, preds, loss, ood_loss = model.train_step_ood(images, intrinsics, extrinsics, labels, ood, mapped_uncertainty, mapped_labels)
                else:
                    outs, preds, loss, ood_loss = model.train_step_ood(images, intrinsics, extrinsics, labels, ood)
            else:
                outs, preds, loss = model.train_step(images, intrinsics, extrinsics, labels)

            step += 1

            if scheduler is not None:
                scheduler.step()

            if step % 50 == 0:
                print(f"[{epoch}] {step} {loss.item()} {time()-t_0}")

                writer.add_scalar('train/step_time', time() - t_0, step)
                writer.add_scalar('train/loss', loss, step)

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

                iou = get_iou(preds.cpu(), labels)

                print(f'[{epoch}] {step} IOU: {iou}')

                for i in range(0, n_classes):
                    writer.add_scalar(f'train/{classes[i]}_iou', iou[i], step)

        aupr = np.mean(total_aupr)
        auroc = np.mean(total_auroc)
        writer.add_scalar('train/aupr', aupr, epoch)
        writer.add_scalar('train/auroc', auroc, epoch)

        model.eval()

        with torch.no_grad():
            total_loss = []
            total_ood_loss = []
            total_id_loss = []
            total_aupr = []
            total_auroc = []
            total_ious = [[] for _ in range(n_classes)]

            for data in val_loader:
                if map_uncertainty:
                    images, intrinsics, extrinsics, labels, ood, mapped_uncertainty, mapped_labels = data
                else:
                    images, intrinsics, extrinsics, labels, ood = data

                t_0 = time()
                ood_loss = None

                if config['ood']:
                    if map_uncertainty:
                        outs = model(images, intrinsics, extrinsics, mapped_uncertainty, mapped_labels)
                        loss, ood_loss = model.loss_ood(outs, labels.to(model.device), ood, mapped_uncertainty, mapped_labels)
                        preds = model.activate(outs)
                    else:
                        outs = model(images, intrinsics, extrinsics)
                        loss, ood_loss = model.loss_ood(outs, labels.to(model.device), ood)
                        preds = model.activate(outs)
                else:
                    outs = model(images, intrinsics, extrinsics)
                    loss = model.loss(outs, labels.to(model.device))
                    preds = model.activate(outs)

                total_loss.append(loss.item())

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

                iou = get_iou(preds.cpu(), labels)

                print(f'[{epoch}] {step} IOU: {iou}')

                for i in range(0, n_classes):
                    total_ious[i].append(iou[i])

            loss = np.mean(total_loss)
            ood_loss = np.mean(total_ood_loss)
            id_loss = np.mean(total_id_loss)
            aupr = np.mean(total_aupr)
            auroc = np.mean(total_auroc)

            writer.add_scalar('val/loss', loss, epoch)
            writer.add_scalar('val/ood_loss', ood_loss, epoch)
            writer.add_scalar('val/id_loss', id_loss, epoch)

            writer.add_scalar('val/aupr', aupr, epoch)
            writer.add_scalar('val/auroc', auroc, epoch)

            for i in range(0, n_classes):
                iou = np.mean(total_ious[i])
                writer.add_scalar(f'val/{classes[i]}_iou', iou, epoch)

        model.save(os.path.join(config['logdir'], f'{epoch}.pt'))
