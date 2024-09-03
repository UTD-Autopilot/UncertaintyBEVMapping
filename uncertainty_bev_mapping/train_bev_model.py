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

    map_uncertainty = config['map_uncertainty'] if 'map_uncertainty' in config else False

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

    model_args = {}
    if 'model' in config:
        model_args.update(config['model'])
    print('model_args:')
    print(json.dumps(model_args, indent=4))

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

        for data in train_loader:
            if map_uncertainty:
                images, intrinsics, extrinsics, labels, ood, mapped_uncertainty = data
            else:
                images, intrinsics, extrinsics, labels, ood = data
            t_0 = time()
            ood_loss = None

            if config['ood']:
                if map_uncertainty:
                    outs, preds, loss, ood_loss = model.train_step_ood(images, intrinsics, extrinsics, labels, ood, mapped_uncertainty)
                else:
                    outs, preds, loss, ood_loss = model.train_step_ood(images, intrinsics, extrinsics, labels, ood)
            else:
                outs, preds, loss = model.train_step(images, intrinsics, extrinsics, labels)

            step += 1

            if scheduler is not None:
                scheduler.step()

            if step % 10 == 0:
                print(f"[{epoch}] {step} {loss.item()} {time()-t_0}")

                writer.add_scalar('train/step_time', time() - t_0, step)
                writer.add_scalar('train/loss', loss, step)

                if ood_loss is not None:
                    writer.add_scalar('train/ood_loss', ood_loss, step)
                    writer.add_scalar('train/id_loss', loss-ood_loss, step)

                if config['ood']:
                    save_unc(model.epistemic(outs) / model.epistemic(outs).max(), ood, config['logdir'], "epistemic.png", "ood.png")
                save_pred(preds, labels, config['logdir'])

            if step % 50 == 0:
                iou = get_iou(preds.cpu(), labels)

                print(f'[{epoch}] {step} IOU: {iou}')

                for i in range(0, n_classes):
                    writer.add_scalar(f'train/{classes[i]}_iou', iou[i], step)

        model.eval()

        if map_uncertainty:
            predictions, ground_truth, oods, aleatoric, epistemic, raw, mapped_uncertainty = run_loader(model, val_loader, map_uncertainty=map_uncertainty)
        else:
            predictions, ground_truth, oods, aleatoric, epistemic, raw = run_loader(model, val_loader, map_uncertainty=map_uncertainty)

        iou = get_iou(predictions, ground_truth)

        for i in range(0, n_classes):
            writer.add_scalar(f'val/{classes[i]}_iou', iou[i], epoch)

        print(f"Validation mIOU: {iou}")

        ood_loss = None

        if config['ood']:
            n_samples = len(raw)
            val_loss = 0
            ood_loss = 0

            for i in range(0, n_samples, config['batch_size']):
                raw_batch = raw[i:i + config['batch_size']].to(model.device)
                ground_truth_batch = ground_truth[i:i + config['batch_size']].to(model.device)
                oods_batch = oods[i:i + config['batch_size']].to(model.device)
                if map_uncertainty:
                    mapped_uncertainty_batch = mapped_uncertainty[i:i + config['batch_size']].to(model.device)

                if map_uncertainty:
                    vl, ol = model.loss_ood(raw_batch, ground_truth_batch, oods_batch, mapped_uncertainty_batch)
                else:
                    vl, ol = model.loss_ood(raw_batch, ground_truth_batch, oods_batch)

                val_loss += vl
                ood_loss += ol

            val_loss /= (n_samples / config['batch_size'])
            ood_loss /= (n_samples / config['batch_size'])

            writer.add_scalar('val/ood_loss', ood_loss, epoch)
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/uce_loss', val_loss - ood_loss, epoch)

            uncertainty_scores = epistemic[:256].squeeze(1)
            uncertainty_labels = oods[:256].bool()

            fpr, tpr, rec, pr, auroc, aupr, _ = roc_pr(uncertainty_scores, uncertainty_labels)
            writer.add_scalar('val/ood_auroc', auroc, epoch)
            writer.add_scalar('val/ood_aupr', aupr, epoch)

            print(f'Validation OOD: AUPR={aupr}, AUROC={auroc}')

            if map_uncertainty:
                fpr, tpr, rec, pr, auroc, aupr, _ = roc_pr(mapped_uncertainty[:256].squeeze(1), uncertainty_labels)
                writer.add_scalar(f"val/mapped_ood_auroc", auroc, epoch)
                writer.add_scalar(f"val/mapped_ood_aupr", aupr, epoch)

        else:
            n_samples = len(raw)
            val_loss = 0

            for i in range(0, n_samples, config['batch_size']):
                raw_batch = raw[i:i + config['batch_size']].to(model.device)
                ground_truth_batch = ground_truth[i:i + config['batch_size']].to(model.device)

                vl = model.loss(raw_batch, ground_truth_batch)

                val_loss += vl

            val_loss /= (n_samples / config['batch_size'])

            writer.add_scalar(f'val/loss', val_loss, epoch)

        if ood_loss is not None:
            print(f"Validation loss: {val_loss}, OOD Reg.: {ood_loss}")
        else:
            print(f"Validation loss: {val_loss}")

        model.save(os.path.join(config['logdir'], f'{epoch}.pt'))
