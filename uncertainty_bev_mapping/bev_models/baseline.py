import torch
from torch import nn
from torch.nn import functional as F
from .model import Model
from .uncertainty import entropy
from .loss import ce_loss, focal_loss


class Baseline(Model):
    def __init__(self, *args, m_in=-23.0, m_out=-5.0, ood_lambda=0.1, **kwargs):
        super(Baseline, self).__init__(*args, **kwargs)

        self.m_in = m_in
        self.m_out = m_out
        self.ood_lambda = ood_lambda

    @staticmethod
    def aleatoric(logits, mode='entropy'):
        if mode == 'aleatoric':
            soft = Baseline.activate(logits)
            max_soft, hard = soft.max(dim=1)
            return (1 - max_soft).unsqueeze(1)
        elif mode == 'entropy':
            return entropy(logits, dim=1)

    @staticmethod
    def epistemic(logits, mode='energy', T=1.0):
        if mode == 'energy':
            energy = -T * torch.logsumexp(logits/T, dim=1)
            norm = (energy - energy.min()) / (energy.max() - energy.min())
            return norm.unsqueeze(1)
        elif mode == 'entropy':
            return entropy(logits)

    @staticmethod
    def activate(logits):
        return torch.softmax(logits, dim=1)

    def loss(self, logits, target, reduction='mean'):
        if self.loss_type == 'ce':
            A = ce_loss(logits, target, weights=self.weights)
        elif self.loss_type == 'focal':
            A = focal_loss(logits, target, weights=self.weights, n=self.gamma)
        else:
            raise NotImplementedError()

        if reduction == 'mean':
            return A.mean()
        else:
            return A

    def loss_ood(self, logits, target, ood):
        ind = ~ood.bool()
        ce = self.loss(logits, target, reduction='none')[ind.squeeze(1)]
        Ec_out = -torch.logsumexp(logits.swapaxes(1, -1)[ood.bool().repeat(1, 2, 1, 1).swapaxes(1, -1)].reshape(-1, 2), dim=1)
        Ec_in = -torch.logsumexp(logits.swapaxes(1, -1)[ind.bool().repeat(1, 2, 1, 1).swapaxes(1, -1)].reshape(-1, 2), dim=1)
        energy = torch.pow(F.relu(Ec_in-self.m_in), 2).mean() + torch.pow(F.relu(self.m_out-Ec_out), 2).mean()

        oodl = self.ood_lambda * energy
        loss = ce.mean() + oodl

        return loss, oodl

    def train_step_ood(self, images, intrinsics, extrinsics, labels, ood):
        self.opt.zero_grad(set_to_none=True)

        if self.scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                outs = self(images, intrinsics, extrinsics)
                loss, oodl = self.loss_ood(outs, labels.to(self.device), ood)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)

            nn.utils.clip_grad_norm_(self.parameters(), 5.0)
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            outs = self(images, intrinsics, extrinsics)
            loss, oodl = self.loss_ood(outs, labels.to(self.device), ood)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 5.0)
            self.opt.step()

        preds = self.activate(outs)
        return outs, preds, loss, oodl

    def forward(self, images, intrinsics, extrinsics):
        if self.tsne:
            print("Returning intermediate")
            return self.backbone(images, intrinsics, extrinsics)

        return self.backbone(images, intrinsics, extrinsics)
