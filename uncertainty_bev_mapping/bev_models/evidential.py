import torch
from torch import nn
from torch.nn import functional as F
from .model import Model
from .loss import uce_loss, u_focal_loss, entropy_reg, ood_reg
from .uncertainty import dissonance, vacuity


class Evidential(Model):
    def __init__(self, *args, beta_lambda=0.001, ood_lambda=0.01, k=64, **kwargs):
        super(Evidential, self).__init__(*args, **kwargs)

        self.beta_lambda = beta_lambda
        self.ood_lambda = ood_lambda
        self.k = k

        print(f"BETA LAMBDA: {self.beta_lambda}")

    @staticmethod
    def aleatoric(alpha, mode='dissonance'):
        if mode == 'aleatoric':
            soft = Evidential.activate(alpha)
            max_soft, hard = soft.max(dim=1)
            return (1 - max_soft).unsqueeze(1)
        elif mode == 'dissonance':
            return dissonance(alpha)

    @staticmethod
    def epistemic(alpha):
        return vacuity(alpha)

    @staticmethod
    def activate(alpha):
        return alpha / torch.sum(alpha, dim=1, keepdim=True)

    def loss(self, alpha, y, reduction='mean'):
        if self.loss_type == 'ce':
            A = uce_loss(alpha, y, weights=self.weights) + entropy_reg(alpha, beta_reg=self.beta_lambda)
        elif self.loss_type == 'focal':
            A = u_focal_loss(alpha, y, weights=self.weights, n=self.gamma)
        else:
            raise NotImplementedError()

        if reduction == 'mean':
            return A.mean()
        else:
            return A

    def loss_ood(self, alpha, y, ood):
        A = self.loss(alpha, y, reduction='none')
        A *= 1 + (self.epistemic(alpha).detach() * self.k)

        oreg = ood_reg(alpha, ood) * self.ood_lambda
        A = A[~ood.bool()].mean()

        A += oreg

        return A, oreg

    def train_step_ood(self, images, intrinsics, extrinsics, labels, ood):
        self.opt.zero_grad(set_to_none=True)

        outs = self(images, intrinsics, extrinsics)
        loss, oodl = self.loss_ood(outs, labels.to(self.device), ood)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 5.0)
        self.opt.step()

        preds = self.activate(outs)
        return outs, preds, loss, oodl

    def forward(self, images, intrinsics, extrinsics, limit=None):
        evidence = self.backbone(images, intrinsics, extrinsics).relu()

        if limit is not None:
            evidence = evidence.clamp(max=limit)
        alpha = evidence + 1.0

        return alpha
