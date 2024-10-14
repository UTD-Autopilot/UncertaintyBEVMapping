import torch
from torch import nn
from torch.nn import functional as F
from .model import Model
from .loss import uce_loss, u_focal_loss, entropy_reg, ood_reg, ood_reg_topk
from .uncertainty import dissonance, vacuity

import einops

class EvidentialTopK(Model):
    def __init__(self, *args, beta_lambda=0.001, ood_lambda=0.01, **kwargs):
        super(EvidentialTopK, self).__init__(*args, **kwargs)

        self.beta_lambda = beta_lambda
        self.ood_lambda = ood_lambda

    @staticmethod
    def aleatoric(alpha, mode='dissonance'):
        if mode == 'aleatoric':
            soft = EvidentialTopK.activate(alpha)
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

    def loss_ood(self, alpha, y, ood, mapped_uncertainty, mapped_labels, top_k=400):
        batch_size = alpha.shape[0]
        channels = alpha.shape[1]
        A = self.loss(alpha, y, reduction='none').squeeze(1)

        epistemic = self.epistemic(alpha)

        mask = (mapped_uncertainty == 1)

        mapped_region_A = A[mask]
        k = min(mapped_region_A.shape[0], top_k * batch_size)

        # print(alpha.shape, epistemic.shape, mapped_uncertainty.shape)
        top_k, top_k_idx = torch.topk(mapped_region_A, k, largest=True)

        top_k_alpha = torch.stack([alpha[:, i][mask][top_k_idx] for i in range(channels)], dim=-1)
        top_k_uncertainty = mapped_uncertainty[mask][top_k_idx]
        # print(top_k_alpha.shape)
        # print(top_k_uncertainty.shape)
        oreg = ood_reg(top_k_alpha, top_k_uncertainty) * self.ood_lambda
        loss = A.mean() + oreg

        return loss, oreg

    def train_step_ood(self, images, intrinsics, extrinsics, labels, ood, mapped_uncertainty, mapped_labels, top_k=400):
        self.opt.zero_grad(set_to_none=True)

        outs = self(images, intrinsics, extrinsics)
        loss, oodl = self.loss_ood(outs, labels.to(self.device), ood, mapped_uncertainty, mapped_labels, top_k=top_k)
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
