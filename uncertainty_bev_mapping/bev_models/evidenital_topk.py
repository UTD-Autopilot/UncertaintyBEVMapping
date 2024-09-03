import torch
from torch import nn
from torch.nn import functional as F
import einops
from .model import Model
from .loss import uce_loss, u_focal_loss, entropy_reg, ood_reg_topk
from .uncertainty import dissonance, vacuity


class EvidentialTopK(Model):
    def __init__(self, *args, beta_lambda=0.001, ood_lambda=0.01, k=64, **kwargs):
        super(EvidentialTopK, self).__init__(*args, **kwargs)

        self.beta_lambda = beta_lambda
        self.ood_lambda = ood_lambda
        self.k = k

        print(f"BETA LAMBDA: {self.beta_lambda}")

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

    def loss_ood(self, alpha, y, ood, mapped_uncertainty, K=40):
        A = self.loss(alpha, y, reduction='none')
        # A *= 1 + (self.epistemic(alpha).detach() * self.k)
        A = A[~ood.bool()].mean()
    
        epistemic = self.epistemic(alpha)
        top_k, top_k_idx = torch.topk(epistemic.reshape(epistemic.shape[0], -1), K)

        reg_alpha = torch.gather(alpha.reshape(alpha.shape[0], alpha.shape[1], -1), 2, einops.repeat(top_k_idx, 'b k -> b c k', c=alpha.shape[1]))
        reg_mapped_uncertainty = torch.gather(mapped_uncertainty.reshape(mapped_uncertainty.shape[0], -1), 1, top_k_idx)

        oreg = ood_reg_topk(reg_alpha, reg_mapped_uncertainty) * self.ood_lambda

        A += oreg

        return A, oreg

    def train_step_ood(self, images, intrinsics, extrinsics, labels, ood, mapped_uncertainty):
        self.opt.zero_grad(set_to_none=True)

        outs = self(images, intrinsics, extrinsics)
        loss, oodl = self.loss_ood(outs, labels.to(self.device), ood, mapped_uncertainty.to(self.device))
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 5.0)
        self.opt.step()

        preds = self.activate(outs)
        return outs, preds, loss, oodl

    def forward(self, images, intrinsics, extrinsics, limit=None):
        if self.tsne:
            print("Returning intermediate")
            return self.backbone(images, intrinsics, extrinsics)

        evidence = self.backbone(images, intrinsics, extrinsics).relu()

        if limit is not None:
            evidence = evidence.clamp(max=limit)
        alpha = evidence + 1

        return alpha

