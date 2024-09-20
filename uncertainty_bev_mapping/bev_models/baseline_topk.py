import torch
from torch import nn
from torch.nn import functional as F
from .model import Model
from .uncertainty import entropy
from .loss import ce_loss, focal_loss


class BaselineTopK(Model):
    def __init__(self, *args, **kwargs):
        super(BaselineTopK, self).__init__(*args, **kwargs)

    @staticmethod
    def aleatoric(logits, mode='entropy'):
        if mode == 'aleatoric':
            soft = BaselineTopK.activate(logits)
            max_soft, hard = soft.max(dim=1)
            return (1 - max_soft).unsqueeze(1)
        elif mode == 'entropy':
            return entropy(logits, dim=1)

    @staticmethod
    def epistemic(logits, mode='energy', T=1.0):
        entropy(logits)

    @staticmethod
    def activate(logits):
        return torch.softmax(logits, dim=1)

    def loss(self, logits, target):
        ce = ce_loss(logits, target, weights=self.weights)
        return ce

    def forward(self, images, intrinsics, extrinsics):
        return self.backbone(images, intrinsics, extrinsics)
