import torch
from torch import nn
from torch.nn import functional as F

class Evidential(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.backbone = model.backbone
        self.classifier = model.classifier
    
    def forward(self, *args, limit=None, **kwargs):
        output = self.model(*args, **kwargs)
        evidence = output.relu()
        if limit is not None:
            evidence = evidence.clamp(max=limit)
        alpha = evidence + 1
        return alpha

    def activate(self, alpha):
        return alpha / torch.sum(alpha, dim=1, keepdim=True)

    def epistemic(self, alpha):
        return vacuity(alpha)
    
    def aleatoric(self, alpha):
        soft = self.activate(alpha)
        max_soft, hard = soft.max(dim=1)
        return (1 - max_soft)

def vacuity(alpha):
    class_num = alpha.shape[1]
    S = torch.sum(alpha, dim=1)
    v = class_num / S
    return v
