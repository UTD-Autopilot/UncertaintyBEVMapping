import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torch 

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

class UCELoss(nn.Module):
    def __init__(self, num_classes, weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.weights = weights
    
    def forward(self, alpha, y):
        y_onehot = F.one_hot(y, num_classes=self.num_classes).permute(0, 3, 1, 2)
        return uce_loss(alpha, y_onehot, weights=self.weights).mean()

def uce_loss(alpha, y, weights=None):
    S = torch.sum(alpha, dim=1, keepdim=True)
    B = y * (torch.digamma(S) - torch.digamma(alpha) + 1e-10)

    if weights is not None:
        B *= weights.view(1, -1, 1, 1)

    A = torch.sum(B, dim=1, keepdim=True)
    return A

def u_focal_loss(alpha, y, weights=None, n=2):
    S = torch.sum(alpha, dim=1, keepdim=True)

    a0 = S
    aj = torch.gather(alpha, 1, torch.argmax(y, dim=1, keepdim=True))

    B = y * torch.exp((torch.lgamma(a0 - aj + n) + torch.lgamma(a0)) -
                      (torch.lgamma(a0 + n) + torch.lgamma(a0 - aj))) * (torch.digamma(a0 + n) - torch.digamma(aj))

    if weights is not None:
        B *= weights.view(1, -1, 1, 1)

    A = torch.sum(B, dim=1, keepdim=True)

    return A

def entropy_reg(alpha, beta_reg=.001):
    alpha = alpha.permute(0, 2, 3, 1)

    reg = D.Dirichlet(alpha).entropy().unsqueeze(1)

    return -beta_reg * reg

def ood_reg(alpha, ood):
    if ood.long().sum() == 0:
        return 0

    alpha = alpha.permute(0, 2, 3, 1)

    alpha_d = D.Dirichlet(alpha)
    target_d = D.Dirichlet(torch.ones_like(alpha))

    reg = D.kl.kl_divergence(alpha_d, target_d).unsqueeze(1)

    return reg[ood.bool()].mean()

def gamma(x):
    return torch.exp(torch.lgamma(x))
