import torch
import torch.nn as nn
from torch.nn import functional as F


class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred:torch.Tensor, target: torch.Tensor, smooth=1):
        N = target.size(0)
        sigmoid = nn.Sigmoid()
        input_flat = sigmoid(pred).view(N, -1)
        target_flat = target.view(N, -1)
        intersection = input_flat * target_flat
        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


class ELDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        smooth = 1

        dice = 0.
        # dice系数的定义
        for i in range(pred.size(1)):
            dice += 2 * (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) / (
                        pred[:, i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                        target[:, i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        dice = dice / pred.size(1)
        # 返回的是dice距离
        return torch.clamp((torch.pow(-torch.log(dice + 1e-5), 0.3)).mean(), 0, 2)


class JaccardLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        smooth = 1

        # jaccard系数的定义
        jaccard = 0.

        for i in range(pred.size(1)):
            jaccard += (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) / (
                        pred[:, i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                        target[:, i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) - (pred[:, i] * target[:, i]).sum(
                    dim=1).sum(dim=1).sum(dim=1) + smooth)

        # 返回的是jaccard距离
        jaccard = jaccard / pred.size(1)
        return torch.clamp((1 - jaccard).mean(), 0, 1)


class SSLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        smooth = 1

        loss = 0.

        for i in range(pred.size(1)):
            s1 = ((pred[:, i] - target[:, i]).pow(2) * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) / (
                        smooth + target[:, i].sum(dim=1).sum(dim=1).sum(dim=1))

            s2 = ((pred[:, i] - target[:, i]).pow(2) * (1 - target[:, i])).sum(dim=1).sum(dim=1).sum(dim=1) / (
                        smooth + (1 - target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1))

            loss += (0.05 * s1 + 0.95 * s2)

        return loss / pred.size(1)


class TverskyLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        smooth = 1

        dice = 0.

        for i in range(pred.size(1)):
            dice += (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) / (
                        (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) +
                        0.3 * (pred[:, i] * (1 - target[:, i])).sum(dim=1).sum(dim=1).sum(dim=1) + 0.7 * (
                                    (1 - pred[:, i]) * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        dice = dice / pred.size(1)
        return torch.clamp((1 - dice).mean(), 0, 2)


class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"    
    def __init__(self, alpha=0.25, gamma=2., device='cuda'):
        super(WeightedFocalLoss, self).__init__()        
        self.alpha = torch.tensor([alpha, 1-alpha], device=device)
        self.gamma = torch.tensor(gamma, device=device)
            
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none').view(-1)
        targets = targets.to(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1)) 
        # print(BCE_loss)
        F_loss = at*(1-torch.exp(-BCE_loss))**self.gamma * BCE_loss
        x = F_loss.mean()
        return x
    

class HybridLoss(nn.Module):
    def __init__(self, dice_weight=0.5, focal_alpha=0.25, focal_gamma=2.):
        super().__init__()

        self.weightfocal_loss = WeightedFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss()
        self.dice_loss_weight = dice_weight

    def forward(self, pred, target):
        return self.dice_loss_weight * self.dice_loss(pred, target) + (1 - self.dice_loss_weight) * self.weightfocal_loss(pred, target)