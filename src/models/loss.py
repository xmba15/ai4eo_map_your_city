import torch
from torch import nn
from torch.nn import functional as F

__all__ = (
    "FocalLoss",
    "DomainClsLoss",
    "FocalLossLabelSmoothing",
)


class FocalLoss(nn.Module):
    """
    Multi-class Focal loss implementation
    """

    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss


class FocalLossLabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1, gamma=2, weight=None):
        super(FocalLossLabelSmoothing, self).__init__()
        self.smoothing = smoothing
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N]
        """
        num_classes = input.size(1)
        target_one_hot = torch.zeros_like(input).scatter(1, target.unsqueeze(1), 1)
        smooth_targets = (1 - self.smoothing) * target_one_hot + self.smoothing / num_classes

        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        focal_weight = (1 - pt) ** self.gamma

        loss = -focal_weight * logpt * smooth_targets
        if self.weight is not None:
            loss = loss * self.weight.unsqueeze(0)

        return loss.sum(dim=1).mean()


class DomainClsLoss(nn.Module):
    def __init__(self):
        super(DomainClsLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target shape do not match"
        total_loss = self.criterion(predict, target)
        return total_loss
