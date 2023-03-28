# -*- coding:utf-8 -*-

# @Filename: criterion
# @Project : CACNN
# @date    : 2020-09-27 07:55
# @Author  : ifeng

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0")


class WBCELoss(nn.Module):
    def __init__(self, w=0.1):
        super(WBCELoss, self).__init__()
        self.weight = w
        self.smooth = 1e-6

    def forward(self, predict, target):
        loss = - target * torch.log(predict + self.smooth) * (1 - self.weight) \
               - (1 - target) * torch.log(1 - predict + self.smooth) * self.weight
        return loss.mean()


class BCEFocalLoss(nn.Module):
    """
    二分类的Focalloss alpha 固定
    """

    def __init__(self, gamma=2, alpha=0.2, reduction='elementwise_mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target):
        if pred.size() != target.size():
            pred = F.interpolate(pred, size=target.size()[-2:], mode='bilinear')

        pt = pred.clamp(min=0.0001, max=9.9999e-1)
        alpha = self.alpha
        loss = -(1 - alpha) * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               alpha * pt ** self.gamma * (1 - target) * torch.log(1 - pt)

        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class Dice_BCE_loss(nn.Module):
    def __init__(self):
        super(Dice_BCE_loss, self).__init__()
        self.bce_loss = BCEFocalLoss()

    def soft_dice_coeff(self, y_pred, y_true):
        smooth = 0.001  # may change
        num = y_pred.size()[0]
        m1 = y_pred.view(num, -1)
        m2 = y_true.view(num, -1)
        intersection = (m1 * m2)

        score = (2. * intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        return score.mean()

    def soft_dice_loss(self, y_pred, y_true):
        loss = 1 - self.soft_dice_coeff(y_pred, y_true)
        return loss

    def __call__(self, y_pred, y_true):
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_pred, y_true)

        return a + b


class F1_BCE_loss(nn.Module):
    def __init__(self, focal=False, beta=1):
        super(F1_BCE_loss, self).__init__()
        if focal is True:
            self.bce_loss = BCEFocalLoss()
        else:
            self.bce_loss = nn.BCELoss()

        self.beta = beta

    def F1_score(self, y_pred, y_true):
        smooth = 1e-6  # may change
        num = y_pred.size()[0]
        m1 = y_pred.view(num, -1)
        m2 = y_true.view(num, -1)

        ### TP
        intersection = (m1 * m2)
        ### precision and recall
        recall = (intersection.sum(1) + smooth) / (m2.sum(1) + smooth)
        precision = (intersection.sum(1) + smooth) / (m1.sum(1) + smooth)
        score = ((1 + self.beta**2) * precision * recall) / (recall + self.beta**2 * precision + smooth)
        return score.mean()

    def F1_loss(self, y_pred, y_true):
        score = self.F1_score(y_pred, y_true)
        loss = 1 - score
        return loss

    def __call__(self, y_pred, y_true):
        if y_pred.shape != y_true.shape:
            y_pred = F.interpolate(y_pred, y_true.size()[-2:], mode='bilinear')

        if len(y_true.size()) < 4:
            y_true = y_true.unsqueeze(1)

        if y_true.max() > 1:
            # ignore !!!
            return 0
        else:
            assert (y_pred.max() <= 1) and (y_pred.min() >= 0), print(y_pred.max(), y_pred.min())
            a = self.bce_loss(y_pred, y_true)
            b = self.F1_loss(y_pred, y_true)

            return a + b




if __name__ == "__main__":
    a = torch.rand(1, 1, 32, 32).to(device)
    a = torch.sigmoid(a)
    b = torch.randint(low=0, high=2, size=(1, 1, 32, 32)).float().to(device)
    print(b)

    loss = F1_BCE_loss()
    print(loss(a, b))
