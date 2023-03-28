import numpy as np
import torch


def cal_kappa(hist):
    if hist.sum() == 0:
        po = 0
        pe = 1
        kappa = 0
    else:
        po = np.diag(hist).sum() / hist.sum()
        pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
        if pe == 1:
            kappa = 0
        else:
            kappa = (po - pe) / (1 - pe)
    return kappa


def cal_fscore(hist):
    assert hist.shape == (2, 2), print(hist.shape)
    TP = hist[1][1]
    FP = hist[1][0]
    FN = hist[0][1]
    TN = hist[0][0]

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    f_score = 2 * precision * recall / (precision + recall)
    return f_score


class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def get_hist(self, label_pred, label_true):
        # 找出标签中需要计算的类别,去掉了背景
        mask = (label_true >= 0) & (label_true < self.num_classes)
        # # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    # 输入：预测值和真实值
    # 语义分割的任务是为每个像素点分配一个label
    def evaluate(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            assert len(lp.flatten()) == len(lt.flatten())
            self.hist += self.get_hist(lp.flatten(), lt.flatten())
            # miou
        iou = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        miou = np.nanmean(iou)
        # dice
        dice = 2 * np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0))
        mdice = np.nanmean(dice)

        # -----------------其他指标------------------------------
        # mean acc
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.nanmean(np.diag(self.hist) / self.hist.sum(axis=1))
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()
        return acc, acc_cls, iou, miou, dice, mdice, fwavacc


class IOUMetric_tensor:
    """
        Class to calculate mean-iou with tensor_type using fast_hist method
        """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = torch.zeros([num_classes, num_classes])

    def get_hist(self, label_pred, label_true):
        # 找出标签中需要计算的类别,去掉了背景
        mask = (label_true >= 0) & (label_true < self.num_classes)
        # # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
        hist = torch.bincount(
            self.num_classes * label_true[mask] +
            label_pred[mask], minlength=self.num_classes ** 2).view(self.num_classes, self.num_classes)
        return hist

    # 输入：预测值和真实值
    # 语义分割的任务是为每个像素点分配一个label
    def evaluate(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            assert len(lp.flatten()) == len(lt.flatten())
            self.hist += self.get_hist(lp.flatten(), lt.flatten())
            # miou
        iou = torch.diag(self.hist) / (self.hist.sum(dim=1) + self.hist.sum(dim=0) - torch.diag(self.hist))
        miou = torch.mean(iou)
        # dice
        dice = 2 * torch.diag(self.hist) / (self.hist.sum(dim=1) + self.hist.sum(dim=0))
        mdice = torch.mean(dice)

        # -----------------其他指标------------------------------
        # mean acc
        acc = torch.diag(self.hist).sum() / self.hist.sum()
        acc_cls = torch.mean(np.diag(self.hist) / self.hist.sum(dim=1))
        freq = self.hist.sum(dim=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()
        return acc, acc_cls, iou, miou, dice, mdice, fwavacc


def eval_hist(hist):
    # hist must be numpy
    kappa = cal_kappa(hist)
    iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    miou = np.nanmean(iou)

    # f_score
    f_score = cal_fscore(hist)

    # mean acc
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)

    return kappa, f_score


