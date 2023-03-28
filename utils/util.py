import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import time
import json
import os
from pathlib import Path
from models import *
from PIL import Image
import numpy as np


def read(img):
    img = Image.open(img)
    return np.asarray(img)


def write(path, img, flag='label'):
    if flag == 'label' and len(img.shape) == 3:
        img = img[:, :, 0]
    img = Image.fromarray(img)
    img.save(path)


def check_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(learning_rate, optimizer, step, length, num_epochs=20):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    stride = num_epochs * length
    lr = learning_rate * (0.1 ** (step // stride))
    if step % stride == 0:
        print("learning_rate change to:%.8f" % (lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def create_save_path(opt):
    save_path = os.path.join(opt.save_path, opt.dataset)
    exp_path = os.path.join(save_path, opt.experiment_name)

    log_path = os.path.join(exp_path, 'log')
    checkpoint_path = os.path.join(exp_path, 'checkpoint')
    predict_test_path = os.path.join(exp_path, 'predict_test')
    predict_train_path = os.path.join(exp_path, 'predict_train')
    # predict_val_path = os.path.join(exp_path, 'predict_val')

    check_dir(opt.save_path)
    check_dir(save_path), check_dir(exp_path), check_dir(log_path), check_dir(checkpoint_path), \
    check_dir(predict_test_path), \
    check_dir(predict_train_path)

    return log_path, checkpoint_path, predict_test_path, predict_train_path


def create_data_path(opt):
    data_inform_path = opt.data_inform_path
    train_txt_path = os.path.join(data_inform_path, 'train_aug.txt')
    val_txt_path = os.path.join(data_inform_path, 'val.txt')
    whole_txt_path = os.path.join(data_inform_path, 'whole.txt')

    return train_txt_path, val_txt_path, whole_txt_path


def create_logger(log_path):
    time_str = time.strftime('%Y-%m-%d-%H-%M')

    log_file = '{}.log'.format(time_str)

    final_log_file = os.path.join(log_path, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(log_path)/'scalar'/time_str
    print('=>creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(tensorboard_log_dir)


def save_pred(x, y, out, save_pred_dir, filename):
    check_dir(save_pred_dir)
    x_path, y_path, cd_path = save_pred_dir + '/x', save_pred_dir + '/y', save_pred_dir + '/cd'
    check_dir(x_path), check_dir(y_path), check_dir(cd_path)

    # save predict dir
    x_path, y_path, cd_path = os.path.join(x_path, filename[0]+'.png'), \
                              os.path.join(y_path, filename[0]+'.png'), \
                              os.path.join(cd_path, filename[0]+'.png')
    write(x_path, (255 * x).astype(np.uint8))
    write(y_path, (255 * y).astype(np.uint8))
    write(cd_path, (255 * out).astype(np.uint8))


def resize_label(label, size):
    if len(label.size()) == 3:
        label = label.unsqueeze(1)
    label = F.interpolate(label, size=(size, size), mode='bilinear', align_corners=True)

    return label


def get_mean_std(data, time):
    if data == 'trainData':
        if time == '1':
            means = [90.32355562, 89.17319967, 80.82956082]
            stds = [47.2191106,  40.74119068, 41.10591621]
        else:
            means = [80.45204145, 81.57964481, 74.75666556]
            stds = [50.52372047, 45.21346179, 48.56336379]

    elif data == 'LEVIR':
        if time == '1':
            means = [114.81641003, 113.90036586, 97.24337607]
            stds = [55.36467751, 52.02340311, 47.59829212]
        else:
            means = [88.10832457, 86.23987012, 73.64794018]
            stds = [40.18651095, 38.75655584, 36.82439213]

    elif data == 'WHUCD':
        if time == '1':
            means = [123.54700451, 113.22367732, 98.57185597]
            stds = [44.23595041, 43.29463682, 44.15438945]
        else:
            means = [123.09675227, 122.99353285, 117.2675075]
            stds = [54.5432597, 51.87598908, 55.94164293]
    else:
        means = 0
        stds = 0
        print('error')

    return means, stds


def Normalize(img, data, time):
    means, std = get_mean_std(data, time)
    img = (img - means) / std

    return img


def Normalize_back(img, data, time):
    means, std = get_mean_std(data, time)

    means = means[:3]
    std = std[:3]

    img = img * std + means

    return img