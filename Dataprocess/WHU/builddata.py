# -*- coding:utf-8 -*-

# @Filename: builddata
# @Project : ContestCD
# @date    : 2021-09-10 10:50
# @Author  : Linshan

from utils import *
import random
from matplotlib import pyplot as plt
import dataset.transform as trans
from torchvision import transforms


def check_pro(gt, size=512):
    return gt.sum() / (size*size)


def check_overlap(gt1, gt2):
    overlap = gt1 * gt2
    if overlap.sum() > 0:
        return True
    else:
        return False


def build(path):
    img_path, label_path = path + '/images_whole', path + '/gt_whole'

    list = os.listdir(img_path)
    random.shuffle(list)

    wait_list = []
    num = 0

    while num < 3000:
        img1 = read(img_path + '/' + list[0])
        gt1 = read(label_path + '/' + list[0])

        img2 = read(img_path + '/' + list[1])
        gt2 = read(label_path + '/' + list[1])

        if check_overlap(gt1, gt2) is False:
            num += 1


def show(img1, img2, gt1, gt2, change, name):
    fig, axs = plt.subplots(1, 5, figsize=(20, 8))

    axs[0].imshow(img1.astype(np.uint8))
    axs[0].axis("off")

    axs[1].imshow(img2.astype(np.uint8))
    axs[1].axis("off")

    axs[2].imshow(gt1.astype(np.uint8), cmap='gray')
    axs[2].axis("off")

    axs[3].imshow(gt2.astype(np.uint8), cmap='gray')
    axs[3].axis("off")

    axs[4].imshow(change.astype(np.uint8), cmap='gray')
    axs[4].axis("off")

    plt.suptitle(os.path.basename(name), y=0.94)
    plt.tight_layout()
    plt.show()
    plt.close()






