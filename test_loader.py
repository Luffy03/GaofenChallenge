# -*- coding:utf-8 -*-

# @Filename: data_test
# @Project : ContestCD
# @date    : 2021-08-27 13:51
# @Author  : Linshan
import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import cv2


def read(img):
    img = Image.open(img)
    return np.asarray(img)


def Normalize(img, time):
    if time == '1':
        means = [90.32355562, 89.17319967, 80.82956082]
        stds = [47.2191106, 40.74119068, 41.10591621]
    else:
        means = [80.45204145, 81.57964481, 74.75666556]
        stds = [50.52372047, 45.21346179, 48.56336379]

    img = (img - means) / stds
    return img


class Dataset_test(Dataset):
    def __init__(self, path):
        self.img_path = path
        self.img_label_path_pairs = self.get_img_label_path_pairs()

    def get_img_label_path_pairs(self):
        img_label_pair_list = []
        list = os.listdir(self.img_path)
        for idx, i in enumerate(list):
            name = i.split('_')[1][:-4]
            if name == '1':
                img1 = os.path.join(self.img_path, i)
                img2 = os.path.join(self.img_path, i.split('_')[0] + '_2.png')
                filename = i.split('_')[0]
                img_label_pair_list.append([img1, img2, filename])

        return img_label_pair_list

    def img_transform(self, img):
        img = img.astype(np.float32).transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        return img

    def __getitem__(self, index):
        item = self.img_label_path_pairs[index]
        img1, img2, name = item

        img1, img2 = read(img1)[:, :, :3], read(img2)[:, :, :3]

        img1, img2 = Normalize(img1, time='1'), Normalize(img2, time='2')
        img1, img2 = self.img_transform(img1), self.img_transform(img2)
        return img1, img2, name

    def __len__(self):

        return len(self.img_label_path_pairs)

