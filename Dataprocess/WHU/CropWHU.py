# -*- coding:utf-8 -*-

# @Filename: CropWHU
# @Project : ContestCD
# @date    : 2021-09-10 09:45
# @Author  : Linshan

from utils import *
import cv2


def fuse(path):
    train_path = path + '/train'
    val_path = path + '/val'
    test_path = path + '/test'
    paths = [train_path, val_path, test_path]

    save_img_path, save_label_path = path + '/images_whole', path + '/gt_whole'
    check_dir(save_img_path), check_dir(save_label_path)

    count = 1
    for p in paths:
        img_path = p + '/image'
        label_path = p + '/label'
        list = os.listdir(img_path)

        for idx, i in enumerate(list):
            print(count)
            img = cv2.imread(img_path + '/' + i)
            label = cv2.imread(label_path + '/' + i)[:, :, 0]
            write(save_img_path + '/' + str(count) + '.png', img, flag='img')
            write(save_label_path + '/' + str(count) + '.png', label, flag='label')
            count += 1



def run(path):
    fuse(path)


if __name__ == '__main__':
    path = '/media/hlf/Luffy/WLS/ContestCD/CDdata/WHU'
    run(path)


