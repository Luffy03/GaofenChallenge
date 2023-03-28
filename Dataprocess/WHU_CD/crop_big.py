# -*- coding:utf-8 -*-

# @Filename: try
# @Project : ContestCD
# @date    : 2021-09-11 10:49
# @Author  : Linshan
import tifffile
from utils import *


def run(path, size=512):
    A_train_img = tifffile.imread('/media/hlf/Luffy/WLS/ContestCD/CDdata/WHUCD/A/whole_image/train/image/2012_train.tif')
    A_train_label = tifffile.imread('/media/hlf/Luffy/WLS/ContestCD/CDdata/WHUCD/A/whole_image/train/label/2012_train.tif')
    A_test_img = tifffile.imread(
        '/media/hlf/Luffy/WLS/ContestCD/CDdata/WHUCD/A/whole_image/test/image/2012_test.tif')
    A_test_label = tifffile.imread(
        '/media/hlf/Luffy/WLS/ContestCD/CDdata/WHUCD/A/whole_image/test/label/2012_test.tif')

    B_train_img = tifffile.imread(
        '/media/hlf/Luffy/WLS/ContestCD/CDdata/WHUCD/B/whole_image/train/image/2016_train.tif')
    B_train_label = tifffile.imread(
        '/media/hlf/Luffy/WLS/ContestCD/CDdata/WHUCD/B/whole_image/train/label/2016_train.tif')
    B_test_img = tifffile.imread(
        '/media/hlf/Luffy/WLS/ContestCD/CDdata/WHUCD/B/whole_image/test/image/2016_test.tif')
    B_test_label = tifffile.imread(
        '/media/hlf/Luffy/WLS/ContestCD/CDdata/WHUCD/B/whole_image/test/label/2016_test.tif')

    change_train = tifffile.imread('/media/hlf/Luffy/WLS/ContestCD/CDdata/WHUCD/label/train/change_label.tif')
    change_test = tifffile.imread('/media/hlf/Luffy/WLS/ContestCD/CDdata/WHUCD/label/test/change_label.tif')

    save_img_path, save_gt_path = path + '/images', path + '/gt'
    check_dir(save_img_path), check_dir(save_gt_path)

    height, width, _ = A_train_img.shape
    h_size = height // size
    w_size = width // size

    count = 1

    for i in range(h_size):
        for j in range(w_size):
            a = A_train_img[i * size:(i + 1) * size, j * size:(j + 1) * size, :3]
            b = B_train_img[i * size:(i + 1) * size, j * size:(j + 1) * size, :3]
            a_gt = A_train_label[i * size:(i + 1) * size, j * size:(j + 1) * size]
            b_gt = B_train_label[i * size:(i + 1) * size, j * size:(j + 1) * size]
            cd = change_train[i * size:(i + 1) * size, j * size:(j + 1) * size]

            assert a.shape == (size, size, 3)
            assert a_gt.shape == (size, size)

            write(save_img_path + '/' + str(count) + '_1.png', a, flag='img')
            write(save_img_path + '/' + str(count) + '_2.png', b, flag='img')

            write(save_gt_path + '/' + str(count) + '_1_label.png', a_gt, flag='label')
            write(save_gt_path + '/' + str(count) + '_2_label.png', b_gt, flag='label')
            write(save_gt_path + '/' + str(count) + '_change.png', cd, flag='label')

            print(count)
            count += 1

    height, width, _ = A_test_img.shape
    h_size = height // size
    w_size = width // size

    for i in range(h_size):
        for j in range(w_size):
            a = A_test_img[i * size:(i + 1) * size, j * size:(j + 1) * size, :3]
            b = B_test_img[i * size:(i + 1) * size, j * size:(j + 1) * size, :3]
            a_gt = A_test_label[i * size:(i + 1) * size, j * size:(j + 1) * size]
            b_gt = B_test_label[i * size:(i + 1) * size, j * size:(j + 1) * size]
            cd = change_test[i * size:(i + 1) * size, j * size:(j + 1) * size]

            assert a.shape == (size, size, 3)
            assert a_gt.shape == (size, size)

            write(save_img_path + '/' + str(count) + '_1.png', a, flag='img')
            write(save_img_path + '/' + str(count) + '_2.png', b, flag='img')

            write(save_gt_path + '/' + str(count) + '_1_label.png', a_gt, flag='label')
            write(save_gt_path + '/' + str(count) + '_2_label.png', b_gt, flag='label')
            write(save_gt_path + '/' + str(count) + '_change.png', cd, flag='label')

            print(count)
            count += 1



if __name__ == '__main__':
    path = '/media/hlf/Luffy/WLS/ContestCD/CDdata/WHUCD'
    run(path)