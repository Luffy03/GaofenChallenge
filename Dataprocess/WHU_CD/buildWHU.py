# -*- coding:utf-8 -*-

# @Filename: build
# @Project : ContestCD
# @date    : 2021-09-10 14:00
# @Author  : Linshan
from utils import *


def fuse(path):
    save_img_path, save_gt_path = path + '/images', path + '/gt'
    check_dir(save_img_path), check_dir(save_gt_path)

    path_a = '/media/hlf/Luffy/WLS/ContestCD/CDdata/WHU_CD/A/splited_images/train'
    path_b = '/media/hlf/Luffy/WLS/ContestCD/CDdata/WHU_CD/A/splited_images/test'

    path_c = '/media/hlf/Luffy/WLS/ContestCD/CDdata/WHU_CD/B/splited_images/train'
    path_d = '/media/hlf/Luffy/WLS/ContestCD/CDdata/WHU_CD/B/splited_images/test'

    count = 1
    img1_path, gt1_path = path_a + '/image', path_a + '/label'
    img2_path, gt2_path = path_c + '/image', path_c + '/label'
    list_train = os.listdir(img1_path)
    for i in list_train:
        print(count)
        img1 = read(img1_path + '/' + i)
        gt1 = read(gt1_path + '/' + i)

        img2 = read(img2_path + '/' + i)
        gt2 = read(gt2_path + '/' + i)

        cd = np.abs(gt1 - gt2)

        write(save_img_path + '/' + str(count) + '_1.png', img1, flag='img')
        write(save_img_path + '/' + str(count) + '_2.png', img2, flag='img')

        write(save_gt_path + '/' + str(count) + '_1_label.png', gt1, flag='label')
        write(save_gt_path + '/' + str(count) + '_2_label.png', gt2, flag='label')
        write(save_gt_path + '/' + str(count) + '_change.png', cd, flag='label')

        count += 1

    img1_path, gt1_path = path_b + '/image', path_b + '/label'
    img2_path, gt2_path = path_d + '/image', path_d + '/label'
    list_train = os.listdir(img1_path)
    for i in list_train:
        print(count)
        img1 = read(img1_path + '/' + i)
        gt1 = read(gt1_path + '/' + i)

        img2 = read(img2_path + '/' + i)
        gt2 = read(gt2_path + '/' + i)

        cd = np.abs(gt1 - gt2)

        write(save_img_path + '/' + str(count) + '_1.png', img1, flag='img')
        write(save_img_path + '/' + str(count) + '_2.png', img2, flag='img')

        write(save_gt_path + '/' + str(count) + '_1_label.png', gt1, flag='label')
        write(save_gt_path + '/' + str(count) + '_2_label.png', gt2, flag='label')
        write(save_gt_path + '/' + str(count) + '_change.png', cd, flag='label')

        count += 1


if __name__ == '__main__':
    path = '/media/hlf/Luffy/WLS/ContestCD/CDdata/WHU_CD'
    fuse(path)