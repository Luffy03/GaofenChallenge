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
    paths = [val_path, train_path, test_path]

    save_img_path, save_label_path = path + '/images_whole', path + '/gt_whole'
    check_dir(save_img_path), check_dir(save_label_path)

    count = 1
    for p in paths:
        A_path = p + '/A'
        B_path = p + '/B'
        gt_path = p + '/label'
        list = os.listdir(A_path)

        for idx, i in enumerate(list):
            print(count)
            a = read(A_path + '/' + i)
            b = read(B_path + '/' + i)
            label = read(gt_path + '/' + i)

            write(save_img_path + '/' + str(count) + '_1.png', a, flag='img')
            write(save_img_path + '/' + str(count) + '_2.png', b, flag='img')
            write(save_label_path + '/' + str(count) + '_change.png', label, flag='label')
            count += 1


def crop(path, size=512):
    img_path, label_path = path + '/images_whole', path + '/gt_whole'

    save_img_path, save_label_path = path + '/images', path + '/gt'
    check_dir(save_img_path), check_dir(save_label_path)
    count = 1

    list = os.listdir(img_path)
    for i in list:
        if i.split('_')[1][:-4] == '1':
            img1 = read(os.path.join(img_path, i))
            img2 = read(os.path.join(img_path, str(i.split('_')[0]) + '_2.png'))
            cd = read(os.path.join(label_path, str(i.split('_')[0]) + '_change.png'))
            gt = cd.copy()
            gt[gt > 0] = 255

            height, width, _ = img1.shape
            h_size = height // size
            w_size = width // size

            for i in range(h_size):
                for j in range(w_size):
                    print(count)

                    img1_cut = img1[i * size:(i + 1) * size, j * size:(j + 1) * size, :]
                    img2_cut = img2[i * size:(i + 1) * size, j * size:(j + 1) * size, :]
                    out = gt[i * size:(i + 1) * size, j * size:(j + 1) * size]

                    assert out.shape == (size, size)
                    assert img1_cut.shape == (size, size, 3)

                    write(save_img_path + '/' + str(count) + '_1.png', img1_cut, flag='img')
                    write(save_img_path + '/' + str(count) + '_2.png', img2_cut, flag='img')

                    write(save_label_path + '/' + str(count) + '_change.png', out, flag='label')

                    count += 1


if __name__ == '__main__':
    path = '/media/hlf/Luffy/WLS/ContestCD/CDdata/LEVIR'
    # fuse(path)
    crop(path)


