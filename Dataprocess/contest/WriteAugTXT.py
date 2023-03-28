# -*- coding:utf-8 -*-

# @Filename: WriteAugTXT
# @Project : ContestCD
# @date    : 2021-08-29 13:10
# @Author  : Linshan

import os
import random


def write_aug(save_path, len=4501):
    old_txt = save_path + '/datafiles/whole.txt'
    whole = []

    with open(old_txt, 'r') as lines:
        for idx, line in enumerate(lines):
            name = line.strip("\n").split(' ')[0]
            whole.append('trainData_' + name)

    train_txt = open(save_path + '/datafiles/train_aug.txt', 'w')

    for j in range(1, len):
        whole.append('augData_' + str(j))

    random.shuffle(whole)

    for idx, i in enumerate(whole):
        train_txt.write(i + '\n')

    train_txt.close()


def write_mosaic(save_path, mosaic_path='/media/hlf/Luffy/WLS/ContestCD/CDdata/trainData/mosaic/images'):
    old_txt = save_path + '/datafiles/whole.txt'
    whole = []

    with open(old_txt, 'r') as lines:
        for idx, line in enumerate(lines):
            name = line.strip("\n").split(' ')[0]
            whole.append('trainData_' + name)

    train_txt = open(save_path + '/datafiles/train_mosaic.txt', 'w')

    list = os.listdir(mosaic_path)
    for j in list:
        if j.split('_')[1][:-4] == '1':
            whole.append('mosaic_' + str(j.split('_')[0]))

    random.shuffle(whole)

    for idx, i in enumerate(whole):
        train_txt.write(i + '\n')

    train_txt.close()


if __name__ == '__main__':
    save_path = '/home/hnu2/WLS/ContestCD'
    write_aug(save_path)

    # write_mosaic(save_path)
