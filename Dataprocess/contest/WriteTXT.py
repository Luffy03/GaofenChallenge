# -*- coding:utf-8 -*-

# @Filename: WriteTXT
# @Project : ContestCD
# @date    : 2021-08-19 17:52
# @Author  : Linshan

import os
import random


def write(save_path):
    train_txt = open(save_path + '/datafiles/train.txt', 'w')
    val_txt = open(save_path + '/datafiles/val.txt', 'w')

    list = []
    for j in range(1, 2001):
        list.append(str(j))

    print(list)
    random.shuffle(list)

    for idx, i in enumerate(list):
        if idx < 0.8 * len(list):
            train_txt.write(i + '\n')

        else:
            val_txt.write(i + '\n')
    train_txt.close()
    val_txt.close()


def write_whole(save_path):
    whole_txt = open(save_path + '/datafiles/whole.txt', 'w')

    list = []
    for j in range(1, 2001):
        list.append(str(j))

    print(list)
    random.shuffle(list)

    for idx, i in enumerate(list):
        whole_txt.write(i + '\n')

    whole_txt.close()


if __name__ == '__main__':
    save_path = '/media/hlf/Luffy/WLS/ContestCD'
    write_whole(save_path)
