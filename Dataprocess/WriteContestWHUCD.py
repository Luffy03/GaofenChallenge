# -*- coding:utf-8 -*-

# @Filename: WriteContestWHUCD
# @Project : ContestCD
# @date    : 2021-09-11 12:57
# @Author  : Linshan


import os
import random


def write_contest_addWHUCD(save_path, len=1828):
    old_txt = save_path + '/datafiles/train.txt'
    whole = []

    with open(old_txt, 'r') as lines:
        for idx, line in enumerate(lines):
            name = line.strip("\n").split(' ')[0]
            whole.append('trainData_' + name)

    train_txt = open(save_path + '/datafiles/Contest_WHUCD.txt', 'w')

    for j in range(1, len):
        whole.append('WHUCD_' + str(j))

    random.shuffle(whole)

    for idx, i in enumerate(whole):
        train_txt.write(i + '\n')

    train_txt.close()


if __name__ == '__main__':
    save_path = '/media/hlf/Luffy/WLS/ContestCD'
    write_contest_addWHUCD(save_path)

