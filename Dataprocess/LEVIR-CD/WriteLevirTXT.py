# -*- coding:utf-8 -*-

# @Filename: WriteLevirTXT
# @Project : ContestCD
# @date    : 2021-09-10 13:01
# @Author  : Linshan
import os
import random


def write_whole(save_path):
    whole_txt = open(save_path + '/datafiles/LEVIR_whole.txt', 'w')

    list = []
    for j in range(1, 2549):
        list.append(str(j))

    print(list)
    random.shuffle(list)

    for idx, i in enumerate(list):
        whole_txt.write('LEVIR' + '_' + i + '\n')

    whole_txt.close()


if __name__ == '__main__':
    save_path = '/media/hlf/Luffy/WLS/ContestCD'
    write_whole(save_path)
