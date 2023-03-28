# -*- coding:utf-8 -*-

# @Filename: WriteLevirTXT
# @Project : ContestCD
# @date    : 2021-09-10 13:01
# @Author  : Linshan
import os
import random


def write(save_path):
    whole_txt = open(save_path + '/datafiles/WHUCD.txt', 'w')
    val_txt = open(save_path + '/datafiles/WHUCD_val.txt', 'w')

    list = []
    for j in range(1, 1828):
        list.append(str(j))

    print(list)
    random.shuffle(list)

    for idx, i in enumerate(list):
        if idx < 0.9 * len(list):
            whole_txt.write('WHUCD' + '_' + i + '\n')
        else:
            val_txt.write('WHUCD' + '_' + i + '\n')

    whole_txt.close()
    val_txt.close()



if __name__ == '__main__':
    save_path = '/media/hlf/Luffy/WLS/ContestCD'
    write(save_path)
