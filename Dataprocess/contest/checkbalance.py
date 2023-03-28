# -*- coding:utf-8 -*-

# @Filename: checkbalance
# @Project : ContestCD
# @date    : 2021-08-19 20:04
# @Author  : Linshan

import os
import numpy as np
from utils import util
from tqdm import tqdm

# path = '/home/hnu2/WLS/ContestCD/CDdata/trainData/gt'
path = '/home/hnu2/WLS/ContestCD/CDdata/trainData/augData_3500/gt'


def check(path, size=512):
    list = os.listdir(path)
    c = 0
    noc = 0

    build = 0
    no_build = 0

    for idx, i in tqdm(enumerate(list)):
        l = util.read(os.path.join(path, i))

        if i.split('_')[1][:-4] == 'change':
            l = util.read(os.path.join(path, i))
            label = util.read(os.path.join(path, i))[:, :]
            lbl = label.copy()
            lbl[label > 0] = 1

            change = lbl.sum()
            no_change = size*size - change
            c += change
            noc += no_change

        else:
            label = util.read(os.path.join(path, i))[:, :]
            lbl = label.copy()
            lbl[label > 0] = 1

            b = lbl.sum()
            no_b = size * size - b
            build += b
            no_build += no_b

    print(noc, c)
    print(c/(c + noc))

    print(no_build, build)
    print(build / (build + no_build))


check(path)
# change 0.020239656448364257
# build 0.08506810569763183

# aug 500
# change 0.2747427673339844
# build 0.1373713836669922