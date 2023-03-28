# -*- coding:utf-8 -*-

# @Filename: FindMean
# @Project : ContestCD
# @date    : 2021-08-19 17:25
# @Author  : Linshan

import os
import numpy as np
import tifffile
from utils import util
import time
# means1 = [123.54700451, 113.22367732,  98.57185597]
# means2 = [123.09675227, 122.99353285, 117.2675075]
path = '/media/hlf/Luffy/WLS/ContestCD/CDdata/WHU_CD/images'
list = os.listdir(path)

k1 = 0
sum1 = np.zeros([3])

k2 = 0
sum2 = np.zeros([3])
for idx, i in enumerate(list):
    name = i.split('_')[1][:-4]
    print(name)
    img = util.read(os.path.join(path, i))
    img = img.reshape(512 * 512, -1)
    mean = np.mean(img, axis=0)

    if name == '1':
        k1 += 1
        sum1 += mean
    else:
        k2 += 1
        sum2 += mean

means1 = sum1 / k1
print('means1:', means1)

means2 = sum2 / k2
print('means2:', means2)


