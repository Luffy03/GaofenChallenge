# -*- coding:utf-8 -*-

# @Filename: FindSTD
# @Project : ContestCD
# @date    : 2021-08-19 17:43
# @Author  : Linshan

import cv2
import os
import numpy as np
import tifffile
from utils import util
means1 = [123.54700451, 113.22367732,  98.57185597]
means2 = [123.09675227, 122.99353285, 117.2675075]

# std1 = [44.23595041, 43.29463682, 44.15438945]
# std2 = [54.5432597,  51.87598908, 55.94164293]

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

    if name == '1':
        x = (img - means1) ** 2
        k1 += 1
        sum1 += np.sum(x, axis=0)
    else:
        x = (img - means2) ** 2
        k2 += 1
        sum2 += np.sum(x, axis=0)


std1 = np.sqrt(sum1 / (k1 * 512 * 512))
print('std1:', std1)

std2 = np.sqrt(sum2 / (k2 * 512 * 512))
print('std2:', std2)


