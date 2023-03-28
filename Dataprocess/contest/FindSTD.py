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
means1 = [90.32355562, 89.17319967, 80.82956082]
means2 = [80.45204145, 81.57964481, 74.75666556]

# std1 = [47.2191106,  40.74119068, 41.10591621]
# std2 = [50.52372047, 45.21346179, 48.56336379]

path = '/media/hlf/Luffy/WLS/ContestCD/CDdata/trainData/images'
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


