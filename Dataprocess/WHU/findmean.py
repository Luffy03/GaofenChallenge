# -*- coding:utf-8 -*-

# @Filename: findmean
# @Project : ContestCD
# @date    : 2021-09-10 10:35
# @Author  : Linshan

import os
import numpy as np
import tifffile
from utils import util
import time
# means = [105.34256577, 114.22842384, 112.52934953]

path = '/media/hlf/Luffy/WLS/ContestCD/CDdata/WHU/images_whole'
list = os.listdir(path)

k = 0
sum = np.zeros([3])

for idx, i in enumerate(list):
    print(idx)
    img = util.read(os.path.join(path, i))
    img = img.reshape(512 * 512, -1)
    mean = np.mean(img, axis=0)

    k += 1
    sum += mean


means = sum / k
print('means:', means)