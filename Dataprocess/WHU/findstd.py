# -*- coding:utf-8 -*-

# @Filename: findstd
# @Project : ContestCD
# @date    : 2021-09-10 10:41
# @Author  : Linshan
import cv2
import os
import numpy as np
import tifffile
from utils import util
means = [105.34256577, 114.22842384, 112.52934953]
# std = [54.1636198,  50.74354362, 54.22707954]

path = '/media/hlf/Luffy/WLS/ContestCD/CDdata/WHU/images_whole'
list = os.listdir(path)

k = 0
sum = np.zeros([3])

for idx, i in enumerate(list):
    print(idx)
    img = util.read(os.path.join(path, i))
    img = img.reshape(512 * 512, -1)

    x = (img - means) ** 2
    k += 1
    sum += np.sum(x, axis=0)


std = np.sqrt(sum / (k * 512 * 512))
print('std:', std)


