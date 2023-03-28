# -*- coding:utf-8 -*-

# @Filename: make_edge
# @Project : ContestCD
# @date    : 2021-08-24 09:53
# @Author  : Linshan

import cv2
import numpy as np
import glob
import os
from utils import *
from utils import util

#源路径
image_path = '/home/hnu2/WLS/ContestCD/CDdata/trainData/augData/gt'
image_list = os.listdir(image_path)
#目标路径
dir_name = '/home/hnu2/WLS/ContestCD/CDdata/trainData/augData/edge'
check_dir(dir_name)

for idx, i in enumerate(image_list):
    print(idx)
    img = util.read(os.path.join(image_path, i))
    canny = cv2.Canny(img, 50, 150)
    cv2.imwrite(dir_name + '/' + i, canny)