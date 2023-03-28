# -*- coding:utf-8 -*-

# @Filename: exp.py
# @Project : ContestCD
# @date    : 2021-08-24 00:02
# @Author  : Linshan
import os
from PIL import Image
from models.network_fpn import Net_fpn
from models.network_uper import Net_uper
import sys
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from test_loader import Dataset_test
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0")
torch.backends.cudnn.benchmark = True  # speed up!!!


def T2npy(input):
    return input[0][0].data.cpu().numpy().astype(np.uint8)


def write(path, img):
    cv2.imwrite(path, img)


def save_output(x, y, out, filename, output_path):
    # save predict dir
    x_path, y_path, cd_path = os.path.join(output_path, filename + '_1_label.png'), \
        os.path.join(output_path, filename + '_2_label.png'), \
        os.path.join(output_path, filename + '_change.png')

    x, y, out = T2npy(x), T2npy(y), T2npy(out)
    write(x_path, 255 * x)
    write(y_path, 255 * y)
    write(cd_path, 255 * out)


def main(argv):
    model = Net_fpn(backbone='convnext_xlarge')

    checkpoint = torch.load('./save/contest/baseline/checkpoint/model_best.pth')
    model.load_state_dict(checkpoint['state_dict'])
    print('load success')
    model = model.to(device)
    model.eval()

    # input_path, output_path = argv[1], argv[2]
    input_path, output_path = './CDdata/trainData/images', './output_path'

    if not os.path.exists(input_path):
        print('no input_path')
    if not os.path.exists(output_path):
        print('no output_path')

    test_dataset = Dataset_test(input_path)
    loader = DataLoader(test_dataset, batch_size=1, num_workers=8,
                        pin_memory=True)

    for batch in tqdm(loader):
        with torch.no_grad():
            img1, img2, name = batch
            img1, img2 = img1.cuda(), img2.cuda()
            with torch.cuda.amp.autocast():
                x, y, out = model(img1, img2)
            x, y, out = (x > 0.5).long(), (y > 0.5).long(), (out > 0.5).long()

            filename = name[0]
            save_output(x, y, out, filename, output_path)

    output_list = os.listdir(output_path)
    length = len(loader) * 3
    if len(output_list) == length:
        print('save success')
    else:
        print(length, len(loader))


if __name__ == '__main__':
    main(sys.argv)


