# -*- coding:utf-8 -*-

# @Filename: predict_levir
# @Project : ContestCD
# @date    : 2021-09-10 11:36
# @Author  : Linshan
import os
from PIL import Image
from models.network_fpn import Net_fpn
import sys
import numpy as np
import torch
from tqdm import tqdm
import ttach as tta
from matplotlib import pyplot as plt
from utils import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0")


def Normalize(img, time):
    if time == '1':
        means = [123.54700451, 113.22367732,  98.57185597]
        stds = [44.23595041, 43.29463682, 44.15438945]
    else:
        means = [123.09675227, 122.99353285, 117.2675075]
        stds = [54.5432597,  51.87598908, 55.94164293]

    img = (img - means) / stds
    return img


def read(img):
    img = Image.open(img)
    return np.asarray(img)


def write(path, img):
    img = img[:, :, 0]
    img = Image.fromarray(img)
    img.save(path)


def input_transform(img):
    img = img.astype(np.float32).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img.unsqueeze(0)


def predict(model, img1, img2):
    img1, img2 = Normalize(img1, time='1'), Normalize(img2, time='2')
    img1, img2 = input_transform(img1), input_transform(img2)

    img1, img2 = img1.contiguous(), img2.contiguous()
    img1, img2 = img1.cuda(non_blocking=True), img2.cuda(non_blocking=True)
    x, y, out = model(img1, img2)
    torch.cuda.synchronize()
    x, y, out = (x > 0.5).long(), (y > 0.5).long(), (out > 0.5).long()
    return x, y, out


def predict_tta(model, img1, img2):
    transforms = tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
            tta.Rotate90(angles=[0, 90, 180, 270]),
        ]
    )

    img1, img2 = Normalize(img1, time='1'), Normalize(img2, time='2')
    img1, img2 = input_transform(img1), input_transform(img2)

    img1, img2 = img1.contiguous(), img2.contiguous()
    img1, img2 = img1.cuda(), img2.cuda()

    xs, ys, outs = [], [], []
    for t in transforms:
        aug_img1, aug_img2 = t.augment_image(img1), t.augment_image(img2)
        aug_x, aug_y, aug_out = model(aug_img1, aug_img2)

        deaug_x, deaug_y, deaug_out = t.deaugment_mask(aug_x), t.deaugment_mask(aug_y), \
                                      t.deaugment_mask(aug_out)
        xs.append(deaug_x), ys.append(deaug_y), outs.append(deaug_out)

    xs, ys, outs = torch.cat(xs, 1), torch.cat(ys, 1), torch.cat(outs, 1)
    x, y, out = torch.mean(xs, dim=1, keepdim=True), \
                torch.mean(ys, dim=1, keepdim=True), torch.mean(outs, dim=1, keepdim=True)
    x, y, out = (x > 0.5).long(), (y > 0.5).long(), (out > 0.5).long()
    return x, y, out


def save_output(x, y, out, filename, output_path):
    x, y, out = x.squeeze(0).permute(1, 2, 0), y.squeeze(0).permute(1, 2, 0), out.squeeze(0).permute(1, 2, 0)

    # save predict dir
    x_path, y_path, cd_path = os.path.join(output_path, filename + '_1_label.png'), \
                              os.path.join(output_path, filename + '_2_label.png'), \
                              os.path.join(output_path, filename + '_change.png')

    x, y, out = x.data.cpu().numpy(), y.data.cpu().numpy(), out.data.cpu().numpy()
    write(x_path, (255 * x).astype(np.uint8))
    write(y_path, (255 * y).astype(np.uint8))
    write(cd_path, (255 * out).astype(np.uint8))


def main(path):
    model = Net_fpn(backbone='resnet34')
    checkpoint = torch.load('/media/hlf/Luffy/WLS/ContestCD/save/contest/baseline/checkpoint/model_best_fpn34aug5000mixup.pth')
    model.load_state_dict(checkpoint['state_dict'])
    print('load success')
    model = model.to(device)
    model.eval()

    input_path, output_path = path + '/images', path + '/predict_WHUCD'
    check_dir(output_path)
    if not os.path.exists(input_path):
        print('no input_path')
    if not os.path.exists(output_path):
        print('no output_path')

    list = os.listdir(input_path)
    for idx, i in enumerate(tqdm(list)):
        with torch.no_grad():
            name = i.split('_')[1][:-4]
            if name == '1':
                img1 = read(os.path.join(input_path, i))
                img2 = read(os.path.join(input_path, i.split('_')[0] + '_2.png'))
                x, y, out = predict(model, img1, img2)
                filename = i.split('_')[0]
                save_output(x, y, out, filename, output_path)
            else:
                pass

    output_list = os.listdir(output_path)
    length = (len(list)//2) * 3
    if len(output_list) == length:
        print('save success')


def show(path):
    img_path = path + '/images'
    gt_path = path + '/gt'
    pre_path = path + '/predict_WHUCD'

    list = os.listdir(img_path)
    for i in list:
        if i.split('_')[1][:-4] == '1':
            img1 = read(os.path.join(img_path, i))
            img2 = read(os.path.join(img_path, str(i.split('_')[0]) + '_2.png'))
            label1 = read(os.path.join(gt_path, i[:-4] + '_label.png'))
            label2 = read(os.path.join(gt_path, str(i.split('_')[0]) + '_2_label.png'))
            gt = read(os.path.join(gt_path, str(i.split('_')[0]) + '_change.png'))

            pre1 = read(os.path.join(pre_path, i[:-4] + '_label.png'))
            pre2 = read(os.path.join(pre_path, str(i.split('_')[0]) + '_2_label.png'))
            pre_cd = read(os.path.join(pre_path, str(i.split('_')[0]) + '_change.png'))

            fig, axs = plt.subplots(2, 4, figsize=(20, 8))

            axs[0][0].imshow(img1.astype(np.uint8))
            axs[0][0].axis("off")

            axs[0][1].imshow(img2.astype(np.uint8))
            axs[0][1].axis("off")

            axs[0][2].imshow(label1.astype(np.uint8), cmap='gray')
            axs[0][2].axis("off")

            axs[0][3].imshow(label2.astype(np.uint8), cmap='gray')
            axs[0][3].axis("off")

            axs[1][0].imshow(gt.astype(np.uint8), cmap='gray')
            axs[1][0].axis("off")

            axs[1][1].imshow(pre1.astype(np.uint8), cmap='gray')
            axs[1][1].axis("off")

            axs[1][2].imshow(pre2.astype(np.uint8), cmap='gray')
            axs[1][2].axis("off")

            axs[1][3].imshow(pre_cd.astype(np.uint8), cmap='gray')
            axs[1][3].axis("off")

            plt.suptitle(os.path.basename(i), y=0.94)
            plt.tight_layout()
            plt.show()
            plt.close()


def to(input):
    output = input.copy()
    output[input > 0] = 1
    return output.astype(np.uint8)

def run_test(path):
    img_path = path + '/images'
    gt_path = path + '/gt'
    pre_path = path + '/predict_WHUCD'

    list = os.listdir(img_path)

    compute_metric = IOUMetric(num_classes=2)
    hist1 = np.zeros([2, 2])
    hist2 = np.zeros([2, 2])
    hist_cd = np.zeros([2, 2])

    for i in tqdm(list):
        if i.split('_')[1][:-4] == '1':
            img1 = read(os.path.join(img_path, i))
            img2 = read(os.path.join(img_path, str(i.split('_')[0]) + '_2.png'))
            label1 = read(os.path.join(gt_path, i[:-4] + '_label.png'))
            label2 = read(os.path.join(gt_path, str(i.split('_')[0]) + '_2_label.png'))
            gt = read(os.path.join(gt_path, str(i.split('_')[0]) + '_change.png'))

            pre1 = read(os.path.join(pre_path, i[:-4] + '_label.png'))
            pre2 = read(os.path.join(pre_path, str(i.split('_')[0]) + '_2_label.png'))
            pre_cd = read(os.path.join(pre_path, str(i.split('_')[0]) + '_change.png'))

            # print(to(pre1).max(), to(label1).max())
            hist1 += compute_metric.get_hist(to(pre1), to(label1))
            hist2 += compute_metric.get_hist(to(pre2), to(label2))
            hist_cd += compute_metric.get_hist(to(pre_cd), to(gt))

    f_score1, f_score2, f_score_cd = cal_fscore(hist1), cal_fscore(hist2), cal_fscore(hist_cd)
    f_score = (f_score1 + f_score2) * 0.2 + f_score_cd * 0.6
    print('f_score1:%s f_score2:%s f_score_cd:%s current_fscore:%s'
                % (f_score1, f_score2, f_score_cd, f_score))


if __name__ == '__main__':
    path = '/media/hlf/Luffy/WLS/ContestCD/CDdata/WHUCD'
    # main(path)
    show(path)
    run_test(path)
    # f_score1:0.8683663135660821 f_score2:0.8453706700325836
    # f_score_cd:0.7043251641455283 current_fscore:0.7653424952070501
