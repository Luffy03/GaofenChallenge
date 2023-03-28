import os
import numpy as np
import tifffile
from utils import *
import cv2
from matplotlib import pyplot as plt
import random


def extend_bbox(x, y, dx, dy, h, w, extend_num=10):
    """
    extend area of the instance to size of extend_num*2
    :return:
    """
    out_x = x - extend_num
    if out_x < 0:
        return None

    out_y = y - extend_num
    if out_y < 0:
        return None

    out_dx = dx + extend_num*2
    if x+out_dx > w-1:
        return None

    out_dy = dy + extend_num * 2
    if y+out_dy > h-1:
        return None

    return out_x, out_y, out_dx, out_dy


def extract(path, size=512, kernal_size=500):
    img_path = path + '/images'
    gt_path = path + '/gt'

    save_path = path + '/cut'
    check_dir(save_path)

    list = os.listdir(img_path)
    random.shuffle(list)
    count = 0

    for idx, i in enumerate(list):
        name = i.split('_')[0]
        img = util.read(os.path.join(img_path, i))
        label = util.read(os.path.join(gt_path, i[:-4]+'_label.png'))
        mask = label.copy()
        mask[label > 0] = 255

        total = np.zeros([size, size]).astype(np.uint8)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        print(len(contours))
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > kernal_size:
                x, y, w, h = cv2.boundingRect(contour)
                item = extend_bbox(x, y, w, h, size, size)
                if item is not None:
                    x, y, w, h = item
                    img_cut = img[y:y + h, x:x + w, :]
                    label_cut = label[y:y + h, x:x + w]

                    cv2.drawContours(total, contours, idx, 255, cv2.FILLED)
                    m = total[y:y + h, x:x + w]
                    check = np.abs(m - label_cut).sum()
                    # print(check)
                    if check == 0:
                        img_cut_path = save_path + '/images'
                        l_path = save_path + '/gt'
                        check_dir(img_cut_path), check_dir(l_path)
                        write(img_cut_path + '/' + str(name) + '_' + str(count) + '.png', img_cut, flag='img')
                        write(l_path + '/' + str(name) + '_' + str(count) + '.png', label_cut)

                        count += 1

                        # fig, axs = plt.subplots(1, 5, figsize=(14, 4))
                        #
                        # axs[0].imshow(img.astype(np.uint8))
                        # axs[0].axis("off")
                        # axs[1].imshow(label.astype(np.uint8), cmap='gray')
                        # axs[1].axis("off")
                        #
                        # axs[2].imshow(img_cut.astype(np.uint8))
                        # axs[2].axis("off")
                        # axs[3].imshow(label_cut.astype(np.uint8), cmap='gray')
                        # axs[3].axis("off")
                        #
                        # axs[4].imshow(total.astype(np.uint8), cmap='gray')
                        # axs[4].axis("off")
                        # plt.suptitle(os.path.basename(i), y=0.94)
                        # plt.tight_layout()
                        # plt.show()
                        # plt.close()


if __name__ == '__main__':
    train_path = '/home/hnu2/WLS/ContestCD/CDdata/trainData'
    extract(train_path)






