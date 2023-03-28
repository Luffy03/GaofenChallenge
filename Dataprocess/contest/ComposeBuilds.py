import ntpath
import os
import random
import numpy as np
import cv2
from utils import *
from matplotlib import pyplot as plt
import dataset.transform as trans
from torchvision import transforms


def blend(src, mask, dst, blend_mode='direct', expand_for_building=True):
    """
    :param src: ndarray h*w*3
    :param mask:ndarray h*w
    :param dst:ndarray h*w*3
    :return:  image blended by src and dst
    """
    assert src.shape[0] == mask.shape[0]
    assert src.shape[0] == dst.shape[0]
    assert src.shape[1] == mask.shape[1]
    assert src.shape[1] == dst.shape[1]
    assert mask.max() == 1
    dh, dw = src.shape[:2]

    # alpha_A
    mask_A = np.zeros([dh, dw],dtype=np.float)
    mask_A[mask == 1] = 1
    #  alpha_B
    mask_B = np.ones([dh, dw], dtype=np.float)
    mask_B[mask == 1] = 0

    if blend_mode is 'direct' or blend_mode == 'poisson':
        expand_for_building = False

    if expand_for_building:
        kernel = np.ones((7, 7), np.uint8)
        mask_expand = cv2.dilate(mask, kernel, iterations=1)
        mask_expand_minus_ = mask_expand - mask
        #  中间过度地带
        mask_A[mask_expand_minus_ == 1] = 1
        mask_B[mask_expand_minus_ == 1] = 0

    if blend_mode == 'poisson':
        center = (dw//2, dh//2)
        mask_ = mask.copy() * 255
        expand = True
        if expand:
            # when levir-cd: d = 9
            # when whu-cd: d=15
            d = 9
            kernel = np.ones((d, d), np.uint8)
            mask_ = cv2.dilate(mask_, kernel, iterations=1)
        out = cv2.seamlessClone(src, dst, mask_, center, cv2.NORMAL_CLONE)
        return out

    elif blend_mode == 'gaussian':
        mask_A = cv2.GaussianBlur(mask_A, (7, 7), 2)
        mask_B = cv2.GaussianBlur(mask_B, (7, 7), 2)

    elif blend_mode == 'box':
        mask_A = cv2.blur(mask_A, (7, 7))
        mask_B = cv2.blur(mask_B, (7, 7))

    # extend to 3D
    mask_A = mask_A[:, :, np.newaxis]
    mask_B = mask_B[:, :, np.newaxis]
    mask_A = np.repeat(mask_A, repeats=3, axis=2)
    mask_B = np.repeat(mask_B, repeats=3, axis=2)

    out = src * mask_A + dst * mask_B
    return out.astype(np.uint8)


def sample_area(labels, dx, dy):
    """
    在h*w的labels中的背景区域中采样一个dx*dy大小的区域,
    最多尝试10次，如果采样失败，则返回None，None
    :param labels: ndarray，h*w，其中labels为0的区域被背景，其余为前景区域
    :return:(x,y)
    """
    h, w = labels.shape[:2]
    # random generate x, y
    y = random.randint(0, h-dy-1)
    x = random.randint(0, w-dx-1)
    #  Determine whether the area is in the labels area
    try_num = 0
    while (try_num < 10):
        if (labels[y:y + dy, x:x + dx].sum()==0):
            return x, y
        else:
            try_num += 1
            y = random.randint(0, h - dy - 1)
            x = random.randint(0, w - dx - 1)
    return None, None


def generate_new_sample(out_img, out_L, img, gt, img_cut, gt_cut, blend_mode='gaussian'):
    """once paste one instance，
    ndarray: a reference value
    :return:
    """
    out_i = out_img.copy()
    out_l = out_L.copy()

    mask_instance = gt_cut
    true_mask = (gt_cut == 255).astype(np.uint8) * 255

    x1, y1, dx, dy = 0, 0, gt_cut.shape[1], gt_cut.shape[0]
    # sample x, y
    x, y = sample_area(gt, dx, dy)
    if x is None:
        # print(None)
        return None

    out_i[y:y + dy, x:x + dx] = blend(src=img_cut[y1:y1 + dy, x1:x1 + dx],
                                      mask=(mask_instance[y1:y1 + dy, x1:x1 + dx] != 0).astype(np.uint8),
                                      dst=img[y:y + dy, x:x + dx],
                                      expand_for_building=True,
                                      blend_mode=blend_mode)

    out_l[y:y + dy, x:x + dx] = true_mask[y1:y1 + dy, x1:x1 + dx]

    return out_i, out_l


def run(path, size=512):
    cut_path = path + '/cut'
    save_path = path + '/mosaic'
    check_dir(save_path)

    img_path = path + '/images'
    gt_path = path + '/gt'

    img_cut_path = cut_path + '/images'
    gt_cut_path = cut_path + '/gt'

    list = []
    for j in range(1, 2001):
        list.append(str(j))
    random.shuffle(list)

    for idx, i in enumerate(list):
        print(idx)
        img1 = util.read(os.path.join(img_path, i + '_1.png'))
        gt1 = util.read(os.path.join(gt_path, i + '_1_label.png'))
        img2 = util.read(os.path.join(img_path, i + '_2.png'))
        gt2 = util.read(os.path.join(gt_path, i + '_2_label.png'))
        cd = util.read(os.path.join(gt_path, i + '_change.png'))

        out_img1, out_img2 = img1.copy(), img2.copy()
        out_L1, out_L2 = gt1.copy(), gt2.copy()

        # the highest number of instance to paste per sample
        M = 3
        num1, num2 = 0, 0
        # check pro
        pro1 = out_L1.sum()/(size * size * 255 + 1)
        pro2 = out_L2.sum() / (size * size * 255 + 1)
        print(pro1)

        # COMPOSE1
        while num1 < 10:
            # cut instance
            list_cut = os.listdir(img_cut_path)
            cut_name = random.choice(list_cut)
            # while cut_name[0] != '1':
            #     cut_name = random.choice(list_cut)
            img_cut = util.read(os.path.join(img_cut_path, cut_name))
            gt_cut = util.read(os.path.join(gt_cut_path, cut_name))

            aug_transform = transforms.Compose([
                trans.Color_Aug()
            ])
            for t in aug_transform.transforms:
                img_cut = t(img_cut)

            results = generate_new_sample(out_img1, out_L1,
                                                      img1, gt1,
                                                      img_cut, gt_cut)
            if results is not None:
                out_img1_, out_L1_ = results

                # check new not overlap !!!
                new = out_L1_ - out_L1
                if (new * gt2).sum() == 0:
                    out_img1, out_L1 = out_img1_, out_L1_
                    pro1 = out_L1.sum() / (size * size * 255)
                    num1 += 1

        print(num1)

        # COMPOSE2
        # while num2 < M and pro2 < 0.05:
        #     out_L2_old = out_L2.copy()
        #     # cut instance
        #     list_cut = os.listdir(img_cut_path)
        #     cut_name = random.choice(list_cut)
        #     while cut_name[0] != '2':
        #         cut_name = random.choice(list_cut)
        #     img_cut = util.read(os.path.join(img_cut_path, cut_name))
        #     gt_cut = util.read(os.path.join(gt_cut_path, cut_name))
        #
        #     results = generate_new_sample(out_img2, out_L2,
        #                                               img2, gt2,
        #                                               img_cut, gt_cut)
        #     if results is not None:
        #         out_img2_, out_L2_ = results
        #
        #         # check new not overlap !!!
        #         new = out_L2_ - out_L2_old
        #         if (new * out_L1).sum() == 0:
        #             out_img2, out_L2 = out_img2_, out_L2_
        #             pro2 = out_L2.sum()/(size*size*255)
        #             num2 += 1

        fig, axs = plt.subplots(2, 5, figsize=(20, 8))

        axs[0][0].imshow(img1.astype(np.uint8))
        axs[0][0].axis("off")
        axs[0][1].imshow(img2.astype(np.uint8))
        axs[0][1].axis("off")
        axs[0][2].imshow(gt1.astype(np.uint8), cmap='gray')
        axs[0][2].axis("off")
        axs[0][3].imshow(gt2.astype(np.uint8), cmap='gray')
        axs[0][3].axis("off")
        axs[0][4].imshow(cd.astype(np.uint8), cmap='gray')
        axs[0][4].axis("off")

        axs[1][0].imshow(out_img1.astype(np.uint8))
        axs[1][0].axis("off")
        axs[1][1].imshow(out_img2.astype(np.uint8))
        axs[1][1].axis("off")
        axs[1][2].imshow(out_L1.astype(np.uint8), cmap='gray')
        axs[1][2].axis("off")
        axs[1][3].imshow(out_L2.astype(np.uint8), cmap='gray')
        axs[1][3].axis("off")
        #
        new1, new2 = out_L1 - gt1, out_L2 - gt2
        new = new1 + new2 + cd
        print(new.shape)
        axs[1][4].imshow(new.astype(np.uint8), cmap='gray')
        axs[1][4].axis("off")

        plt.suptitle(os.path.basename(i), y=0.94)
        plt.tight_layout()
        plt.show()
        plt.close()
        #
        save_img_path, save_gt_path = save_path + '/images', save_path + '/gt'
        check_dir(save_img_path), check_dir(save_gt_path)
        if (new - cd).sum() != 0:
            write(save_img_path + '/' + i + '_1.png', out_img1, flag='img')
            write(save_img_path + '/' + i + '_2.png', out_img2, flag='img')

            write(save_gt_path + '/' + i + '_1_label.png', out_L1)
            write(save_gt_path + '/' + i + '_2_label.png', out_L2)
            write(save_gt_path + '/' + i + '_change.png', new)


if __name__ == '__main__':
    train_path = '/home/hnu2/WLS/ContestCD/CDdata/trainData'
    run(train_path)


