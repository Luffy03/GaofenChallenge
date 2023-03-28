# -*- coding:utf-8 -*-

# @Filename: network_fpn
# @Project : ContestCD
# @date    : 2021-09-08 09:35
# @Author  : Linshan

import math
import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from models.backbone import *
from models.base_tools import *
from utils.criterion import *
from torch.nn import init


def build_smooth_layer(filters, key_channels=128):
    smooth3 = build_conv(filters[3], key_channels)
    smooth2 = build_conv(filters[2], key_channels)
    smooth1 = build_conv(filters[1], key_channels)
    smooth0 = build_conv(filters[0], key_channels)
    return smooth3, smooth2, smooth1, smooth0


class Net_fpn(nn.Module):
    def __init__(self, backbone='resnet34'):
        super(Net_fpn, self).__init__()
        self.backbone = build_backbone(backbone, output_stride=32)
        filters = build_channels(backbone)
        key_channels = 128
        self.img_size = 512

        self.ex_smooth3, self.ex_smooth2, self.ex_smooth1, self.ex_smooth0 = \
            build_smooth_layer(filters, key_channels)
        self.cd_smooth3, self.cd_smooth2, self.cd_smooth1, self.cd_smooth0 = \
            build_smooth_layer(filters, key_channels)

        self.ex_decoder = nn.Sequential(nn.Conv2d(4*key_channels, key_channels, 3, 1, 1),
                                        nn.BatchNorm2d(key_channels),
                                        nn.ReLU(),
                                        nn.Conv2d(key_channels, 1, 3, padding=1),
                                        nn.Sigmoid())

        self.cd_decoder = nn.Sequential(nn.Conv2d(4 * key_channels, key_channels, 3, 1, 1),
                                        nn.BatchNorm2d(key_channels),
                                        nn.ReLU(),
                                        nn.Conv2d(key_channels, 1, 3, padding=1),
                                        nn.Sigmoid())

    def upsample_cat(self, p0, p1, p2, p3):
        # upsample_cat
        p1 = F.interpolate(p1, size=p0.size()[2:], mode='bilinear', align_corners=True)
        p2 = F.interpolate(p2, size=p0.size()[2:], mode='bilinear', align_corners=True)
        p3 = F.interpolate(p3, size=p0.size()[2:], mode='bilinear', align_corners=True)
        return torch.cat([p0, p1, p2, p3], dim=1)

    def top_down_extract(self, f0, f1, f2, f3):
        # fpn-like Top-down
        p3 = self.ex_smooth3(f3)
        p2 = F.interpolate(p3, size=f2.size()[2:], mode='bilinear') + self.ex_smooth2(f2)
        p1 = F.interpolate(p2, size=f1.size()[2:], mode='bilinear') + self.ex_smooth1(f1)
        p0 = F.interpolate(p1, size=f0.size()[2:], mode='bilinear') + self.ex_smooth0(f0)
        return self.upsample_cat(p0, p1, p2, p3)

    def top_down_cd(self, f0, f1, f2, f3):
        # fpn-like Top-down
        p3 = self.cd_smooth3(f3)
        p2 = F.interpolate(p3, size=f2.size()[2:], mode='bilinear') + self.cd_smooth2(f2)
        p1 = F.interpolate(p2, size=f1.size()[2:], mode='bilinear') + self.cd_smooth1(f1)
        p0 = F.interpolate(p1, size=f0.size()[2:], mode='bilinear') + self.cd_smooth0(f0)

        return self.upsample_cat(p0, p1, p2, p3)

    def resize_out(self, output):
        if output.size()[-1] != self.img_size:
            output = F.interpolate(output, size=(self.img_size, self.img_size), mode='bilinear')
        return output

    def forward(self, x, y):
        x0, x1, x2, x3 = self.backbone(x)
        y0, y1, y2, y3 = self.backbone(y)

        cd0, cd1, cd2, cd3 = torch.abs(x0 - y0), torch.abs(x1 - y1), \
                             torch.abs(x2 - y2), torch.abs(x3 - y3)

        x = self.top_down_extract(x0, x1, x2, x3)
        y = self.top_down_extract(y0, y1, y2, y3)
        cd = self.top_down_cd(cd0, cd1, cd2, cd3)

        out_x, out_y = self.ex_decoder(x), self.ex_decoder(y)
        out_cd = self.cd_decoder(cd)
        out_x, out_y, out_cd = self.resize_out(out_x), \
                               self.resize_out(out_y), \
                               self.resize_out(out_cd)

        return out_x, out_y, out_cd

    def forward_loss(self, x, y, label1, label2, label_cd):
        out_x, out_y, out_cd = self.forward(x, y)
        criterion = F1_BCE_loss()
        # criterion = Dice_BCE_loss()

        loss1 = criterion(out_x, label1.float())
        loss2 = criterion(out_y, label2.float())

        # change part
        loss_cd = criterion(out_cd, label_cd.float())

        loss_all = loss1 + loss2 + loss_cd * 2
        return loss_all


if __name__ == "__main__":
    import utils.util as util
    from options import *
    from torch.utils.data import DataLoader
    from dataset import *
    from tqdm import tqdm
    from torchvision import transforms

    opt = Base_Options().parse()
    save_path = os.path.join(opt.save_path, opt.dataset)
    train_txt_path, val_txt_path, test_txt_path = util.create_data_path(opt)
    log_path, checkpoint_path, predict_path, _ = util.create_save_path(opt)

    train_transform = transforms.Compose([
        # trans.Color_Aug(),
        trans.RandomHorizontalFlip(),
        trans.RandomVerticleFlip(),
        trans.RandomRotate90(),
    ])
    dataset = CD_Dataset(opt, val_txt_path, flag='train', transform=train_transform)

    loader = DataLoader(
        dataset=dataset,
        batch_size=2, num_workers=8, pin_memory=True, drop_last=True
    )
    net = Net_fpn('PFSegNet101')
    net.cuda()
    resume = False

    for i in tqdm(loader):
        # put the data from loader to cuda
        img1, img2, label1, label2, change, name = i
        img1, img2, label1, label2, change = img1.cuda(non_blocking=True), \
                                             img2.cuda(non_blocking=True), \
                                             label1.cuda(non_blocking=True), \
                                             label2.cuda(non_blocking=True), \
                                             change.cuda(non_blocking=True),

        # model forward
        # out_x, out_y, out_cd = net.forward(img1, img2)
        loss = net.forward_loss(img1, img2, label1, label2, change)
        pass