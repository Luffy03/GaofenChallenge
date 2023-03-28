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
from models.Uperhead import UPerHead


def build_smooth_layer(filters, key_channels=128):
    smooth3 = build_conv(filters[3], key_channels)
    smooth2 = build_conv(filters[2], key_channels)
    smooth1 = build_conv(filters[1], key_channels)
    smooth0 = build_conv(filters[0], key_channels)
    return smooth3, smooth2, smooth1, smooth0


class Net_uper(nn.Module):
    def __init__(self, backbone='resnet34'):
        super(Net_uper, self).__init__()
        self.backbone = build_backbone(backbone, output_stride=32)
        filters = build_channels(backbone)
        key_channels = 128
        self.img_size = 512

        self.ex_head = UPerHead(in_channels=[256, 512, 1024, 2048],
                    in_index=[0, 1, 2, 3],
                    pool_scales=(1, 2, 3, 6),
                    channels=128,
                    dropout_ratio=0.4,
                    num_classes=1,
                    norm_cfg=dict(type='BN', requires_grad=False),
                    align_corners=False,
                    loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
        self.cd_head = UPerHead(in_channels=[256, 512, 1024, 2048],
                                in_index=[0, 1, 2, 3],
                                pool_scales=(1, 2, 3, 6),
                                channels=128,
                                dropout_ratio=0.4,
                                num_classes=1,
                                norm_cfg=dict(type='BN', requires_grad=False),
                                align_corners=False,
                                loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))

    def resize_out(self, output):
        if output.size()[-1] != self.img_size:
            output = F.interpolate(output, size=(self.img_size, self.img_size), mode='bilinear')
        return output

    def forward(self, x, y):
        x0, x1, x2, x3 = self.backbone(x)
        y0, y1, y2, y3 = self.backbone(y)

        cd0, cd1, cd2, cd3 = torch.abs(x0 - y0), torch.abs(x1 - y1), \
                             torch.abs(x2 - y2), torch.abs(x3 - y3)

        out_x = self.ex_head([x0, x1, x2, x3])
        out_y = self.ex_head([y0, y1, y2, y3])
        out_cd = self.cd_head([cd0, cd1, cd2, cd3])

        out_x, out_y, out_cd = F.sigmoid(out_x), F.sigmoid(out_y), F.sigmoid(out_cd)
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
    net = Net_uper('resnet101')
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