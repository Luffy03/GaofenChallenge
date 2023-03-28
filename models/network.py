import math
import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from models.backbone import *
from models.base_tools import *
from utils.criterion import *


class Net(nn.Module):
    def __init__(self, backbone='resnet34'):
        super(Net, self).__init__()
        self.backbone = build_backbone(backbone, output_stride=32)
        filters = build_channels(backbone)

        # decoder for extraction
        self.dec0, self.dec1, self.dec2, self.dec3, self.dec = build_upsample_decoder(filters)
        self.ex_decoder = build_out_decoder(filters[0]//2)

        # decoder for change detection
        self.dec0_cd, self.dec1_cd, self.dec2_cd, self.dec3_cd, self.dec_cd = build_upsample_decoder(filters)
        self.cd_decoder = build_out_decoder(filters[0]//2)

    def forward(self, x, y):
        x0, x1, x2, x3 = self.backbone(x)
        y0, y1, y2, y3 = self.backbone(y)

        # the T1 decoder
        # Center_1
        # x3 = self.dblock(x3)
        # Decoder_1
        d2_x = self.dec3(x3) + x2
        d1_x = self.dec2(d2_x) + x1
        d0_x = self.dec1(d1_x) + x0
        d_x = self.dec0(d0_x)
        x = self.dec(d_x)

        # the T2 decoder
        # Center_2
        # y3 = self.dblock(y3)
        # Decoder_1
        d2_y = self.dec3(y3) + y2
        d1_y = self.dec2(d2_y) + y1
        d0_y = self.dec1(d1_y) + y0
        d_y = self.dec0(d0_y)
        y = self.dec(d_y)

        # the change branch
        # center_master
        cd3 = torch.abs(x3 - y3)
        # d3 = self.dblock_master(torch.abs(x3 - y3))
        # decoder_master
        cd2 = self.dec3_cd(cd3) + torch.abs(d2_x - d2_y)
        cd1 = self.dec2_cd(cd2) + torch.abs(d1_x - d1_y)
        cd0 = self.dec1_cd(cd1) + torch.abs(d0_x - d0_y)
        cd_ = self.dec0_cd(cd0)
        cd = self.dec_cd(cd_)

        out_x = self.ex_decoder(x)
        out_y = self.ex_decoder(y)
        out_cd = self.cd_decoder(cd)

        return out_x, out_y, out_cd

    def forward_loss(self, x, y, label1, label2, label_cd):
        out_x, out_y, out_cd = self.forward(x, y)

        # focal = BCEFocalLoss()
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
    dataset = CD_Dataset(opt, train_txt_path, flag='train', transform=train_transform)

    loader = DataLoader(
        dataset=dataset,
        batch_size=4, num_workers=8, pin_memory=True, drop_last=True
    )
    net = Net(opt.backbone)
    net.cuda()

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