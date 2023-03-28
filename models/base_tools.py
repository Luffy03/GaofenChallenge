import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from models.backbone import *


def build_conv(in_channels, out_channels, kernal_size=1, stride=1, dilation=1):
    padding = 0 if kernal_size == 1 else dilation
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernal_size, stride=stride, padding=padding, dilation=dilation),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU())


def build_backbone(backbone, output_stride, pretrained=True, in_c=3):
    if backbone == 'resnet50':
        return ResNet50(output_stride, pretrained=pretrained, in_c=in_c)

    elif backbone == 'resnet101':
        return ResNet101(output_stride, pretrained=pretrained, in_c=in_c)

    elif backbone == 'resnet34':
        return ResNet34(output_stride, pretrained=pretrained, in_c=in_c)

    elif backbone == 'resnet18':
        return ResNet18(output_stride, pretrained=pretrained, in_c=in_c)

    elif backbone == 'mit_b0':
        model = mit_b0()
        checkpoint = torch.load('/home/hnu2/WLS/ContestCD/segformer_b0.pth', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        print('load mitb0')
        return model

    elif backbone == 'mit_b1':
        model = mit_b1()
        checkpoint = torch.load('/home/hnu2/WLS/ContestCD/segformer_b1.pth', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        print('load mitb1')
        return model

    elif backbone == 'mit_b2':
        model = mit_b2()
        checkpoint = torch.load('/home/hnu2/WLS/ContestCD/segformer_b2.pth', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        print('load mitb2')
        return model

    elif backbone == 'mit_b3':
        model = mit_b3()
        checkpoint = torch.load('/home/hnu2/WLS/ContestCD/segformer_b3.pth', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        print('load mitb3')
        return model

    elif backbone == 'mit_b4':
        model = mit_b4()
        checkpoint = torch.load('/home/hnu2/WLS/ContestCD/segformer_b4.pth', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        print('load mitb4')
        return model

    elif backbone == 'mit_b5':
        model = mit_b5()
        checkpoint = torch.load('/home/hnu2/WLS/ContestCD/segformer_b5.pth', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        print('load mitb5')
        return model

    elif backbone == 'convnext_base':
        model = ConvNeXt(dims=[128, 256, 512, 1024])
        checkpoint = torch.load('/home/hnu2/WLS/ContestCD/convnext_base.pth', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        print('load convnext')
        return model

    elif backbone == 'convnext_large':
        model = ConvNeXt(dims=[192, 384, 768, 1536])
        # checkpoint = torch.load('/home/hnu2/WLS/ContestCD/convnext_large.pth', map_location=torch.device('cpu'))
        # model.load_state_dict(checkpoint['state_dict'])
        print('load convnext')
        return model

    elif backbone == 'convnext_xlarge':
        model = ConvNeXt(dims=[256, 512, 1024, 2048])
        checkpoint = torch.load('/home/hnu2/WLS/ContestCD/convnext_xlarge.pth', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        print('load convnext_xlarge')
        return model

    elif backbone == 'convnextv2':
        model = convnextv2_base()
        checkpoint = torch.load('/home/hnu2/WLS/ContestCD/convnextv2_base.pth',
                                map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        print('load convnextv2')
        return model

    elif backbone == 'Swin_S':
        model = SwinTransformer()
        checkpoint = torch.load('/userhome/ContestCD/Swin_S.pth', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        print('load pretrained Swin_S transformer')
        return model

    elif backbone == 'Swin_B':
        model = SwinTransformer(embed_dim=128,
                                num_heads=[4, 8, 16, 32])
        checkpoint = torch.load('/userhome/ContestCD/Swin_B.pth', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        print('load pretrained Swin_S transformer')
        return model

    else:
        raise NotImplementedError


def build_channels(backbone):
    if backbone == 'resnet34' or backbone == 'resnet18':
        channels = [64, 128, 256, 512]

    elif backbone == 'mit_b0':
        channels = [32, 64, 160, 256]

    elif backbone == 'mit_b1':
        channels = [64, 128, 320, 512]

    elif backbone == 'mit_b2':
        channels = [64, 128, 320, 512]

    elif backbone == 'mit_b3':
        channels = [64, 128, 320, 512]

    elif backbone == 'mit_b4':
        channels = [64, 128, 320, 512]

    elif backbone == 'mit_b5':
        channels = [64, 128, 320, 512]

    elif backbone == 'Swin_S':
        channels = [96, 192, 384, 768]

    elif backbone == 'Swin_B':
        channels = [128, 256, 512, 1024]

    elif backbone == 'convnext_base':
        channels = [128, 256, 512, 1024]

    elif backbone == 'convnext_large':
        channels = [192, 384, 768, 1536]

    elif backbone == 'convnext_xlarge':
        channels = [256, 512, 1024, 2048]

    elif backbone == 'convnextv2':
        channels = [128, 256, 512, 1024]

    else:
        channels = [256, 512, 1024, 2048]

    return channels


def build_out_decoder(channels):
    return nn.Sequential(nn.Conv2d(channels, 1, 3, padding=1),
                         nn.Sigmoid())
