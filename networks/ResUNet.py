import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import cv2


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernalsize=(3, 3)):
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernalsize, stride=(1, 1), padding=(kernalsize[0]//2, kernalsize[1]//2), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernalsize, stride=(1, 1), padding=(kernalsize[0]//2, kernalsize[1]//2), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.in_channels == self.out_channels:
            out = F.relu(self.bn2(x + self.conv2(out)))
        else:
            out = F.relu(self.bn2(self.conv2(out)))
        return out


class BasicBlock_light(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock_light, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.in_channels == self.out_channels:
            out = F.relu(self.bn1(self.conv1(x) + x))
        else:
            out = F.relu(self.bn1(self.conv1(x)))
        return out


class UT_ClassifyNet(nn.Module):
    def __init__(self, in_channels=2, medium_channels=[64, 128, 256, 512], num_block=[2, 2, 2, 2], num_class=2):
        super(UT_ClassifyNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, medium_channels[0], kernel_size=(3, 3), stride=(2, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(medium_channels[0])
        self.Resblocks1 = []
        for i in range(num_block[0]):
            self.Resblocks1.append(BasicBlock(medium_channels[0], medium_channels[0]))
        self.Resblocks1 = nn.Sequential(*self.Resblocks1)
        self.conv2a = nn.Conv2d(medium_channels[0], medium_channels[1], kernel_size=(3, 3), stride=(2, 1), padding=1)
        self.conv2b = nn.Conv2d(medium_channels[1], medium_channels[1], kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.bn2 = nn.BatchNorm2d(medium_channels[1])
        self.Resblocks2 = []
        for i in range(num_block[1]):
            self.Resblocks2.append(BasicBlock(medium_channels[1], medium_channels[1]))
        self.Resblocks2 = nn.Sequential(*self.Resblocks2)
        self.conv3 = nn.Conv2d(medium_channels[1], medium_channels[2], kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.bn3 = nn.BatchNorm2d(medium_channels[2])
        self.Resblocks3 = []
        for i in range(num_block[2]):
            self.Resblocks3.append(BasicBlock(medium_channels[2], medium_channels[2]))
        self.Resblocks3 = nn.Sequential(*self.Resblocks3)
        self.conv4 = nn.Conv2d(medium_channels[2], medium_channels[3], kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.bn4 = nn.BatchNorm2d(medium_channels[3])
        self.Resblocks4 = []
        for i in range(num_block[3]):
            self.Resblocks4.append(BasicBlock(medium_channels[3], medium_channels[3]))
        self.Resblocks4 = nn.Sequential(*self.Resblocks4)
        self.out_ds = nn.Conv2d(medium_channels[3], 2 * medium_channels[3], kernel_size=(4, 4), stride=(2, 2))
        self.out_layer = nn.Linear(1024 * 7 * 3, num_class)

    def forward(self, x):
        out_1 = F.relu(self.bn1(self.conv1(x)))
        out_1_b = self.Resblocks1(out_1)
        out_2 = F.relu(self.conv2b(self.conv2a(out_1_b)))
        out_2_b = self.Resblocks2(out_2)
        out_3 = F.relu(self.conv3(out_2_b))
        out_3_b = self.Resblocks3(out_3)
        out_4 = F.relu(self.conv4(out_3_b))
        out_4_b = self.Resblocks4(out_4)
        out_ds = self.out_ds(out_4_b)
        out = self.out_layer(out_ds.view(out_ds.size(0), -1))
        return out



class UT_ClassifyNet_Light(nn.Module):
    def __init__(self, in_channels=2, medium_channels=[8, 16, 16, 32], num_block=[2, 2, 1, 1], num_class=2):
        super(UT_ClassifyNet_Light, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, medium_channels[0], kernel_size=(3, 3), stride=(2, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(medium_channels[0])
        self.Resblocks1 = []
        for i in range(num_block[0]):
            self.Resblocks1.append(BasicBlock_light(medium_channels[0], medium_channels[0]))
        self.Resblocks1 = nn.Sequential(*self.Resblocks1)
        self.conv2a = nn.Conv2d(medium_channels[0], medium_channels[1], kernel_size=(3, 3), stride=(2, 1), padding=1)
        self.conv2b = nn.Conv2d(medium_channels[1], medium_channels[1], kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.bn2 = nn.BatchNorm2d(medium_channels[1])
        self.Resblocks2 = []
        for i in range(num_block[1]):
            self.Resblocks2.append(BasicBlock_light(medium_channels[1], medium_channels[1]))
        self.Resblocks2 = nn.Sequential(*self.Resblocks2)
        self.conv3 = nn.Conv2d(medium_channels[1], medium_channels[2], kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.bn3 = nn.BatchNorm2d(medium_channels[2])
        self.Resblocks3 = []
        for i in range(num_block[2]):
            self.Resblocks3.append(BasicBlock_light(medium_channels[2], medium_channels[2]))
        self.Resblocks3 = nn.Sequential(*self.Resblocks3)
        self.conv4 = nn.Conv2d(medium_channels[2], medium_channels[3], kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.bn4 = nn.BatchNorm2d(medium_channels[3])
        self.Resblocks4 = []
        for i in range(num_block[3]):
            self.Resblocks4.append(BasicBlock_light(medium_channels[3], medium_channels[3]))
        self.Resblocks4 = nn.Sequential(*self.Resblocks4)
        self.out_ds = nn.Conv2d(medium_channels[3], 2 * medium_channels[3], kernel_size=(4, 4), stride=(2, 2))
        self.out_layer = nn.Linear(64 * 7 * 3, num_class)

    def forward(self, x):
        out_1 = F.relu(self.bn1(self.conv1(x)))
        out_1_b = self.Resblocks1(out_1)
        out_2 = F.relu(self.conv2b(self.conv2a(out_1_b)))
        out_2_b = self.Resblocks2(out_2)
        out_3 = F.relu(self.conv3(out_2_b))
        out_3_b = self.Resblocks3(out_3)
        out_4 = F.relu(self.conv4(out_3_b))
        out_4_b = self.Resblocks4(out_4)
        out_ds = self.out_ds(out_4_b)
        out = self.out_layer(out_ds.view(out_ds.size(0), -1))
        return out


class UT_SegmentationNet_Normal(nn.Module):
    def __init__(self, in_channels=2, medium_channels=[96, 192, 384, 768, 1536], num_class=5):
        super(UT_SegmentationNet_Normal, self).__init__()
        # 下采样
        self.conv1 = nn.Conv2d(in_channels, medium_channels[0], kernel_size=(3, 3), stride=(2, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(medium_channels[0])
        self.double_conv_1 = BasicBlock(medium_channels[0], medium_channels[0])
        self.conv2 = nn.Conv2d(medium_channels[0], medium_channels[1], kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.double_conv_2 = BasicBlock(medium_channels[1], medium_channels[1])
        self.bn2 = nn.BatchNorm2d(medium_channels[1])
        self.conv3 = nn.Conv2d(medium_channels[1], medium_channels[2], kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.double_conv_3 = BasicBlock(medium_channels[2], medium_channels[2])
        self.bn3 = nn.BatchNorm2d(medium_channels[2])
        self.conv4 = nn.Conv2d(medium_channels[2], medium_channels[3], kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.bn4 = nn.BatchNorm2d(medium_channels[3])
        self.double_conv_4 = BasicBlock(medium_channels[3], medium_channels[3])
        self.conv5 = nn.Conv2d(medium_channels[3], medium_channels[4], kernel_size=(3, 3), stride=(2, 2), padding=1)

        # 上采样
        self.up_conv1 = nn.ConvTranspose2d(medium_channels[4], medium_channels[3], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
        self.double_up_conv1 = BasicBlock(medium_channels[4], medium_channels[3])
        self.up_conv2 = nn.ConvTranspose2d(medium_channels[3], medium_channels[2], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
        self.double_up_conv2 = BasicBlock(medium_channels[3], medium_channels[2])
        self.up_conv3 = nn.ConvTranspose2d(medium_channels[2], medium_channels[1], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=(1, 0))
        self.double_up_conv3 = BasicBlock(medium_channels[2], medium_channels[1])
        self.up_conv4 = nn.ConvTranspose2d(medium_channels[1], medium_channels[0], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=(1, 0))
        self.double_up_conv4 = BasicBlock(medium_channels[1], medium_channels[0])

        # 整合复原
        self.catconv = nn.ConvTranspose2d(medium_channels[0], num_class, kernel_size=(3, 3), stride=(2, 1), bias=True, padding=1, output_padding=(1, 0))

    def forward(self, x):
        out_1 = F.relu(self.bn1(self.conv1(x)))
        out_1_b = self.double_conv_1(out_1)
        out_2 = F.relu(self.conv2(out_1_b))
        out_2_b = self.double_conv_2(out_2)
        out_3 = F.relu(self.conv3(out_2_b))
        out_3_b = self.double_conv_3(out_3)
        out_4 = F.relu(self.conv4(out_3_b))
        out_4_b = self.double_conv_4(out_4)
        out_5 = F.relu(self.conv5(out_4_b))

        up_1 = F.relu(self.up_conv1(out_5))
        cat_up1 = self.double_up_conv1(torch.concat([out_4_b, up_1], dim=1))
        up_2 = F.relu(self.up_conv2(cat_up1))
        cat_up2 = self.double_up_conv2(torch.concat([out_3_b, up_2], dim=1))
        up_3 = F.relu(self.up_conv3(cat_up2))
        cat_up3 = self.double_up_conv3(torch.concat([out_2_b, up_3], dim=1))
        up_4 = F.relu(self.up_conv4(cat_up3))
        cat_up4 = self.double_up_conv4(torch.concat([out_1_b, up_4], dim=1))
        output = self.catconv(cat_up4)
        return output

class UT_SegmentationNet_Normal_deep(nn.Module):
    def __init__(self, in_channels=2, medium_channels=[64, 128, 256, 512, 1024], num_class=5):
        super(UT_SegmentationNet_Normal_deep, self).__init__()
        # 下采样
        self.conv1 = nn.Conv2d(in_channels, medium_channels[0], kernel_size=(5, 3), stride=(3, 1), padding=(2, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(medium_channels[0])
        self.double_conv_1 = BasicBlock(medium_channels[0], medium_channels[0])
        self.conv2 = nn.Conv2d(medium_channels[0], medium_channels[1], kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.double_conv_2 = BasicBlock(medium_channels[1], medium_channels[1])
        self.bn2 = nn.BatchNorm2d(medium_channels[1])
        self.conv3 = nn.Conv2d(medium_channels[1], medium_channels[2], kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.double_conv_3 = BasicBlock(medium_channels[2], medium_channels[2])
        self.bn3 = nn.BatchNorm2d(medium_channels[2])
        self.conv4 = nn.Conv2d(medium_channels[2], medium_channels[3], kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.bn4 = nn.BatchNorm2d(medium_channels[3])
        self.double_conv_4 = BasicBlock(medium_channels[3], medium_channels[3])
        self.conv5 = nn.Conv2d(medium_channels[3], medium_channels[4], kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.double_conv_5 = BasicBlock(medium_channels[4], medium_channels[4], kernalsize=(3, 3))
        self.double_conv_6 = BasicBlock(medium_channels[4], medium_channels[4], kernalsize=(3, 3))
        # self.conv6 = nn.Conv2d(medium_channels[4], medium_channels[5], kernel_size=(3, 3), stride=(2, 2), padding=1)


        # 上采样
        # self.up_conv0 = nn.ConvTranspose2d(medium_channels[5], medium_channels[4], kernel_size=(5, 5), stride=(2, 2), padding=1, output_padding=1)
        # self.double_up_conv0 = BasicBlock(medium_channels[5], medium_channels[4])
        self.up_conv1 = nn.ConvTranspose2d(medium_channels[4], medium_channels[3], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
        self.double_up_conv1 = BasicBlock(medium_channels[4], medium_channels[3])
        self.up_conv2 = nn.ConvTranspose2d(medium_channels[3], medium_channels[2], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=(0, 1))
        self.double_up_conv2 = BasicBlock(medium_channels[3], medium_channels[2])
        self.up_conv3 = nn.ConvTranspose2d(medium_channels[2], medium_channels[1], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=(1, 0))
        self.double_up_conv3 = BasicBlock(medium_channels[2], medium_channels[1])
        self.up_conv4 = nn.ConvTranspose2d(medium_channels[1], medium_channels[0], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=(0, 0))
        self.double_up_conv4 = BasicBlock(medium_channels[1], medium_channels[0])

        # 整合复原
        self.catconv = nn.ConvTranspose2d(medium_channels[0], num_class, kernel_size=(5, 3), stride=(3, 1), bias=True, padding=(2, 1), output_padding=(1, 0))

    def forward(self, x):
        out_1 = F.relu(self.bn1(self.conv1(x)))
        out_1_b = self.double_conv_1(out_1)
        out_2 = F.relu(self.conv2(out_1_b))
        out_2_b = self.double_conv_2(out_2)
        out_3 = F.relu(self.conv3(out_2_b))
        out_3_b = self.double_conv_3(out_3)
        out_4 = F.relu(self.conv4(out_3_b))
        out_4_b = self.double_conv_4(out_4)
        out_5 = F.relu(self.conv5(out_4_b))
        out_5 = self.double_conv_5(out_5) + out_5
        out_5 = self.double_conv_6(out_5) + out_5

        # up_0 = F.relu(self.up_conv0(out_6))
        # cat_up0 = self.double_up_conv0(torch.concat([out_5_b, up_0], dim=1))

        up_1 = F.relu(self.up_conv1(out_5))
        cat_up1 = self.double_up_conv1(torch.concat([out_4_b, up_1], dim=1))
        up_2 = F.relu(self.up_conv2(cat_up1))
        cat_up2 = self.double_up_conv2(torch.concat([out_3_b, up_2], dim=1))
        up_3 = F.relu(self.up_conv3(cat_up2))
        cat_up3 = self.double_up_conv3(torch.concat([out_2_b, up_3], dim=1))
        up_4 = F.relu(self.up_conv4(cat_up3))
        cat_up4 = self.double_up_conv4(torch.concat([out_1_b, up_4], dim=1))
        output = self.catconv(cat_up4)
        return output

class UT_SegmentationNet_light_deep(nn.Module):
    def __init__(self, in_channels=2, medium_channels=[32, 64, 128, 256, 512], num_class=5):
        super(UT_SegmentationNet_light_deep, self).__init__()
        # 下采样
        self.conv1 = nn.Conv2d(in_channels, medium_channels[0], kernel_size=(5, 3), stride=(3, 1), padding=(2, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(medium_channels[0])
        self.double_conv_1 = BasicBlock(medium_channels[0], medium_channels[0])
        self.conv2 = nn.Conv2d(medium_channels[0], medium_channels[1], kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.double_conv_2 = BasicBlock(medium_channels[1], medium_channels[1])
        self.bn2 = nn.BatchNorm2d(medium_channels[1])
        self.conv3 = nn.Conv2d(medium_channels[1], medium_channels[2], kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.double_conv_3 = BasicBlock(medium_channels[2], medium_channels[2])
        self.bn3 = nn.BatchNorm2d(medium_channels[2])
        self.conv4 = nn.Conv2d(medium_channels[2], medium_channels[3], kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.bn4 = nn.BatchNorm2d(medium_channels[3])
        self.double_conv_4 = BasicBlock(medium_channels[3], medium_channels[3])
        self.conv5 = nn.Conv2d(medium_channels[3], medium_channels[4], kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.double_conv_5 = BasicBlock(medium_channels[4], medium_channels[4], kernalsize=(3, 3))
        self.double_conv_6 = BasicBlock(medium_channels[4], medium_channels[4], kernalsize=(3, 3))
        # self.conv6 = nn.Conv2d(medium_channels[4], medium_channels[5], kernel_size=(3, 3), stride=(2, 2), padding=1)


        # 上采样
        # self.up_conv0 = nn.ConvTranspose2d(medium_channels[5], medium_channels[4], kernel_size=(5, 5), stride=(2, 2), padding=1, output_padding=1)
        # self.double_up_conv0 = BasicBlock(medium_channels[5], medium_channels[4])
        self.up_conv1 = nn.ConvTranspose2d(medium_channels[4], medium_channels[3], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
        self.double_up_conv1 = BasicBlock(medium_channels[4], medium_channels[3])
        self.up_conv2 = nn.ConvTranspose2d(medium_channels[3], medium_channels[2], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=(0, 1))
        self.double_up_conv2 = BasicBlock(medium_channels[3], medium_channels[2])
        self.up_conv3 = nn.ConvTranspose2d(medium_channels[2], medium_channels[1], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=(1, 0))
        self.double_up_conv3 = BasicBlock(medium_channels[2], medium_channels[1])
        self.up_conv4 = nn.ConvTranspose2d(medium_channels[1], medium_channels[0], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=(0, 0))
        self.double_up_conv4 = BasicBlock(medium_channels[1], medium_channels[0])

        # 整合复原
        self.catconv = nn.ConvTranspose2d(medium_channels[0], num_class, kernel_size=(5, 3), stride=(3, 1), bias=True, padding=(2, 1), output_padding=(1, 0))

    def forward(self, x):
        out_1 = F.relu(self.bn1(self.conv1(x)))
        out_1_b = self.double_conv_1(out_1)
        out_2 = F.relu(self.conv2(out_1_b))
        out_2_b = self.double_conv_2(out_2)
        out_3 = F.relu(self.conv3(out_2_b))
        out_3_b = self.double_conv_3(out_3)
        out_4 = F.relu(self.conv4(out_3_b))
        out_4_b = self.double_conv_4(out_4)
        out_5 = F.relu(self.conv5(out_4_b))
        out_5 = self.double_conv_5(out_5) + out_5
        out_5 = self.double_conv_6(out_5) + out_5

        # up_0 = F.relu(self.up_conv0(out_6))
        # cat_up0 = self.double_up_conv0(torch.concat([out_5_b, up_0], dim=1))

        up_1 = F.relu(self.up_conv1(out_5))
        cat_up1 = self.double_up_conv1(torch.concat([out_4_b, up_1], dim=1))
        up_2 = F.relu(self.up_conv2(cat_up1))
        cat_up2 = self.double_up_conv2(torch.concat([out_3_b, up_2], dim=1))
        up_3 = F.relu(self.up_conv3(cat_up2))
        cat_up3 = self.double_up_conv3(torch.concat([out_2_b, up_3], dim=1))
        up_4 = F.relu(self.up_conv4(cat_up3))
        cat_up4 = self.double_up_conv4(torch.concat([out_1_b, up_4], dim=1))
        output = self.catconv(cat_up4)
        return output

class UT_SegmentationNet_tiny_deep(nn.Module):
    def __init__(self, in_channels=2, medium_channels=[16, 32, 64, 128, 256], num_class=5):
        super(UT_SegmentationNet_tiny_deep, self).__init__()
        # 下采样
        self.conv1 = nn.Conv2d(in_channels, medium_channels[0], kernel_size=(5, 3), stride=(3, 1), padding=(2, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(medium_channels[0])
        self.double_conv_1 = BasicBlock(medium_channels[0], medium_channels[0])
        self.conv2 = nn.Conv2d(medium_channels[0], medium_channels[1], kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.double_conv_2 = BasicBlock(medium_channels[1], medium_channels[1])
        self.bn2 = nn.BatchNorm2d(medium_channels[1])
        self.conv3 = nn.Conv2d(medium_channels[1], medium_channels[2], kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.double_conv_3 = BasicBlock(medium_channels[2], medium_channels[2])
        self.bn3 = nn.BatchNorm2d(medium_channels[2])
        self.conv4 = nn.Conv2d(medium_channels[2], medium_channels[3], kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.bn4 = nn.BatchNorm2d(medium_channels[3])
        self.double_conv_4 = BasicBlock(medium_channels[3], medium_channels[3])
        self.conv5 = nn.Conv2d(medium_channels[3], medium_channels[4], kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.double_conv_5 = BasicBlock(medium_channels[4], medium_channels[4], kernalsize=(3, 3))
        self.double_conv_6 = BasicBlock(medium_channels[4], medium_channels[4], kernalsize=(3, 3))
        # self.conv6 = nn.Conv2d(medium_channels[4], medium_channels[5], kernel_size=(3, 3), stride=(2, 2), padding=1)


        # 上采样
        # self.up_conv0 = nn.ConvTranspose2d(medium_channels[5], medium_channels[4], kernel_size=(5, 5), stride=(2, 2), padding=1, output_padding=1)
        # self.double_up_conv0 = BasicBlock(medium_channels[5], medium_channels[4])
        self.up_conv1 = nn.ConvTranspose2d(medium_channels[4], medium_channels[3], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
        self.double_up_conv1 = BasicBlock(medium_channels[4], medium_channels[3])
        self.up_conv2 = nn.ConvTranspose2d(medium_channels[3], medium_channels[2], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=(0, 1))
        self.double_up_conv2 = BasicBlock(medium_channels[3], medium_channels[2])
        self.up_conv3 = nn.ConvTranspose2d(medium_channels[2], medium_channels[1], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=(1, 0))
        self.double_up_conv3 = BasicBlock(medium_channels[2], medium_channels[1])
        self.up_conv4 = nn.ConvTranspose2d(medium_channels[1], medium_channels[0], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=(0, 0))
        self.double_up_conv4 = BasicBlock(medium_channels[1], medium_channels[0])

        # 整合复原
        self.catconv = nn.ConvTranspose2d(medium_channels[0], num_class, kernel_size=(5, 3), stride=(3, 1), bias=True, padding=(2, 1), output_padding=(1, 0))

    def forward(self, x):
        out_1 = F.relu(self.bn1(self.conv1(x)))
        out_1_b = self.double_conv_1(out_1)
        out_2 = F.relu(self.conv2(out_1_b))
        out_2_b = self.double_conv_2(out_2)
        out_3 = F.relu(self.conv3(out_2_b))
        out_3_b = self.double_conv_3(out_3)
        out_4 = F.relu(self.conv4(out_3_b))
        out_4_b = self.double_conv_4(out_4)
        out_5 = F.relu(self.conv5(out_4_b))
        out_5 = self.double_conv_5(out_5) + out_5
        out_5 = self.double_conv_6(out_5) + out_5

        # up_0 = F.relu(self.up_conv0(out_6))
        # cat_up0 = self.double_up_conv0(torch.concat([out_5_b, up_0], dim=1))

        up_1 = F.relu(self.up_conv1(out_5))
        cat_up1 = self.double_up_conv1(torch.concat([out_4_b, up_1], dim=1))
        up_2 = F.relu(self.up_conv2(cat_up1))
        cat_up2 = self.double_up_conv2(torch.concat([out_3_b, up_2], dim=1))
        up_3 = F.relu(self.up_conv3(cat_up2))
        cat_up3 = self.double_up_conv3(torch.concat([out_2_b, up_3], dim=1))
        up_4 = F.relu(self.up_conv4(cat_up3))
        cat_up4 = self.double_up_conv4(torch.concat([out_1_b, up_4], dim=1))
        output = self.catconv(cat_up4)
        return output

# class UT_SegmentationNet_Light(nn.Module):
#     def __init__(self, in_channels=2, medium_channels=[16, 32, 64, 128, 256], num_class=5):
#         super(UT_SegmentationNet_Light, self).__init__()
#         # 下采样
#         self.conv1 = nn.Conv2d(in_channels, medium_channels[0], kernel_size=(3, 3), stride=(2, 1), padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(medium_channels[0])
#         self.double_conv_1 = BasicBlock_light(medium_channels[0], medium_channels[0])
#         self.conv2 = nn.Conv2d(medium_channels[0], medium_channels[1], kernel_size=(3, 3), stride=(2, 2), padding=1)
#         self.double_conv_2 = BasicBlock_light(medium_channels[1], medium_channels[1])
#         self.bn2 = nn.BatchNorm2d(medium_channels[1])
#         self.conv3 = nn.Conv2d(medium_channels[1], medium_channels[2], kernel_size=(3, 3), stride=(2, 2), padding=1)
#         self.double_conv_3 = BasicBlock_light(medium_channels[2], medium_channels[2])
#         self.bn3 = nn.BatchNorm2d(medium_channels[2])
#         self.conv4 = nn.Conv2d(medium_channels[2], medium_channels[3], kernel_size=(3, 3), stride=(2, 2), padding=1)
#         self.bn4 = nn.BatchNorm2d(medium_channels[3])
#         self.double_conv_4 = BasicBlock_light(medium_channels[3], medium_channels[3])
#         self.conv5 = nn.Conv2d(medium_channels[3], medium_channels[4], kernel_size=(3, 3), stride=(2, 2), padding=1)
#
#         # 上采样
#         self.up_conv1 = nn.ConvTranspose2d(medium_channels[4], medium_channels[3], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
#         self.double_up_conv1 = BasicBlock_light(medium_channels[4], medium_channels[3])
#         self.up_conv2 = nn.ConvTranspose2d(medium_channels[3], medium_channels[2], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
#         self.double_up_conv2 = BasicBlock_light(medium_channels[3], medium_channels[2])
#         self.up_conv3 = nn.ConvTranspose2d(medium_channels[2], medium_channels[1], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=(1, 0))
#         self.double_up_conv3 = BasicBlock_light(medium_channels[2], medium_channels[1])
#         self.up_conv4 = nn.ConvTranspose2d(medium_channels[1], medium_channels[0], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=(1, 0))
#         self.double_up_conv4 = BasicBlock_light(medium_channels[1], medium_channels[0])
#
#         # 整合复原
#         self.catconv = nn.ConvTranspose2d(medium_channels[0], num_class, kernel_size=(3, 3), stride=(2, 1), bias=True, padding=1, output_padding=(1, 0))
#
#     def forward(self, x):
#         out_1 = F.relu(self.bn1(self.conv1(x)))
#         out_1_b = self.double_conv_1(out_1)
#         out_2 = F.relu(self.conv2(out_1_b))
#         out_2_b = self.double_conv_2(out_2)
#         out_3 = F.relu(self.conv3(out_2_b))
#         out_3_b = self.double_conv_3(out_3)
#         out_4 = F.relu(self.conv4(out_3_b))
#         out_4_b = self.double_conv_4(out_4)
#         out_5 = F.relu(self.conv5(out_4_b))
#
#         up_1 = F.relu(self.up_conv1(out_5))
#         cat_up1 = self.double_up_conv1(torch.concat([out_4_b, up_1], dim=1))
#         up_2 = F.relu(self.up_conv2(cat_up1))
#         cat_up2 = self.double_up_conv2(torch.concat([out_3_b, up_2], dim=1))
#         up_3 = F.relu(self.up_conv3(cat_up2))
#         cat_up3 = self.double_up_conv3(torch.concat([out_2_b, up_3], dim=1))
#         up_4 = F.relu(self.up_conv4(cat_up3))
#         cat_up4 = self.double_up_conv4(torch.concat([out_1_b, up_4], dim=1))
#         output = self.catconv(cat_up4)
#         return output

class UT_SegmentationNet_tiny(nn.Module):
    def __init__(self, in_channels=2, medium_channels=[16, 32, 64, 128, 256], num_class=5):
        super(UT_SegmentationNet_tiny, self).__init__()
        # 下采样
        self.conv1 = nn.Conv2d(in_channels, medium_channels[0], kernel_size=(3, 3), stride=(2, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(medium_channels[0])
        self.double_conv_1 = BasicBlock_light(medium_channels[0], medium_channels[0])
        self.conv2 = nn.Conv2d(medium_channels[0], medium_channels[1], kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.double_conv_2 = BasicBlock_light(medium_channels[1], medium_channels[1])
        self.bn2 = nn.BatchNorm2d(medium_channels[1])
        self.conv3 = nn.Conv2d(medium_channels[1], medium_channels[2], kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.double_conv_3 = BasicBlock_light(medium_channels[2], medium_channels[2])
        self.bn3 = nn.BatchNorm2d(medium_channels[2])
        self.conv4 = nn.Conv2d(medium_channels[2], medium_channels[3], kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.bn4 = nn.BatchNorm2d(medium_channels[3])
        self.double_conv_4 = BasicBlock_light(medium_channels[3], medium_channels[3])
        self.conv5 = nn.Conv2d(medium_channels[3], medium_channels[4], kernel_size=(3, 3), stride=(2, 2), padding=1)

        # 上采样
        self.up_conv1 = nn.ConvTranspose2d(medium_channels[4], medium_channels[3], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
        self.double_up_conv1 = BasicBlock_light(medium_channels[4], medium_channels[3])
        self.up_conv2 = nn.ConvTranspose2d(medium_channels[3], medium_channels[2], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
        self.double_up_conv2 = BasicBlock_light(medium_channels[3], medium_channels[2])
        self.up_conv3 = nn.ConvTranspose2d(medium_channels[2], medium_channels[1], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=(1, 0))
        self.double_up_conv3 = BasicBlock_light(medium_channels[2], medium_channels[1])
        self.up_conv4 = nn.ConvTranspose2d(medium_channels[1], medium_channels[0], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=(1, 0))
        self.double_up_conv4 = BasicBlock_light(medium_channels[1], medium_channels[0])

        # 整合复原
        self.catconv = nn.ConvTranspose2d(medium_channels[0], num_class, kernel_size=(3, 3), stride=(2, 1), bias=True, padding=1, output_padding=(1, 0))

    def forward(self, x):
        out_1 = F.relu(self.bn1(self.conv1(x)))
        out_1_b = self.double_conv_1(out_1)
        out_2 = F.relu(self.conv2(out_1_b))
        out_2_b = self.double_conv_2(out_2)
        out_3 = F.relu(self.conv3(out_2_b))
        out_3_b = self.double_conv_3(out_3)
        out_4 = F.relu(self.conv4(out_3_b))
        out_4_b = self.double_conv_4(out_4)
        out_5 = F.relu(self.conv5(out_4_b))

        up_1 = F.relu(self.up_conv1(out_5))
        cat_up1 = self.double_up_conv1(torch.concat([out_4_b, up_1], dim=1))
        up_2 = F.relu(self.up_conv2(cat_up1))
        cat_up2 = self.double_up_conv2(torch.concat([out_3_b, up_2], dim=1))
        up_3 = F.relu(self.up_conv3(cat_up2))
        cat_up3 = self.double_up_conv3(torch.concat([out_2_b, up_3], dim=1))
        up_4 = F.relu(self.up_conv4(cat_up3))
        cat_up4 = self.double_up_conv4(torch.concat([out_1_b, up_4], dim=1))
        output = self.catconv(cat_up4)
        return output


class UT_SegmentationNet_light(nn.Module):
    def __init__(self, in_channels=2, medium_channels=[32, 64, 128, 256, 512], num_class=5):
        super(UT_SegmentationNet_light, self).__init__()
        # 下采样
        self.conv1 = nn.Conv2d(in_channels, medium_channels[0], kernel_size=(3, 3), stride=(2, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(medium_channels[0])
        self.double_conv_1 = BasicBlock_light(medium_channels[0], medium_channels[0])
        self.conv2 = nn.Conv2d(medium_channels[0], medium_channels[1], kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.double_conv_2 = BasicBlock_light(medium_channels[1], medium_channels[1])
        self.bn2 = nn.BatchNorm2d(medium_channels[1])
        self.conv3 = nn.Conv2d(medium_channels[1], medium_channels[2], kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.double_conv_3 = BasicBlock_light(medium_channels[2], medium_channels[2])
        self.bn3 = nn.BatchNorm2d(medium_channels[2])
        self.conv4 = nn.Conv2d(medium_channels[2], medium_channels[3], kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.bn4 = nn.BatchNorm2d(medium_channels[3])
        self.double_conv_4 = BasicBlock_light(medium_channels[3], medium_channels[3])
        self.conv5 = nn.Conv2d(medium_channels[3], medium_channels[4], kernel_size=(3, 3), stride=(2, 2), padding=1)

        # 上采样
        self.up_conv1 = nn.ConvTranspose2d(medium_channels[4], medium_channels[3], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
        self.double_up_conv1 = BasicBlock_light(medium_channels[4], medium_channels[3])
        self.up_conv2 = nn.ConvTranspose2d(medium_channels[3], medium_channels[2], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
        self.double_up_conv2 = BasicBlock_light(medium_channels[3], medium_channels[2])
        self.up_conv3 = nn.ConvTranspose2d(medium_channels[2], medium_channels[1], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=(1, 0))
        self.double_up_conv3 = BasicBlock_light(medium_channels[2], medium_channels[1])
        self.up_conv4 = nn.ConvTranspose2d(medium_channels[1], medium_channels[0], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=(1, 0))
        self.double_up_conv4 = BasicBlock_light(medium_channels[1], medium_channels[0])

        # 整合复原
        self.catconv = nn.ConvTranspose2d(medium_channels[0], num_class, kernel_size=(3, 3), stride=(2, 1), bias=True, padding=1, output_padding=(1, 0))

    def forward(self, x):
        out_1 = F.relu(self.bn1(self.conv1(x)))
        out_1_b = self.double_conv_1(out_1)
        out_2 = F.relu(self.conv2(out_1_b))
        out_2_b = self.double_conv_2(out_2)
        out_3 = F.relu(self.conv3(out_2_b))
        out_3_b = self.double_conv_3(out_3)
        out_4 = F.relu(self.conv4(out_3_b))
        out_4_b = self.double_conv_4(out_4)
        out_5 = F.relu(self.conv5(out_4_b))

        up_1 = F.relu(self.up_conv1(out_5))
        cat_up1 = self.double_up_conv1(torch.concat([out_4_b, up_1], dim=1))
        up_2 = F.relu(self.up_conv2(cat_up1))
        cat_up2 = self.double_up_conv2(torch.concat([out_3_b, up_2], dim=1))
        up_3 = F.relu(self.up_conv3(cat_up2))
        cat_up3 = self.double_up_conv3(torch.concat([out_2_b, up_3], dim=1))
        up_4 = F.relu(self.up_conv4(cat_up3))
        cat_up4 = self.double_up_conv4(torch.concat([out_1_b, up_4], dim=1))
        output = self.catconv(cat_up4)
        return output


class UT_SegmentationNet_light_shallow(nn.Module):
    def __init__(self, in_channels=2, medium_channels=[32, 64, 128, 256], num_class=5):
        super(UT_SegmentationNet_light_shallow, self).__init__()
        # 下采样
        self.conv1 = nn.Conv2d(in_channels, medium_channels[0], kernel_size=(3, 3), stride=(2, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(medium_channels[0])
        self.double_conv_1 = BasicBlock_light(medium_channels[0], medium_channels[0])
        self.conv2 = nn.Conv2d(medium_channels[0], medium_channels[1], kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.double_conv_2 = BasicBlock_light(medium_channels[1], medium_channels[1])
        self.bn2 = nn.BatchNorm2d(medium_channels[1])
        self.conv3 = nn.Conv2d(medium_channels[1], medium_channels[2], kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.double_conv_3 = BasicBlock_light(medium_channels[2], medium_channels[2])
        self.bn3 = nn.BatchNorm2d(medium_channels[2])
        self.conv4 = nn.Conv2d(medium_channels[2], medium_channels[3], kernel_size=(3, 3), stride=(2, 2), padding=1)
        # self.bn4 = nn.BatchNorm2d(medium_channels[3])
        # self.double_conv_4 = BasicBlock_light(medium_channels[3], medium_channels[3])
        # self.conv5 = nn.Conv2d(medium_channels[3], medium_channels[4], kernel_size=(3, 3), stride=(2, 2), padding=1)

        # 上采样
        # self.up_conv1 = nn.ConvTranspose2d(medium_channels[4], medium_channels[3], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
        # self.double_up_conv1 = BasicBlock_light(medium_channels[4], medium_channels[3])
        self.up_conv2 = nn.ConvTranspose2d(medium_channels[3], medium_channels[2], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
        self.double_up_conv2 = BasicBlock_light(medium_channels[3], medium_channels[2])
        self.up_conv3 = nn.ConvTranspose2d(medium_channels[2], medium_channels[1], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=(1, 0))
        self.double_up_conv3 = BasicBlock_light(medium_channels[2], medium_channels[1])
        self.up_conv4 = nn.ConvTranspose2d(medium_channels[1], medium_channels[0], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=(1, 0))
        self.double_up_conv4 = BasicBlock_light(medium_channels[1], medium_channels[0])

        # 整合复原
        self.catconv = nn.ConvTranspose2d(medium_channels[0], num_class, kernel_size=(3, 3), stride=(2, 1), bias=True, padding=1, output_padding=(1, 0))

    def forward(self, x):
        out_1 = F.relu(self.bn1(self.conv1(x)))
        out_1_b = self.double_conv_1(out_1)
        out_2 = F.relu(self.conv2(out_1_b))
        out_2_b = self.double_conv_2(out_2)
        out_3 = F.relu(self.conv3(out_2_b))
        out_3_b = self.double_conv_3(out_3)
        out_4 = F.relu(self.conv4(out_3_b))
        # out_4_b = self.double_conv_4(out_4)
        # out_5 = F.relu(self.conv5(out_4_b))
        #
        # up_1 = F.relu(self.up_conv1(out_5))
        # cat_up1 = self.double_up_conv1(torch.concat([out_4_b, up_1], dim=1))
        up_2 = F.relu(self.up_conv2(out_4))
        cat_up2 = self.double_up_conv2(torch.concat([out_3_b, up_2], dim=1))
        up_3 = F.relu(self.up_conv3(cat_up2))
        cat_up3 = self.double_up_conv3(torch.concat([out_2_b, up_3], dim=1))
        up_4 = F.relu(self.up_conv4(cat_up3))
        cat_up4 = self.double_up_conv4(torch.concat([out_1_b, up_4], dim=1))
        output = self.catconv(cat_up4)
        return output


class UT_SegmentationNet_Normal_shallow(nn.Module):
    def __init__(self, in_channels=2, medium_channels=[96, 192, 384, 768], num_class=5):
        super(UT_SegmentationNet_Normal_shallow, self).__init__()
        # 下采样
        self.conv1 = nn.Conv2d(in_channels, medium_channels[0], kernel_size=(3, 3), stride=(2, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(medium_channels[0])
        self.double_conv_1 = BasicBlock_light(medium_channels[0], medium_channels[0])
        self.conv2 = nn.Conv2d(medium_channels[0], medium_channels[1], kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.double_conv_2 = BasicBlock_light(medium_channels[1], medium_channels[1])
        self.bn2 = nn.BatchNorm2d(medium_channels[1])
        self.conv3 = nn.Conv2d(medium_channels[1], medium_channels[2], kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.double_conv_3 = BasicBlock_light(medium_channels[2], medium_channels[2])
        self.bn3 = nn.BatchNorm2d(medium_channels[2])
        self.conv4 = nn.Conv2d(medium_channels[2], medium_channels[3], kernel_size=(3, 3), stride=(2, 2), padding=1)
        # self.bn4 = nn.BatchNorm2d(medium_channels[3])
        # self.double_conv_4 = BasicBlock_light(medium_channels[3], medium_channels[3])
        # self.conv5 = nn.Conv2d(medium_channels[3], medium_channels[4], kernel_size=(3, 3), stride=(2, 2), padding=1)

        # 上采样
        # self.up_conv1 = nn.ConvTranspose2d(medium_channels[4], medium_channels[3], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
        # self.double_up_conv1 = BasicBlock_light(medium_channels[4], medium_channels[3])
        self.up_conv2 = nn.ConvTranspose2d(medium_channels[3], medium_channels[2], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
        self.double_up_conv2 = BasicBlock_light(medium_channels[3], medium_channels[2])
        self.up_conv3 = nn.ConvTranspose2d(medium_channels[2], medium_channels[1], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=(1, 0))
        self.double_up_conv3 = BasicBlock_light(medium_channels[2], medium_channels[1])
        self.up_conv4 = nn.ConvTranspose2d(medium_channels[1], medium_channels[0], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=(1, 0))
        self.double_up_conv4 = BasicBlock_light(medium_channels[1], medium_channels[0])

        # 整合复原
        self.catconv = nn.ConvTranspose2d(medium_channels[0], num_class, kernel_size=(3, 3), stride=(2, 1), bias=True, padding=1, output_padding=(1, 0))

    def forward(self, x):
        out_1 = F.relu(self.bn1(self.conv1(x)))
        out_1_b = self.double_conv_1(out_1)
        out_2 = F.relu(self.conv2(out_1_b))
        out_2_b = self.double_conv_2(out_2)
        out_3 = F.relu(self.conv3(out_2_b))
        out_3_b = self.double_conv_3(out_3)
        out_4 = F.relu(self.conv4(out_3_b))
        # out_4_b = self.double_conv_4(out_4)
        # out_5 = F.relu(self.conv5(out_4_b))
        #
        # up_1 = F.relu(self.up_conv1(out_5))
        # cat_up1 = self.double_up_conv1(torch.concat([out_4_b, up_1], dim=1))
        up_2 = F.relu(self.up_conv2(out_4))
        cat_up2 = self.double_up_conv2(torch.concat([out_3_b, up_2], dim=1))
        up_3 = F.relu(self.up_conv3(cat_up2))
        cat_up3 = self.double_up_conv3(torch.concat([out_2_b, up_3], dim=1))
        up_4 = F.relu(self.up_conv4(cat_up3))
        cat_up4 = self.double_up_conv4(torch.concat([out_1_b, up_4], dim=1))
        output = self.catconv(cat_up4)
        return output


class UT_SegmentationNet_tiny_shallow(nn.Module):
    def __init__(self, in_channels=2, medium_channels=[16, 32, 64, 128], num_class=5):
        super(UT_SegmentationNet_tiny_shallow, self).__init__()
        # 下采样
        self.conv1 = nn.Conv2d(in_channels, medium_channels[0], kernel_size=(3, 3), stride=(2, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(medium_channels[0])
        self.double_conv_1 = BasicBlock_light(medium_channels[0], medium_channels[0])
        self.conv2 = nn.Conv2d(medium_channels[0], medium_channels[1], kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.double_conv_2 = BasicBlock_light(medium_channels[1], medium_channels[1])
        self.bn2 = nn.BatchNorm2d(medium_channels[1])
        self.conv3 = nn.Conv2d(medium_channels[1], medium_channels[2], kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.double_conv_3 = BasicBlock_light(medium_channels[2], medium_channels[2])
        self.bn3 = nn.BatchNorm2d(medium_channels[2])
        self.conv4 = nn.Conv2d(medium_channels[2], medium_channels[3], kernel_size=(3, 3), stride=(2, 2), padding=1)
        # self.bn4 = nn.BatchNorm2d(medium_channels[3])
        # self.double_conv_4 = BasicBlock_light(medium_channels[3], medium_channels[3])
        # self.conv5 = nn.Conv2d(medium_channels[3], medium_channels[4], kernel_size=(3, 3), stride=(2, 2), padding=1)

        # 上采样
        # self.up_conv1 = nn.ConvTranspose2d(medium_channels[4], medium_channels[3], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
        # self.double_up_conv1 = BasicBlock_light(medium_channels[4], medium_channels[3])
        self.up_conv2 = nn.ConvTranspose2d(medium_channels[3], medium_channels[2], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
        self.double_up_conv2 = BasicBlock_light(medium_channels[3], medium_channels[2])
        self.up_conv3 = nn.ConvTranspose2d(medium_channels[2], medium_channels[1], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=(1, 0))
        self.double_up_conv3 = BasicBlock_light(medium_channels[2], medium_channels[1])
        self.up_conv4 = nn.ConvTranspose2d(medium_channels[1], medium_channels[0], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=(1, 0))
        self.double_up_conv4 = BasicBlock_light(medium_channels[1], medium_channels[0])

        # 整合复原
        self.catconv = nn.ConvTranspose2d(medium_channels[0], num_class, kernel_size=(3, 3), stride=(2, 1), bias=True, padding=1, output_padding=(1, 0))

    def forward(self, x):
        out_1 = F.relu(self.bn1(self.conv1(x)))
        out_1_b = self.double_conv_1(out_1)
        out_2 = F.relu(self.conv2(out_1_b))
        out_2_b = self.double_conv_2(out_2)
        out_3 = F.relu(self.conv3(out_2_b))
        out_3_b = self.double_conv_3(out_3)
        out_4 = F.relu(self.conv4(out_3_b))
        # out_4_b = self.double_conv_4(out_4)
        # out_5 = F.relu(self.conv5(out_4_b))
        #
        # up_1 = F.relu(self.up_conv1(out_5))
        # cat_up1 = self.double_up_conv1(torch.concat([out_4_b, up_1], dim=1))
        up_2 = F.relu(self.up_conv2(out_4))
        cat_up2 = self.double_up_conv2(torch.concat([out_3_b, up_2], dim=1))
        up_3 = F.relu(self.up_conv3(cat_up2))
        cat_up3 = self.double_up_conv3(torch.concat([out_2_b, up_3], dim=1))
        up_4 = F.relu(self.up_conv4(cat_up3))
        cat_up4 = self.double_up_conv4(torch.concat([out_1_b, up_4], dim=1))
        output = self.catconv(cat_up4)
        return output

#
# class UT_Segmentation_DepthPriorNet(nn.Module):
#
#     def __init__(self, in_channels=2, medium_channels=[64, 128, 256, 512, 1024], num_class=16):
#         super(UT_Segmentation_DepthPriorNet, self).__init__()
#         # 下采样
#         self.conv1 = nn.Conv2d(in_channels, medium_channels[0], kernel_size=(3, 3), stride=(2, 1), padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(medium_channels[0])
#         self.double_conv_1 = BasicBlock(medium_channels[0], medium_channels[0])
#         self.conv2 = nn.Conv2d(medium_channels[0], medium_channels[1], kernel_size=(3, 3), stride=(2, 2), padding=1)
#         self.double_conv_2 = BasicBlock(medium_channels[1], medium_channels[1])
#         self.bn2 = nn.BatchNorm2d(medium_channels[1])
#         self.conv3 = nn.Conv2d(medium_channels[1], medium_channels[2], kernel_size=(3, 3), stride=(2, 2), padding=1)
#         self.double_conv_3 = BasicBlock(medium_channels[2], medium_channels[2])
#         self.bn3 = nn.BatchNorm2d(medium_channels[2])
#         self.conv4 = nn.Conv2d(medium_channels[2], medium_channels[3], kernel_size=(3, 3), stride=(2, 2), padding=1)
#         self.bn4 = nn.BatchNorm2d(medium_channels[3])
#         self.double_conv_4 = BasicBlock(medium_channels[3], medium_channels[3])
#         self.conv5 = nn.Conv2d(medium_channels[3], medium_channels[4], kernel_size=(3, 3), stride=(2, 2), padding=1)
#
#         # 上采样
#         self.up_conv1 = nn.ConvTranspose2d(medium_channels[4], medium_channels[3], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
#         self.double_up_conv1 = BasicBlock(medium_channels[4], medium_channels[3])
#         self.up_conv2 = nn.ConvTranspose2d(medium_channels[3], medium_channels[2], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
#         self.double_up_conv2 = BasicBlock(medium_channels[3], medium_channels[2])
#         self.up_conv3 = nn.ConvTranspose2d(medium_channels[2], medium_channels[1], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=(1, 0))
#         self.double_up_conv3 = BasicBlock(medium_channels[2], medium_channels[1])
#         self.up_conv4 = nn.ConvTranspose2d(medium_channels[1], medium_channels[0], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=(1, 0))
#         self.double_up_conv4 = BasicBlock(medium_channels[1], medium_channels[0])
#
#         # 整合复原
#         self.catconv1 = nn.ConvTranspose2d(medium_channels[0], num_class, kernel_size=(3, 3), stride=(2, 1), bias=True, padding=1, output_padding=(1, 0))   # 预测：0 vs. 1
#         self.catconv2 = nn.Conv2d(num_class + 1, num_class, kernel_size=(3, 3), stride=(1, 1), bias=True, padding=1)        # 1 预测：0 vs. 2
#         self.catconv3 = nn.Conv2d(num_class + 1 + 2, num_class, kernel_size=(3, 3), stride=(1, 1), bias=True, padding=1)        # 1cat2 预测 0 vs. 3
#         self.catconv4 = nn.Conv2d(num_class + 1 + 4, 2, kernel_size=(3, 3), stride=(1, 1), bias=True, padding=1)                    # 1cat2cat3 预测 0 vs. 4
#
#     def forward(self, x):
#         out_1 = F.relu(self.bn1(self.conv1(x)))
#         out_1_b = self.double_conv_1(out_1)
#         out_2 = F.relu(self.conv2(out_1_b))
#         out_2_b = self.double_conv_2(out_2)
#         out_3 = F.relu(self.conv3(out_2_b))
#         out_3_b = self.double_conv_3(out_3)
#         out_4 = F.relu(self.conv4(out_3_b))
#         out_4_b = self.double_conv_4(out_4)
#         out_5 = F.relu(self.conv5(out_4_b))
#
#         up_1 = F.relu(self.up_conv1(out_5))
#         cat_up1 = self.double_up_conv1(torch.concat([out_4_b, up_1], dim=1))
#         up_2 = F.relu(self.up_conv2(cat_up1))
#         cat_up2 = self.double_up_conv2(torch.concat([out_3_b, up_2], dim=1))
#         up_3 = F.relu(self.up_conv3(cat_up2))
#         cat_up3 = self.double_up_conv3(torch.concat([out_2_b, up_3], dim=1))
#         up_4 = F.relu(self.up_conv4(cat_up3))
#         cat_up4 = self.double_up_conv4(torch.concat([out_1_b, up_4], dim=1))
#         output_1 = self.catconv1(cat_up4)
#         pred_01 = output_1[:, :2, :, :]
#         output_2 = self.catconv2(F.relu(torch.concat([x[:, 0, :, :].unsqueeze(1), output_1], dim=1)))
#         pred_02 = output_2[:, :2, :, :]
#         output_3 = self.catconv3(F.relu(torch.concat([x[:, 0, :, :].unsqueeze(1), pred_01, output_2], dim=1)))
#         pred_03 = output_3[:, :2, :, :]
#         pred_04 = self.catconv4(F.relu(torch.concat([x[:, 0, :, :].unsqueeze(1), pred_01, pred_02, output_3], dim=1)))
#         return torch.concat([pred_01, pred_02, pred_03, pred_04], dim=1)
#
#
# class UT_Segmentation_DepthPriorNet_Light(nn.Module):
#     def __init__(self, in_channels=2, medium_channels=[12, 24, 48, 96, 192], num_class=8):
#         super(UT_Segmentation_DepthPriorNet_Light, self).__init__()
#         # 下采样
#         self.conv1 = nn.Conv2d(in_channels, medium_channels[0], kernel_size=(3, 3), stride=(2, 1), padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(medium_channels[0])
#         self.double_conv_1 = BasicBlock_light(medium_channels[0], medium_channels[0])
#         self.conv2 = nn.Conv2d(medium_channels[0], medium_channels[1], kernel_size=(3, 3), stride=(2, 2), padding=1)
#         self.double_conv_2 = BasicBlock_light(medium_channels[1], medium_channels[1])
#         self.bn2 = nn.BatchNorm2d(medium_channels[1])
#         self.conv3 = nn.Conv2d(medium_channels[1], medium_channels[2], kernel_size=(3, 3), stride=(2, 2), padding=1)
#         self.double_conv_3 = BasicBlock_light(medium_channels[2], medium_channels[2])
#         self.bn3 = nn.BatchNorm2d(medium_channels[2])
#         self.conv4 = nn.Conv2d(medium_channels[2], medium_channels[3], kernel_size=(3, 3), stride=(2, 2), padding=1)
#         self.bn4 = nn.BatchNorm2d(medium_channels[3])
#         self.double_conv_4 = BasicBlock_light(medium_channels[3], medium_channels[3])
#         self.conv5 = nn.Conv2d(medium_channels[3], medium_channels[4], kernel_size=(3, 3), stride=(2, 2), padding=1)
#
#         # 上采样
#         self.up_conv1 = nn.ConvTranspose2d(medium_channels[4], medium_channels[3], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
#         self.double_up_conv1 = BasicBlock_light(medium_channels[4], medium_channels[3])
#         self.up_conv2 = nn.ConvTranspose2d(medium_channels[3], medium_channels[2], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
#         self.double_up_conv2 = BasicBlock_light(medium_channels[3], medium_channels[2])
#         self.up_conv3 = nn.ConvTranspose2d(medium_channels[2], medium_channels[1], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=(1, 0))
#         self.double_up_conv3 = BasicBlock_light(medium_channels[2], medium_channels[1])
#         self.up_conv4 = nn.ConvTranspose2d(medium_channels[1], medium_channels[0], kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=(1, 0))
#         self.double_up_conv4 = BasicBlock_light(medium_channels[1], medium_channels[0])
#
#         # 整合复原
#         self.catconv1 = nn.ConvTranspose2d(medium_channels[0], num_class, kernel_size=(3, 3), stride=(2, 1), bias=True, padding=1, output_padding=(1, 0))   # 预测：0 vs. 1
#         self.catconv2 = nn.Conv2d(num_class + 1, num_class, kernel_size=(3, 3), stride=(1, 1), bias=True, padding=1)        # 1 预测：0 vs. 2
#         self.catconv3 = nn.Conv2d(num_class + 1 + 2, num_class, kernel_size=(3, 3), stride=(1, 1), bias=True, padding=1)        # 1cat2 预测 0 vs. 3
#         self.catconv4 = nn.Conv2d(num_class + 1 + 4, 2, kernel_size=(3, 3), stride=(1, 1), bias=True, padding=1)                    # 1cat2cat3 预测 0 vs. 4
#
#     def forward(self, x):
#         out_1 = F.relu(self.bn1(self.conv1(x)))
#         out_1_b = self.double_conv_1(out_1)
#         out_2 = F.relu(self.conv2(out_1_b))
#         out_2_b = self.double_conv_2(out_2)
#         out_3 = F.relu(self.conv3(out_2_b))
#         out_3_b = self.double_conv_3(out_3)
#         out_4 = F.relu(self.conv4(out_3_b))
#         out_4_b = self.double_conv_4(out_4)
#         out_5 = F.relu(self.conv5(out_4_b))
#
#         up_1 = F.relu(self.up_conv1(out_5))
#         cat_up1 = self.double_up_conv1(torch.concat([out_4_b, up_1], dim=1))
#         up_2 = F.relu(self.up_conv2(cat_up1))
#         cat_up2 = self.double_up_conv2(torch.concat([out_3_b, up_2], dim=1))
#         up_3 = F.relu(self.up_conv3(cat_up2))
#         cat_up3 = self.double_up_conv3(torch.concat([out_2_b, up_3], dim=1))
#         up_4 = F.relu(self.up_conv4(cat_up3))
#         cat_up4 = self.double_up_conv4(torch.concat([out_1_b, up_4], dim=1))
#         output_1 = self.catconv1(cat_up4)
#         pred_01 = output_1[:, :2, :, :]
#         output_2 = self.catconv2(F.relu(torch.concat([x[:, 0, :, :].unsqueeze(1), output_1], dim=1)))
#         pred_02 = output_2[:, :2, :, :]
#         output_3 = self.catconv3(F.relu(torch.concat([x[:, 0, :, :].unsqueeze(1), pred_01, output_2], dim=1)))
#         pred_03 = output_3[:, :2, :, :]
#         pred_04 = self.catconv4(F.relu(torch.concat([x[:, 0, :, :].unsqueeze(1), pred_01, pred_02, output_3], dim=1)))
#         return torch.concat([pred_01, pred_02, pred_03, pred_04], dim=1)
#

class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, alpha=[0.2, 0.8, 0.8, 1.5], gamma=2, reduction='mean', device='cuda'):
        """
        :param alpha: 权重系数列表，三分类中第0类权重0.2，第1类权重0.3，第2类权重0.5
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        self.alpha = torch.tensor(alpha).to(device)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()

#
# class Multi_2Class_FocalLossWithAlpha(nn.Module):
#     def __init__(self, alpha=[0.8, 0.7, 1.0, 0.4], gamma=2, reduction='mean', device='cuda'):
#         """
#         :param alpha: 权重系数列表，三分类中第0类权重0.2，第1类权重0.3，第2类权重0.5
#         :param gamma: 困难样本挖掘的gamma
#         :param reduction:
#         """
#         super(Multi_2Class_FocalLossWithAlpha, self).__init__()
#         self.alpha = torch.tensor(alpha).to(device)
#         self.gamma = gamma
#         self.reduction = reduction
#
#     def forward(self, pred, target_volume):
#         class1, class2, class3, class4 = pred[:, 0:2, :, :], pred[:, 2:4, :, :], pred[:, 4:6, :, :], pred[:, 6:8, :, :]
#         loss_01 = F.cross_entropy(class1, target_volume[:, 0, :, :], reduction='none')
#         loss_02 = F.cross_entropy(class2, target_volume[:, 1, :, :], reduction='none')
#         loss_03 = F.cross_entropy(class3, target_volume[:, 2, :, :], reduction='none')
#         loss_04 = F.cross_entropy(class4, target_volume[:, 3, :, :], reduction='none')
#
#         pt1 = torch.exp(-loss_01)
#         focal_loss1 = self.alpha[0] * (1 - pt1) ** self.gamma * loss_01
#         pt2 = torch.exp(-loss_02)
#         focal_loss2 = self.alpha[1] * (1 - pt2) ** self.gamma * loss_02
#         pt3 = torch.exp(-loss_03)
#         focal_loss3 = self.alpha[2] * (1 - pt3) ** self.gamma * loss_03
#         pt4 = torch.exp(-loss_04)
#         focal_loss4 = self.alpha[3] * (1 - pt4) ** self.gamma * loss_04
#
#         if self.reduction == 'mean':
#             return focal_loss1.mean() + focal_loss2.mean() + focal_loss3.mean() + focal_loss4.mean()
#         elif self.reduction == 'sum':
#             return focal_loss1.sum() + focal_loss2.sum() + focal_loss3.sum() + focal_loss4.sum()


class ClassifyDataloader(Dataset):
    def __init__(self, file_dir, type="Train"):
        self.all_files = os.listdir(file_dir)
        if type == 'Train':
            self.all_inputs = [os.path.join(file_dir, file) for file in self.all_files if file.endswith("_t.npy")]
        elif type == 'Valid':
            self.all_inputs = [os.path.join(file_dir, file) for file in self.all_files if file.endswith("_v.npy")]

    def __len__(self):
        return len(self.all_inputs)

    def __getitem__(self, item):
        filename = self.all_inputs[item]
        array = np.load(filename)[:, :512, :].astype('float32')
        label = np.array([0, 0], dtype='float32')
        label[int(filename.split('\\')[-1][0])] = 1.0
        label = torch.from_numpy(label).cuda()
        array_torch = torch.from_numpy(array).cuda()
        return array_torch, label, filename


class SegDataloader(Dataset):
    def __init__(self, file_dir, type="Train", mode=0, opt='ndep', length=512):
        self.all_files = os.listdir(file_dir)
        self.mode = mode
        if type == 'Train':
            self.all_inputs = [os.path.join(file_dir, file) for file in self.all_files if file.endswith("_t.npy")]
        elif type == 'Valid':
            self.all_inputs = [os.path.join(file_dir, file) for file in self.all_files if (file.endswith("_v.npy"))] # or file.endswith("_e.npy")
        elif type == 'Eval':
            self.all_inputs = [os.path.join(file_dir, file) for file in self.all_files if file.endswith("_e.npy")]
        self.opt = opt
        self.ilength = length
        self.all_sets = []
        for fname in self.all_inputs:
            array = np.load(fname)
            if self.mode == 0:
                if self.opt.split('-')[0] == 'yz':
                    array_torch = torch.from_numpy(array[:3, :, :].astype('float32')).cuda()
                    label = torch.from_numpy(array[3, :, :].astype('int64')).cuda()
                else:
                    array_torch = torch.from_numpy(array[:2, :, :].astype('float32')).cuda()
                    label = torch.from_numpy(array[2, :, :].astype('int64')).cuda()
            else:
                array_torch = torch.from_numpy(array[:2, :, :].astype('float32')).cuda()
                label = torch.from_numpy(array[3:, :, :].astype('int64')).cuda()
            array_torch = (array_torch / 128) - 1.0
            self.all_sets.append([array_torch, label])

    def __len__(self):
        return len(self.all_inputs)

    def __getitem__(self, item):
        data_item = self.all_sets[item]
        return data_item[0], data_item[1], self.all_inputs[item]


def GenerateDataloader(file_dir, type="Train", model='Classify', opt='dep', batch_size=16, mode=0, shuffle=True):
    if model == 'seg':
        return DataLoader(SegDataloader(file_dir, type, mode, opt=opt, length=512), batch_size=batch_size, shuffle=shuffle)
    else:
        return DataLoader(ClassifyDataloader(file_dir, type), batch_size=batch_size, shuffle=shuffle)