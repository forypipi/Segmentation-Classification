from collections import OrderedDict

import torch
from torch import nn


class UNet3D(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.level1_down = nn.Sequential()
        self.level1_down.add_module("level1_down_conv1", self.ConvBlock(in_channels=2, out_channels=32))
        self.level1_down.add_module("level1_down_conv2", self.ConvBlock(in_channels=32, out_channels=64))

        self.level1_up = nn.Sequential()
        self.level1_up.add_module("level1_up_conv1", self.ConvBlock(in_channels=64+128, out_channels=64))
        self.level1_up.add_module("level1_up_conv2", self.ConvBlock(in_channels=64, out_channels=64))
        self.level1_up.add_module("level1_up_conv3", self.ConvBlock(in_channels=64, out_channels=1))

        self.level2_down = nn.Sequential()
        self.level2_down.add_module("level2_down_conv1", self.ConvBlock(in_channels=64, out_channels=64))
        self.level2_down.add_module("level2_down_conv2", self.ConvBlock(in_channels=64, out_channels=128))

        self.level2_up = nn.Sequential()
        self.level2_up.add_module("level2_up_conv1", self.ConvBlock(in_channels=128+256, out_channels=128))
        self.level2_up.add_module("level2_up_conv2", self.ConvBlock(in_channels=128, out_channels=128))

        self.level3_down = nn.Sequential()
        self.level3_down.add_module("level3_down_conv1", self.ConvBlock(in_channels=128, out_channels=128))
        self.level3_down.add_module("level3_down_conv2", self.ConvBlock(in_channels=128, out_channels=256))

        self.level3_up = nn.Sequential()
        self.level3_up.add_module("level3_up_conv1", self.ConvBlock(in_channels=256+512, out_channels=256))
        self.level3_up.add_module("level3_up_conv2", self.ConvBlock(in_channels=256, out_channels=256))

        self.level4_conv1 = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(5, 5, 5), padding=(2, 2, 2))
        self.level4_conv2 = nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(5, 5, 5), padding=(2, 2, 2))

        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))   # 400 to 200 to 100 to 50

        self.unpool_4_3 = nn.ConvTranspose3d(in_channels=512, out_channels=512, kernel_size=(2,2,2), stride=(2,2,2))
        self.unpool_3_2 = nn.ConvTranspose3d(in_channels=256, out_channels=256, kernel_size=(2,2,2), stride=(2,2,2))
        self.unpool_2_1 = nn.ConvTranspose3d(in_channels=128, out_channels=128, kernel_size=(2,2,2), stride=(2,2,2))

        self.relu = nn.ReLU()
        self.glo_avr = nn.AdaptiveAvgPool3d(1)
        # self.softmax = nn.Softmax()
        self.linear = nn.Sequential(OrderedDict([
            ("Linear1", nn.Linear(in_features=256 + 1 + 3, out_features=32)),
            ("Linear2", nn.Linear(in_features=32, out_features=4))
        ]))
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, img, sex, age):
        """

        :param img: image
        :param sex: sex
        :param age: age
        :return: segmentation, classfication
        """
        print(f"image: {img.shape}")
        x1_down_out = self.level1_down(img)
        x2_down_in = self.maxpool2(x1_down_out)
        x2_down_out = self.level2_down(x2_down_in)
        x3_down_in = self.maxpool2(x2_down_out)
        x3_down_out = self.level3_down(x3_down_in)
        x4_in = self.maxpool2(x3_down_out)
        x4_center = self.level4_conv1(x4_in)
        x4_out = self.level4_conv2(x4_center)
        x3_up_in = torch.cat((x3_down_out, self.unpool_4_3(x4_out)), dim=1)
        x3_up_out = self.level3_up(x3_up_in)
        x2_up_in = torch.cat((x2_down_out, self.unpool_3_2(x3_up_out)), dim=1)
        x2_up_out = self.level2_up(x2_up_in)
        x1_up_in = torch.cat((x1_down_out, self.unpool_2_1(x2_up_out)), dim=1)
        segmentation = self.sigmoid(self.level1_up(x1_up_in))

        img_feature = self.glo_avr(x4_center)
        img_feature_1d = torch.reshape(img_feature, (-1, 256))
        sex = torch.reshape(sex, (-1, 3))
        age = torch.reshape(age, (-1, 1))
        classify_in = torch.cat((img_feature_1d, sex, age), dim=1)
        classify_out = self.softmax(self.linear(classify_in))

        return segmentation, classify_out

    def ConvBlock(self, in_channels, out_channels, kernel_size=(5, 5, 5), padding=(2, 2, 2)):
        conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        bn = nn.BatchNorm3d(num_features=out_channels)
        relu = nn.ReLU()
        sequence = nn.Sequential()
        sequence.add_module("Conv3D", conv)
        sequence.add_module("Batch Normalization", bn)
        sequence.add_module("relu", relu)
        return sequence


class VNet (nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.level1_down = nn.Sequential()
        self.level1_down.add_module("level1_down_conv1", self.ResBlock(in_channels=2, out_channels=16))

        self.level1_up = nn.Sequential()
        self.level1_up.add_module("level1_up_conv1", self.ResBlock(in_channels=16+16, out_channels=32))

        self.down_1to2 = self.DownSampling(in_channels=16, out_channels=32)
        self.level2_down = nn.Sequential()
        self.level2_down.add_module("level2_down_conv1", self.ResBlock(in_channels=32, out_channels=32))
        self.level2_down.add_module("level2_down_conv2", self.ResBlock(in_channels=32, out_channels=32))

        self.level2_up = nn.Sequential()
        self.level2_up.add_module("level2_up_conv1", self.ResBlock(in_channels=32+32, out_channels=64))
        self.level2_up.add_module("level2_up_conv2", self.ResBlock(in_channels=64, out_channels=64))
        self.up_2to1 = self.UpSampling(in_channels=64, out_channels=16)

        self.down_2to3 = self.DownSampling(in_channels=32, out_channels=64)
        self.level3_down = nn.Sequential()
        self.level3_down.add_module("level3_down_conv1", self.ResBlock(in_channels=64, out_channels=64))
        self.level3_down.add_module("level3_down_conv2", self.ResBlock(in_channels=64, out_channels=64))
        self.level3_down.add_module("level3_down_conv3", self.ResBlock(in_channels=64, out_channels=64))

        self.level3_up = nn.Sequential()
        self.level3_up.add_module("level3_up_conv1", self.ResBlock(in_channels=64+64, out_channels=128))
        self.level3_up.add_module("level3_up_conv2", self.ResBlock(in_channels=128, out_channels=128))
        self.level3_up.add_module("level3_up_conv3", self.ResBlock(in_channels=128, out_channels=128))
        self.up_3to2 = self.UpSampling(in_channels=128, out_channels=32)

        self.down_3to4 = self.DownSampling(in_channels=64, out_channels=128)
        self.level4_down = nn.Sequential()
        self.level4_down.add_module("level4_down_conv1", self.ResBlock(in_channels=128, out_channels=128))
        self.level4_down.add_module("level4_down_conv2", self.ResBlock(in_channels=128, out_channels=128))
        self.level4_down.add_module("level4_down_conv3", self.ResBlock(in_channels=128, out_channels=128))

        self.level4_up = nn.Sequential()
        self.level4_up.add_module("level4_up_conv1", self.ResBlock(in_channels=128+128, out_channels=256))
        self.level4_up.add_module("level4_up_conv2", self.ResBlock(in_channels=256, out_channels=256))
        self.level4_up.add_module("level4_up_conv3", self.ResBlock(in_channels=256, out_channels=256))
        self.up_4to3 = self.UpSampling(in_channels=256, out_channels=64)

        self.down_4to5 = self.DownSampling(in_channels=128, out_channels=256)
        self.level5 = nn.Sequential()
        self.level5.add_module("level5_conv1", self.ResBlock(in_channels=256, out_channels=256))
        self.level5.add_module("level5_conv2", self.ResBlock(in_channels=256, out_channels=256))
        self.level5.add_module("level5_conv3", self.ResBlock(in_channels=256, out_channels=256))
        self.up_5to4 = self.UpSampling(in_channels=256, out_channels=128)

        self.conv1kernel = self.ResBlock(in_channels=16+16, out_channels=1, kernel_size=(1,1,1), padding=(0,0,0))
        self.prelu = nn.PReLU()
        self.glo_avr = nn.AdaptiveAvgPool3d(1)
        # self.softmax = nn.Softmax()
        self.linear = nn.Sequential(OrderedDict([
            ("Linear1", nn.Linear(in_features=256 + 1 + 3, out_features=32)),
            ("Linear2", nn.Linear(in_features=32, out_features=4))
        ]))
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, img, sex, age):
        """

        :param img: image
        :param sex: sex
        :param age: age
        :return: segmentation, classfication
        """
        print(f"image: {img.shape}")
        # x1_down_out = img + self.level1_down(img)     # for input is 1 channel
        x1_down_out = self.level1_down(img)
        x2_down_in = self.down_1to2(x1_down_out)
        x2_down_out = x2_down_in + self.level2_down(x2_down_in)
        x3_down_in = self.down_2to3(x2_down_out)
        x3_down_out = x3_down_in + self.level3_down(x3_down_in)
        x4_down_in = self.down_3to4(x3_down_out)
        x4_down_out = x4_down_in + self.level4_down(x4_down_in)
        x5_down_in = self.down_4to5(x4_down_out)
        x5_down_out = self.level5(x5_down_in)
        x4_up_in = torch.cat((x4_down_out, self.up_5to4(x5_down_out)), dim=1)
        x4_up_out = x4_up_in + self.level4_up(x4_up_in)
        x3_up_in = torch.cat((x3_down_out, self.up_4to3(x4_up_out)), dim=1)
        x3_up_out = x3_up_in + self.level3_up(x3_up_in)
        x2_up_in = torch.cat((x2_down_out, self.up_3to2(x3_up_out)), dim=1)
        x2_up_out = x2_up_in + self.level2_up(x2_up_in)
        x1_up_in = torch.cat((x1_down_out, self.up_2to1(x2_up_out)), dim=1)
        segmentation = self.sigmoid(self.conv1kernel(x1_up_in + self.level1_up(x1_up_in)))

        img_feature = self.glo_avr(x5_down_out)
        img_feature_1d = torch.reshape(img_feature, (-1, 256))
        sex = torch.reshape(sex, (-1, 3))
        age = torch.reshape(age, (-1, 1))
        classify_in = torch.cat((img_feature_1d, sex, age), dim=1)
        classify_out = self.softmax(self.linear(classify_in))

        return segmentation, classify_out

    def ResBlock(self, in_channels, out_channels, kernel_size=(5, 5, 5), padding=(2, 2, 2)):
        conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        bn = nn.BatchNorm3d(num_features=out_channels)
        prelu = nn.PReLU()
        sequence = nn.Sequential()
        sequence.add_module("Conv3D", conv)
        sequence.add_module("Batch Normalization", bn)
        sequence.add_module("prelu", prelu)
        return sequence

    def DownSampling(self, in_channels, out_channels, kernel_size=(2,2,2), stride=(2,2,2)):
        conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        prelu = nn.PReLU()
        sequence = nn.Sequential()
        sequence.add_module("DownConv3D", conv)
        sequence.add_module("prelu", prelu)
        return sequence

    def UpSampling(self, in_channels, out_channels, kernel_size=(2,2,2), stride=(2,2,2)):
        conv = nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        prelu = nn.PReLU()
        sequence = nn.Sequential()
        sequence.add_module("DownConv3D", conv)
        sequence.add_module("prelu", prelu)
        return sequence