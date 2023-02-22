from collections import OrderedDict

import torch
from torch import nn

class UNet3D(nn.Module):
    def __init__(self, channel_factor=1) -> None:
        super().__init__()
        self.channel_factor = channel_factor
        self.level1_down = nn.Sequential()
        self.level1_down.add_module("level1_down_conv1", self.ConvBlock(in_channels=2, out_channels=16*channel_factor))
        # self.level1_down.add_module("level1_down_conv2", self.ConvBlock(in_channels=32, out_channels=64))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.level1_up = nn.Sequential()
        self.level1_up.add_module("level1_up_conv1", self.ConvBlock(in_channels=16*channel_factor+32*channel_factor, out_channels=16*channel_factor))
        self.level1_up.add_module("level1_up_conv2", self.ConvBlock(in_channels=16*channel_factor, out_channels=16*channel_factor))
        self.level1_up.add_module("level1_up_conv3", self.ConvBlock(in_channels=16*channel_factor, out_channels=1))

        self.level2_down = nn.Sequential()
        self.level2_down.add_module("level2_down_conv1", self.ConvBlock(in_channels=16*channel_factor, out_channels=16*channel_factor))
        self.level2_down.add_module("level2_down_conv2", self.ConvBlock(in_channels=16*channel_factor, out_channels=32*channel_factor))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.level2_up = nn.Sequential()
        self.level2_up.add_module("level2_up_conv1", self.ConvBlock(in_channels=32*channel_factor+64*channel_factor, out_channels=32*channel_factor))
        self.level2_up.add_module("level2_up_conv2", self.ConvBlock(in_channels=32*channel_factor, out_channels=32*channel_factor))

        self.level3_down = nn.Sequential()
        self.level3_down.add_module("level3_down_conv1", self.ConvBlock(in_channels=32*channel_factor, out_channels=32*channel_factor))
        self.level3_down.add_module("level3_down_conv2", self.ConvBlock(in_channels=32*channel_factor, out_channels=64*channel_factor))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.level3_up = nn.Sequential()
        self.level3_up.add_module("level3_up_conv1", self.ConvBlock(in_channels=64*channel_factor+128*channel_factor, out_channels=64*channel_factor))
        self.level3_up.add_module("level3_up_conv2", self.ConvBlock(in_channels=64*channel_factor, out_channels=64*channel_factor))

        self.level4 = nn.Sequential()
        self.level4.add_module("level4_conv1", self.ConvBlock(in_channels=64*channel_factor, out_channels=64*channel_factor))
        self.level4.add_module("level4_conv1", self.ConvBlock(in_channels=64*channel_factor, out_channels=128*channel_factor))

        self.unpool_4_3 = nn.ConvTranspose3d(in_channels=128*channel_factor, out_channels=128*channel_factor, kernel_size=(2,2,2), stride=(2,2,2))
        self.unpool_3_2 = nn.ConvTranspose3d(in_channels=64*channel_factor, out_channels=64*channel_factor, kernel_size=(2,2,2), stride=(2,2,2))
        self.unpool_2_1 = nn.ConvTranspose3d(in_channels=32*channel_factor, out_channels=32*channel_factor, kernel_size=(2,2,2), stride=(2,2,2))

        self.relu = nn.ReLU()
        self.glo_avr = nn.AdaptiveAvgPool3d(1)
        # self.softmax = nn.Softmax()
        self.linear = nn.Sequential(OrderedDict([
            ("Linear1", nn.Linear(in_features=(128+64+32+16)*channel_factor + 1 + 3, out_features=32)),
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
        x1_down_out = self.level1_down(img)
        x2_down_in = self.maxpool1(x1_down_out)
        x2_down_out = self.level2_down(x2_down_in)
        x3_down_in = self.maxpool2(x2_down_out)
        x3_down_out = self.level3_down(x3_down_in)
        x4_in = self.maxpool3(x3_down_out)
        x4_out = self.level4(x4_in)
        x3_up_in = torch.cat((x3_down_out, self.unpool_4_3(x4_out)), dim=1)
        x3_up_out = self.level3_up(x3_up_in)
        x2_up_in = torch.cat((x2_down_out, self.unpool_3_2(x3_up_out)), dim=1)
        x2_up_out = self.level2_up(x2_up_in)
        x1_up_in = torch.cat((x1_down_out, self.unpool_2_1(x2_up_out)), dim=1)
        segmentation = self.sigmoid(self.level1_up(x1_up_in))

        img_feature1 = self.glo_avr(x1_down_out)
        img_feature_level1 = torch.reshape(img_feature1, (-1, 16*self.channel_factor))
        img_feature2 = self.glo_avr(x2_down_out)
        img_feature_level2 = torch.reshape(img_feature2, (-1, 32*self.channel_factor))
        img_feature3 = self.glo_avr(x3_down_out)
        img_feature_level3 = torch.reshape(img_feature3, (-1, 64*self.channel_factor))
        img_feature4 = self.glo_avr(x4_out)
        img_feature_level4 = torch.reshape(img_feature4, (-1, 128*self.channel_factor))
        sex = torch.reshape(sex, (-1, 3))
        age = torch.reshape(age, (-1, 1))
        classify_in = torch.cat((img_feature_level1, img_feature_level2, img_feature_level3, img_feature_level4, sex, age), dim=1)
        classify_out = self.softmax(self.linear(classify_in))

        return segmentation, classify_out

    def ConvBlock(self, in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)):
        conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        # bn = nn.BatchNorm3d(num_features=out_channels)
        relu = nn.ReLU()
        sequence = nn.Sequential()
        sequence.add_module("Conv3D", conv)
        # sequence.add_module("Batch Normalization", bn)
        sequence.add_module("relu", relu)
        return sequence


class VNet(nn.Module):
    def __init__(self, channel_factor=1) -> None:
        super().__init__()
        self.channel_factor = channel_factor
        self.level1_down = nn.Sequential()
        self.level1_down.add_module("level1_down_conv1", self.ResBlock(in_channels=2, out_channels=16*channel_factor))

        self.level1_up = nn.Sequential()
        self.level1_up.add_module("level1_up_conv1", self.ResBlock(in_channels=16*channel_factor+16*channel_factor, out_channels=32*channel_factor))

        self.down_1to2 = self.DownSampling(in_channels=16*channel_factor, out_channels=32*channel_factor)
        self.level2_down = nn.Sequential()
        self.level2_down.add_module("level2_down_conv1", self.ResBlock(in_channels=32*channel_factor, out_channels=32*channel_factor))
        self.level2_down.add_module("level2_down_conv2", self.ResBlock(in_channels=32*channel_factor, out_channels=32*channel_factor))

        self.level2_up = nn.Sequential()
        self.level2_up.add_module("level2_up_conv1", self.ResBlock(in_channels=32*channel_factor+32*channel_factor, out_channels=64*channel_factor))
        self.level2_up.add_module("level2_up_conv2", self.ResBlock(in_channels=64*channel_factor, out_channels=64*channel_factor))
        self.up_2to1 = self.UpSampling(in_channels=64*channel_factor, out_channels=16*channel_factor)

        self.down_2to3 = self.DownSampling(in_channels=32*channel_factor, out_channels=64*channel_factor)
        self.level3_down = nn.Sequential()
        self.level3_down.add_module("level3_down_conv1", self.ResBlock(in_channels=64*channel_factor, out_channels=64*channel_factor))
        self.level3_down.add_module("level3_down_conv2", self.ResBlock(in_channels=64*channel_factor, out_channels=64*channel_factor))
        self.level3_down.add_module("level3_down_conv3", self.ResBlock(in_channels=64*channel_factor, out_channels=64*channel_factor))

        self.level3_up = nn.Sequential()
        self.level3_up.add_module("level3_up_conv1", self.ResBlock(in_channels=64*channel_factor+64*channel_factor, out_channels=128*channel_factor))
        self.level3_up.add_module("level3_up_conv2", self.ResBlock(in_channels=128*channel_factor, out_channels=128*channel_factor))
        self.level3_up.add_module("level3_up_conv3", self.ResBlock(in_channels=128*channel_factor, out_channels=128*channel_factor))
        self.up_3to2 = self.UpSampling(in_channels=128*channel_factor, out_channels=32*channel_factor)

        self.down_3to4 = self.DownSampling(in_channels=64*channel_factor, out_channels=128*channel_factor)
        self.level4_down = nn.Sequential()
        self.level4_down.add_module("level4_down_conv1", self.ResBlock(in_channels=128*channel_factor, out_channels=128*channel_factor))
        self.level4_down.add_module("level4_down_conv2", self.ResBlock(in_channels=128*channel_factor, out_channels=128*channel_factor))
        self.level4_down.add_module("level4_down_conv3", self.ResBlock(in_channels=128*channel_factor, out_channels=128*channel_factor))

        self.level4_up = nn.Sequential()
        self.level4_up.add_module("level4_up_conv1", self.ResBlock(in_channels=128*channel_factor+128*channel_factor, out_channels=256*channel_factor))
        self.level4_up.add_module("level4_up_conv2", self.ResBlock(in_channels=256*channel_factor, out_channels=256*channel_factor))
        self.level4_up.add_module("level4_up_conv3", self.ResBlock(in_channels=256*channel_factor, out_channels=256*channel_factor))
        self.up_4to3 = self.UpSampling(in_channels=256*channel_factor, out_channels=64*channel_factor)

        self.down_4to5 = self.DownSampling(in_channels=128*channel_factor, out_channels=256*channel_factor)
        self.level5 = nn.Sequential()
        self.level5.add_module("level5_conv1", self.ResBlock(in_channels=256*channel_factor, out_channels=256*channel_factor))
        self.level5.add_module("level5_conv2", self.ResBlock(in_channels=256*channel_factor, out_channels=256*channel_factor))
        self.level5.add_module("level5_conv3", self.ResBlock(in_channels=256*channel_factor, out_channels=256*channel_factor))
        self.up_5to4 = self.UpSampling(in_channels=256*channel_factor, out_channels=128*channel_factor)

        self.conv1kernel = self.ResBlock(in_channels=16*channel_factor+16*channel_factor, out_channels=1, kernel_size=(1,1,1), padding=(0,0,0))
        self.prelu = nn.PReLU()
        self.glo_avr = nn.AdaptiveAvgPool3d(1)
        # self.softmax = nn.Softmax()
        self.linear = nn.Sequential(OrderedDict([
            ("Linear1", nn.Linear(in_features=256*channel_factor + 1 + 3, out_features=32)),
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

    def ResBlock(self, in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)):
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
        sequence.add_module("DeConv3D", conv)
        sequence.add_module("prelu", prelu)
        return sequence


class AttentionUNetblock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionUNetblock, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

class AttentionUNet3D(nn.Module):
    def __init__(self, img_channel=2, output_channel=1):
        super(AttentionUNet3D, self).__init__()

        n1 = 16
        channels = [n1, n1 * 2, n1 * 4, n1 * 8]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.level1_down = nn.Sequential(
            self.ConvBlock(img_channel, channels[0]),
            self.ConvBlock(img_channel, channels[0])
        )

        self.level2_down = nn.Sequential(
            self.ConvBlock(channels[0], channels[1]),
            self.ConvBlock(channels[1], channels[1])
        )

        self.level3_down = nn.Sequential(
            self.ConvBlock(channels[1], channels[2]),
            self.ConvBlock(channels[2], channels[2])
        )

        self.level4_down = nn.Sequential(
            self.ConvBlock(channels[2], channels[3]),
            self.ConvBlock(channels[3], channels[3])
        )

        self.level4_Up = self.UpSampling(channels[3], channels[2])
        self.Att3 = AttentionUNetblock(F_g=channels[2], F_l=channels[2], F_int=channels[2])
        self.Up_conv3 = self.ConvBlock(2*channels[2], channels[1])

        self.level3_Up = self.UpSampling(channels[2], channels[1])
        self.Att2 = AttentionUNetblock(F_g=channels[1], F_l=channels[1], F_int=channels[1])
        self.Up_conv2 = self.ConvBlock(2*channels[1], channels[0])

        self.level2_Up = self.UpSampling(channels[1], channels[0])
        self.Att1 = AttentionUNetblock(F_g=channels[0], F_l=channels[0], F_int=channels[0])
        self.Up_conv1 = self.ConvBlock(2*channels[0], output_channel, kernel_size=(1, 1, 1), padding=(0, 0, 0))

        self.seg = torch.nn.Sigmoid()

    def forward(self, img, sex, age):
        """
        :param img: image
        :param sex: sex
        :param age: age
        :return: segmentation, classfication
        """
        level1_down_out = self.level1_down(img)

        level2_down_in = self.Maxpool1(level1_down_out)
        level2_down_out = self.level2_down(level2_down_in)

        level3_down_in = self.Maxpool2(level2_down_out)
        level3_down_out = self.level3_down(level3_down_in)

        level4_down_in = self.Maxpool3(level3_down_out)
        level4_down_out = self.level4_down(level4_down_in)

        level3_Att_Up_in = self.level4_Up(level4_down_out)
        level3_Att_out = self.Att3(g=level3_Att_Up_in, x=level3_down_out)
        level3_Conv_in = torch.cat((level3_Att_out, level3_Att_Up_in), dim=1)
        level3_Conv_out = self.Up_conv3(level3_Conv_in)

        level2_Att_Up_in = self.level3_Up(level3_Conv_out)
        level2_Att_out = self.Att2(g=level2_Att_Up_in, x=level2_down_out)
        level2_Conv_in = torch.cat((level2_Att_out, level2_Att_Up_in), dim=1)
        level2_Conv_out = self.Up_conv2(level2_Conv_in)

        level1_Att_Up_in = self.level2_Up(level2_Conv_out)
        level1_Att_out = self.Att1(g=level1_Att_Up_in, x=level1_down_out)
        level1_Conv_in = torch.cat((level1_Att_out, level1_Att_Up_in), dim=1)
        level1_Conv_out = self.Up_conv1(level1_Conv_in)

        seg = self.seg(level1_Conv_out)
        
        return out

    def ConvBlock(self, in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)):
        conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        # bn = nn.BatchNorm3d(num_features=out_channels)
        relu = nn.ReLU()
        sequence = nn.Sequential()
        sequence.add_module("Conv3D", conv)
        # sequence.add_module("Batch Normalization", bn)
        sequence.add_module("relu", relu)
        return sequence

    def DownSampling(self, in_channels, out_channels, kernel_size=(2,2,2), stride=(2,2,2)):
        conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        relu = nn.PReLU()
        sequence = nn.Sequential()
        sequence.add_module("DownConv3D", conv)
        sequence.add_module("relu", relu)
        return sequence

    def UpSampling(self, in_channels, out_channels, kernel_size=(2,2,2), stride=(2,2,2)):
        conv = nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        relu = nn.ReLU()
        sequence = nn.Sequential()
        sequence.add_module("DownConv3D", conv)
        sequence.add_module("relu", relu)
        return sequence