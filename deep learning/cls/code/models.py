import torch
from torch import nn

class UNet(nn.Module):
    def __init__(self, channel_factor=1) -> None:
        super().__init__()
        n = 16 * channel_factor
        self.channels = (n, n * 2, n * 4, n * 8)
        self.channel_factor = channel_factor
        self.level1_down = nn.Sequential(
            self.ConvBlock(in_channels=2, out_channels=self.channels[0]),
            # self.ConvBlock(in_channels=self.channels[0], out_channels=self.channels[1]),
        )

        self.level2_down = nn.Sequential(
            self.ConvBlock(in_channels=self.channels[0], out_channels=self.channels[0]),
            self.ConvBlock(in_channels=self.channels[0], out_channels=self.channels[1]),
        )

        self.level3_down = nn.Sequential(
            self.ConvBlock(in_channels=self.channels[1], out_channels=self.channels[1]),
            self.ConvBlock(in_channels=self.channels[1], out_channels=self.channels[2]),
        )

        self.level4 = nn.Sequential(
            self.ConvBlock(in_channels=self.channels[2], out_channels=self.channels[2]),
            self.ConvBlock(in_channels=self.channels[2], out_channels=self.channels[3])
        )

        self.maxpool1 = nn.MaxPool3d(kernel_size=(4, 4, 4), stride=(4, 4, 4))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(4, 4, 4), stride=(4, 4, 4))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(4, 4, 4), stride=(4, 4, 4))

        self.relu = nn.ReLU()
        self.glo_avr = nn.AdaptiveAvgPool3d(1)
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.channels[3]*2*2*2 + 1 + 3,
                      out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=4)
        )
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

        img_feature4 = x4_out.view(x4_out.size(0), -1)
        sex = torch.reshape(sex, (-1, 3))
        age = torch.reshape(age, (-1, 1))
        classify_in = torch.cat(
            (img_feature4, sex, age), dim=1)
        classify_out = self.softmax(self.linear(classify_in))

        return classify_out

    def ConvBlock(self, in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)):
        conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        bn = nn.BatchNorm3d(num_features=out_channels)
        relu = nn.ReLU()
        sequence = nn.Sequential()
        sequence.add_module("Conv3D", conv)
        sequence.add_module("Batch Normalization", bn)
        sequence.add_module("relu", relu)
        return sequence

class VNet(nn.Module):
    def __init__(self, channel_factor=1) -> None:
        super().__init__()

        n = 16 * channel_factor
        self.channels = (n, n * 2, n * 4, n * 8)

        self.channel_factor = channel_factor
        self.level1_down = nn.Sequential(
            self.ConvBlock(in_channels=2, out_channels=self.channels[0])
        )

        self.down_1to2 = self.DownSampling(in_channels=self.channels[0], out_channels=self.channels[1])
        self.level2_down = nn.Sequential(
            ResBlock(in_channels=self.channels[1], out_channels=self.channels[1]),
            ResBlock(in_channels=self.channels[1], out_channels=self.channels[1]),
        )

        self.down_2to3 = self.DownSampling(in_channels=self.channels[1], out_channels=self.channels[2])
        self.level3_down = nn.Sequential(
            ResBlock(in_channels=self.channels[2], out_channels=self.channels[2]),
            ResBlock(in_channels=self.channels[2], out_channels=self.channels[2]),
        )

        self.down_3to4 = self.DownSampling(in_channels=self.channels[2], out_channels=self.channels[3])
        self.level4_down = nn.Sequential(
            ResBlock(in_channels=self.channels[3], out_channels=self.channels[3]),
            ResBlock(in_channels=self.channels[3], out_channels=self.channels[3]),
        )

        self.level1_conv = self.ConvBlock(in_channels=self.channels[0], out_channels=self.channels[1])
        self.level2_conv = self.ConvBlock(in_channels=self.channels[1], out_channels=self.channels[2])
        self.level3_conv = self.ConvBlock(in_channels=self.channels[2], out_channels=self.channels[1])
        
        self.maxpool1 = nn.MaxPool3d(kernel_size=(4, 4, 4), stride=(4, 4, 4))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(4, 4, 4), stride=(4, 4, 4))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(4, 4, 4), stride=(4, 4, 4))

        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax()
        self.linear = nn.Sequential(
            nn.Linear(in_features= self.channels[3]*2*2*2 + 1 + 3, out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=4)
        )
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
        x2_down_out = self.level2_down(x2_down_in)
        x3_down_in = self.down_2to3(x2_down_out)
        x3_down_out = self.level3_down(x3_down_in)
        x4_down_in = self.down_3to4(x3_down_out)
        x4_down_out = self.level4_down(x4_down_in)

        img_feature4 = x4_down_out.view(x4_down_out.size(0), -1)

        sex = torch.reshape(sex, (-1, 3))
        age = torch.reshape(age, (-1, 1))
        classify_in = torch.cat((img_feature4, sex, age), dim=1)
        classify_out = self.softmax(self.linear(classify_in))

        return classify_out
    
    def DownSampling(self, in_channels, out_channels, kernel_size=(4, 4, 4), stride=(4, 4, 4)):
        conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        relu = nn.ReLU()
        sequence = nn.Sequential()
        sequence.add_module("DownConv3D", conv)
        sequence.add_module("relu", relu)
        return sequence
    
    def ConvBlock(self, in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)):
        conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        bn = nn.BatchNorm3d(num_features=out_channels)
        relu = nn.ReLU()
        sequence = nn.Sequential()
        sequence.add_module("Conv3D", conv)
        sequence.add_module("Batch Normalization", bn)
        sequence.add_module("relu", relu)
        return sequence

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)):
        super(ResBlock, self).__init__()
        self.ConBlock  = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm3d(num_features=out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return x + self.ConBlock(x)

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
    
    



class AttentionUNet(nn.Module):
    def __init__(self, img_channel=2, output_channel=1):
        super(AttentionUNet, self).__init__()

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
        self.Up_conv3 = self.ConvBlock(2 * channels[2], channels[1])

        self.level3_Up = self.UpSampling(channels[2], channels[1])
        self.Att2 = AttentionUNetblock(F_g=channels[1], F_l=channels[1], F_int=channels[1])
        self.Up_conv2 = self.ConvBlock(2 * channels[1], channels[0])

        self.level2_Up = self.UpSampling(channels[1], channels[0])
        self.Att1 = AttentionUNetblock(F_g=channels[0], F_l=channels[0], F_int=channels[0])
        self.Up_conv1 = self.ConvBlock(2 * channels[0], output_channel, kernel_size=(1, 1, 1), padding=(0, 0, 0))

        self.seg = torch.nn.Sigmoid()

        self.prelu = nn.PReLU()
        self.glo_avr = nn.AdaptiveAvgPool3d(1)
        # self.softmax = nn.Softmax()
        self.linear = nn.Sequential(
            nn.Linear(in_features=(self.channels[0] + self.channels[1] + self.channels[2] + self.channels[3]) + 1 + 3, out_features=32),
            nn.Linear(in_features=32, out_features=4)
        )
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

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

        img_feature1 = torch.reshape(self.glo_avr(level1_down_out), (-1, self.channels[0]))
        img_feature2 = torch.reshape(self.glo_avr(level2_down_out), (-1, self.channels[1]))
        img_feature3 = torch.reshape(self.glo_avr(level3_down_out), (-1, self.channels[2]))
        img_feature4 = torch.reshape(self.glo_avr(level4_down_out), (-1, self.channels[3]))
        sex = torch.reshape(sex, (-1, 3))
        age = torch.reshape(age, (-1, 1))
        classify_in = torch.cat((img_feature1, img_feature2, img_feature3, img_feature4, sex, age), dim=1)
        classify_out = self.softmax(self.linear(classify_in))

        return classify_out

    def ConvBlock(self, in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)):
        conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        # bn = nn.BatchNorm3d(num_features=out_channels)
        relu = nn.ReLU()
        sequence = nn.Sequential()
        sequence.add_module("Conv3D", conv)
        # sequence.add_module("Batch Normalization", bn)
        sequence.add_module("relu", relu)
        return sequence

    def DownSampling(self, in_channels, out_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2)):
        conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        relu = nn.ReLU()
        sequence = nn.Sequential()
        sequence.add_module("DownConv3D", conv)
        sequence.add_module("relu", relu)
        return sequence

    def UpSampling(self, in_channels, out_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2)):
        conv = nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                  stride=stride)
        relu = nn.ReLU()
        sequence = nn.Sequential()
        sequence.add_module("DownConv3D", conv)
        sequence.add_module("relu", relu)
        return sequence