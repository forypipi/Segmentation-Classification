import torch
from torch import nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, dilation=1, downsample=None, drop_rate=0):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv3 = nn.Conv3d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):

        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """定义BasicBlock残差块类
        
        参数：
            in_channels (int): 输入的Feature Map的通道数
            out_channels (int): 第一个卷积层输出的Feature Map的通道数
            stride (int, optional): 第一个卷积层的步长
            downsample (nn.Sequential, optional): 旁路下采样的操作
        注意：
            残差块输出的Feature Map的通道数是out_channels*expansion
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], input_size=(128, 128, 128), no_cuda = False, num_classes=4):
        self.in_channels = 32
        self.no_cuda = no_cuda
        self.num_classes = num_classes
        self.expansion = block.expansion
        super(ResNet18, self).__init__()
        # self.conv1 = nn.Conv3d(1, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = nn.Conv3d(1, self.in_channels, kernel_size=3, stride=2, padding=1, bias=False)

        self.bn1 = nn.BatchNorm3d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, out_channel=self.in_channels, blocks=layers[0])                         # ResBlock change C to 64*4=256 by 1*1*1 kernel
        self.layer2 = self._make_layer(block, out_channel=self.in_channels * 2, blocks=layers[1], stride=2)              # downsampled by conv3d
        self.layer3 = self._make_layer(block, out_channel=self.in_channels * 2, blocks=layers[2], stride=2)
        # self.layer4 = self._make_layer(block, out_channel=512, blocks=layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((4, 4, 4))
        self.fc = nn.Linear(self.in_channels*self.expansion*4*4*4, self.num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
        #     elif isinstance(m, nn.BatchNorm3d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer(self, block, out_channel, blocks, stride=1, dilation=1):
        """
        :param block: ResBlock
        :param planes: up layer channels
        :param blocks: block number in this layer
        :param stride: 
        :param dilation:
        :return: nn.Sequential, ResBlock of this layer
        """
        downsample = None
        if stride != 1 or self.in_channels != out_channel * block.expansion:        # block.expansion = 1
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channel * block.expansion, kernel_size=stride, stride=stride, bias=False), 
                nn.BatchNorm3d(out_channel * block.expansion),
                )

        layers = []
        layers.append(block(self.in_channels, out_channel, stride=stride, downsample=downsample))     # downsample channel = planes * 1
        self.in_channels = out_channel * block.expansion         # prepare for next resblock
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    test = torch.rand(1, 2, 128, 128, 128).cuda()
    model = ResNet50().cuda()
    # print(model)
    print(model(test).shape)
