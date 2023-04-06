import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, nchan):
        super().__init__()
        self.conv = nn.Conv3d(nchan, nchan, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(nchan)
        self.relu = nn.PReLU(nchan)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


def make_nConv(nchan, depth):
    layers = []
    for _ in range(depth):
        layers.append(Conv(nchan))
    return nn.Sequential(*layers)


class InputLayer(nn.Module):
    def __init__(self, inChans, outChans):
        super().__init__()
        self.conv = nn.Conv3d(inChans, outChans, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(outChans)
        self.relu = nn.PReLU(outChans)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class DownLayer(nn.Module):
    def __init__(self, inChans, nConvs):
        super().__init__()
        outChans = 2 * inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.down_bn = nn.BatchNorm3d(outChans)
        self.down_relu = nn.PReLU(outChans)
        self.ops = make_nConv(outChans, nConvs)

    def forward(self, x):
        down = self.down_relu(self.down_bn(self.down_conv(x)))
        out = self.ops(down)
        # out = torch.add(out, down)
        return out


class UpLayer(nn.Module):
    def __init__(self, inChans, nConvs, depthest_layer=False):
        super().__init__()
        if not depthest_layer:
            outChans = inChans // 2
        else:
            outChans = inChans
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.up_bn = nn.BatchNorm3d(outChans // 2)
        self.up_relu = nn.PReLU(outChans // 2)
        self.conv_relu = nn.PReLU(outChans)
        self.ops = make_nConv(outChans, nConvs)

    def forward(self, x, skip):
        # print(x.shape, skip.shape)
        up_sample = self.up_relu(self.up_bn(self.up_conv(x)))
        concat = torch.cat((up_sample, skip), 1)
        # print(up_sample.shape, skip.shape)
        out = self.ops(concat)
        # out = self.conv_relu(torch.add(out, concat))
        return out


class OutputLayer(nn.Module):
    def __init__(self, inChans):
        super().__init__()
        self.conv = nn.Conv3d(inChans, 1, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(1)
        self.relu = nn.PReLU(1)

    def forward(self, x):
        # convolve 32 down to 1 channels
        out = self.relu(self.bn(self.conv(x)))
        return out


class VNet(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, inChans=1, depth=3):
        super().__init__()
        self.in_tr = InputLayer(inChans=inChans, outChans=16)
        self.up_layer1 = UpLayer(inChans=64, nConvs=1)
        self.depth = depth
        if self.depth not in [2, 3, 4]:
            raise Exception("depth should in [2, 3]")
        self.down_layer2 = DownLayer(inChans=16, nConvs=2)
        self.up_layer2 = UpLayer(inChans=128, nConvs=2)
        if self.depth == 2:
            self.down_layer2 = DownLayer(inChans=16, nConvs=2)
            self.up_layer2 = UpLayer(inChans=64, nConvs=2, depthest_layer=True)
            self.down_layer3 = DownLayer(inChans=32, nConvs=2)
        if self.depth == 3:
            self.down_layer3 = DownLayer(inChans=32, nConvs=2)
            self.up_layer3 = UpLayer(inChans=128, nConvs=2, depthest_layer=True)
            self.down_layer4 = DownLayer(inChans=64, nConvs=2)
        if self.depth == 4:
            self.down_layer3 = DownLayer(inChans=32, nConvs=2)
            self.up_layer3 = UpLayer(inChans=256, nConvs=2)
            self.down_layer4 = DownLayer(inChans=64, nConvs=2)
            self.up_layer4 = UpLayer(inChans=256, nConvs=2, depthest_layer=True)
            self.down_layer5 = DownLayer(inChans=128, nConvs=2)
        self.outlayer1 = OutputLayer(inChans=32)
        self.outlayer2 = OutputLayer(inChans=64)
        self.outlayer3 = OutputLayer(inChans=128)
        self.outlayer4 = OutputLayer(inChans=256)


    def forward(self, x):
        down_layer1 = self.in_tr(x)
        down_layer2 = self.down_layer2(down_layer1)
        if self.depth >= 2:
            down_layer3 = self.down_layer3(down_layer2)
            if self.depth >= 3:
                down_layer4 = self.down_layer4(down_layer3)
                if self.depth == 4:
                    down_layer5 = self.down_layer5(down_layer4)

        if self.depth == 4:
            up_layer4 = self.up_layer4(down_layer5, down_layer4)
            up_layer3 = self.up_layer3(up_layer4, down_layer3)
            up_layer2 = self.up_layer2(up_layer3, down_layer2)
        if self.depth == 3:
            up_layer3 = self.up_layer3(down_layer4, down_layer3)
            up_layer2 = self.up_layer2(up_layer3, down_layer2)
        if self.depth == 2:
            up_layer2 = self.up_layer2(down_layer3, down_layer2)
        up_layer1 = self.up_layer1(up_layer2, down_layer1)

        if self.depth == 4:
            return self.outlayer1(up_layer1), self.outlayer2(up_layer2), self.outlayer3(up_layer3), self.outlayer4(up_layer4)
        if self.depth == 3:
            return self.outlayer1(up_layer1), self.outlayer2(up_layer2), self.outlayer3(up_layer3)
        if self.depth == 2:
            return self.outlayer1(up_layer1), self.outlayer2(up_layer2)



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device='cpu'
    model = VNet(inChans=2, depth=2).to(device)

    input = torch.randn(2, 2, 128, 128, 128)  # BCDHW
    input = input.to(device)
    out_list = model(input)
    print("output.shape:", [out.shape for out in out_list])  # 2, 1, 128, 128, 128