import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class _DenseLayer_3d(nn.Sequential):
    """DenseNet layer(bottleNeck)"""
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):    # bn=4, grow=32, drop=0.5
        super().__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)) # k to 4k
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)) # 4k to k
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)    # drop half during training
        return torch.cat([x, new_features], dim=1)

class _DenseBlock_3D(nn.Sequential):
    """DenseBlock"""
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):    # bn=4, grow=32, drop=0.5
        super(_DenseBlock_3D, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer_3d(num_input_features+i*growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module(f"denselayer {i+1}", layer)

class _Transition_3d(nn.Sequential):
    """1*1*1 conv + 2*2*2 avg pool"""
    def __init__(self, num_input_feature, num_output_features):
        super(_Transition_3d, self).__init__()
        self.add_module("norm", nn.BatchNorm3d(num_input_feature))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv3d(num_input_feature, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module("pool", nn.AvgPool3d(2, stride=2))
    
class DenseNet121_3d(nn.Module):
    """DenseNet121-BC model"""
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, compression_rate=0.5, drop_rate=0, num_classes=4):
        """
        :param growth_rate: (int) number of filters used in DenseLayer
        :param block_config: (list of 4 ints) number of layers in each DenseBlock
        :param num_init_features: (int) number of filters in the first Conv3d
        :param bn_size: (int) the factor using in the bottleneck layer
        :param compression_rate: (float) the compression rate used in Transition Layer
        :param drop_rate: (float) the drop rate after each DenseLayer
        :param num_classes: (int) number of classes for classification
        """
        super(DenseNet121_3d, self).__init__()
        # first Conv3d
        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv3d(2, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ("norm0", nn.BatchNorm3d(num_init_features)),
            ("relu0", nn.ReLU(inplace=True)),
            ("pool0", nn.MaxPool3d(3, stride=2, padding=1))
        ]))

        # DenseBlock
        num_features = num_init_features    # 64
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock_3D(num_layers, num_features, bn_size, growth_rate, drop_rate)   # bn=4, grow=32, drop=0.5
            self.features.add_module(f"denseblock {i+1}", block)
            num_features += num_layers * growth_rate    # 6*32=192 → 96+12*32=480 → 240+24*32=1008 → 504+16*32=1016
            if i != len(block_config) - 1:
                transition = _Transition_3d(num_features, int(num_features * compression_rate))     # comppress channel to half
                self.features.add_module(f"transition {i+1}", transition)
                num_features = int(num_features * compression_rate)     # 192 → 96, 480 → 240, 1008 → 504, 512 → 508

        # final BN+ReLU
        self.features.add_module("norm5", nn.BatchNorm3d(num_features))
        self.features.add_module("relu5", nn.ReLU(inplace=True))

        # classification layer
        self.classifier = nn.Linear(num_features + 1+ 3, num_classes)

        # params initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.bias, 0)
        #         nn.init.constant_(m.weight, 1)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x, sex, age):
        features = self.features(x)
        out = F.avg_pool3d(features, 4, stride=1).view(features.size(0), -1)
        sex = torch.reshape(sex, (-1, 3))
        age = torch.reshape(age, (-1, 1))
        out = torch.cat((out, sex, age), dim=1)
        out = self.classifier(out)
        return out

