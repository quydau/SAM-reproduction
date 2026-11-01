"""Wide ResNet for 32x32 CIFAR images.

Adapted from https://github.com/xternalz/WideResNet-pytorch/tree/master
"""

import torch
from torch import nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, 1, 1, bias=False)
        self.drop_rate = drop_rate
        self.equal_in_out = in_planes == out_planes
        self.conv_shortcut = (
            None if self.equal_in_out else nn.Conv2d(in_planes, out_planes, 1, stride, 0, bias=False)
        )

    def forward(self, x):
        activated = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(activated)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        # Identity shortcut taps the raw input; projection shortcut taps the activated input.
        shortcut = x if self.equal_in_out else self.conv_shortcut(activated)
        return torch.add(shortcut, out)


class NetworkBlock(nn.Sequential):
    def __init__(self, num_layers, in_planes, out_planes, stride, drop_rate=0.0):
        layers = [
            BasicBlock(in_planes if i == 0 else out_planes, out_planes, stride if i == 0 else 1, drop_rate)
            for i in range(num_layers)
        ]
        super().__init__(*layers)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, drop_rate=0.0):
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError(f"WideResNet depth must be 6n+4, got {depth}")
        n = (depth - 4) // 6
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.conv1 = nn.Conv2d(3, channels[0], 3, 1, 1, bias=False)
        self.block1 = NetworkBlock(n, channels[0], channels[1], 1, drop_rate)
        self.block2 = NetworkBlock(n, channels[1], channels[2], 2, drop_rate)
        self.block3 = NetworkBlock(n, channels[2], channels[3], 2, drop_rate)
        self.bn1 = nn.BatchNorm2d(channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(channels[3], num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block3(self.block2(self.block1(out)))
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)
