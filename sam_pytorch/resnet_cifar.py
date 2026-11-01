"""CIFAR ResNet (He et al., 2016, Section 4.2), used as ResNet-32 for §3.3.

Architecture for depth 6n+2 (ResNet-32: n=5):
  3x3 conv (16) -> 3 stages of n BasicBlocks with 16/32/64 channels
  (stride 2 at the start of stages 2 and 3) -> global avg-pool -> FC.

Blocks are post-activation (conv-BN-ReLU-conv-BN, add, ReLU) as in the
original paper. Shortcuts that change shape use "option A": identity with
stride-2 subsampling and zero-padded channels, so the model has no projection
parameters (~0.46M parameters for ResNet-32). Weights start from a random
Kaiming initialization; nothing is pretrained.
"""

from torch import nn
from torch.nn import functional as F


class _OptionAShortcut(nn.Module):
    """Subsample spatially and zero-pad channels (He et al. option A)."""

    def __init__(self, extra_channels: int):
        super().__init__()
        self.extra_channels = extra_channels

    def forward(self, x):
        x = x[:, :, ::2, ::2]
        pad = self.extra_channels // 2
        return F.pad(x, (0, 0, 0, 0, pad, self.extra_channels - pad))


class BasicBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = (nn.Identity() if stride == 1 and in_planes == planes
                         else _OptionAShortcut(planes - in_planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNetCIFAR(nn.Module):
    def __init__(self, depth: int = 32, num_classes: int = 10):
        super().__init__()
        if (depth - 2) % 6 != 0:
            raise ValueError(f"CIFAR ResNet depth must be 6n+2, got {depth}")
        n = (depth - 2) // 6
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        layers, in_planes = [], 16
        for stage, planes in enumerate((16, 32, 64)):
            for i in range(n):
                stride = 2 if stage > 0 and i == 0 else 1
                layers.append(BasicBlock(in_planes, planes, stride))
                in_planes = planes
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(64, num_classes)

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)
