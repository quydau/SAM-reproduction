"""Models used by the CIFAR-10 and fine-tuning experiments."""

from math import sqrt

from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class WideResNetBlock(nn.Module):
    """Pre-activation residual block used by the original CIFAR WRN."""

    def __init__(self, in_channels, channels, stride=1, activate_before_residual=False):
        super().__init__()
        self.activate_before_residual = activate_before_residual
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.stride = stride
        self.in_channels = in_channels
        self.channels = channels

    def forward(self, x):
        activated = self.relu1(self.bn1(x))
        residual = activated if self.activate_before_residual else x
        out = self.conv1(activated)
        out = self.conv2(self.relu2(self.bn2(out)))

        if self.stride != 1:
            residual = nn.functional.avg_pool2d(residual, self.stride, self.stride)
        if self.channels > self.in_channels:
            padding = self.channels - self.in_channels
            residual = nn.functional.pad(residual, (0, 0, 0, 0, 0, padding))
        return out + residual


class WideResNetGroup(nn.Sequential):
    def __init__(self, in_channels, channels, blocks, stride=1, activate_before_residual=False):
        layers = [
            WideResNetBlock(
                in_channels,
                channels,
                stride=stride,
                activate_before_residual=activate_before_residual,
            )
        ]
        layers.extend(WideResNetBlock(channels, channels) for _ in range(blocks - 1))
        super().__init__(*layers)


class WideResNet28x10(nn.Module):
    """WRN-28-10 for 32x32 CIFAR images."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.group1 = WideResNetGroup(16, 160, 4, activate_before_residual=True)
        self.group2 = WideResNetGroup(160, 320, 4, stride=2)
        self.group3 = WideResNetGroup(320, 640, 4, stride=2)
        self.bn = nn.BatchNorm2d(640)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(640, num_classes)
        self._initialize_weights(num_classes)

    def _initialize_weights(self, num_classes):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        bound = 1 / sqrt(num_classes)
        nn.init.uniform_(self.classifier.weight, -bound, bound)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.relu(self.bn(x))
        x = nn.functional.avg_pool2d(x, x.shape[-2:]).flatten(1)
        return self.classifier(x)


class EfficientNetB0(nn.Module):
    """Torchvision EfficientNet-B0 with a dataset-specific classifier."""

    def __init__(self, num_classes, pretrained_weights="IMAGENET1K_V1"):
        super().__init__()
        weights = None if pretrained_weights is None else EfficientNet_B0_Weights[pretrained_weights]
        self.network = efficientnet_b0(weights=weights)
        in_features = self.network.classifier[1].in_features
        self.network.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.network(x)


def create_model(model_name, num_classes, pretrained_weights="IMAGENET1K_V1"):
    """Build the model named in an experiment config."""
    if model_name == "wrn28x10":
        return WideResNet28x10(num_classes)
    if model_name == "efficientnet_b0":
        return EfficientNetB0(num_classes, pretrained_weights)
    raise ValueError(f"Unsupported model: {model_name}")
