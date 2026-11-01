"""Models used by the CIFAR-10 and fine-tuning experiments."""

from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from .wideresnet import WideResNet


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
    if model_name == "wrn16x8":
        return WideResNet(depth=16, num_classes=num_classes, widen_factor=8)
    if model_name == "efficientnet_b0":
        return EfficientNetB0(num_classes, pretrained_weights)
    raise ValueError(f"Unsupported model: {model_name}")
