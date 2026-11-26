"""WRN-16-8 and EfficientNet-B0 models used by the experiments."""

from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from .resnet_cifar import ResNetCIFAR
from .wideresnet import WideResNet


class EfficientNetB0(nn.Module):
    """EfficientNet-B0 with torchvision or timm pretrained weights."""

    def __init__(self, num_classes, pretrained_weights="IMAGENET1K_V1"):
        super().__init__()
        if pretrained_weights in (None, "IMAGENET1K_V1", "DEFAULT"):
            weights = (None if pretrained_weights is None else
                       EfficientNet_B0_Weights[pretrained_weights])
            self.network = efficientnet_b0(weights=weights)
            in_features = self.network.classifier[1].in_features
            self.network.classifier[1] = nn.Linear(in_features, num_classes)
        else:
            import timm

            self.network = timm.create_model(
                pretrained_weights, pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.network(x)


def create_model(model_name, num_classes, pretrained_weights="IMAGENET1K_V1"):
    """Build the model named in an experiment config."""
    if model_name == "wrn16x8":
        return WideResNet(depth=16, num_classes=num_classes, widen_factor=8)
    if model_name == "resnet32":
        return ResNetCIFAR(depth=32, num_classes=num_classes)
    if model_name == "efficientnet_b0":
        return EfficientNetB0(num_classes, pretrained_weights)
    raise ValueError(f"Unsupported model: {model_name}")
