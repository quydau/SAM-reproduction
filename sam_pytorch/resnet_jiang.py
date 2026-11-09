"""ResNet-32 theo đúng code của Jiang et al. 2020 (MentorMix).

Nguồn: google-research/google-research, mentormix/code/resnet_model.py
(build_model, _residual, _batch_norm, _conv, _fully_connected) và
cifar_train_mentormix.py (num_residual_units=5, use_bottleneck=False,
relu_leakiness=0.1).

Khác ResNet-32 chuẩn của He et al. (resnet_cifar.py) ở chỗ mạng RỘNG hơn nhiều:

  init_conv (3x3, 3 -> 64)                                    ->  1 lớp
  stage 1: 5 block, 128 kênh, ảnh 32x32                      -> 10 lớp conv
  stage 2: 5 block, 256 kênh, ảnh 16x16 (stride 2)           -> 10 lớp conv
  stage 3: 5 block, 512 kênh, ảnh  8x8  (stride 2)           -> 10 lớp conv
  BN -> leaky ReLU -> global avg-pool -> fc (512 -> 10)      ->  1 lớp
  Tổng: 32 lớp có trọng số, ~29.4M tham số.

Block là pre-activation (BN-ReLU trước conv), ReLU là leaky ReLU 0.1.
Skip connection khi đổi kênh: avg-pool (kernel = stride) rồi pad kênh bằng 0.
"""

import torch
from torch import nn
from torch.nn import functional as F

LEAKINESS = 0.1      # relu_leakiness=0.1
BN_MOMENTUM = 0.1    # TF moving average decay 0.9  <=>  PyTorch momentum 0.1
BN_EPS = 1e-3        # tf.nn.batch_normalization(..., 0.001)


class PoolPadSkipConnection(nn.Module):
    """Skip connection khi đổi số kênh: avg-pool rồi pad kênh bằng 0 hai bên.

    Code gốc: tf.nn.avg_pool(orig_x, stride, stride, 'VALID') rồi tf.pad kênh
    (out - in) // 2 mỗi bên.
    """

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.stride = stride
        self.pad_each_side = (out_channels - in_channels) // 2

    def forward(self, x):
        if self.stride > 1:
            x = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
        pad = self.pad_each_side
        # F.pad đọc từ chiều cuối về đầu: (W_trái, W_phải, H_trên, H_dưới, C_trước, C_sau)
        return F.pad(x, (0, 0, 0, 0, pad, pad), "constant", 0)


class PreActBlock(nn.Module):
    """Block ``_residual`` của Jiang (pre-activation, 2 conv 3x3).

    activate_before_residual=True (chỉ block đầu của stage 1):
        x = BN(LeakyReLU(x)); skip lấy x SAU kích hoạt   (đúng thứ tự code gốc)
    activate_before_residual=False (mọi block còn lại):
        skip lấy x TRƯỚC kích hoạt; x = LeakyReLU(BN(x))
    Sau đó: conv1 -> BN -> LeakyReLU -> conv2, cộng skip (không ReLU sau phép cộng).
    """

    def __init__(self, in_channels, out_channels, stride, activate_before_residual=False):
        super().__init__()
        self.activate_before_residual = activate_before_residual
        self.bn1 = nn.BatchNorm2d(in_channels, eps=BN_EPS, momentum=BN_MOMENTUM)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=BN_EPS, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        if in_channels == out_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = PoolPadSkipConnection(in_channels, out_channels, stride)

    def forward(self, x):
        if self.activate_before_residual:
            x = F.leaky_relu(x, LEAKINESS)
            x = self.bn1(x)
            skip = self.skip_connection(x)
        else:
            skip = self.skip_connection(x)
            x = self.bn1(x)
            x = F.leaky_relu(x, LEAKINESS)

        out = self.conv1(x)
        out = self.bn2(out)
        out = F.leaky_relu(out, LEAKINESS)
        out = self.conv2(out)

        return out + skip


class ResNet32Jiang(nn.Module):
    """ResNet-32 rộng của Jiang et al. 2020 (kênh 64 -> 128/256/512)."""

    def __init__(self, num_classes=10):
        super().__init__()

        # ----- Lớp đầu: (B, 3, 32, 32) -> (B, 64, 32, 32) -----
        self.init_conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # ----- Stage 1: 128 kênh, ảnh 32x32 -----
        self.stage1 = nn.Sequential(
            PreActBlock(64, 128, stride=1, activate_before_residual=True),
            PreActBlock(128, 128, stride=1),
            PreActBlock(128, 128, stride=1),
            PreActBlock(128, 128, stride=1),
            PreActBlock(128, 128, stride=1),
        )

        # ----- Stage 2: 256 kênh, ảnh 16x16 -----
        self.stage2 = nn.Sequential(
            PreActBlock(128, 256, stride=2),
            PreActBlock(256, 256, stride=1),
            PreActBlock(256, 256, stride=1),
            PreActBlock(256, 256, stride=1),
            PreActBlock(256, 256, stride=1),
        )

        # ----- Stage 3: 512 kênh, ảnh 8x8 -----
        self.stage3 = nn.Sequential(
            PreActBlock(256, 512, stride=2),
            PreActBlock(512, 512, stride=1),
            PreActBlock(512, 512, stride=1),
            PreActBlock(512, 512, stride=1),
            PreActBlock(512, 512, stride=1),
        )

        # ----- unit_last + logit -----
        self.final_bn = nn.BatchNorm2d(512, eps=BN_EPS, momentum=BN_MOMENTUM)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

        # ----- Khởi tạo (giống code gốc) -----
        # conv: normal(0, sqrt(2 / (k*k*out))) = Kaiming normal, fan_out.
        # fc:   uniform_unit_scaling(factor=1.0) = U(-sqrt(3/fan_in), +sqrt(3/fan_in)); bias 0.
        # BN:   gamma = 1, beta = 0.
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        fc_scale = (3.0 / 512) ** 0.5
        nn.init.uniform_(self.fc.weight, -fc_scale, fc_scale)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.init_conv(x)   # (B,  64, 32, 32)
        x = self.stage1(x)      # (B, 128, 32, 32)
        x = self.stage2(x)      # (B, 256, 16, 16)
        x = self.stage3(x)      # (B, 512,  8,  8)

        x = self.final_bn(x)
        x = F.leaky_relu(x, LEAKINESS)
        x = self.avgpool(x)     # (B, 512, 1, 1)
        x = torch.flatten(x, 1) # (B, 512)
        x = self.fc(x)          # (B, num_classes)
        return x


if __name__ == "__main__":
    model = ResNet32Jiang()
    print("Số tham số:", sum(p.numel() for p in model.parameters()))
    print("Output:", model(torch.randn(4, 3, 32, 32)).shape)
