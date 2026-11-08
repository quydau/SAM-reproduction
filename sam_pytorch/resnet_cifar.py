"""CIFAR ResNet-32 (He et al., 2016, Section 4.2), viết tường minh từng lớp.

Cấu trúc (32 lớp có trọng số):
  conv1 (3x3, 16 kênh)                                   ->  1 lớp
  layer1: 5 BasicBlock, 16 kênh, ảnh 32x32               -> 10 lớp conv
  layer2: 5 BasicBlock, 32 kênh, ảnh 16x16 (stride 2)    -> 10 lớp conv
  layer3: 5 BasicBlock, 64 kênh, ảnh  8x8  (stride 2)    -> 10 lớp conv
  global avg-pool -> fc (64 -> num_classes)              ->  1 lớp
  Tổng: 1 + 10 + 10 + 10 + 1 = 32

Block là post-activation: conv-BN-ReLU-conv-BN, cộng skip connection, rồi ReLU.
Skip connection khi đổi kích thước dùng "option A" (không có tham số):
lấy mẫu cách 1 pixel và pad thêm kênh bằng 0 -> ~0.46M tham số.
"""

import torch
from torch import nn
from torch.nn import functional as F


class OptionASkipConnection(nn.Module):
    """Skip connection option A: giảm ảnh một nửa + pad kênh bằng 0.

    Ví dụ vào (B, 16, 32, 32), cần ra 32 kênh:
      x[:, :, ::2, ::2]  -> (B, 16, 16, 16)  bỏ bớt pixel (giống stride 2)
      pad 8 kênh trước + 8 kênh sau -> (B, 32, 16, 16)
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pad_front = (out_channels - in_channels) // 2
        self.pad_back = (out_channels - in_channels) - self.pad_front

    def forward(self, x):
        x = x[:, :, ::2, ::2]
        # F.pad đọc từ chiều cuối về đầu: (W_trái, W_phải, H_trên, H_dưới, C_trước, C_sau)
        return F.pad(x, (0, 0, 0, 0, self.pad_front, self.pad_back), "constant", 0)


class BasicBlock(nn.Module):
    """out = ReLU( F(x) + skip_connection(x) ),  F(x) = conv-BN-ReLU-conv-BN."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride == 1 and in_channels == out_channels:
            self.skip_connection = nn.Identity()  # giữ nguyên x
        else:
            self.skip_connection = OptionASkipConnection(in_channels, out_channels)  # làm x khớp với F(x)

    def forward(self, x):
        skip = self.skip_connection(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + skip  # phép cộng residual (không in-place để không ghi đè skip)
        out = self.relu(out)
        return out


class ResNetCIFAR(nn.Module):
    """ResNet-32 cho CIFAR. Tham số `depth` giữ lại để khớp models.py, chỉ nhận 32."""

    def __init__(self, depth=32, num_classes=10):
        super().__init__()
        if depth != 32:
            raise ValueError(f"Chỉ hỗ trợ ResNet-32, nhận depth={depth}")

        # ----- Lớp đầu: (B, 3, 32, 32) -> (B, 16, 32, 32) -----
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # ----- Stage 1: 16 kênh, ảnh 32x32, không giảm kích thước -----
        self.layer1 = nn.Sequential(
            BasicBlock(16, 16, stride=1),
            BasicBlock(16, 16, stride=1),
            BasicBlock(16, 16, stride=1),
            BasicBlock(16, 16, stride=1),
            BasicBlock(16, 16, stride=1),
        )

        # ----- Stage 2: 32 kênh, ảnh 16x16; block đầu tăng kênh + stride 2 -----
        self.layer2 = nn.Sequential(
            BasicBlock(16, 32, stride=2),
            BasicBlock(32, 32, stride=1),
            BasicBlock(32, 32, stride=1),
            BasicBlock(32, 32, stride=1),
            BasicBlock(32, 32, stride=1),
        )

        # ----- Stage 3: 64 kênh, ảnh 8x8; block đầu tăng kênh + stride 2 -----
        self.layer3 = nn.Sequential(
            BasicBlock(32, 64, stride=2),
            BasicBlock(64, 64, stride=1),
            BasicBlock(64, 64, stride=1),
            BasicBlock(64, 64, stride=1),
            BasicBlock(64, 64, stride=1),
        )

        # ----- Phân loại -----
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # (B, 64, 8, 8) -> (B, 64, 1, 1)
        self.fc = nn.Linear(64, num_classes)

        # ----- Khởi tạo trọng số (giống code gốc SAM, sam_jax/models/utils.py) -----
        # conv: variance_scaling(2.0, 'fan_out', 'normal') = Kaiming normal, fan_out.
        # fc:   U(-s, s) với s = 1 / sqrt(num_classes); bias = 0.
        # BN:   scale = 1, bias = 0.
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        fc_scale = 1.0 / num_classes ** 0.5
        nn.init.uniform_(self.fc.weight, -fc_scale, fc_scale)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.conv1(x)       # (B, 16, 32, 32)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)      # (B, 16, 32, 32)
        x = self.layer2(x)      # (B, 32, 16, 16)
        x = self.layer3(x)      # (B, 64,  8,  8)

        x = self.avgpool(x)     # (B, 64,  1,  1)
        x = torch.flatten(x, 1) # (B, 64)
        x = self.fc(x)          # (B, num_classes)
        return x


if __name__ == "__main__":
    model = ResNetCIFAR(depth=32)
    print("Số tham số:", sum(p.numel() for p in model.parameters()))
    print("Output:", model(torch.randn(4, 3, 32, 32)).shape)
