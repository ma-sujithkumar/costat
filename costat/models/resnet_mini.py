"""A small ResNet with basic residual blocks and tanh activations.

The residual connections give the quantiser deeper, batch-normalised weight
distributions to fit, which is where the heavier-tailed candidate families in
the controller tend to earn their keep.
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Two 3x3 convolutions with a tanh-activated identity (or projected) skip."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1: nn.Conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(out_channels)
        self.act1: nn.Tanh = nn.Tanh()
        self.conv2: nn.Conv2d = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2: nn.BatchNorm2d = nn.BatchNorm2d(out_channels)
        self.act2: nn.Tanh = nn.Tanh()

        self.shortcut: nn.Module = nn.Identity()
        needs_projection: bool = stride != 1 or in_channels != out_channels
        if needs_projection:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        residual: torch.Tensor = self.shortcut(input_batch)
        out: torch.Tensor = self.act1(self.bn1(self.conv1(input_batch)))
        out = self.bn2(self.conv2(out))
        return self.act2(out + residual)


class ResNetMini(nn.Module):
    """Stem plus two residual stages (16 then 32 channels) and a linear head."""

    def __init__(self, input_channels: int, num_classes: int) -> None:
        super().__init__()
        self.stem: nn.Sequential = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        )
        self.stage1: BasicBlock = BasicBlock(16, 16, stride=1)
        self.stage2: BasicBlock = BasicBlock(16, 32, stride=2)
        self.pool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(output_size=1)
        self.head: nn.Linear = nn.Linear(32, num_classes)

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.stem(input_batch)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.pool(out)
        flattened: torch.Tensor = torch.flatten(out, start_dim=1)
        return self.head(flattened)


def build_resnet_mini(input_channels: int, num_classes: int) -> nn.Module:
    """Factory entry point referenced from config/models.json."""
    return ResNetMini(input_channels=input_channels, num_classes=num_classes)
