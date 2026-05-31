"""A compact VGG-style network: stacked 3x3 convolutions with tanh and pooling.

Small enough to train on CPU in a few minutes while still giving the quantiser
several convolutional and fully connected layers with distinct weight shapes.
"""

import torch
import torch.nn as nn


class VGGMini(nn.Module):
    """Two VGG blocks (16 and 32 channels) followed by a small classifier."""

    def __init__(self, input_channels: int, num_classes: int) -> None:
        super().__init__()
        self.features: nn.Sequential = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier: nn.Sequential = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
            nn.Tanh(),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        feature_maps: torch.Tensor = self.features(input_batch)
        flattened: torch.Tensor = torch.flatten(feature_maps, start_dim=1)
        return self.classifier(flattened)


def build_vgg_mini(input_channels: int, num_classes: int) -> nn.Module:
    """Factory entry point referenced from config/models.json."""
    return VGGMini(input_channels=input_channels, num_classes=num_classes)
