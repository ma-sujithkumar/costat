"""Classic LeNet-5, the reference model from the CoStat paper.

The tanh activations are kept as explicit modules so the profiler can hook them
and the piecewise-linear approximation step can swap them out cleanly.
"""

import torch
import torch.nn as nn


class LeNet5(nn.Module):
    """LeNet-5 for 32x32 single-channel inputs (six tanh non-linearities)."""

    def __init__(self, input_channels: int, num_classes: int) -> None:
        super().__init__()
        self.features: nn.Sequential = nn.Sequential(
            nn.Conv2d(input_channels, 6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
        )
        self.classifier: nn.Sequential = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        feature_maps: torch.Tensor = self.features(input_batch)
        flattened: torch.Tensor = torch.flatten(feature_maps, start_dim=1)
        return self.classifier(flattened)


def build_lenet5(input_channels: int, num_classes: int) -> nn.Module:
    """Factory entry point referenced from config/models.json."""
    return LeNet5(input_channels=input_channels, num_classes=num_classes)
