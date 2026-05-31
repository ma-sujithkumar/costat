"""Floating-point training - Stage 1 of the flow.

Nothing exotic here on purpose: a trained FP32 baseline is just the starting
point that the distribution-aware quantiser later compresses.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from costat.utils.logging_utils import get_logger


class Trainer:
    """Trains a model with Adam and reports progress per epoch."""

    def __init__(self, device: str, learning_rate: float) -> None:
        self.device: str = device
        self.learning_rate: float = learning_rate
        self.loss_function: nn.Module = nn.CrossEntropyLoss()
        self.logger = get_logger()

    def train(self, model: nn.Module, train_loader: DataLoader, epochs: int) -> nn.Module:
        """Train in place for the given number of epochs and return the model."""
        model.to(self.device)
        optimizer: torch.optim.Optimizer = torch.optim.Adam(
            model.parameters(), lr=self.learning_rate
        )
        for epoch_index in range(epochs):
            model.train()
            running_loss: float = 0.0
            batch_count: int = 0
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                logits: torch.Tensor = model(images)
                loss: torch.Tensor = self.loss_function(logits, labels)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item())
                batch_count += 1
            mean_loss: float = running_loss / max(batch_count, 1)
            self.logger.info(
                "train => epoch %d/%d mean_loss=%.4f", epoch_index + 1, epochs, mean_loss
            )
        return model
