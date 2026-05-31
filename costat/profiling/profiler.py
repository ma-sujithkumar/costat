"""Profile a trained model: collect every weight tensor and the pre-activation
values that flow into each tanh.

Weights are read straight from the parameters. Pre-activations are gathered with
forward hooks on the tanh modules over a handful of calibration batches, exactly
the z = Wx + b signal the paper places its activation breakpoints against.
"""

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle

from costat.utils.logging_utils import get_logger

# Layer types whose weight tensors get quantised.
QUANTISABLE_TYPES = (nn.Conv2d, nn.Linear)


class LayerProfile:
    """Flattened sample values for one named layer (weights or pre-activations)."""

    def __init__(self, layer_name: str, samples: np.ndarray) -> None:
        self.layer_name: str = layer_name
        self.samples: np.ndarray = samples


class ModelProfiler:
    """Collects weight and pre-activation samples for a single model."""

    def __init__(self, model: nn.Module, device: str) -> None:
        self.model: nn.Module = model
        self.device: str = device
        self.logger = get_logger()

    def collect_weights(self) -> List[LayerProfile]:
        """Return one profile per quantisable layer, holding its flat weights."""
        weight_profiles: List[LayerProfile] = []
        for module_name, module in self.model.named_modules():
            if isinstance(module, QUANTISABLE_TYPES):
                weight_values: np.ndarray = (
                    module.weight.detach().cpu().numpy().reshape(-1)
                )
                weight_profiles.append(LayerProfile(module_name, weight_values))
                self.logger.debug(
                    "weights => layer=%s count=%d", module_name, weight_values.size
                )
        return weight_profiles

    def collect_preactivations(
        self, data_loader: DataLoader, num_batches: int
    ) -> List[LayerProfile]:
        """Hook every tanh and record its incoming pre-activation values.

        Args:
            data_loader: source of calibration inputs (the test loader is fine).
            num_batches: how many batches to accumulate before stopping.

        Returns:
            One LayerProfile per tanh module, holding the sampled inputs.
        """
        captured: Dict[str, List[np.ndarray]] = {}
        handles: List[RemovableHandle] = []

        def make_hook(layer_name: str):
            def hook(module: nn.Module, inputs, output) -> None:
                preactivation: np.ndarray = inputs[0].detach().cpu().numpy().reshape(-1)
                captured.setdefault(layer_name, []).append(preactivation)

            return hook

        for module_name, module in self.model.named_modules():
            if isinstance(module, nn.Tanh):
                captured[module_name] = []
                handles.append(module.register_forward_hook(make_hook(module_name)))

        self.model.eval()
        with torch.no_grad():
            for batch_index, (images, _labels) in enumerate(data_loader):
                if batch_index >= num_batches:
                    break
                self.model(images.to(self.device))

        for handle in handles:
            handle.remove()

        preactivation_profiles: List[LayerProfile] = []
        for layer_name, chunks in captured.items():
            if not chunks:
                continue
            stacked: np.ndarray = np.concatenate(chunks)
            preactivation_profiles.append(LayerProfile(layer_name, stacked))
            self.logger.debug(
                "preact => layer=%s count=%d", layer_name, stacked.size
            )
        return preactivation_profiles
