"""Evaluation helpers: measure accuracy and apply the two quantisation stages to
a copy of a trained model.

Quantisation here is post-training: the model is cloned, its weights (or tanh
modules) are replaced with their quantised counterparts, and accuracy is measured.
This keeps the original baseline untouched between configurations.
"""

import copy
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from costat.controller.controller import LayerFit
from costat.profiling.profiler import QUANTISABLE_TYPES
from costat.quantization.pwl_activation import (
    PWLApproximation,
    PWLTanh,
)
from costat.quantization.weight_quantizer import (
    DistributionAwareQuantizer,
    UniformQuantizer,
)


def evaluate_accuracy(model: nn.Module, test_loader: DataLoader, device: str) -> float:
    """Return top-1 test accuracy as a percentage."""
    model.to(device)
    model.eval()
    correct: int = 0
    total: int = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions: torch.Tensor = model(images).argmax(dim=1)
            correct += int((predictions == labels.to(device)).sum().item())
            total += labels.size(0)
    return 100.0 * correct / max(total, 1)


def apply_uniform_weight_quantization(
    model: nn.Module, bit_width: int
) -> nn.Module:
    """Clone the model and uniformly quantise every Conv/Linear weight."""
    quantizer: UniformQuantizer = UniformQuantizer()
    clone: nn.Module = copy.deepcopy(model)
    for module in clone.modules():
        if isinstance(module, QUANTISABLE_TYPES):
            original_weight: np.ndarray = module.weight.detach().cpu().numpy()
            quantized_flat: np.ndarray = quantizer.quantize(
                original_weight.reshape(-1), bit_width
            )
            _write_weight(module, quantized_flat, original_weight.shape)
    return clone


def apply_distribution_aware_weight_quantization(
    model: nn.Module,
    bit_width: int,
    weight_fits: Dict[str, LayerFit],
    quantizer: DistributionAwareQuantizer,
) -> nn.Module:
    """Clone the model and quantise each layer with its own fitted distribution."""
    clone: nn.Module = copy.deepcopy(model)
    for module_name, module in clone.named_modules():
        if isinstance(module, QUANTISABLE_TYPES) and module_name in weight_fits:
            frozen = weight_fits[module_name].selected.frozen_distribution
            original_weight: np.ndarray = module.weight.detach().cpu().numpy()
            quantized_flat: np.ndarray = quantizer.quantize(
                original_weight.reshape(-1), frozen, bit_width
            )
            _write_weight(module, quantized_flat, original_weight.shape)
    return clone


def apply_pwl_activations(
    model: nn.Module, approximations_by_layer: Dict[str, PWLApproximation]
) -> nn.Module:
    """Clone the model and replace each tanh with its PWL approximation."""
    clone: nn.Module = copy.deepcopy(model)
    for parent_name, parent_module in clone.named_modules():
        for child_name, child_module in parent_module.named_children():
            if not isinstance(child_module, nn.Tanh):
                continue
            full_name: str = (
                child_name if not parent_name else parent_name + "." + child_name
            )
            approximation = approximations_by_layer.get(full_name)
            if approximation is not None:
                setattr(parent_module, child_name, PWLTanh(approximation))
    return clone


def _write_weight(
    module: nn.Module, quantized_flat: np.ndarray, original_shape: tuple
) -> None:
    """Copy a flattened quantised weight array back into a module."""
    reshaped: np.ndarray = quantized_flat.reshape(original_shape)
    module.weight.data = torch.tensor(reshaped, dtype=module.weight.dtype)


def list_quantisable_layer_names(model: nn.Module) -> List[str]:
    """Return the module names whose weights participate in quantisation."""
    return [
        module_name
        for module_name, module in model.named_modules()
        if isinstance(module, QUANTISABLE_TYPES)
    ]
