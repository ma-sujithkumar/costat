"""Name-driven model factory.

The mapping from a builder name (as written in config/models.json) to the actual
callable lives in a dictionary, so constructing a model is a lookup rather than
a chain of if/elif branches.
"""

from typing import Any, Callable, Dict

import torch.nn as nn

from costat.models.lenet5 import build_lenet5
from costat.models.resnet_mini import build_resnet_mini
from costat.models.vgg_mini import build_vgg_mini

BuilderType = Callable[[int, int], nn.Module]

BUILDER_REGISTRY: Dict[str, BuilderType] = {
    "build_lenet5": build_lenet5,
    "build_vgg_mini": build_vgg_mini,
    "build_resnet_mini": build_resnet_mini,
}


def create_model(model_name: str, models_config: Dict[str, Any]) -> nn.Module:
    """Instantiate a model by its registry name.

    Args:
        model_name: key under the "models" object in models.json.
        models_config: the parsed models.json content.

    Returns:
        An initialised nn.Module ready for training.
    """
    model_table: Dict[str, Any] = models_config["models"]
    if model_name not in model_table:
        raise KeyError("Unknown model name: " + model_name)
    spec: Dict[str, Any] = model_table[model_name]
    builder_name: str = spec["builder"]
    if builder_name not in BUILDER_REGISTRY:
        raise KeyError("No builder registered for: " + builder_name)
    builder: BuilderType = BUILDER_REGISTRY[builder_name]
    return builder(spec["input_channels"], spec["num_classes"])
