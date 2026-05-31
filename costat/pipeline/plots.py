"""Benchmark figures.

All plotting lives here so the orchestrator stays focused on the numbers. The
non-interactive Agg backend is selected explicitly so the code runs headless.
"""

import os
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from costat.utils.io_utils import ensure_dir

UNIFORM_COLOR: str = "#b0b0b0"
DIST_AWARE_COLOR: str = "#1f77b4"
BASELINE_COLOR: str = "#d62728"


def plot_accuracy_vs_bitwidth(
    display_name: str,
    bit_widths: List[int],
    uniform_accuracy: List[float],
    dist_aware_accuracy: List[float],
    fp32_accuracy: float,
    plots_dir: str,
) -> str:
    """Grouped bars of uniform vs distribution-aware accuracy per bit-width."""
    ensure_dir(plots_dir)
    positions: np.ndarray = np.arange(len(bit_widths))
    bar_width: float = 0.35
    figure, axis = plt.subplots(figsize=(6.0, 4.0))
    axis.bar(positions - bar_width / 2, uniform_accuracy, bar_width,
             label="Uniform", color=UNIFORM_COLOR)
    axis.bar(positions + bar_width / 2, dist_aware_accuracy, bar_width,
             label="Distribution-aware", color=DIST_AWARE_COLOR)
    axis.axhline(fp32_accuracy, color=BASELINE_COLOR, linestyle="--",
                 label="FP32 baseline")
    axis.set_xticks(positions)
    axis.set_xticklabels(["INT" + str(bit) for bit in bit_widths])
    axis.set_ylabel("Test accuracy (%)")
    axis.set_title(display_name + " => weight quantisation")
    axis.legend(loc="lower right", fontsize=8)
    axis.grid(axis="y", linestyle=":", alpha=0.5)
    output_path: str = os.path.join(plots_dir, "accuracy_vs_bitwidth_" + display_name + ".png")
    figure.tight_layout()
    figure.savefig(output_path, dpi=130)
    plt.close(figure)
    return output_path


def plot_selected_distributions(
    display_name: str,
    layer_names: List[str],
    selected_names: List[str],
    plots_dir: str,
) -> str:
    """Show which family the controller picked for each weight layer."""
    ensure_dir(plots_dir)
    unique_families: List[str] = sorted(set(selected_names))
    family_to_y: Dict[str, int] = {name: index for index, name in enumerate(unique_families)}
    y_positions: List[int] = [family_to_y[name] for name in selected_names]
    figure, axis = plt.subplots(figsize=(7.0, 4.0))
    axis.scatter(range(len(layer_names)), y_positions, color=DIST_AWARE_COLOR, s=60)
    axis.set_yticks(range(len(unique_families)))
    axis.set_yticklabels(unique_families)
    axis.set_xticks(range(len(layer_names)))
    axis.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=7)
    axis.set_title(display_name + " => selected distribution per layer")
    axis.grid(axis="y", linestyle=":", alpha=0.5)
    output_path: str = os.path.join(plots_dir, "selected_distributions_" + display_name + ".png")
    figure.tight_layout()
    figure.savefig(output_path, dpi=130)
    plt.close(figure)
    return output_path


def plot_pwl_error(
    display_name: str,
    layer_names: List[str],
    uniform_error: List[float],
    dist_aware_error: List[float],
    plots_dir: str,
) -> str:
    """Per-tanh density-weighted approximation error: uniform vs dist-aware."""
    ensure_dir(plots_dir)
    positions: np.ndarray = np.arange(len(layer_names))
    bar_width: float = 0.35
    figure, axis = plt.subplots(figsize=(7.0, 4.0))
    axis.bar(positions - bar_width / 2, uniform_error, bar_width,
             label="Uniform", color=UNIFORM_COLOR)
    axis.bar(positions + bar_width / 2, dist_aware_error, bar_width,
             label="Distribution-aware", color=DIST_AWARE_COLOR)
    axis.set_xticks(positions)
    axis.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=7)
    axis.set_ylabel("Density-weighted mean abs error")
    axis.set_title(display_name + " => PWL tanh approximation error")
    axis.legend(loc="upper right", fontsize=8)
    axis.grid(axis="y", linestyle=":", alpha=0.5)
    output_path: str = os.path.join(plots_dir, "pwl_error_" + display_name + ".png")
    figure.tight_layout()
    figure.savefig(output_path, dpi=130)
    plt.close(figure)
    return output_path


def plot_weight_fit(
    display_name: str,
    layer_name: str,
    weight_samples: np.ndarray,
    frozen_distribution: Any,
    selected_name: str,
    plots_dir: str,
) -> str:
    """Overlay the fitted density on the empirical histogram for one layer."""
    ensure_dir(plots_dir)
    figure, axis = plt.subplots(figsize=(6.0, 4.0))
    axis.hist(weight_samples, bins=120, density=True, color=UNIFORM_COLOR,
              alpha=0.7, label="Empirical weights")
    grid: np.ndarray = np.linspace(weight_samples.min(), weight_samples.max(), 400)
    axis.plot(grid, frozen_distribution.pdf(grid), color=DIST_AWARE_COLOR,
              linewidth=2.0, label="Fitted " + selected_name)
    axis.set_xlabel("Weight value")
    axis.set_ylabel("Density")
    axis.set_title(display_name + " => " + layer_name + " fit")
    axis.legend(loc="upper right", fontsize=8)
    safe_layer: str = layer_name.replace(".", "_")
    output_path: str = os.path.join(
        plots_dir, "weight_fit_" + display_name + "_" + safe_layer + ".png"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=130)
    plt.close(figure)
    return output_path
