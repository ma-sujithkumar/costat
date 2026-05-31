"""Piecewise-linear (PWL) approximation of tanh.

Both strategies build the same kind of object - a set of breakpoints and the
slope / intercept of each segment - and differ only in where the breakpoints sit:

* uniform breakpoints      - evenly spaced across [-clip, +clip] (baseline).
* distribution-aware ones  - placed at equal-probability quantiles of the layer's
  pre-activation distribution, so segments cluster where inputs are dense (the
  steep middle of tanh) instead of being wasted on the flat saturation tails.

PWLTanh wraps the result as an nn.Module so it can replace nn.Tanh in place for an
end-to-end accuracy check.
"""

from typing import Any

import numpy as np
import torch
import torch.nn as nn


class PWLApproximation:
    """A piecewise-linear curve fitted to tanh at a fixed set of breakpoints."""

    def __init__(self, breakpoints: np.ndarray) -> None:
        self.breakpoints: np.ndarray = breakpoints
        target_values: np.ndarray = np.tanh(breakpoints)
        segment_widths: np.ndarray = np.diff(breakpoints)
        self.slopes: np.ndarray = np.diff(target_values) / segment_widths
        self.intercepts: np.ndarray = target_values[:-1] - self.slopes * breakpoints[:-1]

    def evaluate(self, inputs: np.ndarray) -> np.ndarray:
        """Apply the PWL curve, saturating outside the breakpoint range."""
        clamped: np.ndarray = np.clip(inputs, self.breakpoints[0], self.breakpoints[-1])
        segment_index: np.ndarray = np.searchsorted(self.breakpoints, clamped, side="right") - 1
        segment_index = np.clip(segment_index, 0, self.slopes.size - 1)
        return self.slopes[segment_index] * clamped + self.intercepts[segment_index]

    def density_weighted_error(self, preactivation_samples: np.ndarray) -> float:
        """Mean |PWL(x) - tanh(x)| over the real pre-activations (density-weighted)."""
        approximation: np.ndarray = self.evaluate(preactivation_samples)
        exact: np.ndarray = np.tanh(preactivation_samples)
        return float(np.mean(np.abs(approximation - exact)))


def build_uniform_breakpoints(num_segments: int, clip: float) -> np.ndarray:
    """Evenly spaced breakpoints across [-clip, +clip]."""
    return np.linspace(-clip, clip, num_segments + 1)


def build_distribution_aware_breakpoints(
    frozen_distribution: Any, num_segments: int, margin: float, clip: float
) -> np.ndarray:
    """Equal-probability breakpoints from the pre-activation distribution.

    Falls back to uniform spacing if the fit yields non-finite or collapsed
    quantiles, so a pathological layer can never break the approximation.
    """
    probabilities: np.ndarray = np.linspace(margin, 1.0 - margin, num_segments + 1)
    breakpoints: np.ndarray = frozen_distribution.ppf(probabilities)
    if not np.all(np.isfinite(breakpoints)):
        return build_uniform_breakpoints(num_segments, clip)
    breakpoints = np.clip(breakpoints, -clip, clip)
    # Clipping can map several quantiles onto the same bound (a wide fit pins
    # many points at +/- clip). Dropping the duplicates avoids zero-width
    # segments, which would otherwise give infinite slopes and NaN outputs.
    breakpoints = np.unique(breakpoints)
    if breakpoints.size < 3 or breakpoints[-1] <= breakpoints[0]:
        return build_uniform_breakpoints(num_segments, clip)
    return breakpoints


class PWLTanh(nn.Module):
    """Drop-in replacement for nn.Tanh that evaluates a fixed PWL approximation."""

    def __init__(self, approximation: PWLApproximation) -> None:
        super().__init__()
        self.register_buffer(
            "breakpoints", torch.tensor(approximation.breakpoints, dtype=torch.float32)
        )
        self.register_buffer(
            "slopes", torch.tensor(approximation.slopes, dtype=torch.float32)
        )
        self.register_buffer(
            "intercepts", torch.tensor(approximation.intercepts, dtype=torch.float32)
        )

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        clamped: torch.Tensor = torch.clamp(
            input_batch, float(self.breakpoints[0]), float(self.breakpoints[-1])
        )
        # Interior breakpoints define the segment boundaries for the bucketise.
        segment_index: torch.Tensor = torch.searchsorted(
            self.breakpoints[1:-1].contiguous(), clamped
        )
        segment_index = torch.clamp(segment_index, 0, self.slopes.numel() - 1)
        segment_slope: torch.Tensor = self.slopes[segment_index]
        segment_intercept: torch.Tensor = self.intercepts[segment_index]
        return segment_slope * clamped + segment_intercept
