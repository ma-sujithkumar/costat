"""Weight quantisers.

Two strategies share the same call shape so the benchmark can swap them freely:

* UniformQuantizer    - evenly spaced levels across the weight range (baseline).
* DistributionAwareQuantizer - levels placed at equal-probability quantiles of
  the layer's fitted distribution, with Lloyd-Max optimal reconstruction values
  (the conditional mean of each bin). It reads everything it needs through the
  frozen distribution's ppf / pdf, so it never branches on which family won.
"""

from typing import Any

import numpy as np


class UniformQuantizer:
    """Even b-bit quantisation over the observed [min, max] weight range."""

    def quantize(self, values: np.ndarray, bit_width: int) -> np.ndarray:
        num_levels: int = 2 ** bit_width
        lower: float = float(values.min())
        upper: float = float(values.max())
        if upper <= lower:
            return values.copy()
        step: float = (upper - lower) / (num_levels - 1)
        level_indices: np.ndarray = np.round((values - lower) / step)
        level_indices = np.clip(level_indices, 0, num_levels - 1)
        return lower + level_indices * step


class DistributionAwareQuantizer:
    """Equal-probability levels from a fitted distribution, Lloyd-Max centroids."""

    def __init__(self, clip_quantile: float, integration_points: int) -> None:
        self.clip_quantile: float = clip_quantile
        self.integration_points: int = integration_points

    def quantize(
        self, values: np.ndarray, frozen_distribution: Any, bit_width: int
    ) -> np.ndarray:
        num_levels: int = 2 ** bit_width
        bin_edges: np.ndarray = self._equal_probability_edges(frozen_distribution, num_levels)
        # A degenerate fit (flat or non-monotone edges) falls back to uniform.
        if bin_edges is None:
            return UniformQuantizer().quantize(values, bit_width)
        centroids: np.ndarray = self._bin_centroids(frozen_distribution, bin_edges)
        clipped: np.ndarray = np.clip(values, bin_edges[0], bin_edges[-1])
        # searchsorted maps each weight to the bin whose centroid it adopts.
        bin_index: np.ndarray = np.searchsorted(bin_edges, clipped, side="right") - 1
        bin_index = np.clip(bin_index, 0, num_levels - 1)
        return centroids[bin_index]

    def _equal_probability_edges(self, frozen_distribution: Any, num_levels: int):
        """Place num_levels + 1 boundaries at equal-probability quantiles."""
        probabilities: np.ndarray = np.linspace(
            self.clip_quantile, 1.0 - self.clip_quantile, num_levels + 1
        )
        edges: np.ndarray = frozen_distribution.ppf(probabilities)
        if not np.all(np.isfinite(edges)) or edges[-1] <= edges[0]:
            return None
        # Guard against numerical ties so every bin keeps a positive width.
        monotone_edges: np.ndarray = np.maximum.accumulate(edges)
        if monotone_edges[-1] <= monotone_edges[0]:
            return None
        return monotone_edges

    def _bin_centroids(self, frozen_distribution: Any, bin_edges: np.ndarray) -> np.ndarray:
        """Lloyd-Max reconstruction value for each bin: E[w | bin] under the fit."""
        num_bins: int = bin_edges.size - 1
        centroids: np.ndarray = np.empty(num_bins, dtype=np.float64)
        for bin_position in range(num_bins):
            left_edge: float = bin_edges[bin_position]
            right_edge: float = bin_edges[bin_position + 1]
            grid: np.ndarray = np.linspace(left_edge, right_edge, self.integration_points)
            density: np.ndarray = frozen_distribution.pdf(grid)
            mass: float = float(np.trapezoid(density, grid))
            if mass <= 0.0:
                # Empty-mass bin: fall back to its midpoint.
                centroids[bin_position] = 0.5 * (left_edge + right_edge)
                continue
            weighted: float = float(np.trapezoid(grid * density, grid))
            centroids[bin_position] = weighted / mass
        return centroids
