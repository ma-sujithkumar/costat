"""Candidate parametric families and their maximum-likelihood fits.

Each candidate wraps a scipy.stats distribution. Fitting returns a frozen
distribution object, so downstream code can call the same .pdf / .cdf / .ppf
interface regardless of which family won - that uniform interface is what lets
the quantiser stay generic instead of hardcoding Gaussian and Laplacian cases.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.stats as stats

from costat.utils.logging_utils import get_logger


class FittedCandidate:
    """A fitted candidate: its name, frozen distribution and free-parameter count."""

    def __init__(
        self,
        name: str,
        frozen_distribution: Any,
        parameters: Tuple[float, ...],
        num_params: int,
    ) -> None:
        self.name: str = name
        self.frozen_distribution: Any = frozen_distribution
        self.parameters: Tuple[float, ...] = parameters
        self.num_params: int = num_params


class CandidateLibrary:
    """Builds and fits the configured set of candidate distributions."""

    def __init__(self, candidates_config: Dict[str, Any], max_fit_samples: int) -> None:
        self.max_fit_samples: int = max_fit_samples
        self.logger = get_logger()
        # Keep only the enabled entries, preserving config order.
        self.specs: List[Dict[str, Any]] = [
            spec for spec in candidates_config["candidates"] if spec.get("enabled", True)
        ]
        self.random_state: np.random.Generator = np.random.default_rng(seed=0)

    def _subsample(self, samples: np.ndarray) -> np.ndarray:
        """Cap the sample count so heavy MLE fits (t, gennorm) stay fast."""
        if samples.size <= self.max_fit_samples:
            return samples
        indices: np.ndarray = self.random_state.choice(
            samples.size, size=self.max_fit_samples, replace=False
        )
        return samples[indices]

    def fit_all(self, samples: np.ndarray) -> List[FittedCandidate]:
        """Fit every enabled candidate to the (subsampled) data.

        Args:
            samples: flattened weight or pre-activation values for one layer.

        Returns:
            The candidates that fitted successfully, in config order.
        """
        fit_samples: np.ndarray = self._subsample(samples)
        fitted: List[FittedCandidate] = []
        for spec in self.specs:
            candidate: Optional[FittedCandidate] = self._fit_one(spec, fit_samples)
            if candidate is not None:
                fitted.append(candidate)
        if not fitted:
            raise RuntimeError("No candidate distribution could be fitted to the layer.")
        return fitted

    def _fit_one(
        self, spec: Dict[str, Any], samples: np.ndarray
    ) -> Optional[FittedCandidate]:
        """Fit a single candidate, returning None if the MLE is degenerate."""
        distribution = getattr(stats, spec["scipy_name"])
        # A runtime guard (not an import guard): some families fail to converge on
        # pathological layers, and we simply drop those rather than abort the run.
        try:
            parameters: Tuple[float, ...] = distribution.fit(samples)
            frozen = distribution(*parameters)
            # Reject fits that produced non-finite parameters or zero scale.
            if not np.all(np.isfinite(parameters)) or parameters[-1] <= 0.0:
                return None
        except Exception as fit_error:  # noqa: BLE001 - intentional runtime guard
            self.logger.debug("fit failed => %s : %s", spec["name"], str(fit_error))
            return None
        return FittedCandidate(
            name=spec["name"],
            frozen_distribution=frozen,
            parameters=parameters,
            num_params=spec["num_params"],
        )
