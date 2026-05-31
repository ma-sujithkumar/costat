"""Goodness-of-fit metrics used to rank candidate distributions.

Every metric is normalised to "lower is better" so the controller can aggregate
them by rank without worrying about direction. The metric name in scoring.json
is mapped to its function through a dictionary, keeping the call site branch-free.
"""

from typing import Callable, Dict

import numpy as np
import scipy.stats as stats

from costat.controller.candidates import FittedCandidate

EPSILON: float = 1e-12


class ScoringContext:
    """Everything a metric needs about one layer's data, bundled together."""

    def __init__(self, samples: np.ndarray, histogram_bins: int) -> None:
        self.samples: np.ndarray = samples
        self.histogram_bins: int = histogram_bins
        self.random_state: np.random.Generator = np.random.default_rng(seed=1)


def kl_divergence(context: ScoringContext, candidate: FittedCandidate) -> float:
    """KL( empirical || fitted ) over a normalised histogram of the samples."""
    counts, bin_edges = np.histogram(context.samples, bins=context.histogram_bins, density=False)
    empirical_prob: np.ndarray = counts.astype(np.float64)
    empirical_prob /= max(empirical_prob.sum(), EPSILON)
    bin_centers: np.ndarray = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    fitted_density: np.ndarray = candidate.frozen_distribution.pdf(bin_centers)
    fitted_prob: np.ndarray = fitted_density.astype(np.float64)
    fitted_prob /= max(fitted_prob.sum(), EPSILON)
    mask: np.ndarray = empirical_prob > 0.0
    ratio: np.ndarray = empirical_prob[mask] / np.maximum(fitted_prob[mask], EPSILON)
    return float(np.sum(empirical_prob[mask] * np.log(ratio)))


def bayesian_information_criterion(
    context: ScoringContext, candidate: FittedCandidate
) -> float:
    """BIC = k*ln(n) - 2*log-likelihood. The k*ln(n) term penalises families
    that buy a better fit with extra parameters (Student-t, gennorm, skew-normal)."""
    sample_count: int = context.samples.size
    log_likelihood: float = float(
        np.sum(candidate.frozen_distribution.logpdf(context.samples))
    )
    penalty: float = candidate.num_params * np.log(max(sample_count, 1))
    return penalty - 2.0 * log_likelihood


def ks_statistic(context: ScoringContext, candidate: FittedCandidate) -> float:
    """Kolmogorov-Smirnov distance between empirical and fitted CDFs."""
    result = stats.kstest(context.samples, candidate.frozen_distribution.cdf)
    return float(result.statistic)


def wasserstein(context: ScoringContext, candidate: FittedCandidate) -> float:
    """Earth-mover distance between the samples and a draw from the fitted family."""
    reference: np.ndarray = candidate.frozen_distribution.rvs(
        size=context.samples.size, random_state=context.random_state
    )
    return float(stats.wasserstein_distance(context.samples, reference))


MetricType = Callable[[ScoringContext, FittedCandidate], float]

METRIC_REGISTRY: Dict[str, MetricType] = {
    "kl_divergence": kl_divergence,
    "bic": bayesian_information_criterion,
    "ks_statistic": ks_statistic,
    "wasserstein": wasserstein,
}
