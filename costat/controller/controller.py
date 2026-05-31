"""The distribution controller.

For each layer it fits the full candidate family, scores every fit with several
complementary metrics, and then selects a winner by aggregating the metric
*ranks* rather than trusting one raw number.

Why rank aggregation: KL divergence on its own is sensitive to histogram binning
and rewards extra parameters for free. By making each metric vote through its
rank and folding in a BIC complexity penalty, a heavier-tailed family is only
chosen when it agrees across criteria - which is what makes the selection robust
instead of a coin toss between close fits.
"""

from typing import Any, Dict, List

import numpy as np

from costat.controller.candidates import CandidateLibrary, FittedCandidate
from costat.controller.scoring import METRIC_REGISTRY, ScoringContext
from costat.utils.logging_utils import get_logger


class LayerFit:
    """The controller's full decision for one layer."""

    def __init__(
        self,
        layer_name: str,
        selected: FittedCandidate,
        candidates: List[FittedCandidate],
        metric_scores: Dict[str, Dict[str, float]],
        rank_sums: Dict[str, float],
    ) -> None:
        self.layer_name: str = layer_name
        self.selected: FittedCandidate = selected
        self.candidates: List[FittedCandidate] = candidates
        self.metric_scores: Dict[str, Dict[str, float]] = metric_scores
        self.rank_sums: Dict[str, float] = rank_sums


class DistributionController:
    """Fits, scores and selects a distribution for each profiled layer."""

    def __init__(
        self,
        candidates_config: Dict[str, Any],
        scoring_config: Dict[str, Any],
        histogram_bins: int,
        max_fit_samples: int,
    ) -> None:
        self.library: CandidateLibrary = CandidateLibrary(candidates_config, max_fit_samples)
        self.histogram_bins: int = histogram_bins
        self.max_fit_samples: int = max_fit_samples
        self.logger = get_logger()
        # Only the enabled metrics participate, with their configured weights.
        self.active_metrics: List[Dict[str, Any]] = [
            metric for metric in scoring_config["metrics"] if metric.get("enabled", True)
        ]
        self.random_state: np.random.Generator = np.random.default_rng(seed=2)

    def _score_sample(self, samples: np.ndarray) -> np.ndarray:
        """Cap the sample count used for scoring so KS / Wasserstein stay quick."""
        if samples.size <= self.max_fit_samples:
            return samples
        indices: np.ndarray = self.random_state.choice(
            samples.size, size=self.max_fit_samples, replace=False
        )
        return samples[indices]

    def select(self, layer_name: str, samples: np.ndarray) -> LayerFit:
        """Run the full fit-score-rank pipeline for one layer's samples."""
        fitted_candidates: List[FittedCandidate] = self.library.fit_all(samples)
        scoring_samples: np.ndarray = self._score_sample(samples)
        context: ScoringContext = ScoringContext(scoring_samples, self.histogram_bins)

        # metric_scores[metric_name][candidate_name] = raw score (lower is better).
        metric_scores: Dict[str, Dict[str, float]] = {}
        for metric in self.active_metrics:
            metric_name: str = metric["name"]
            metric_function = METRIC_REGISTRY[metric_name]
            metric_scores[metric_name] = {
                candidate.name: metric_function(context, candidate)
                for candidate in fitted_candidates
            }

        rank_sums: Dict[str, float] = self._aggregate_ranks(fitted_candidates, metric_scores)
        winner_name: str = min(rank_sums, key=rank_sums.get)
        selected: FittedCandidate = next(
            candidate for candidate in fitted_candidates if candidate.name == winner_name
        )
        self.logger.debug(
            "select => layer=%s winner=%s rank_sum=%.3f",
            layer_name,
            winner_name,
            rank_sums[winner_name],
        )
        return LayerFit(layer_name, selected, fitted_candidates, metric_scores, rank_sums)

    def _aggregate_ranks(
        self,
        candidates: List[FittedCandidate],
        metric_scores: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        """Turn raw metric scores into a weighted sum of per-metric ranks.

        Rank 1 is best (lowest raw score). Ties share the average rank so no
        candidate is arbitrarily favoured by input order.
        """
        weight_by_metric: Dict[str, float] = {
            metric["name"]: float(metric.get("weight", 1.0)) for metric in self.active_metrics
        }
        rank_sums: Dict[str, float] = {candidate.name: 0.0 for candidate in candidates}
        for metric_name, scores in metric_scores.items():
            ranks: Dict[str, float] = self._rank_ascending(scores)
            weight: float = weight_by_metric[metric_name]
            for candidate_name, rank_value in ranks.items():
                rank_sums[candidate_name] += weight * rank_value
        return rank_sums

    @staticmethod
    def _rank_ascending(scores: Dict[str, float]) -> Dict[str, float]:
        """Average-rank the scores so equal values get the same rank."""
        names: List[str] = list(scores.keys())
        values: np.ndarray = np.array([scores[name] for name in names], dtype=np.float64)
        order: np.ndarray = np.argsort(values, kind="mergesort")
        ranks: np.ndarray = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
        # Average ties so duplicated scores do not bias the aggregate.
        unique_values: np.ndarray = np.unique(values)
        for value in unique_values:
            tie_mask: np.ndarray = values == value
            if np.count_nonzero(tie_mask) > 1:
                ranks[tie_mask] = ranks[tie_mask].mean()
        return {names[index]: float(ranks[index]) for index in range(len(names))}
