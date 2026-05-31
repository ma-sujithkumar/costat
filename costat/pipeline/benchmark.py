"""Top-level benchmark: run the whole CoStat software flow on each model.

Per model the steps are: train FP32, profile weights and pre-activations, let the
controller pick a distribution per layer, then compare uniform against
distribution-aware quantisation for both the weights (INT2 / INT4) and the tanh
piecewise-linear approximation. Results are written to CSV, printed as a table,
and visualised under plots/.
"""

import os
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from costat.controller.controller import DistributionController, LayerFit
from costat.data.mnist import build_dataloaders
from costat.models.factory import create_model
from costat.pipeline import plots
from costat.pipeline.evaluator import (
    apply_distribution_aware_weight_quantization,
    apply_pwl_activations,
    apply_uniform_weight_quantization,
    evaluate_accuracy,
)
from costat.pipeline.trainer import Trainer
from costat.profiling.profiler import LayerProfile, ModelProfiler
from costat.quantization.pwl_activation import (
    PWLApproximation,
    build_distribution_aware_breakpoints,
    build_uniform_breakpoints,
)
from costat.quantization.weight_quantizer import DistributionAwareQuantizer
from costat.utils.config_loader import ConfigLoader
from costat.utils.io_utils import ensure_dir, write_csv
from costat.utils.logging_utils import get_logger


class CoStatBenchmark:
    """Drives training, profiling, selection and evaluation for every model."""

    def __init__(self, config: ConfigLoader) -> None:
        self.config: ConfigLoader = config
        self.logger = get_logger()
        self.device: str = config.get_str("training", "device")
        self.plots_dir: str = ensure_dir(config.get_path("plots_dir"))
        self.results_dir: str = ensure_dir(config.get_path("results_dir"))
        self.bit_widths: List[int] = config.get_int_list("quantization", "bit_widths")

        self.models_config: Dict[str, Any] = ConfigLoader.load_json(
            config.get_str("benchmark", "models_config")
        )
        candidates_config: Dict[str, Any] = ConfigLoader.load_json(
            config.get_str("controller", "candidates_config")
        )
        scoring_config: Dict[str, Any] = ConfigLoader.load_json(
            config.get_str("controller", "scoring_config")
        )
        self.controller: DistributionController = DistributionController(
            candidates_config=candidates_config,
            scoring_config=scoring_config,
            histogram_bins=config.get_int("controller", "histogram_bins"),
            max_fit_samples=config.get_int("controller", "max_fit_samples"),
        )
        self.weight_quantizer: DistributionAwareQuantizer = DistributionAwareQuantizer(
            clip_quantile=config.get_float("quantization", "clip_quantile"),
            integration_points=config.get_int("quantization", "centroid_integration_points"),
        )
        self.weight_rows: List[Dict[str, Any]] = []
        self.pwl_rows: List[Dict[str, Any]] = []
        self.selection_rows: List[Dict[str, Any]] = []

    def run(self) -> None:
        """Run the benchmark for every model listed in the configuration."""
        torch.manual_seed(self.config.get_int("training", "seed"))
        np.random.seed(self.config.get_int("training", "seed"))
        train_loader, test_loader = build_dataloaders(
            data_dir=self.config.get_path("data_dir"),
            batch_size=self.config.get_int("training", "batch_size"),
            num_workers=self.config.get_int("training", "num_workers"),
        )
        model_names: List[str] = self.config.get_str_list("benchmark", "models")
        for model_name in model_names:
            self._run_single_model(model_name, train_loader, test_loader)

        self._write_reports()
        self._print_summary()

    def _run_single_model(
        self, model_name: str, train_loader: DataLoader, test_loader: DataLoader
    ) -> None:
        model_spec: Dict[str, Any] = self.models_config["models"][model_name]
        display_name: str = model_spec["display_name"]
        self.logger.info("==== model => %s ====", display_name)

        model = create_model(model_name, self.models_config)
        trainer: Trainer = Trainer(
            device=self.device,
            learning_rate=self.config.get_float("training", "learning_rate"),
        )
        # A model may override the default epoch count (deeper tanh nets such as
        # the residual model need a few more passes to reach a fair baseline).
        epochs: int = model_spec.get("epochs", self.config.get_int("training", "epochs"))
        trainer.train(model, train_loader, epochs)
        fp32_accuracy: float = evaluate_accuracy(model, test_loader, self.device)
        self.logger.info("%s => FP32 accuracy %.2f%%", display_name, fp32_accuracy)

        profiler: ModelProfiler = ModelProfiler(model, self.device)
        weight_profiles: List[LayerProfile] = profiler.collect_weights()
        preactivation_profiles: List[LayerProfile] = profiler.collect_preactivations(
            test_loader, self.config.get_int("training", "calibration_batches")
        )

        weight_fits: Dict[str, LayerFit] = self._fit_layers(weight_profiles)
        preactivation_fits: Dict[str, LayerFit] = self._fit_layers(preactivation_profiles)

        self._evaluate_weight_quantisation(
            display_name, model, test_loader, fp32_accuracy, weight_fits
        )
        self._evaluate_pwl_activation(
            display_name, model, test_loader, preactivation_profiles, preactivation_fits
        )
        self._record_selection(display_name, weight_fits)
        self._plot_example_fit(display_name, weight_profiles, weight_fits)

    def _fit_layers(self, profiles: List[LayerProfile]) -> Dict[str, LayerFit]:
        """Run the controller on each profiled layer."""
        fits: Dict[str, LayerFit] = {}
        for profile in profiles:
            fits[profile.layer_name] = self.controller.select(
                profile.layer_name, profile.samples
            )
        return fits

    def _evaluate_weight_quantisation(
        self,
        display_name: str,
        model: torch.nn.Module,
        test_loader: DataLoader,
        fp32_accuracy: float,
        weight_fits: Dict[str, LayerFit],
    ) -> None:
        uniform_accuracy: List[float] = []
        dist_aware_accuracy: List[float] = []
        for bit_width in self.bit_widths:
            uniform_model = apply_uniform_weight_quantization(model, bit_width)
            dist_model = apply_distribution_aware_weight_quantization(
                model, bit_width, weight_fits, self.weight_quantizer
            )
            uniform_score: float = evaluate_accuracy(uniform_model, test_loader, self.device)
            dist_score: float = evaluate_accuracy(dist_model, test_loader, self.device)
            uniform_accuracy.append(uniform_score)
            dist_aware_accuracy.append(dist_score)
            self.weight_rows.append(
                {
                    "model": display_name,
                    "bit_width": "INT" + str(bit_width),
                    "fp32_accuracy": round(fp32_accuracy, 2),
                    "uniform_accuracy": round(uniform_score, 2),
                    "dist_aware_accuracy": round(dist_score, 2),
                    "gain": round(dist_score - uniform_score, 2),
                }
            )
            self.logger.info(
                "%s INT%d => uniform %.2f%% | dist-aware %.2f%% | gain %.2f",
                display_name, bit_width, uniform_score, dist_score,
                dist_score - uniform_score,
            )
        plots.plot_accuracy_vs_bitwidth(
            display_name, self.bit_widths, uniform_accuracy,
            dist_aware_accuracy, fp32_accuracy, self.plots_dir,
        )

    def _evaluate_pwl_activation(
        self,
        display_name: str,
        model: torch.nn.Module,
        test_loader: DataLoader,
        preactivation_profiles: List[LayerProfile],
        preactivation_fits: Dict[str, LayerFit],
    ) -> None:
        num_segments: int = self.config.get_int("pwl", "num_segments")
        clip: float = self.config.get_float("pwl", "tanh_clip")
        margin: float = self.config.get_float("pwl", "boundary_margin")

        uniform_breakpoints = build_uniform_breakpoints(num_segments, clip)
        uniform_approx: PWLApproximation = PWLApproximation(uniform_breakpoints)

        uniform_by_layer: Dict[str, PWLApproximation] = {}
        dist_by_layer: Dict[str, PWLApproximation] = {}
        layer_names: List[str] = []
        uniform_errors: List[float] = []
        dist_errors: List[float] = []

        for profile in preactivation_profiles:
            layer_name: str = profile.layer_name
            frozen = preactivation_fits[layer_name].selected.frozen_distribution
            dist_breakpoints = build_distribution_aware_breakpoints(
                frozen, num_segments, margin, clip
            )
            dist_approx: PWLApproximation = PWLApproximation(dist_breakpoints)
            uniform_by_layer[layer_name] = uniform_approx
            dist_by_layer[layer_name] = dist_approx

            uniform_error: float = uniform_approx.density_weighted_error(profile.samples)
            dist_error: float = dist_approx.density_weighted_error(profile.samples)
            layer_names.append(layer_name)
            uniform_errors.append(uniform_error)
            dist_errors.append(dist_error)
            self.pwl_rows.append(
                {
                    "model": display_name,
                    "layer": layer_name,
                    "uniform_error": round(uniform_error, 5),
                    "dist_aware_error": round(dist_error, 5),
                    "error_reduction_pct": round(
                        100.0 * (uniform_error - dist_error) / max(uniform_error, 1e-9), 1
                    ),
                }
            )

        uniform_model = apply_pwl_activations(model, uniform_by_layer)
        dist_model = apply_pwl_activations(model, dist_by_layer)
        uniform_acc: float = evaluate_accuracy(uniform_model, test_loader, self.device)
        dist_acc: float = evaluate_accuracy(dist_model, test_loader, self.device)
        self.logger.info(
            "%s PWL tanh (N=%d) => uniform %.2f%% | dist-aware %.2f%%",
            display_name, num_segments, uniform_acc, dist_acc,
        )
        self.pwl_rows.append(
            {
                "model": display_name,
                "layer": "END_TO_END_ACCURACY",
                "uniform_error": round(uniform_acc, 2),
                "dist_aware_error": round(dist_acc, 2),
                "error_reduction_pct": round(dist_acc - uniform_acc, 2),
            }
        )
        plots.plot_pwl_error(
            display_name, layer_names, uniform_errors, dist_errors, self.plots_dir
        )

    def _record_selection(
        self, display_name: str, weight_fits: Dict[str, LayerFit]
    ) -> None:
        layer_names: List[str] = list(weight_fits.keys())
        selected_names: List[str] = [
            weight_fits[name].selected.name for name in layer_names
        ]
        for layer_name, selected_name in zip(layer_names, selected_names):
            self.selection_rows.append(
                {
                    "model": display_name,
                    "layer": layer_name,
                    "selected_distribution": selected_name,
                }
            )
        plots.plot_selected_distributions(
            display_name, layer_names, selected_names, self.plots_dir
        )

    def _plot_example_fit(
        self,
        display_name: str,
        weight_profiles: List[LayerProfile],
        weight_fits: Dict[str, LayerFit],
    ) -> None:
        """Plot the histogram-and-fit for the largest weight layer as an example."""
        largest_profile: LayerProfile = max(
            weight_profiles, key=lambda profile: profile.samples.size
        )
        layer_fit: LayerFit = weight_fits[largest_profile.layer_name]
        plots.plot_weight_fit(
            display_name,
            largest_profile.layer_name,
            largest_profile.samples,
            layer_fit.selected.frozen_distribution,
            layer_fit.selected.name,
            self.plots_dir,
        )

    def _write_reports(self) -> None:
        write_csv(self.weight_rows, os.path.join(self.results_dir, "weight_quantisation.csv"))
        write_csv(self.pwl_rows, os.path.join(self.results_dir, "pwl_activation.csv"))
        write_csv(self.selection_rows, os.path.join(self.results_dir, "layer_selection.csv"))
        self.logger.info("reports => written to %s", self.results_dir)

    def _print_summary(self) -> None:
        print("")
        print("==================== CoStat weight quantisation ====================")
        print(
            "{:<14}{:<8}{:>10}{:>12}{:>14}{:>8}".format(
                "model", "bits", "fp32", "uniform", "dist-aware", "gain"
            )
        )
        for row in self.weight_rows:
            print(
                "{:<14}{:<8}{:>10.2f}{:>12.2f}{:>14.2f}{:>8.2f}".format(
                    row["model"], row["bit_width"], row["fp32_accuracy"],
                    row["uniform_accuracy"], row["dist_aware_accuracy"], row["gain"],
                )
            )
        print("====================================================================")
