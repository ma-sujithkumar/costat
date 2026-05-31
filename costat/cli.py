"""Command-line entry point for the CoStat benchmark.

Usage:
    python run.py            run with the default config/config.ini
    python run.py -v         same, with debug-level logging
    python run.py --config /path/to/config.ini
"""

import argparse
import os
from typing import Optional

from costat.pipeline.benchmark import CoStatBenchmark
from costat.utils.config_loader import ConfigLoader
from costat.utils.logging_utils import configure_logger


def _default_config_path() -> str:
    """Locate config/config.ini relative to the project root (not hardcoded)."""
    package_dir: str = os.path.dirname(os.path.abspath(__file__))
    project_root: str = os.path.dirname(package_dir)
    return os.path.join(project_root, "config", "config.ini")


def parse_arguments() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="CoStat distribution-aware quantisation benchmark."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.ini (defaults to the project's config/config.ini).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )
    return parser.parse_args()


def main() -> None:
    arguments: argparse.Namespace = parse_arguments()
    configure_logger(verbose=arguments.verbose)

    config_path: Optional[str] = arguments.config
    if config_path is None:
        config_path = _default_config_path()

    config: ConfigLoader = ConfigLoader(config_path)
    benchmark: CoStatBenchmark = CoStatBenchmark(config)
    benchmark.run()


if __name__ == "__main__":
    main()
