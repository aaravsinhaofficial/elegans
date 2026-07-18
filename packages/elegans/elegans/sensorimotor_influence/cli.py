"""Command-line entry point for the complete toy validation study."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .plotting import make_all_figures
from .study import run_full_study

DEFAULT_OUTPUT = Path("artifacts/sensorimotor_influence")
LOGGER = logging.getLogger("sensorimotor_influence")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate an online action-conditioned predictive-information estimator "
            "in a scalar linear-Gaussian cable world."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--phase-steps", type=int, default=1_500)
    parser.add_argument("--base-seed", type=int, default=751_000)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument(
        "--robustness-seeds",
        type=int,
        default=None,
        help="Independent seeds per gain/noise cell (defaults to --seeds).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a small development smoke study (6 seeds, 500-step phases).",
    )
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run validation, save seed-level tables, and render the four figures."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    arguments = parse_arguments()
    seeds = 6 if arguments.quick else arguments.seeds
    phase_steps = 500 if arguments.quick else arguments.phase_steps
    bootstrap_replicates = 500 if arguments.quick else arguments.bootstrap_replicates
    robustness_seeds = 6 if arguments.quick else arguments.robustness_seeds
    results = run_full_study(
        output_directory=arguments.output,
        n_seeds=seeds,
        phase_steps=phase_steps,
        base_seed=arguments.base_seed,
        bootstrap_replicates=bootstrap_replicates,
        robustness_seeds=robustness_seeds,
        progress=lambda message: LOGGER.info("[sensorimotor-influence] %s", message),
    )
    if not arguments.no_figures:
        figure_directory = arguments.output / "figures"
        make_all_figures(results, figure_directory)
        LOGGER.info("[sensorimotor-influence] figures: %s", figure_directory)
    supported = sum(
        bool(result.get("supported"))
        for name, result in results.hypothesis_results.items()
        if name.startswith("H")
    )
    LOGGER.info(
        "[sensorimotor-influence] complete: %s/8 prespecified hypotheses supported; results: %s",
        supported,
        arguments.output,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
