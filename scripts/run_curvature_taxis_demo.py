"""Run the minimal bilateral-sensing curvature taxis demonstration.

The source bearing from the requested start is 19.983 degrees, so an exact 20-degree
no-steering agent reaches the source by geometry alone.  The default comparison uses a modest
30-degree offset to make steering necessary while preserving the requested setup otherwise.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from elegans.curvature_taxis import CurvatureTaxisConfig, save_demo_artifacts

DEFAULT_COMPARISON_HEADING_DEGREES = 30.0
DEFAULT_HEADING_COUNT = 20


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exports/curvature_taxis"),
        help="Directory for the figure, CSV traces, sweep, and JSON summary.",
    )
    parser.add_argument(
        "--initial-heading-degrees",
        type=float,
        default=DEFAULT_COMPARISON_HEADING_DEGREES,
        help="Initial heading for the controller/baseline comparison (default: 30).",
    )
    parser.add_argument(
        "--heading-count",
        type=int,
        default=DEFAULT_HEADING_COUNT,
        help="Number of evenly spaced headings in the robustness sweep (default: 20).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the demo, save all outputs, and print its machine-readable summary."""
    args = parse_args()
    config = replace(
        CurvatureTaxisConfig(),
        initial_heading_degrees=args.initial_heading_degrees,
    )
    summary = save_demo_artifacts(
        args.output_dir,
        config,
        heading_count=args.heading_count,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
