#!/usr/bin/env python3
"""Compare experiment tracking outputs between two OCR evaluation runs.

Writes static regression artifacts to --output-dir:
  - regression_report.md
  - failing_pages.csv
  - regression_comparison.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from production.experiment_tracking import compare_tracking_runs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two OCR experiment runs")
    parser.add_argument("--current-dir", required=True, help="Current run directory")
    parser.add_argument("--baseline-dir", required=True, help="Baseline run directory")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for regression artifacts (default: current-dir)",
    )
    parser.add_argument("--top-n", type=int, default=10, help="Top N improved/regressed pages")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    current_dir = Path(args.current_dir)
    baseline_dir = Path(args.baseline_dir)
    output_dir = Path(args.output_dir) if args.output_dir else current_dir

    outputs = compare_tracking_runs(
        current_dir=current_dir,
        baseline_dir=baseline_dir,
        output_dir=output_dir,
        top_n=max(1, int(args.top_n)),
    )

    print("Comparison complete")
    for key, value in outputs.items():
        print(f"{key}: {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
