#!/usr/bin/env python3
"""Compatibility wrapper for archive/tools/run_baseline_eval.py."""

from __future__ import annotations

from pathlib import Path

from _archive_wrapper import run_archive_main


ROOT = Path(__file__).resolve().parents[1]
IMPL_PATH = ROOT / "archive" / "tools" / "run_baseline_eval.py"


def main() -> int:
    return run_archive_main(
        IMPL_PATH,
        "_archive_run_baseline_eval",
        wrapper_name="run_baseline_eval",
        missing_dependency_note="this archived script requires additional legacy dependencies.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
