#!/usr/bin/env python3
"""Compatibility wrapper for archive/tools/run_gold_eval.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMPL_PATH = ROOT / "archive" / "tools" / "run_gold_eval.py"


def main() -> int:
    if "--help" in sys.argv or "-h" in sys.argv:
        print("run_gold_eval compatibility wrapper")
        print("Delegates to archive/tools/run_gold_eval.py")
        print("Note: this archived script requires additional legacy modules that may not be installed.")

    if not IMPL_PATH.exists():
        raise FileNotFoundError(f"Missing implementation: {IMPL_PATH}")

    spec = importlib.util.spec_from_file_location("_archive_run_gold_eval", IMPL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec for {IMPL_PATH}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        print(f"Missing dependency for archived evaluator: {exc}")
        print("Use run_pipeline.py and test_pipeline.py for current supported workflows.")
        return 2
    except SystemExit as exc:
        print("Archived evaluator exited during startup (likely missing legacy dependencies).")
        if isinstance(exc.code, int):
            return exc.code
        return 2

    entry = getattr(module, "main", None)
    if entry is None:
        raise AttributeError("archive/tools/run_gold_eval.py does not define main()")

    result = entry()
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
