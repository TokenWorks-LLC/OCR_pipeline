#!/usr/bin/env python3
"""Shared helpers for compatibility wrappers that delegate to archive scripts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def run_archive_main(
    impl_path: Path,
    module_name: str,
    *,
    wrapper_name: str,
    missing_dependency_note: str,
) -> int:
    """Load an archive script dynamically and run its ``main`` entrypoint."""
    if "--help" in sys.argv or "-h" in sys.argv:
        print(f"{wrapper_name} compatibility wrapper")
        print(f"Delegates to {impl_path.relative_to(Path.cwd())}")
        print(f"Note: {missing_dependency_note}")

    if not impl_path.exists():
        raise FileNotFoundError(f"Missing implementation: {impl_path}")

    spec = importlib.util.spec_from_file_location(module_name, impl_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec for {impl_path}")

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
        raise AttributeError(f"{impl_path} does not define main()")

    result = entry()
    return int(result) if isinstance(result, int) else 0
