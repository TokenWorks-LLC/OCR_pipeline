#!/usr/bin/env python3
"""Compatibility wrapper for the protected page-text pipeline implementation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMPL_PATH = ROOT / ".merge_protect" / "tools" / "run_page_text.py"


def _load_impl_main():
    if not IMPL_PATH.exists():
        raise FileNotFoundError(f"Missing implementation: {IMPL_PATH}")

    spec = importlib.util.spec_from_file_location("_protected_run_page_text", IMPL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec for {IMPL_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    main_fn = getattr(module, "main", None)
    if main_fn is None:
        raise AttributeError("Protected run_page_text implementation does not define main()")
    return main_fn


def main() -> int:
    impl_main = _load_impl_main()
    result = impl_main()
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
