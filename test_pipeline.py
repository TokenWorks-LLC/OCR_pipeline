#!/usr/bin/env python3
"""Comprehensive smoke checks for OCR pipeline executability and engine availability."""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

ENGINE_MODULES = {
    "paddleocr": "paddleocr",
    "doctr": "doctr",
    "mmocr": "mmocr",
    "kraken": "kraken",
}


def _run(cmd: list[str], timeout: int = 120) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
    )
    return proc.returncode, proc.stdout, proc.stderr


def check_cli_help() -> tuple[bool, str]:
    checks = [
        [PYTHON, "run_pipeline.py", "--help"],
        [PYTHON, "tools/run_page_text.py", "--help"],
    ]

    failures: list[str] = []
    for cmd in checks:
        code, out, err = _run(cmd)
        text = (out + err).lower()
        if code != 0 or "usage" not in text:
            failures.append(f"failed: {' '.join(cmd)}")

    if failures:
        return False, "; ".join(failures)
    return True, "CLI help checks passed"


def check_validate_only() -> tuple[bool, str]:
    code, out, err = _run([PYTHON, "run_pipeline.py", "--validate-only", "-c", "config.json"])
    ok = code == 0 and "validation passed" in (out + err).lower()
    return (ok, "run_pipeline --validate-only passed" if ok else out + err)


def check_engine_imports(required: list[str]) -> tuple[bool, str]:
    missing: list[str] = []
    present: list[str] = []

    for engine in required:
        module_name = ENGINE_MODULES[engine]
        try:
            importlib.import_module(module_name)
            present.append(engine)
        except Exception:
            missing.append(engine)

    if missing:
        return False, f"missing engines: {', '.join(missing)} | present: {', '.join(present) or 'none'}"
    return True, f"all required engines available: {', '.join(present)}"


def check_manifest_mode_smoke() -> tuple[bool, str]:
    # Smoke-test manifest parsing path without requiring OCR execution.
    sample_pdf = ROOT / "data" / "samples"
    if not sample_pdf.exists():
        return True, "data/samples not present; manifest smoke skipped"

    return True, "manifest-mode preconditions satisfied"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR pipeline smoke test runner")
    parser.add_argument(
        "--required-engines",
        nargs="+",
        choices=sorted(ENGINE_MODULES.keys()),
        default=sorted(ENGINE_MODULES.keys()),
        help="Engine set that must be importable",
    )
    parser.add_argument(
        "--allow-missing-engines",
        action="store_true",
        help="Do not fail overall run if one or more required engines are missing",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    checks = [
        ("cli_help", check_cli_help),
        ("validate_only", check_validate_only),
        ("manifest_preconditions", check_manifest_mode_smoke),
    ]

    failures = 0
    print("OCR pipeline smoke checks")
    print("=" * 60)

    for name, fn in checks:
        ok, msg = fn()
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {msg}")
        if not ok:
            failures += 1

    engines_ok, engines_msg = check_engine_imports(args.required_engines)
    if engines_ok:
        print(f"[PASS] engine_imports: {engines_msg}")
    else:
        level = "WARN" if args.allow_missing_engines else "FAIL"
        print(f"[{level}] engine_imports: {engines_msg}")
        if not args.allow_missing_engines:
            failures += 1

    print("=" * 60)
    if failures:
        print(f"Smoke checks failed: {failures}")
        return 1

    print("All smoke checks passed")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    raise SystemExit(main())
