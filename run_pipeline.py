#!/usr/bin/env python3
"""Compatibility entrypoint for OCR pipeline execution.

This script preserves the long-standing `run_pipeline.py` command while routing
execution to the maintained page-text pipeline at `tools/run_page_text.py`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RUN_PAGE_TEXT_PATH = ROOT / "tools" / "run_page_text.py"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _resolve_default_output_root(config: dict[str, Any]) -> str:
    return (
        config.get("output", {}).get("output_directory")
        or config.get("paths", {}).get("output_dir")
        or "reports/output"
    )


def _resolve_default_inputs(config: dict[str, Any]) -> str:
    return (
        config.get("input", {}).get("input_directory")
        or config.get("paths", {}).get("input_dir")
        or "data/input_pdfs"
    )


def _resolve_engine(config: dict[str, Any], explicit_engine: str | None) -> str:
    if explicit_engine:
        return explicit_engine

    configured = config.get("ocr", {}).get("engine") if isinstance(config, dict) else None
    if configured:
        return str(configured)

    return "ensemble"


def _validate_only(config_path: Path | None, input_dir: str | None, input_file: str | None) -> int:
    errors: list[str] = []

    if not RUN_PAGE_TEXT_PATH.exists():
        errors.append(f"Missing runner: {RUN_PAGE_TEXT_PATH}")

    if config_path is not None and not config_path.exists():
        errors.append(f"Config file not found: {config_path}")

    if input_dir and not Path(input_dir).exists():
        errors.append(f"Input directory not found: {input_dir}")

    if input_file and not Path(input_file).exists():
        errors.append(f"Input file not found: {input_file}")

    if errors:
        print("Validation failed:")
        for err in errors:
            print(f"- {err}")
        return 1

    print("Validation passed.")
    if config_path is not None:
        print(f"Config: {config_path}")
    print(f"Runner: {RUN_PAGE_TEXT_PATH}")
    return 0


def _build_manifest_for_single_pdf(pdf_path: Path) -> str:
    pages = [1]
    try:
        import fitz  # type: ignore

        with fitz.open(str(pdf_path)) as doc:
            page_count = len(doc)
        pages = list(range(1, max(page_count, 1) + 1))
    except Exception:
        # Fall back to first page when page counting dependencies are unavailable.
        pages = [1]

    fd, manifest_path = tempfile.mkstemp(prefix="ocr_single_pdf_", suffix=".tsv")
    os.close(fd)
    manifest_file = Path(manifest_path)
    with manifest_file.open("w", encoding="utf-8") as fh:
        for page in pages:
            fh.write(f"{pdf_path}\t{page}\n")
    return manifest_path


def _build_run_page_text_args(args: argparse.Namespace, cfg: dict[str, Any]) -> tuple[list[str], list[Path]]:
    temp_paths: list[Path] = []
    output_root = args.output_dir or _resolve_default_output_root(cfg)
    selected_engine = _resolve_engine(cfg, args.engine)

    if args.input_file:
        input_path = Path(args.input_file).resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        manifest_path = _build_manifest_for_single_pdf(input_path)
        temp_paths.append(Path(manifest_path))
        run_args = [
            "--manifest",
            manifest_path,
            "--output-root",
            output_root,
            "--prefer-text-layer",
        ]
    else:
        inputs_dir = args.input_dir or _resolve_default_inputs(cfg)
        run_args = [
            "--inputs",
            inputs_dir,
            "--output-root",
            output_root,
            "--prefer-text-layer",
        ]

    selected_engine_lower = str(selected_engine).lower()
    if selected_engine_lower.startswith("paddle"):
        run_args.extend(["--ocr-fallback", "paddle"])
    elif selected_engine_lower.startswith("ensemble") or selected_engine_lower.startswith("multi"):
        run_args.extend(["--ocr-fallback", "ensemble"])

    if args.status_bar:
        run_args.append("--status-bar")

    if args.profile:
        run_args.extend(["--profile", args.profile])

    if args.progress_csv:
        run_args.extend(["--progress-csv", args.progress_csv])

    return run_args, temp_paths


def _call_run_page_text(mapped_args: list[str]) -> int:
    spec = importlib.util.spec_from_file_location("_compat_run_page_text", RUN_PAGE_TEXT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load runner module from {RUN_PAGE_TEXT_PATH}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    main_fn = getattr(mod, "main", None)
    if main_fn is None:
        raise AttributeError("tools/run_page_text.py does not define main()")

    original_argv = sys.argv[:]
    try:
        sys.argv = [str(RUN_PAGE_TEXT_PATH)] + mapped_args
        result = main_fn()
    finally:
        sys.argv = original_argv

    return int(result) if isinstance(result, int) else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "OCR pipeline compatibility entrypoint. Supports legacy run_pipeline flags "
            "and routes execution to tools/run_page_text.py."
        )
    )
    parser.add_argument("-c", "--config", default="config.json", help="Path to config JSON")
    parser.add_argument("--input-dir", help="Input directory containing PDFs")
    parser.add_argument("--input-file", help="Single PDF file path")
    parser.add_argument("--output-dir", help="Output root directory")
    parser.add_argument("--engine", help="OCR engine name (compatibility flag)")
    parser.add_argument("--profile", help="Akkadian detection profile JSON")
    parser.add_argument("--progress-csv", help="Optional progress CSV output path")
    parser.add_argument("--status-bar", action="store_true", help="Display progress bar")
    parser.add_argument("--validate-only", action="store_true", help="Validate environment and config")
    parser.add_argument("--dry-run", action="store_true", help="Print mapped command without executing")
    parser.add_argument(
        "passthrough",
        nargs=argparse.REMAINDER,
        help="Extra flags forwarded directly to tools/run_page_text.py",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    cfg_path = Path(args.config) if args.config else None
    cfg: dict[str, Any] = {}
    if cfg_path is not None and cfg_path.exists():
        cfg = _load_json(cfg_path)

    if args.validate_only:
        return _validate_only(cfg_path if cfg_path and args.config else None, args.input_dir, args.input_file)

    mapped_args, temp_paths = _build_run_page_text_args(args, cfg)
    if args.passthrough:
        mapped_args.extend(args.passthrough)

    if args.dry_run:
        print("Mapped command:")
        print("python tools/run_page_text.py " + " ".join(mapped_args))
        return 0

    try:
        return _call_run_page_text(mapped_args)
    finally:
        for tmp_path in temp_paths:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
