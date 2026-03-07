"""Unit tests for OCR pipeline compatibility entrypoints and smoke harness."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [PYTHON, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )


def test_run_pipeline_help():
    proc = _run("run_pipeline.py", "--help")
    assert proc.returncode == 0
    assert "usage" in (proc.stdout + proc.stderr).lower()


def test_run_pipeline_validate_only():
    proc = _run("run_pipeline.py", "--validate-only", "-c", "config.json")
    assert proc.returncode == 0
    assert "validation passed" in (proc.stdout + proc.stderr).lower()


def test_run_page_text_wrapper_help():
    proc = _run("tools/run_page_text.py", "--help")
    assert proc.returncode == 0
    assert "usage" in (proc.stdout + proc.stderr).lower()


def test_test_pipeline_help():
    proc = _run("test_pipeline.py", "--help")
    assert proc.returncode == 0
    assert "usage" in (proc.stdout + proc.stderr).lower()
