"""Failure-semantics regression tests for the page-text pipeline."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [PYTHON, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
        check=False,
    )


def _write_pdf(pdf_path: Path, pages: list[str]) -> None:
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    for page_text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), page_text, fontsize=11)
    doc.save(str(pdf_path))
    doc.close()


def _read_csv(path: Path, encoding: str = "utf-8") -> list[dict[str, str]]:
    with path.open("r", encoding=encoding, newline="") as fh:
        return list(csv.DictReader(fh))


def test_all_pages_failed_returns_nonzero(tmp_path: Path):
    missing_pdf = tmp_path / "missing.pdf"
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(f"pdf_path\tpage_no\n{missing_pdf}\t1\n", encoding="utf-8")

    output_dir = tmp_path / "outputs"
    proc = _run(
        "tools/run_page_text.py",
        "--manifest",
        str(manifest),
        "--output-root",
        str(output_dir),
        "--prefer-text-layer",
    )

    assert proc.returncode != 0
    rows = _read_csv(output_dir / "client_page_text.csv", encoding="utf-8-sig")
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"


def test_empty_rate_threshold_breaches_return_nonzero(tmp_path: Path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir(parents=True)
    good_pdf = input_dir / "good.pdf"
    _write_pdf(good_pdf, ["This page should extract from text layer."])

    missing_pdf = input_dir / "missing.pdf"
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "pdf_path\tpage_no\n"
        f"{good_pdf}\t1\n"
        f"{missing_pdf}\t1\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "outputs"
    proc = _run(
        "tools/run_page_text.py",
        "--manifest",
        str(manifest),
        "--output-root",
        str(output_dir),
        "--prefer-text-layer",
        "--max-empty-rate",
        "0.4",
    )

    assert proc.returncode != 0
    progress_rows = _read_csv(output_dir / "progress.csv")
    statuses = {row["status"] for row in progress_rows}
    assert "success" in statuses
    assert "failed" in statuses


def test_no_usable_ensemble_engine_still_succeeds_for_text_layer_pages(tmp_path: Path):
    """No usable OCR engine must not fail a run whose pages all used the text layer.

    OCR fallback is opportunistic: the engine-availability gate should only fire
    when a page actually needed OCR. A page served entirely by an accepted text
    layer does not depend on the ensemble, so a broken/empty engine config must
    not gate-fail the run in the default (non-strict) mode. Operators who require
    full engine readiness opt in via ``--strict-readiness`` (covered by
    ``test_strict_readiness_flag_emits_failure_reason``).
    """
    input_dir = tmp_path / "inputs"
    input_dir.mkdir(parents=True)
    sample_pdf = input_dir / "sample.pdf"
    _write_pdf(sample_pdf, ["Text layer page for readiness check."])

    profile = tmp_path / "bad_profile.json"
    profile.write_text('{"engines":{"enabled":["nonexistent_engine"]}}', encoding="utf-8")

    output_dir = tmp_path / "outputs"
    proc = _run(
        "tools/run_page_text.py",
        "--inputs",
        str(input_dir),
        "--output-root",
        str(output_dir),
        "--prefer-text-layer",
        "--ocr-fallback",
        "ensemble",
        "--profile",
        str(profile),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "engine_availability_gate" not in (proc.stdout + proc.stderr)

    rows = _read_csv(output_dir / "client_page_text.csv", encoding="utf-8-sig")
    assert len(rows) == 1
    assert rows[0]["status"] == "success"

    payload = json.loads((output_dir / "run_quality.json").read_text(encoding="utf-8"))
    assert payload.get("launch_gate", {}).get("should_fail_run", True) is False


def test_strict_readiness_flag_emits_failure_reason(tmp_path: Path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir(parents=True)
    sample_pdf = input_dir / "sample.pdf"
    _write_pdf(sample_pdf, ["Strict readiness test page."])

    profile = tmp_path / "strict_bad_profile.json"
    profile.write_text('{"engines":{"enabled":["nonexistent_engine"]}}', encoding="utf-8")

    output_dir = tmp_path / "outputs"
    proc = _run(
        "tools/run_page_text.py",
        "--inputs",
        str(input_dir),
        "--output-root",
        str(output_dir),
        "--prefer-text-layer",
        "--ocr-fallback",
        "ensemble",
        "--strict-readiness",
        "--profile",
        str(profile),
    )

    assert proc.returncode != 0
    assert "strict_readiness_failed" in (proc.stdout + proc.stderr)


def test_launch_gate_production_mode_can_fail_on_avg_quality_threshold(tmp_path: Path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir(parents=True)
    sample_pdf = input_dir / "sample.pdf"
    _write_pdf(sample_pdf, ["Launch-gate production mode quality threshold validation page."])

    output_dir = tmp_path / "outputs"
    proc = _run(
        "tools/run_page_text.py",
        "--inputs",
        str(input_dir),
        "--output-root",
        str(output_dir),
        "--prefer-text-layer",
        "--launch-gate-mode",
        "production",
        "--gate-min-avg-quality",
        "0.99",
    )

    assert proc.returncode != 0
    assert "launch_gate_failed" in (proc.stdout + proc.stderr)


def test_launch_gate_beta_mode_is_review_only_for_quality_thresholds(tmp_path: Path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir(parents=True)
    sample_pdf = input_dir / "sample.pdf"
    _write_pdf(sample_pdf, ["Launch-gate beta mode review-oriented quality threshold validation page."])

    output_dir = tmp_path / "outputs"
    proc = _run(
        "tools/run_page_text.py",
        "--inputs",
        str(input_dir),
        "--output-root",
        str(output_dir),
        "--prefer-text-layer",
        "--launch-gate-mode",
        "beta",
        "--gate-min-avg-quality",
        "0.99",
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr

    payload = json.loads((output_dir / "run_quality.json").read_text(encoding="utf-8"))
    launch_gate = payload.get("launch_gate", {})
    assert launch_gate.get("review_required", False) is True
    assert launch_gate.get("should_fail_run", False) is False
