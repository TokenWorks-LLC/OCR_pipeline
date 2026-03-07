"""End-to-end regression tests for page-text pipeline workflows.

These tests generate small PDFs at runtime so they do not depend on large
binary fixtures. They validate the real CLI entrypoints and CSV artifacts.
"""

from __future__ import annotations

import csv
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


def test_run_page_text_inputs_mode_generates_expected_outputs(tmp_path: Path):
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir(parents=True)

    sample_pdf = input_dir / "sample_inputs_mode.pdf"
    _write_pdf(
        sample_pdf,
        [
            "This page is English prose only.\nNo Akkadian content is expected.",
            "sa-ra-am i-na-an e-li-su DUMU\nplain trailing text",
        ],
    )

    proc = _run(
        "tools/run_page_text.py",
        "--inputs",
        str(input_dir),
        "--output-root",
        str(output_dir),
        "--prefer-text-layer",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    output_csv = output_dir / "client_page_text.csv"
    progress_csv = output_dir / "progress.csv"
    assert output_csv.exists()
    assert progress_csv.exists()

    rows = _read_csv(output_csv, encoding="utf-8-sig")
    assert len(rows) == 2
    assert [row["page"] for row in rows] == ["1", "2"]
    assert rows[0]["pdf_name"] == sample_pdf.name
    assert rows[1]["pdf_name"] == sample_pdf.name
    assert rows[0]["has_akkadian"] == "false"
    assert rows[1]["has_akkadian"] == "true"

    progress_rows = _read_csv(progress_csv)
    assert len(progress_rows) == 2
    assert list(progress_rows[0].keys()) == [
        "pdf_name",
        "page",
        "ms",
        "used_text_layer",
        "has_akkadian",
        "timestamp",
    ]


def test_run_page_text_manifest_mode_parses_header_and_comments(tmp_path: Path):
    input_dir = tmp_path / "manifest_inputs"
    output_dir = tmp_path / "manifest_outputs"
    input_dir.mkdir(parents=True)

    sample_pdf = input_dir / "sample_manifest_mode.pdf"
    _write_pdf(sample_pdf, ["Single-page manifest test document."])

    manifest_path = tmp_path / "manifest.tsv"
    manifest_path.write_text(
        "pdf_path\tpage_no\n"
        "# comment line should be ignored\n"
        f"{sample_pdf}\t1\n",
        encoding="utf-8",
    )

    proc = _run(
        "tools/run_page_text.py",
        "--manifest",
        str(manifest_path),
        "--output-root",
        str(output_dir),
        "--prefer-text-layer",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    rows = _read_csv(output_dir / "client_page_text.csv", encoding="utf-8-sig")
    assert len(rows) == 1
    assert rows[0]["pdf_name"] == sample_pdf.name
    assert rows[0]["page"] == "1"


def test_run_pipeline_input_file_mode_writes_expected_page_rows(tmp_path: Path):
    pdf_path = tmp_path / "compat_single_file.pdf"
    output_dir = tmp_path / "compat_outputs"
    _write_pdf(
        pdf_path,
        [
            "Compatibility wrapper page one.",
            "Compatibility wrapper page two.",
        ],
    )

    proc = _run(
        "run_pipeline.py",
        "--input-file",
        str(pdf_path),
        "--output-dir",
        str(output_dir),
        "-c",
        "config.json",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    rows = _read_csv(output_dir / "client_page_text.csv", encoding="utf-8-sig")
    assert len(rows) == 2
    assert [row["page"] for row in rows] == ["1", "2"]
    assert all(row["pdf_name"] == pdf_path.name for row in rows)


def test_run_pipeline_missing_input_file_returns_error(tmp_path: Path):
    missing_pdf = tmp_path / "does_not_exist.pdf"
    output_dir = tmp_path / "unused_output"

    proc = _run(
        "run_pipeline.py",
        "--input-file",
        str(missing_pdf),
        "--output-dir",
        str(output_dir),
        "-c",
        "config.json",
    )

    assert proc.returncode != 0
    assert "Input file not found" in (proc.stdout + proc.stderr)


def test_run_pipeline_dry_run_maps_engine_and_output_dir(tmp_path: Path):
    output_dir = tmp_path / "dry_run_output"

    proc = _run(
        "run_pipeline.py",
        "--input-dir",
        "data/input",
        "--output-dir",
        str(output_dir),
        "--engine",
        "paddleocr",
        "--dry-run",
        "-c",
        "config.json",
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    text = proc.stdout + proc.stderr
    assert "Mapped command:" in text
    assert "--inputs data/input" in text
    assert f"--output-root {output_dir}" in text
    assert "--ocr-fallback paddle" in text


def test_run_pipeline_dry_run_uses_configured_ensemble_by_default(tmp_path: Path):
    output_dir = tmp_path / "ensemble_output"

    proc = _run(
        "run_pipeline.py",
        "--input-dir",
        "data/input",
        "--output-dir",
        str(output_dir),
        "--dry-run",
        "-c",
        "config.json",
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    text = proc.stdout + proc.stderr
    assert "Mapped command:" in text
    assert f"--output-root {output_dir}" in text
    assert "--ocr-fallback ensemble" in text


def test_run_pipeline_validate_only_missing_input_dir_returns_error(tmp_path: Path):
    missing_dir = tmp_path / "missing_input_dir"

    proc = _run(
        "run_pipeline.py",
        "--validate-only",
        "--input-dir",
        str(missing_dir),
        "-c",
        "config.json",
    )

    assert proc.returncode != 0
    assert "Input directory not found" in (proc.stdout + proc.stderr)


def test_build_manifest_expands_ranges_and_writes_rows(tmp_path: Path):
    pdf_root = tmp_path / "pdfs"
    pdf_root.mkdir(parents=True)

    sample_pdf = pdf_root / "sample.pdf"
    _write_pdf(sample_pdf, ["page 1", "page 2", "page 3"])

    csv_path = tmp_path / "gold.csv"
    csv_path.write_text("PDF LINK,PAGE\nsample.pdf,1-3\n", encoding="utf-8")

    manifest_path = tmp_path / "manifest.tsv"
    proc = _run(
        "tools/build_manifest.py",
        "--csv",
        str(csv_path),
        "--pdf-root",
        str(pdf_root),
        "--out",
        str(manifest_path),
        "--expand-ranges",
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert manifest_path.exists()
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    assert lines == [
        f"{sample_pdf.resolve()}\t1",
        f"{sample_pdf.resolve()}\t2",
        f"{sample_pdf.resolve()}\t3",
    ]


def test_build_manifest_strict_missing_pdf_returns_error(tmp_path: Path):
    pdf_root = tmp_path / "pdfs"
    pdf_root.mkdir(parents=True)

    csv_path = tmp_path / "gold.csv"
    csv_path.write_text("PDF LINK,PAGE\nmissing.pdf,4\n", encoding="utf-8")

    manifest_path = tmp_path / "manifest.tsv"
    proc = _run(
        "tools/build_manifest.py",
        "--csv",
        str(csv_path),
        "--pdf-root",
        str(pdf_root),
        "--out",
        str(manifest_path),
        "--strict",
    )

    assert proc.returncode != 0
    text = proc.stdout + proc.stderr
    assert "Missing PDFs referenced in CSV" in text
    assert "missing.pdf" in text
