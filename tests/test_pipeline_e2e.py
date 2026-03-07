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
