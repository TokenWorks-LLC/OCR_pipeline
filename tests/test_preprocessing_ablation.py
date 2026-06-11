from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _write_pdf(pdf_path: Path, text: str) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=11)
    doc.save(str(pdf_path))
    doc.close()


def _write_lightweight_profile(profile_path: Path) -> None:
    payload = {
        "rendering": {"dpi": 300},
        "engines": {"enabled": []},
    }
    profile_path.write_text(json.dumps(payload), encoding="utf-8")


def test_preprocessing_ablation_runner_generates_outputs(tmp_path: Path):
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "ablation"
    inputs.mkdir(parents=True)

    pdf_path = inputs / "sample.pdf"
    _write_pdf(pdf_path, "Sample OCR text")

    gold_csv = tmp_path / "gold.csv"
    with gold_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pdf_name", "page", "ground_truth_text"])
        writer.writeheader()
        writer.writerow({"pdf_name": pdf_path.name, "page": "1", "ground_truth_text": "Sample OCR text"})

    profile_path = tmp_path / "light_profile.json"
    _write_lightweight_profile(profile_path)

    proc = subprocess.run(
        [
            PYTHON,
            "tools/run_preprocessing_ablation.py",
            "--inputs",
            str(inputs),
            "--output-dir",
            str(outputs),
            "--profiles",
            "unknown_safe_default",
            "--max-pages",
            "1",
            "--gold-csv",
            str(gold_csv),
            "--profile",
            str(profile_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    per_page = outputs / "preprocessing_ablation_per_page.csv"
    summary = outputs / "preprocessing_ablation_summary.csv"
    by_page_type = outputs / "preprocessing_ablation_by_page_type.csv"
    outputs_jsonl = outputs / "preprocessing_ablation_outputs.jsonl"

    assert per_page.exists()
    assert summary.exists()
    assert by_page_type.exists()
    assert outputs_jsonl.exists()

    with per_page.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["profile_requested"] == "unknown_safe_default"
    assert rows[0]["profile_applied"]
