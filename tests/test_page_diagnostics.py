"""Unit and integration tests for per-page diagnostics."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from production.page_diagnostics import CV2_AVAILABLE, PageDiagnosticsAnalyzer


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [PYTHON, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=240,
        check=False,
    )


def _write_text_pdf(pdf_path: Path, pages: list[str]) -> None:
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    for page_text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), page_text, fontsize=11)
    doc.save(str(pdf_path))
    doc.close()


def _write_blank_pdf(pdf_path: Path) -> None:
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()


def _write_skew_image_pdf(pdf_path: Path, angle: float = 12.0) -> None:
    fitz = pytest.importorskip("fitz")
    pillow = pytest.importorskip("PIL.Image")
    image_draw = pytest.importorskip("PIL.ImageDraw")

    image = pillow.new("L", (1200, 1600), 255)
    drawer = image_draw.Draw(image)
    for y in range(140, 1480, 70):
        drawer.rectangle((170, y, 1040, y + 9), fill=0)

    rotated = image.rotate(angle, expand=True, fillcolor=255)
    image_path = pdf_path.with_suffix(".png")
    rotated.save(image_path)

    doc = fitz.open()
    page = doc.new_page(width=rotated.width, height=rotated.height)
    page.insert_image(fitz.Rect(0, 0, rotated.width, rotated.height), filename=str(image_path))
    doc.save(str(pdf_path))
    doc.close()


def test_blank_page_detection(tmp_path: Path):
    pdf = tmp_path / "blank.pdf"
    _write_blank_pdf(pdf)

    analyzer = PageDiagnosticsAnalyzer(dpi=300)
    diagnostics, text = analyzer.inspect_page(str(pdf), 0)

    assert text == ""
    assert diagnostics.is_mostly_blank
    assert diagnostics.text_layer_char_count == 0
    assert diagnostics.foreground_ratio <= 0.02


def test_text_layer_acceptance_and_rejection(tmp_path: Path):
    good_pdf = tmp_path / "good.pdf"
    junk_pdf = tmp_path / "junk.pdf"

    _write_text_pdf(good_pdf, ["This is a normal born-digital text layer with meaningful words and spacing."])
    _write_text_pdf(junk_pdf, ["%%%%%%@@@@@@######$$$$$$%%%%%%@@@@@@######$$$$$$"])

    analyzer = PageDiagnosticsAnalyzer(dpi=300)

    good_diag, _ = analyzer.inspect_page(str(good_pdf), 0)
    junk_diag, _ = analyzer.inspect_page(str(junk_pdf), 0)

    assert good_diag.text_layer_usable
    assert good_diag.text_layer_accepted
    assert good_diag.text_layer_rejected_reason == ""
    assert good_diag.text_layer_quality_score >= 0.6
    assert not good_diag.text_layer_suspicious_reasons

    assert not junk_diag.text_layer_usable
    assert not junk_diag.text_layer_accepted
    assert junk_diag.text_layer_rejected_reason
    assert junk_diag.text_layer_quality_score <= 0.6
    assert junk_diag.text_layer_suspicious_reasons


def test_skew_estimation_sanity(tmp_path: Path):
    if not CV2_AVAILABLE:
        pytest.skip("cv2 is required for skew estimation sanity checks")

    pdf = tmp_path / "skewed.pdf"
    _write_skew_image_pdf(pdf, angle=12.0)

    analyzer = PageDiagnosticsAnalyzer(dpi=300)
    diagnostics, _ = analyzer.inspect_page(str(pdf), 0)

    assert diagnostics.estimated_skew_degrees is not None
    assert 2.0 <= abs(float(diagnostics.estimated_skew_degrees)) <= 30.0


def test_born_digital_pdf_detection(tmp_path: Path):
    pdf = tmp_path / "born_digital.pdf"
    _write_text_pdf(pdf, ["Born digital content should produce a usable text layer and no OCR fallback requirement."])

    analyzer = PageDiagnosticsAnalyzer(dpi=300)
    diagnostics, _ = analyzer.inspect_page(str(pdf), 0)

    assert diagnostics.is_born_digital
    assert diagnostics.recommended_ocr_strategy == "text_layer"


def test_diagnostics_output_schema(tmp_path: Path):
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir(parents=True)

    sample_pdf = input_dir / "schema_test.pdf"
    _write_text_pdf(sample_pdf, ["Schema validation page with normal text layer content."])

    proc = _run(
        "tools/run_page_text.py",
        "--inputs",
        str(input_dir),
        "--output-root",
        str(output_dir),
        "--prefer-text-layer",
        "--language-hint",
        "English",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    diagnostics_path = output_dir / "page_diagnostics.jsonl"
    schema_path = ROOT / "config" / "page_diagnostics_schema.json"

    assert diagnostics_path.exists()
    assert schema_path.exists()

    lines = diagnostics_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines
    record = json.loads(lines[0])

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    required = schema.get("required", [])
    for key in required:
        assert key in record, key


def test_pipeline_writes_ensemble_artifacts(tmp_path: Path):
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir(parents=True)

    sample_pdf = input_dir / "ensemble_artifacts.pdf"
    _write_text_pdf(sample_pdf, ["Artifact validation page with stable text output."])

    proc = _run(
        "tools/run_page_text.py",
        "--inputs",
        str(input_dir),
        "--output-root",
        str(output_dir),
        "--prefer-text-layer",
        "--language-hint",
        "English",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    ensemble_jsonl = output_dir / "ensemble_output.jsonl"
    per_engine_jsonl = output_dir / "per_engine_output.jsonl"
    disagreement_json = output_dir / "disagreement_report.json"
    confusion_json = output_dir / "confusion_matrix.json"
    final_output_json = output_dir / "client_page_text.json"
    document_quality_jsonl = output_dir / "document_quality.jsonl"
    run_quality_json = output_dir / "run_quality.json"

    assert ensemble_jsonl.exists() and ensemble_jsonl.stat().st_size > 0
    assert per_engine_jsonl.exists() and per_engine_jsonl.stat().st_size > 0
    assert disagreement_json.exists() and disagreement_json.stat().st_size > 0
    assert confusion_json.exists() and confusion_json.stat().st_size > 0
    assert final_output_json.exists() and final_output_json.stat().st_size > 0
    assert document_quality_jsonl.exists() and document_quality_jsonl.stat().st_size > 0
    assert run_quality_json.exists() and run_quality_json.stat().st_size > 0

    first_ensemble_record = json.loads(ensemble_jsonl.read_text(encoding="utf-8").splitlines()[0])
    assert first_ensemble_record["pdf_name"] == "ensemble_artifacts.pdf"
    assert "metadata" in first_ensemble_record

    first_engine_record = json.loads(per_engine_jsonl.read_text(encoding="utf-8").splitlines()[0])
    assert first_engine_record["pdf_name"] == "ensemble_artifacts.pdf"
    assert "engine" in first_engine_record

    disagreement_payload = json.loads(disagreement_json.read_text(encoding="utf-8"))
    confusion_payload = json.loads(confusion_json.read_text(encoding="utf-8"))
    final_payload = json.loads(final_output_json.read_text(encoding="utf-8"))

    assert "pages_analyzed" in disagreement_payload
    assert "pair_counts" in confusion_payload
    assert isinstance(final_payload, list)
    assert final_payload

    final_record = final_payload[0]
    assert final_record.get("raw_text", "")
    assert final_record.get("cleaned_text", "")
    assert final_record.get("corrected_text", "")
    assert final_record.get("adapter_used", "")
    assert "corrections_applied" in final_record
    assert "lexicon_coverage" in final_record
    assert "unknown_token_rate" in final_record
    assert "protected_character_changes" in final_record
    assert "needs_human_review" in final_record
    assert "page_quality_score" in final_record
    assert "document_quality_score" in final_record
    assert "quality_class" in final_record
    assert "quality_reasons" in final_record
    assert "failed_gate" in final_record
    assert "gate_reason" in final_record

    run_quality_payload = json.loads(run_quality_json.read_text(encoding="utf-8"))
    assert "run_quality" in run_quality_payload
    assert "launch_gate" in run_quality_payload
    assert "quality_thresholds" in run_quality_payload

    layout_jsonl = output_dir / "layout_regions.jsonl"
    layout_record = json.loads(layout_jsonl.read_text(encoding="utf-8").splitlines()[0])
    regions = layout_record.get("layout", {}).get("regions", [])
    assert regions
    assert "postprocessing" in regions[0]
    assert "raw_text" in regions[0]["postprocessing"]
    assert "region_quality_score" in regions[0]
    assert "quality_class" in regions[0]
    assert "needs_review" in regions[0]
