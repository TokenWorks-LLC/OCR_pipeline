"""Tests for multilingual gold-set evaluation metrics and outputs."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = ROOT / "tools" / "evaluate_gold.py"


def _load_evaluator_module():
    spec = importlib.util.spec_from_file_location("evaluate_gold", EVALUATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load evaluator from {EVALUATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evaluate_gold = _load_evaluator_module()


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_basic_error_rates():
    assert evaluate_gold._char_error_rate("abc", "abc") == pytest.approx(0.0)
    assert evaluate_gold._char_error_rate("abc", "axc") == pytest.approx(1.0 / 3.0)

    assert evaluate_gold._word_error_rate("a b c", "a b c") == pytest.approx(0.0)
    assert evaluate_gold._word_error_rate("a b c", "a x c") == pytest.approx(1.0 / 3.0)


def test_normalization_options_reduce_case_and_punctuation_noise():
    cfg = evaluate_gold.NormalizationConfig(
        unicode_form="NFKC",
        whitespace_mode="collapse",
        strip_punctuation=True,
        casefold=True,
    )
    normalized = evaluate_gold._normalize_for_metrics("  Über,   Text! ", cfg)
    assert normalized == "über text"


def test_optional_akkadian_metrics_are_available_but_not_global():
    reference = "a-na LUGAL ša ī [x]"
    hypothesis = "a-na LUGAL sa i x"

    metrics = evaluate_gold._akkadian_optional_metrics(reference, hypothesis)
    assert "akkadian_special_char_preservation_rate" in metrics
    assert "akkadian_macron_preservation_rate" in metrics
    assert metrics["akkadian_hyphenated_sign_preservation_rate"] == pytest.approx(1.0)
    assert metrics["akkadian_special_char_preservation_rate"] < 1.0


def test_document_type_metric_helpers():
    ref = "Title\nkey: value\n1, row"
    hyp = "Title\nkey: value\n1, row"

    assert evaluate_gold._line_order_similarity(ref, hyp) == pytest.approx(1.0)
    assert evaluate_gold._paragraph_order_similarity(ref, hyp) == pytest.approx(1.0)
    assert evaluate_gold._header_footer_mismatch(ref, hyp) is False

    kv_ref = set(evaluate_gold._extract_key_value_pairs(ref))
    kv_hyp = set(evaluate_gold._extract_key_value_pairs(hyp))
    assert evaluate_gold._set_f1(kv_ref, kv_hyp) == pytest.approx(1.0)
    assert evaluate_gold._structured_json_similarity(ref, hyp) == pytest.approx(1.0)


def test_evaluator_generates_required_artifacts(tmp_path: Path):
    gold_csv = tmp_path / "gold.csv"
    ocr_csv = tmp_path / "ocr.csv"
    progress_csv = tmp_path / "progress.csv"
    output_dir = tmp_path / "out"

    _write_csv(
        gold_csv,
        [
            "pdf_name",
            "page",
            "ground_truth_text",
            "dataset_id",
            "language_primary",
            "languages_present",
            "script_type",
            "document_type",
            "layout_type",
            "has_transliteration",
        ],
        [
            {
                "pdf_name": "sample_en.pdf",
                "page": 1,
                "ground_truth_text": "Hello world.",
                "dataset_id": "local_gold",
                "language_primary": "english",
                "languages_present": "english",
                "script_type": "latin",
                "document_type": "article",
                "layout_type": "single_column",
                "has_transliteration": "false",
            },
            {
                "pdf_name": "sample_akk.pdf",
                "page": 1,
                "ground_truth_text": "a-na LUGAL ša ī",
                "dataset_id": "funsd_demo",
                "language_primary": "akkadian",
                "languages_present": "akkadian,english",
                "script_type": "transliteration",
                "document_type": "form",
                "layout_type": "two_column",
                "has_transliteration": "true",
            },
        ],
    )

    _write_csv(
        ocr_csv,
        [
            "pdf_name",
            "page",
            "page_text",
            "status",
            "failure_reason",
            "runtime_ms",
            "confidence",
            "extraction_method",
            "engine_statuses",
            "detected_orientation_angle",
            "detected_orientation_class",
            "detected_rotation_base_angle",
            "detected_skew_angle",
            "detected_layout_type",
            "detected_column_count",
            "detected_has_columns",
        ],
        [
            {
                "pdf_name": "sample_en.pdf",
                "page": 1,
                "page_text": "Hello world.",
                "status": "success",
                "failure_reason": "",
                "runtime_ms": 25,
                "confidence": 0.95,
                "extraction_method": "text_layer",
                "engine_statuses": "{}",
                "detected_orientation_angle": 0,
                "detected_orientation_class": "upright",
                "detected_rotation_base_angle": 0,
                "detected_skew_angle": 0,
                "detected_layout_type": "single_column",
                "detected_column_count": 1,
                "detected_has_columns": "false",
            },
            {
                "pdf_name": "sample_akk.pdf",
                "page": 1,
                "page_text": "",
                "status": "timed_out",
                "failure_reason": "page_timeout_1000ms",
                "runtime_ms": 1000,
                "confidence": "",
                "extraction_method": "ocr_ensemble",
                "engine_statuses": '{"paddle": {"status": "timed_out", "reason": "inference timeout"}}',
                "detected_orientation_angle": 88.6,
                "detected_orientation_class": "rotated_90_cw_skewed",
                "detected_rotation_base_angle": 90,
                "detected_skew_angle": -1.4,
                "detected_layout_type": "multi_column",
                "detected_column_count": 2,
                "detected_has_columns": "true",
            },
        ],
    )

    _write_csv(
        progress_csv,
        [
            "pdf_name",
            "page",
            "ms",
            "status",
            "failure_reason",
            "detected_orientation_angle",
            "detected_orientation_class",
            "detected_layout_type",
            "detected_column_count",
            "detected_has_columns",
        ],
        [
            {
                "pdf_name": "sample_en.pdf",
                "page": 1,
                "ms": 25,
                "status": "success",
                "failure_reason": "",
                "detected_orientation_angle": 0,
                "detected_orientation_class": "upright",
                "detected_layout_type": "single_column",
                "detected_column_count": 1,
                "detected_has_columns": "false",
            },
            {
                "pdf_name": "sample_akk.pdf",
                "page": 1,
                "ms": 1000,
                "status": "timed_out",
                "failure_reason": "page_timeout_1000ms",
                "detected_orientation_angle": 88.6,
                "detected_orientation_class": "rotated_90_cw_skewed",
                "detected_layout_type": "multi_column",
                "detected_column_count": 2,
                "detected_has_columns": "true",
            },
        ],
    )

    exit_code = evaluate_gold.main(
        [
            "--ocr-csv",
            str(ocr_csv),
            "--gold-csv",
            str(gold_csv),
            "--progress-csv",
            str(progress_csv),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert exit_code == 0

    required_files = [
        "evaluation_summary.csv",
        "per_page_metrics.csv",
        "per_engine_metrics.csv",
        "per_language_metrics.csv",
        "per_layout_metrics.csv",
        "per_detected_orientation_metrics.csv",
        "per_detected_layout_metrics.csv",
        "confusion_matrix.json",
        "run_metadata.json",
    ]
    for filename in required_files:
        assert (output_dir / filename).exists(), filename

    summary_rows = _read_csv(output_dir / "evaluation_summary.csv")
    assert len(summary_rows) == 1
    summary = summary_rows[0]
    assert float(summary["empty_output_rate"]) == pytest.approx(0.5)
    assert float(summary["timeout_rate"]) == pytest.approx(0.5)

    per_page = _read_csv(output_dir / "per_page_metrics.csv")
    assert len(per_page) == 2
    # Optional Akkadian metrics should be present in schema and computed where applicable.
    assert "akkadian_special_char_preservation_rate" in per_page[0]
    assert "metric_family" in per_page[0]
    assert "line_order_similarity" in per_page[0]
    assert "key_value_f1" in per_page[0]
    assert "target_scope_valid" in per_page[0]
    assert "original_page_key" in per_page[0]
    assert "normalized_page_key" in per_page[0]
    assert "original_document_key" in per_page[0]
    assert "normalized_document_key" in per_page[0]
    assert "key_normalization_applied" in per_page[0]
    assert "key_normalization_warnings" in per_page[0]
    assert "detected_orientation_class" in per_page[0]
    assert "detected_has_columns" in per_page[0]
    assert any(row["detected_orientation_class"] == "rotated_90_cw_skewed" for row in per_page)
    assert any(row["detected_has_columns"] == "true" for row in per_page)
    assert any(row["metric_family"] == "local_scope_validated" for row in per_page)
    assert any(row["metric_family"] == "form_structured_primary" for row in per_page)


def test_baseline_comparison_reports_improvements(tmp_path: Path):
    gold_csv = tmp_path / "gold.csv"
    baseline_ocr_csv = tmp_path / "baseline_ocr.csv"
    current_ocr_csv = tmp_path / "current_ocr.csv"
    baseline_dir = tmp_path / "baseline"
    current_dir = tmp_path / "current"

    _write_csv(
        gold_csv,
        [
            "pdf_name",
            "page",
            "ground_truth_text",
            "language_primary",
            "languages_present",
            "script_type",
            "document_type",
            "layout_type",
        ],
        [
            {
                "pdf_name": "sample_fr.pdf",
                "page": 1,
                "ground_truth_text": "Bonjour à tous.",
                "language_primary": "french",
                "languages_present": "french",
                "script_type": "latin",
                "document_type": "article",
                "layout_type": "single_column",
            },
            {
                "pdf_name": "sample_ar.pdf",
                "page": 1,
                "ground_truth_text": "هذا نص عربي",
                "language_primary": "arabic",
                "languages_present": "arabic",
                "script_type": "arabic",
                "document_type": "article",
                "layout_type": "single_column",
            },
        ],
    )

    _write_csv(
        baseline_ocr_csv,
        [
            "pdf_name",
            "page",
            "page_text",
            "status",
            "runtime_ms",
            "failure_reason",
            "extraction_method",
        ],
        [
            {
                "pdf_name": "sample_fr.pdf",
                "page": 1,
                "page_text": "Bonjou a tous",
                "status": "success",
                "runtime_ms": 50,
                "failure_reason": "",
                "extraction_method": "text_layer",
            },
            {
                "pdf_name": "sample_ar.pdf",
                "page": 1,
                "page_text": "",
                "status": "failed",
                "runtime_ms": 120,
                "failure_reason": "empty",
                "extraction_method": "failed",
            },
        ],
    )

    _write_csv(
        current_ocr_csv,
        [
            "pdf_name",
            "page",
            "page_text",
            "status",
            "runtime_ms",
            "failure_reason",
            "extraction_method",
        ],
        [
            {
                "pdf_name": "sample_fr.pdf",
                "page": 1,
                "page_text": "Bonjour à tous.",
                "status": "success",
                "runtime_ms": 45,
                "failure_reason": "",
                "extraction_method": "text_layer",
            },
            {
                "pdf_name": "sample_ar.pdf",
                "page": 1,
                "page_text": "هذا نص عربي",
                "status": "success",
                "runtime_ms": 100,
                "failure_reason": "",
                "extraction_method": "text_layer",
            },
        ],
    )

    assert (
        evaluate_gold.main(
            [
                "--ocr-csv",
                str(baseline_ocr_csv),
                "--gold-csv",
                str(gold_csv),
                "--output-dir",
                str(baseline_dir),
            ]
        )
        == 0
    )

    assert (
        evaluate_gold.main(
            [
                "--ocr-csv",
                str(current_ocr_csv),
                "--gold-csv",
                str(gold_csv),
                "--output-dir",
                str(current_dir),
                "--baseline-dir",
                str(baseline_dir),
            ]
        )
        == 0
    )

    with (current_dir / "run_metadata.json").open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)

    baseline = metadata.get("baseline_comparison")
    assert baseline is not None
    assert baseline.get("improved_metrics")
    assert baseline.get("improved_pages") is not None


def test_malformed_rows_fail_fast_by_default(tmp_path: Path):
    gold_csv = tmp_path / "gold_malformed.csv"
    ocr_csv = tmp_path / "ocr_valid.csv"
    output_dir = tmp_path / "out"

    _write_csv(
        gold_csv,
        ["PDF LINK", "PAGE", "HANDTYPED"],
        [
            {"PDF LINK": "", "PAGE": "2", "HANDTYPED": "malformed gold row should fail"},
            {"PDF LINK": "valid_doc.pdf", "PAGE": "1", "HANDTYPED": "valid row"},
        ],
    )

    _write_csv(
        ocr_csv,
        ["pdf_name", "page", "page_text", "status", "failure_reason", "runtime_ms", "extraction_method"],
        [
            {
                "pdf_name": "valid_doc.pdf",
                "page": "1",
                "page_text": "valid row",
                "status": "success",
                "failure_reason": "",
                "runtime_ms": "10",
                "extraction_method": "text_layer",
            }
        ],
    )

    exit_code = evaluate_gold.main(
        [
            "--ocr-csv",
            str(ocr_csv),
            "--gold-csv",
            str(gold_csv),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert exit_code == 2
    assert (output_dir / "malformed_rows.csv").exists()


def test_malformed_rows_can_be_skipped_in_permissive_mode(tmp_path: Path):
    gold_csv = tmp_path / "gold_malformed.csv"
    ocr_csv = tmp_path / "ocr_valid.csv"
    output_dir = tmp_path / "out"

    _write_csv(
        gold_csv,
        ["PDF LINK", "PAGE", "HANDTYPED"],
        [
            {"PDF LINK": "", "PAGE": "2", "HANDTYPED": "malformed gold row skipped"},
            {"PDF LINK": "valid_doc.pdf", "PAGE": "1", "HANDTYPED": "valid row"},
        ],
    )

    _write_csv(
        ocr_csv,
        ["pdf_name", "page", "page_text", "status", "failure_reason", "runtime_ms", "extraction_method"],
        [
            {
                "pdf_name": "valid_doc.pdf",
                "page": "1",
                "page_text": "valid row",
                "status": "success",
                "failure_reason": "",
                "runtime_ms": "10",
                "extraction_method": "text_layer",
            }
        ],
    )

    exit_code = evaluate_gold.main(
        [
            "--ocr-csv",
            str(ocr_csv),
            "--gold-csv",
            str(gold_csv),
            "--output-dir",
            str(output_dir),
            "--permissive-malformed",
        ]
    )
    assert exit_code == 0

    per_page = _read_csv(output_dir / "per_page_metrics.csv")
    assert len(per_page) == 1
    assert per_page[0]["page_key"] == "valid_doc_page_1"
    assert (output_dir / "malformed_rows.csv").exists()


def test_key_normalization_matches_unicode_and_path_variants(tmp_path: Path):
    gold_csv = tmp_path / "gold.csv"
    ocr_csv = tmp_path / "ocr.csv"
    output_dir = tmp_path / "out"

    _write_csv(
        gold_csv,
        ["pdf_name", "page", "ground_truth_text", "language_primary", "layout_type"],
        [
            {
                "pdf_name": "Docs/Attinger - ÁK.pdf",
                "page": "1",
                "ground_truth_text": "Normalized key matching text.",
                "language_primary": "english",
                "layout_type": "single_column",
            }
        ],
    )

    _write_csv(
        ocr_csv,
        ["pdf_name", "page", "page_text", "status", "failure_reason", "runtime_ms", "extraction_method", "page_key"],
        [
            {
                "pdf_name": "docs\\attinger - a\u0301k.pdf",
                "page": "1",
                "page_text": "Normalized key matching text.",
                "status": "success",
                "failure_reason": "",
                "runtime_ms": "10",
                "extraction_method": "text_layer",
                "page_key": "docs\\Attinger - A\u0301K_page_1",
            }
        ],
    )

    exit_code = evaluate_gold.main(
        [
            "--ocr-csv",
            str(ocr_csv),
            "--gold-csv",
            str(gold_csv),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert exit_code == 0

    per_page = _read_csv(output_dir / "per_page_metrics.csv")
    assert len(per_page) == 1

    row = per_page[0]
    assert row["page_key"] == "attinger_ak_page_1"
    assert row["normalized_page_key"] == "attinger_ak_page_1"
    assert row["normalized_document_key"] == "attinger_ak"
    assert row["key_normalization_applied"].strip().lower() == "true"
