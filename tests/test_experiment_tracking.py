"""Tests for experiment tracking dashboard and regression comparison."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from production.experiment_tracking import build_tracking_dashboard, compare_tracking_runs


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


def _sample_per_page_rows() -> list[dict[str, object]]:
    return [
        {
            "page_key": "doc_a_page_1",
            "pdf_name": "doc_a.pdf",
            "page": 1,
            "status": "success",
            "failure_reason": "",
            "extraction_method": "ocr_paddle",
            "runtime_ms": 120.0,
            "cer": 0.12,
            "wer": 0.22,
            "normalized_cer": 0.12,
            "normalized_wer": 0.22,
            "empty_output": False,
            "timed_out": False,
            "language_primary": "english",
            "languages_present": "english",
            "script_type": "latin",
            "document_type": "article",
            "layout_type": "single_column",
            "scan_quality": "high",
            "expected_difficulty": "easy",
            "page_quality_score": 0.89,
            "quality_class": "production_quality",
            "diacritic_preservation_rate": 1.0,
        },
        {
            "page_key": "doc_b_page_2",
            "pdf_name": "doc_b.pdf",
            "page": 2,
            "status": "timed_out",
            "failure_reason": "page_timeout_1000ms",
            "extraction_method": "ocr_ensemble",
            "runtime_ms": 1000.0,
            "cer": 0.42,
            "wer": 0.56,
            "normalized_cer": 0.42,
            "normalized_wer": 0.56,
            "empty_output": True,
            "timed_out": True,
            "language_primary": "akkadian",
            "languages_present": "akkadian|english",
            "script_type": "transliteration",
            "document_type": "edition",
            "layout_type": "two_column",
            "scan_quality": "noisy",
            "expected_difficulty": "hard",
            "page_quality_score": 0.41,
            "quality_class": "failed_ocr",
            "diacritic_preservation_rate": 0.55,
        },
    ]


def _summary_row() -> dict[str, object]:
    return {
        "matched_pages": 2,
        "cer_mean": 0.27,
        "wer_mean": 0.39,
        "empty_output_rate": 0.5,
        "timeout_rate": 0.5,
        "failed_page_rate": 0.0,
        "runtime_ms_mean": 560.0,
        "runtime_ms_p90": 912.0,
    }


def test_run_metadata_writing_and_dashboard_outputs(tmp_path: Path):
    output_dir = tmp_path / "current"
    ocr_csv = tmp_path / "ocr.csv"
    gold_csv = tmp_path / "gold.csv"

    _write_csv(
        ocr_csv,
        [
            "pdf_name",
            "page",
            "page_text",
            "recommended_preprocessing_profile",
            "applied_preprocessing_profile",
            "adapter_used",
        ],
        [
            {
                "pdf_name": "doc_a.pdf",
                "page": 1,
                "page_text": "hello",
                "recommended_preprocessing_profile": "clean_scan",
                "applied_preprocessing_profile": "clean_scan",
                "adapter_used": "english",
            },
            {
                "pdf_name": "doc_b.pdf",
                "page": 2,
                "page_text": "a-na LUGAL",
                "recommended_preprocessing_profile": "noisy_scan",
                "applied_preprocessing_profile": "noisy_scan",
                "adapter_used": "akkadian_transliteration",
            },
        ],
    )

    _write_csv(
        gold_csv,
        ["pdf_name", "page", "ground_truth_text", "language_primary", "script_type", "document_type", "layout_type"],
        [
            {
                "pdf_name": "doc_a.pdf",
                "page": 1,
                "ground_truth_text": "hello",
                "language_primary": "english",
                "script_type": "latin",
                "document_type": "article",
                "layout_type": "single_column",
            }
        ],
    )

    outputs = build_tracking_dashboard(
        output_dir=output_dir,
        per_page_rows=_sample_per_page_rows(),
        evaluation_summary_row=_summary_row(),
        ocr_csv_path=ocr_csv,
        gold_csv_path=gold_csv,
        run_id="run-test-001",
        config_file="config_eval_advanced_v3.json",
        engine_versions_arg='{"paddleocr": "2.7.0"}',
        model_versions_arg='{"kraken": "en_best.mlmodel"}',
        gold_set_version="gold-v1",
    )

    for key in [
        "metrics_summary_md",
        "metrics_summary_csv",
        "metrics_by_language_csv",
        "metrics_by_layout_csv",
        "metrics_by_engine_csv",
        "regression_report_md",
        "failing_pages_csv",
        "experiment_run_metadata_json",
        "experiment_tracking_json",
    ]:
        assert Path(outputs[key]).exists(), key

    metadata = json.loads((output_dir / "experiment_run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["run_id"] == "run-test-001"
    assert metadata["config_file"] == "config_eval_advanced_v3.json"
    assert metadata["gold_set_version"] == "gold-v1"
    assert "git_commit" in metadata
    assert sorted(metadata["preprocessing_profiles"]) == ["clean_scan", "noisy_scan"]
    assert sorted(metadata["postprocessing_adapters"]) == ["akkadian_transliteration", "english"]


def test_grouping_outputs_include_language_layout_and_engine(tmp_path: Path):
    output_dir = tmp_path / "grouped"
    outputs = build_tracking_dashboard(
        output_dir=output_dir,
        per_page_rows=_sample_per_page_rows(),
        evaluation_summary_row=_summary_row(),
        ocr_csv_path=None,
        gold_csv_path=None,
    )

    language_rows = _read_csv(Path(outputs["metrics_by_language_csv"]))
    layout_rows = _read_csv(Path(outputs["metrics_by_layout_csv"]))
    engine_rows = _read_csv(Path(outputs["metrics_by_engine_csv"]))

    languages = {row["language_primary"] for row in language_rows}
    layouts = {row["layout_type"] for row in layout_rows}
    engines = {row["engine"] for row in engine_rows}

    assert {"english", "akkadian"}.issubset(languages)
    assert {"single_column", "two_column"}.issubset(layouts)
    assert {"paddle", "ensemble"}.issubset(engines)


def test_baseline_comparison_and_regression_report_generation(tmp_path: Path):
    baseline_dir = tmp_path / "baseline"
    current_dir = tmp_path / "current"

    baseline_rows = _sample_per_page_rows()
    baseline_rows[0]["cer"] = 0.22
    baseline_rows[0]["wer"] = 0.30
    baseline_rows[0]["normalized_cer"] = 0.22
    baseline_rows[0]["normalized_wer"] = 0.30
    baseline_rows[1]["cer"] = 0.30
    baseline_rows[1]["wer"] = 0.40
    baseline_rows[1]["normalized_cer"] = 0.30
    baseline_rows[1]["normalized_wer"] = 0.40

    current_rows = _sample_per_page_rows()

    build_tracking_dashboard(
        output_dir=baseline_dir,
        per_page_rows=baseline_rows,
        evaluation_summary_row=_summary_row(),
        ocr_csv_path=None,
        gold_csv_path=None,
    )
    _write_csv(
        baseline_dir / "per_page_metrics.csv",
        list(baseline_rows[0].keys()),
        baseline_rows,
    )

    build_tracking_dashboard(
        output_dir=current_dir,
        per_page_rows=current_rows,
        evaluation_summary_row=_summary_row(),
        ocr_csv_path=None,
        gold_csv_path=None,
        baseline_dir=baseline_dir,
    )
    _write_csv(
        current_dir / "per_page_metrics.csv",
        list(current_rows[0].keys()),
        current_rows,
    )

    outputs = compare_tracking_runs(
        current_dir=current_dir,
        baseline_dir=baseline_dir,
        output_dir=current_dir,
        top_n=5,
    )

    regression_text = Path(outputs["regression_report_md"]).read_text(encoding="utf-8")
    assert "Overall Improvement/Regression" in regression_text
    assert "Pages That Regressed Most" in regression_text

    failing_rows = _read_csv(Path(outputs["failing_pages_csv"]))
    assert failing_rows
    assert any(row["page_key"] == "doc_b_page_2" for row in failing_rows)


def test_missing_metric_handling_does_not_crash(tmp_path: Path):
    output_dir = tmp_path / "missing"
    rows = [
        {
            "page_key": "missing_page_1",
            "pdf_name": "missing.pdf",
            "page": 1,
            "status": "success",
            "extraction_method": "text_layer",
            "language_primary": "unknown",
            "layout_type": "unknown",
        }
    ]

    outputs = build_tracking_dashboard(
        output_dir=output_dir,
        per_page_rows=rows,
        evaluation_summary_row={},
        ocr_csv_path=None,
        gold_csv_path=None,
    )

    summary_rows = _read_csv(Path(outputs["metrics_summary_csv"]))
    assert len(summary_rows) == 1
    assert summary_rows[0]["pages"] == "1"
    assert Path(outputs["regression_report_md"]).exists()


def test_overall_regression_prioritizes_quality_over_runtime(tmp_path: Path):
    baseline_dir = tmp_path / "baseline_weighted"
    current_dir = tmp_path / "current_weighted"

    baseline_rows = _sample_per_page_rows()
    baseline_rows[0]["cer"] = 0.10
    baseline_rows[0]["wer"] = 0.18
    baseline_rows[0]["normalized_cer"] = 0.10
    baseline_rows[0]["normalized_wer"] = 0.18
    baseline_rows[0]["runtime_ms"] = 5000.0
    baseline_rows[1]["cer"] = 0.20
    baseline_rows[1]["wer"] = 0.28
    baseline_rows[1]["normalized_cer"] = 0.20
    baseline_rows[1]["normalized_wer"] = 0.28
    baseline_rows[1]["runtime_ms"] = 6000.0
    baseline_rows[1]["empty_output"] = False
    baseline_rows[1]["status"] = "success"

    current_rows = _sample_per_page_rows()
    current_rows[0]["cer"] = 0.60
    current_rows[0]["wer"] = 0.78
    current_rows[0]["normalized_cer"] = 0.60
    current_rows[0]["normalized_wer"] = 0.78
    current_rows[0]["runtime_ms"] = 80.0
    current_rows[1]["cer"] = 0.70
    current_rows[1]["wer"] = 0.88
    current_rows[1]["normalized_cer"] = 0.70
    current_rows[1]["normalized_wer"] = 0.88
    current_rows[1]["runtime_ms"] = 90.0
    current_rows[1]["empty_output"] = False
    current_rows[1]["status"] = "success"

    build_tracking_dashboard(
        output_dir=baseline_dir,
        per_page_rows=baseline_rows,
        evaluation_summary_row=_summary_row(),
        ocr_csv_path=None,
        gold_csv_path=None,
    )
    _write_csv(
        baseline_dir / "per_page_metrics.csv",
        list(baseline_rows[0].keys()),
        baseline_rows,
    )

    build_tracking_dashboard(
        output_dir=current_dir,
        per_page_rows=current_rows,
        evaluation_summary_row=_summary_row(),
        ocr_csv_path=None,
        gold_csv_path=None,
        baseline_dir=baseline_dir,
    )
    _write_csv(
        current_dir / "per_page_metrics.csv",
        list(current_rows[0].keys()),
        current_rows,
    )

    outputs = compare_tracking_runs(
        current_dir=current_dir,
        baseline_dir=baseline_dir,
        output_dir=current_dir,
        top_n=3,
    )

    payload = json.loads(Path(outputs["regression_comparison_json"]).read_text(encoding="utf-8"))
    assert payload["overall"]["overall"] == "faster_but_regressed"
    assert payload["overall"]["quality_status"] == "regressed"
    assert payload["overall"]["runtime_status"] == "improved"


def test_compare_tracking_runs_matches_pages_with_normalized_keys(tmp_path: Path):
    baseline_dir = tmp_path / "baseline_norm"
    current_dir = tmp_path / "current_norm"

    baseline_rows = [
        {
            "page_key": "Dóc-B\\page_1",
            "pdf_name": "docs/Dóc-B.pdf",
            "page": 1,
            "status": "success",
            "failure_reason": "",
            "extraction_method": "ocr_paddle",
            "runtime_ms": 100,
            "cer": 0.10,
            "wer": 0.15,
            "normalized_cer": 0.10,
            "normalized_wer": 0.15,
            "language_primary": "english",
            "layout_type": "single_column",
            "empty_output": False,
            "timed_out": False,
        }
    ]
    current_rows = [
        {
            "page_key": "doc_b_page_1",
            "pdf_name": "DOCS/doc_b.pdf",
            "page": 1,
            "status": "failed",
            "failure_reason": "empty",
            "extraction_method": "ocr_paddle",
            "runtime_ms": 80,
            "cer": 0.60,
            "wer": 0.70,
            "normalized_cer": 0.60,
            "normalized_wer": 0.70,
            "language_primary": "english",
            "layout_type": "single_column",
            "empty_output": True,
            "timed_out": False,
        }
    ]

    _write_csv(baseline_dir / "per_page_metrics.csv", list(baseline_rows[0].keys()), baseline_rows)
    _write_csv(current_dir / "per_page_metrics.csv", list(current_rows[0].keys()), current_rows)
    _write_csv(
        baseline_dir / "metrics_summary.csv",
        ["matched_pages", "cer_mean", "wer_mean", "failed_page_rate", "runtime_ms_mean"],
        [{"matched_pages": 1, "cer_mean": 0.10, "wer_mean": 0.15, "failed_page_rate": 0.0, "runtime_ms_mean": 100}],
    )
    _write_csv(
        current_dir / "metrics_summary.csv",
        ["matched_pages", "cer_mean", "wer_mean", "failed_page_rate", "runtime_ms_mean"],
        [{"matched_pages": 1, "cer_mean": 0.60, "wer_mean": 0.70, "failed_page_rate": 1.0, "runtime_ms_mean": 80}],
    )

    outputs = compare_tracking_runs(
        current_dir=current_dir,
        baseline_dir=baseline_dir,
        output_dir=current_dir,
        top_n=3,
    )

    payload = json.loads(Path(outputs["regression_comparison_json"]).read_text(encoding="utf-8"))
    regressed = payload["pages"]["regressed_pages"]
    assert regressed
    assert regressed[0]["normalized_page_key"] == "doc_b_page_1"


def test_release_recommendation_uses_policy_status_labels(tmp_path: Path):
    baseline_dir = tmp_path / "baseline_policy"
    paddle_dir = tmp_path / "paddle_policy"
    ensemble_dir = tmp_path / "ensemble_policy"

    baseline_rows = [
        {
            "page_key": "doc_page_1",
            "pdf_name": "doc.pdf",
            "page": 1,
            "status": "success",
            "failure_reason": "",
            "extraction_method": "ocr_paddle",
            "runtime_ms": 100,
            "cer": 0.40,
            "wer": 0.50,
            "normalized_cer": 0.40,
            "normalized_wer": 0.50,
            "empty_output": False,
            "timed_out": False,
            "language_primary": "english",
            "layout_type": "single_column",
        }
    ]

    paddle_rows = [
        {
            **baseline_rows[0],
            "runtime_ms": 240,
            "cer": 0.20,
            "wer": 0.30,
            "normalized_cer": 0.20,
            "normalized_wer": 0.30,
            "status": "success",
        }
    ]

    ensemble_rows = [
        {
            **baseline_rows[0],
            "runtime_ms": 60,
            "cer": 0.70,
            "wer": 0.85,
            "normalized_cer": 0.70,
            "normalized_wer": 0.85,
            "status": "success",
        }
    ]

    _write_csv(baseline_dir / "per_page_metrics.csv", list(baseline_rows[0].keys()), baseline_rows)
    _write_csv(paddle_dir / "per_page_metrics.csv", list(paddle_rows[0].keys()), paddle_rows)
    _write_csv(ensemble_dir / "per_page_metrics.csv", list(ensemble_rows[0].keys()), ensemble_rows)

    baseline_summary = {"matched_pages": 1, "cer_mean": 0.40, "wer_mean": 0.50, "runtime_ms_mean": 100, "failed_page_rate": 0.0, "timeout_rate": 0.0, "empty_output_rate": 0.0}
    paddle_summary = {"matched_pages": 1, "cer_mean": 0.20, "wer_mean": 0.30, "runtime_ms_mean": 240, "failed_page_rate": 0.0, "timeout_rate": 0.0, "empty_output_rate": 0.0}
    ensemble_summary = {"matched_pages": 1, "cer_mean": 0.70, "wer_mean": 0.85, "runtime_ms_mean": 60, "failed_page_rate": 0.0, "timeout_rate": 0.0, "empty_output_rate": 0.0}

    build_tracking_dashboard(
        output_dir=baseline_dir,
        per_page_rows=baseline_rows,
        evaluation_summary_row=baseline_summary,
        ocr_csv_path=None,
        gold_csv_path=None,
        run_id="fast_v2_baseline",
    )

    build_tracking_dashboard(
        output_dir=paddle_dir,
        per_page_rows=paddle_rows,
        evaluation_summary_row=paddle_summary,
        ocr_csv_path=None,
        gold_csv_path=None,
        baseline_dir=baseline_dir,
        run_id="two_pass_controlled_paddle",
    )

    build_tracking_dashboard(
        output_dir=ensemble_dir,
        per_page_rows=ensemble_rows,
        evaluation_summary_row=ensemble_summary,
        ocr_csv_path=None,
        gold_csv_path=None,
        baseline_dir=baseline_dir,
        run_id="two_pass_controlled_ensemble",
    )

    paddle_summary_rows = _read_csv(paddle_dir / "metrics_summary.csv")
    ensemble_summary_rows = _read_csv(ensemble_dir / "metrics_summary.csv")

    assert paddle_summary_rows[0]["production_recommendation_status"] == "use_two_pass_paddle_default"
    assert ensemble_summary_rows[0]["production_recommendation_status"] == "reject_ensemble_due_to_quality_regression"
