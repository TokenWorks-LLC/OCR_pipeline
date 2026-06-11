from __future__ import annotations

import json
from pathlib import Path

import run_pipeline as rp


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4\n% mocked\n")


def test_collect_rerun_candidates_no_failed_pages_no_candidates(tmp_path: Path):
    _touch(tmp_path / "doc_ok.pdf")
    rows = [
        {
            "pdf_name": "doc_ok.pdf",
            "page": "1",
            "status": "success",
            "page_text": "This is valid extracted content.",
            "input_file": "doc_ok.pdf",
        }
    ]

    candidates, reason_map = rp._collect_rerun_candidates(
        rows,
        input_dir=str(tmp_path),
        rerun_failed_pages=True,
        fallback_on_empty=False,
        fallback_on_low_quality=False,
        low_text_threshold=24,
        low_confidence_threshold=0.15,
    )

    assert candidates == []
    assert reason_map == {}


def test_collect_rerun_candidates_empty_output_triggers_rerun(tmp_path: Path):
    _touch(tmp_path / "doc_empty.pdf")
    rows = [
        {
            "pdf_name": "doc_empty.pdf",
            "page": "3",
            "status": "success",
            "page_text": "",
            "input_file": "doc_empty.pdf",
        }
    ]

    candidates, reason_map = rp._collect_rerun_candidates(
        rows,
        input_dir=str(tmp_path),
        rerun_failed_pages=False,
        fallback_on_empty=True,
        fallback_on_low_quality=False,
        low_text_threshold=24,
        low_confidence_threshold=0.15,
    )

    assert len(candidates) == 1
    assert candidates[0]["pdf_name"] == "doc_empty.pdf"
    assert candidates[0]["page"] == 3
    assert "empty_output" in candidates[0]["reasons"]
    assert candidates[0]["rerun_candidate_class"] == "empty_output"
    assert int(candidates[0]["rerun_candidate_score"]) > 0
    assert reason_map[("doc_empty.pdf", 3)] == sorted(set(candidates[0]["reasons"]))


def test_collect_rerun_candidates_failed_status_triggers_rerun(tmp_path: Path):
    _touch(tmp_path / "doc_failed.pdf")
    rows = [
        {
            "pdf_name": "doc_failed.pdf",
            "page": "9",
            "status": "failed",
            "page_text": "",
            "input_file": "doc_failed.pdf",
        }
    ]

    candidates, reason_map = rp._collect_rerun_candidates(
        rows,
        input_dir=str(tmp_path),
        rerun_failed_pages=True,
        fallback_on_empty=False,
        fallback_on_low_quality=False,
        low_text_threshold=24,
        low_confidence_threshold=0.15,
    )

    assert len(candidates) == 1
    assert candidates[0]["pdf_name"] == "doc_failed.pdf"
    assert candidates[0]["page"] == 9
    assert "status:failed" in candidates[0]["reasons"]
    assert candidates[0]["rerun_candidate_class"] == "critical_failed"
    assert reason_map[("doc_failed.pdf", 9)] == sorted(set(candidates[0]["reasons"]))


def test_merge_two_pass_rows_prefers_better_second_pass_result():
    pass1_rows = [
        {
            "pdf_name": "doc.pdf",
            "page": "1",
            "status": "failed",
            "page_text": "",
            "runtime_ms": "120",
            "extraction_method": "text_layer",
        }
    ]
    pass2_rows = [
        {
            "pdf_name": "doc.pdf",
            "page": "1",
            "status": "success",
            "page_text": "Recovered text from fallback OCR.",
            "runtime_ms": "410",
            "extraction_method": "ocr",
        }
    ]

    merged, stats = rp._merge_two_pass_rows(
        pass1_rows,
        pass2_rows,
        {("doc.pdf", 1): ["status:failed"]},
        fallback_engine="ensemble",
    )

    assert len(merged) == 1
    row = merged[0]
    assert row["status"] == "success"
    assert row["final_output_source"] == "second_pass_ensemble"
    assert row["first_pass_status"] == "failed"
    assert row["second_pass_status"] == "success"
    assert row["final_status"] == "success"
    assert row["fallback_reason"] == "status:failed"
    assert row["fallback_engine"] == "ensemble"
    assert row["pass_number"] == "2"
    assert stats["fallback_selected_pages"] == 1
    assert stats["still_failing_after_fallback"] == 0


def test_merge_two_pass_rows_reports_still_failing_after_second_pass():
    pass1_rows = [
        {
            "pdf_name": "doc.pdf",
            "page": "2",
            "status": "failed",
            "page_text": "",
            "runtime_ms": "80",
            "extraction_method": "text_layer",
        }
    ]
    pass2_rows = [
        {
            "pdf_name": "doc.pdf",
            "page": "2",
            "status": "failed",
            "page_text": "",
            "runtime_ms": "330",
            "extraction_method": "ocr",
        }
    ]

    merged, stats = rp._merge_two_pass_rows(
        pass1_rows,
        pass2_rows,
        {("doc.pdf", 2): ["status:failed", "empty_output"]},
        fallback_engine="paddle",
    )

    assert len(merged) == 1
    row = merged[0]
    assert row["status"] == "failed"
    assert row["final_output_source"] == "first_pass_kept_due_to_failed_fallback"
    assert row["first_pass_status"] == "failed"
    assert row["second_pass_status"] == "failed"
    assert row["final_status"] == "failed"
    assert row["fallback_rejected_reason"] == "second_pass_failed_or_empty"
    assert row["fallback_reason"] in {
        "empty_output|status:failed",
        "status:failed|empty_output",
    }
    assert row["fallback_engine"] == "paddle"
    assert row["pass_number"] == "1"
    assert stats["fallback_selected_pages"] == 0
    assert stats["still_failing_after_fallback"] == 1


def test_merge_two_pass_rows_ignores_non_candidate_second_pass_rows():
    pass1_rows = [
        {
            "pdf_name": "doc.pdf",
            "page": "1",
            "status": "failed",
            "page_text": "",
            "runtime_ms": "100",
            "extraction_method": "text_layer",
        },
        {
            "pdf_name": "doc.pdf",
            "page": "2",
            "status": "success",
            "page_text": "Stable page text.",
            "runtime_ms": "70",
            "extraction_method": "text_layer",
        },
    ]
    pass2_rows = [
        {
            "pdf_name": "doc.pdf",
            "page": "1",
            "status": "success",
            "page_text": "Recovered candidate page.",
            "runtime_ms": "300",
            "extraction_method": "ocr",
        },
        {
            "pdf_name": "doc.pdf",
            "page": "2",
            "status": "success",
            "page_text": "Unexpected pass2 overwrite.",
            "runtime_ms": "999",
            "extraction_method": "ocr",
        },
    ]

    merged, _ = rp._merge_two_pass_rows(
        pass1_rows,
        pass2_rows,
        {("doc.pdf", 1): ["status:failed"]},
        fallback_engine="paddle",
    )

    assert len(merged) == 2
    row_page_1 = next(row for row in merged if row["page"] == "1")
    row_page_2 = next(row for row in merged if row["page"] == "2")

    assert row_page_1["final_output_source"] == "second_pass_paddle"
    assert row_page_2["final_output_source"] == "first_pass_fast"
    assert row_page_2["page_text"] == "Stable page text."
    assert row_page_2["second_pass_status"] == "not_rerun"


def test_merge_two_pass_rows_does_not_select_junky_longer_second_pass_text():
    pass1_rows = [
        {
            "pdf_name": "doc.pdf",
            "page": "7",
            "status": "success",
            "page_text": "Readable first pass text with normal spacing.",
            "runtime_ms": "90",
            "extraction_method": "text_layer",
        }
    ]
    pass2_rows = [
        {
            "pdf_name": "doc.pdf",
            "page": "7",
            "status": "success",
            "page_text": "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",
            "runtime_ms": "280",
            "extraction_method": "ocr",
        }
    ]

    merged, _ = rp._merge_two_pass_rows(
        pass1_rows,
        pass2_rows,
        {("doc.pdf", 7): ["quality_class:weak_ocr"]},
        fallback_engine="paddle",
    )

    row = merged[0]
    assert row["final_output_source"] == "first_pass_kept_due_to_suspicious_fallback"
    assert row["fallback_rejected_reason"] == "suspicious_second_pass_output"


def test_apply_rerun_budget_limits_respects_max_rerun_pages():
    candidates = [
        {"pdf_name": "a.pdf", "page": 1, "reasons": ["status:failed"]},
        {"pdf_name": "a.pdf", "page": 2, "reasons": ["status:timed_out"]},
        {"pdf_name": "a.pdf", "page": 3, "reasons": ["low_text_length"]},
        {"pdf_name": "a.pdf", "page": 4, "reasons": ["quality_class:weak_ocr"]},
    ]

    selected, skipped, meta = rp._apply_rerun_budget_limits(
        candidates,
        total_pages=20,
        max_rerun_pages=2,
        max_rerun_page_ratio=1.0,
        fallback_budget_mode="strict",
    )

    assert len(selected) == 2
    assert len(skipped) == 2
    assert meta["rerun_candidates_total"] == 4
    assert meta["rerun_attempted"] == 2
    assert selected[0]["page"] in {1, 2}
    assert selected[1]["page"] in {1, 2}


def test_apply_rerun_budget_limits_respects_max_rerun_ratio():
    candidates = [
        {"pdf_name": "doc.pdf", "page": page, "reasons": ["status:failed"]}
        for page in range(1, 9)
    ]

    selected, skipped, meta = rp._apply_rerun_budget_limits(
        candidates,
        total_pages=10,
        max_rerun_pages=0,
        max_rerun_page_ratio=0.2,
        fallback_budget_mode="strict",
    )

    assert len(selected) == 2
    assert len(skipped) == 6
    assert meta["rerun_budget_cap"] == 2


def test_merge_two_pass_rows_timeout_keep_first_pass_behavior():
    pass1_rows = [
        {
            "pdf_name": "doc.pdf",
            "page": "5",
            "status": "success",
            "page_text": "Reasonable first-pass text.",
            "runtime_ms": "90",
            "extraction_method": "text_layer",
        }
    ]
    pass2_rows = [
        {
            "pdf_name": "doc.pdf",
            "page": "5",
            "status": "timed_out",
            "failure_reason": "page_timeout",
            "page_text": "",
            "runtime_ms": "30050",
            "extraction_method": "ocr",
        }
    ]

    merged, stats = rp._merge_two_pass_rows(
        pass1_rows,
        pass2_rows,
        {("doc.pdf", 5): ["status:failed"]},
        fallback_engine="ensemble",
        second_pass_timeout_action="keep_first_pass",
        timed_out_keys={("doc.pdf", 5)},
    )

    row = merged[0]
    assert row["final_output_source"] == "first_pass_kept_due_to_timeout"
    assert row["pass_number"] == "1"
    assert row["second_pass_status"] == "timed_out"
    assert row["final_selection_reason"] == "second_pass_timeout_keep_first_pass"
    assert stats["rerun_timed_out"] == 1


def test_merge_two_pass_rows_marks_budget_exhaustion_and_skipped_pages():
    pass1_rows = [
        {
            "pdf_name": "doc.pdf",
            "page": "1",
            "status": "failed",
            "page_text": "",
            "runtime_ms": "100",
            "extraction_method": "text_layer",
        },
        {
            "pdf_name": "doc.pdf",
            "page": "2",
            "status": "failed",
            "page_text": "",
            "runtime_ms": "105",
            "extraction_method": "text_layer",
        },
    ]
    pass2_rows = [
        {
            "pdf_name": "doc.pdf",
            "page": "1",
            "status": "success",
            "page_text": "Recovered text",
            "runtime_ms": "290",
            "extraction_method": "ocr",
        }
    ]

    merged, stats = rp._merge_two_pass_rows(
        pass1_rows,
        pass2_rows,
        {
            ("doc.pdf", 1): ["status:failed"],
            ("doc.pdf", 2): ["status:failed"],
        },
        fallback_engine="paddle",
        budget_skipped_keys={("doc.pdf", 2)},
        second_pass_budget_exhausted=True,
    )

    row_page_1 = next(row for row in merged if row["page"] == "1")
    row_page_2 = next(row for row in merged if row["page"] == "2")

    assert row_page_1["final_output_source"] == "second_pass_paddle"
    assert row_page_2["final_output_source"] == "first_pass_kept_due_to_budget"
    assert row_page_2["second_pass_status"] == "skipped_budget"
    assert row_page_2["final_selection_reason"] == "second_pass_skipped_due_to_budget"
    assert stats["rerun_skipped_budget"] == 1
    assert stats["second_pass_budget_exhausted"] is True


def test_run_budgeted_second_pass_default_optimization_disabled(tmp_path: Path, monkeypatch):
    calls: list[list[str]] = []

    def _fake_run_page_text(args: list[str]) -> int:
        calls.append(list(args))
        manifest_path = Path(args[args.index("--manifest") + 1])
        output_root = Path(args[args.index("--output-root") + 1])
        output_root.mkdir(parents=True, exist_ok=True)

        page_rows = []
        progress_rows = []
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            pdf_path, page_value = line.split("\t", 1)
            page = int(page_value)
            pdf_name = Path(pdf_path).name
            page_rows.append(
                {
                    "pdf_name": pdf_name,
                    "page": str(page),
                    "status": "success",
                    "failure_reason": "",
                    "extraction_method": "ocr",
                    "runtime_ms": "25",
                    "page_text": f"Recovered text for page {page}.",
                }
            )
            progress_rows.append(
                {
                    "pdf_name": pdf_name,
                    "page": str(page),
                    "ms": "25",
                    "status": "success",
                    "failure_reason": "",
                    "extraction_method": "ocr",
                }
            )

        rp._write_csv_rows(
            output_root / "client_page_text.csv",
            page_rows,
            ["pdf_name", "page", "status", "failure_reason", "extraction_method", "runtime_ms", "page_text"],
            encoding="utf-8-sig",
        )
        rp._write_csv_rows(
            output_root / "progress.csv",
            progress_rows,
            ["pdf_name", "page", "ms", "status", "failure_reason", "extraction_method"],
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(rp, "_call_run_page_text", _fake_run_page_text)

    candidates = [
        {
            "pdf_name": "doc_a.pdf",
            "page": 1,
            "pdf_path": str(tmp_path / "doc_a.pdf"),
            "reasons": ["status:failed"],
        },
        {
            "pdf_name": "doc_b.pdf",
            "page": 2,
            "pdf_path": str(tmp_path / "doc_b.pdf"),
            "reasons": ["status:failed"],
        },
    ]

    pass2_rows, pass2_progress_rows, stats = rp._run_budgeted_second_pass(
        candidates=candidates,
        output_root=tmp_path / "two_pass",
        ocr_fallback_mode="paddle",
        engine_mode_label="ocr-only",
        profile=None,
        status_bar=False,
        max_second_pass_ms_per_page=0,
        max_total_second_pass_ms=0,
        fallback_budget_mode="strict",
    )

    assert len(pass2_rows) == 2
    assert len(pass2_progress_rows) == 2
    assert len(calls) == 2
    assert stats["optimization_enabled"] is False
    assert stats["optimization_name"] == ""
    assert stats["optimization_skipped_reason"] == "optimization_flag_disabled"
    assert stats["second_pass_invocation_count"] == 2
    assert stats["estimated_backend_initialization_count"] == 2


def test_run_budgeted_second_pass_warm_reuse_batches_invocations(tmp_path: Path, monkeypatch):
    calls: list[list[str]] = []

    def _fake_run_page_text(args: list[str]) -> int:
        calls.append(list(args))
        manifest_path = Path(args[args.index("--manifest") + 1])
        output_root = Path(args[args.index("--output-root") + 1])
        output_root.mkdir(parents=True, exist_ok=True)

        page_rows = []
        progress_rows = []
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            pdf_path, page_value = line.split("\t", 1)
            page = int(page_value)
            pdf_name = Path(pdf_path).name
            page_rows.append(
                {
                    "pdf_name": pdf_name,
                    "page": str(page),
                    "status": "success",
                    "failure_reason": "",
                    "extraction_method": "ocr",
                    "runtime_ms": "30",
                    "page_text": f"Recovered text for page {page}.",
                }
            )
            progress_rows.append(
                {
                    "pdf_name": pdf_name,
                    "page": str(page),
                    "ms": "30",
                    "status": "success",
                    "failure_reason": "",
                    "extraction_method": "ocr",
                }
            )

        rp._write_csv_rows(
            output_root / "client_page_text.csv",
            page_rows,
            ["pdf_name", "page", "status", "failure_reason", "extraction_method", "runtime_ms", "page_text"],
            encoding="utf-8-sig",
        )
        rp._write_csv_rows(
            output_root / "progress.csv",
            progress_rows,
            ["pdf_name", "page", "ms", "status", "failure_reason", "extraction_method"],
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(rp, "_call_run_page_text", _fake_run_page_text)

    candidates = []
    for page in range(1, 6):
        candidates.append(
            {
                "pdf_name": "doc.pdf",
                "page": page,
                "pdf_path": str(tmp_path / "doc.pdf"),
                "reasons": ["status:failed"],
            }
        )

    pass2_rows, pass2_progress_rows, stats = rp._run_budgeted_second_pass(
        candidates=candidates,
        output_root=tmp_path / "two_pass",
        ocr_fallback_mode="ensemble",
        engine_mode_label="balanced",
        profile=None,
        status_bar=False,
        max_second_pass_ms_per_page=0,
        max_total_second_pass_ms=0,
        fallback_budget_mode="strict",
        enable_backend_warm_reuse=True,
        backend_warm_batch_size=2,
    )

    assert len(pass2_rows) == 5
    assert len(pass2_progress_rows) == 5
    assert len(calls) == 3
    assert stats["optimization_enabled"] is True
    assert stats["optimization_name"] == "backend_model_warm_reuse"
    assert stats["optimization_skipped_reason"] == ""
    assert stats["second_pass_invocation_count"] == 3
    assert stats["estimated_backend_initialization_count"] == 3


def test_merge_two_pass_rows_emits_optimization_audit_fields():
    pass1_rows = [
        {
            "pdf_name": "doc.pdf",
            "page": "1",
            "status": "success",
            "page_text": "Baseline text.",
            "runtime_ms": "90",
            "extraction_method": "text_layer",
        }
    ]

    merged, _ = rp._merge_two_pass_rows(
        pass1_rows,
        [],
        {},
        fallback_engine="ensemble",
        optimization_enabled=True,
        optimization_name="backend_model_warm_reuse",
        optimization_timeout_policy="strict",
        optimization_worker_count=1,
    )

    row = merged[0]
    assert row["optimization_enabled"] == "true"
    assert row["optimization_name"] == "backend_model_warm_reuse"
    assert row["worker_count"] == "1"
    assert row["timeout_policy"] == "strict"
    assert row["performance_trace"]
    trace = json.loads(row["performance_trace"])
    assert trace["optimization_enabled"] is True
    assert trace["optimization_name"] == "backend_model_warm_reuse"
