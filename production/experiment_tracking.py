from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import socket
import subprocess
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from production.key_normalization import build_key_provenance


LOWER_IS_BETTER_METRICS = {
    "cer_mean",
    "cer_median",
    "cer_p90",
    "wer_mean",
    "wer_median",
    "wer_p90",
    "empty_page_rate",
    "timeout_rate",
    "failed_page_rate",
    "runtime_ms_mean",
    "runtime_ms_p90",
    "runtime_ms_p95",
}


METRIC_WEIGHTS = {
    # Quality and correctness metrics carry stronger weight than speed.
    "cer_mean": 3.0,
    "cer_median": 2.5,
    "cer_p90": 3.0,
    "wer_mean": 3.0,
    "wer_median": 2.5,
    "wer_p90": 3.0,
    "empty_page_rate": 3.0,
    "timeout_rate": 2.0,
    "failed_page_rate": 3.0,
    "quality_score_mean": 2.0,
    # Runtime is important but should not hide OCR quality regressions.
    "runtime_ms_mean": 1.0,
    "runtime_ms_p90": 1.0,
}


QUALITY_METRICS = {
    "cer_mean",
    "cer_median",
    "cer_p90",
    "wer_mean",
    "wer_median",
    "wer_p90",
    "quality_score_mean",
}


RELIABILITY_METRICS = {
    "empty_page_rate",
    "timeout_rate",
    "failed_page_rate",
}


RUNTIME_METRICS = {
    "runtime_ms_mean",
    "runtime_ms_p90",
    "runtime_ms_p95",
}


STATUS_EPSILON = 0.005


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t", "on"}


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / float(len(values))


def _quantile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    frac = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * frac


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _read_csv_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _load_single_csv_row(path: Path | None) -> dict[str, Any] | None:
    rows = _read_csv_rows(path)
    if not rows:
        return None
    return rows[0]


def _write_csv(path: Path, rows: list[dict[str, Any]], preferred_first: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    observed: set[str] = set()

    for row in rows:
        observed.update(row.keys())

    for name in preferred_first:
        if name in observed:
            fieldnames.append(name)
            observed.remove(name)

    fieldnames.extend(sorted(observed))

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=True, indent=2)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(_json_dumps(payload) + "\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _category_score(entries: list[dict[str, Any]], metrics: set[str]) -> float:
    selected = [entry for entry in entries if str(entry.get("metric", "")) in metrics]
    if not selected:
        return 0.0
    total_weight = sum(float(entry.get("weight", 1.0)) for entry in selected)
    if total_weight <= 0:
        return 0.0
    weighted_sum = sum(float(entry.get("weighted_score", 0.0)) for entry in selected)
    return weighted_sum / total_weight


def _status_from_score(score: float, epsilon: float = STATUS_EPSILON) -> str:
    if score > epsilon:
        return "improved"
    if score < -epsilon:
        return "regressed"
    return "unchanged"


def _final_status_from_components(quality_status: str, runtime_status: str, reliability_status: str) -> str:
    if quality_status == "regressed" or reliability_status == "regressed":
        if runtime_status == "improved":
            return "faster_but_regressed"
        return "regressed"
    if quality_status == "improved" and runtime_status == "regressed":
        return "quality_improved_runtime_regressed"
    if quality_status == "improved" or reliability_status == "improved":
        if runtime_status == "regressed":
            return "quality_up_runtime_down"
        return "improved"
    if runtime_status == "improved":
        return "runtime_improved"
    if runtime_status == "regressed":
        return "runtime_regressed"
    return "unchanged"


def _parse_list_field(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
    return [part.strip() for part in re.split(r"[;,|/]", text) if part.strip()]


def _parse_json_or_file(value: str | None) -> dict[str, Any]:
    if not value:
        return {}

    candidate = Path(value)
    if candidate.exists() and candidate.is_file():
        try:
            return json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"raw_path": str(candidate)}

    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}
    except json.JSONDecodeError:
        return {"raw": value}


def _file_sha256(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(8192)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _git_value(args: list[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except OSError:
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _detect_gpu_info() -> dict[str, Any]:
    if not shutil.which("nvidia-smi"):
        return {"gpu_available": False, "gpus": []}

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except OSError:
        return {"gpu_available": False, "gpus": []}

    if completed.returncode != 0:
        return {"gpu_available": False, "gpus": []}

    gpus: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if not parts or not parts[0]:
            continue
        gpu_record: dict[str, Any] = {"name": parts[0]}
        if len(parts) > 1:
            gpu_record["driver_version"] = parts[1]
        if len(parts) > 2:
            memory = _safe_int(parts[2])
            gpu_record["memory_mb"] = memory
        gpus.append(gpu_record)

    return {"gpu_available": bool(gpus), "gpus": gpus}


def _derive_engine_name(row: dict[str, Any]) -> str:
    extraction_method = str(row.get("extraction_method", "") or "").strip().lower()
    if not extraction_method:
        return "unknown"
    if extraction_method == "text_layer":
        return "text_layer"
    if extraction_method.startswith("ocr_"):
        suffix = extraction_method[4:]
        if suffix.startswith("paddle"):
            return "paddle"
        if suffix.startswith("ensemble"):
            return "ensemble"
        return suffix or "ocr"
    return extraction_method


def _page_key_provenance(row: dict[str, Any]) -> dict[str, Any]:
    provenance = build_key_provenance(
        page_key=row.get("page_key", ""),
        document_key=(
            row.get("pdf_name", "")
            or row.get("original_document_key", "")
            or row.get("input_file", "")
        ),
        page=row.get("page", ""),
        page_reference=row.get("page_reference", ""),
        unicode_form="NFKC",
    )
    return provenance.to_dict()


def _run_family(summary: dict[str, Any], comparison: dict[str, Any] | None) -> str:
    tokens = " ".join(
        str(item or "")
        for item in (
            summary.get("run_id", ""),
            summary.get("config_file", ""),
            summary.get("engine", ""),
            summary.get("extraction_method", ""),
            (comparison or {}).get("final_status", ""),
        )
    ).lower()

    if "ensemble" in tokens:
        return "ensemble"
    if "two_pass" in tokens and "paddle" in tokens:
        return "paddle_two_pass"
    if "paddle" in tokens:
        return "paddle"
    return "unknown"


def _core_metrics_from_per_page(per_page_rows: list[dict[str, Any]], summary_row: dict[str, Any]) -> dict[str, Any]:
    cer_values = [float(val) for val in (_safe_float(row.get("cer")) for row in per_page_rows) if val is not None]
    wer_values = [float(val) for val in (_safe_float(row.get("wer")) for row in per_page_rows) if val is not None]
    runtime_values = [float(val) for val in (_safe_float(row.get("runtime_ms")) for row in per_page_rows) if val is not None]

    total_pages = len(per_page_rows)
    failed_pages = sum(1 for row in per_page_rows if str(row.get("status", "")).strip().lower() == "failed")
    empty_pages = sum(1 for row in per_page_rows if _safe_bool(row.get("empty_output")))
    timeout_pages = sum(1 for row in per_page_rows if _safe_bool(row.get("timed_out")))

    page_quality_scores = [
        float(val)
        for val in (_safe_float(row.get("page_quality_score")) for row in per_page_rows)
        if val is not None
    ]

    return {
        "pages": total_pages,
        "cer_mean": _safe_mean(cer_values),
        "cer_median": _quantile(cer_values, 0.50),
        "cer_p90": _quantile(cer_values, 0.90),
        "wer_mean": _safe_mean(wer_values),
        "wer_median": _quantile(wer_values, 0.50),
        "wer_p90": _quantile(wer_values, 0.90),
        "empty_page_rate": (
            (empty_pages / float(total_pages))
            if total_pages
            else _safe_float(summary_row.get("empty_output_rate"))
        ),
        "timeout_rate": (
            (timeout_pages / float(total_pages))
            if total_pages
            else _safe_float(summary_row.get("timeout_rate"))
        ),
        "failed_page_rate": (
            (failed_pages / float(total_pages))
            if total_pages
            else _safe_float(summary_row.get("failed_page_rate"))
        ),
        "runtime_ms_mean": _safe_mean(runtime_values) or _safe_float(summary_row.get("runtime_ms_mean")),
        "runtime_ms_p90": _quantile(runtime_values, 0.90) or _safe_float(summary_row.get("runtime_ms_p90")),
        "runtime_ms_p95": _quantile(runtime_values, 0.95) or _safe_float(summary_row.get("runtime_ms_p95")),
        "quality_score_mean": _safe_mean(page_quality_scores),
        "quality_score_median": _quantile(page_quality_scores, 0.50),
        "quality_score_p90": _quantile(page_quality_scores, 0.90),
    }


def _quality_distribution(per_page_rows: list[dict[str, Any]]) -> dict[str, Any]:
    class_counts: Counter[str] = Counter()
    score_bins = {
        "failed_ocr": 0,
        "weak_ocr": 0,
        "usable_with_review": 0,
        "production_quality": 0,
        "unknown": 0,
    }

    score_values: list[float] = []
    for row in per_page_rows:
        quality_class = str(row.get("quality_class", "") or "").strip()
        if quality_class:
            class_counts[quality_class] += 1

        score = _safe_float(row.get("page_quality_score"))
        if score is None:
            if not quality_class:
                score_bins["unknown"] += 1
            continue

        score_values.append(float(score))
        if score >= 0.85:
            score_bins["production_quality"] += 1
        elif score >= 0.70:
            score_bins["usable_with_review"] += 1
        elif score >= 0.50:
            score_bins["weak_ocr"] += 1
        else:
            score_bins["failed_ocr"] += 1

    return {
        "classes": dict(class_counts),
        "score_bins": score_bins,
        "score_mean": _safe_mean(score_values),
        "score_median": _quantile(score_values, 0.50),
        "score_p90": _quantile(score_values, 0.90),
    }


def _domain_specific_metrics(per_page_rows: list[dict[str, Any]]) -> dict[str, Any]:
    diacritic_values = [
        float(val)
        for val in (_safe_float(row.get("diacritic_preservation_rate")) for row in per_page_rows)
        if val is not None
    ]
    translit_unknown_values = [
        float(val)
        for val in (
            _safe_float(row.get("akkadian_unknown_transliteration_token_rate"))
            for row in per_page_rows
        )
        if val is not None
    ]
    lexicon_coverage_values = [
        float(val)
        for val in (_safe_float(row.get("lexicon_coverage")) for row in per_page_rows)
        if val is not None
    ]
    protected_changes_values = [
        float(val)
        for val in (_safe_float(row.get("protected_character_changes")) for row in per_page_rows)
        if val is not None
    ]

    token_validity_values = [max(0.0, 1.0 - value) for value in translit_unknown_values]

    return {
        "diacritic_preservation_mean": _safe_mean(diacritic_values),
        "transliteration_token_validity_mean": _safe_mean(token_validity_values),
        "lexicon_coverage_mean": _safe_mean(lexicon_coverage_values),
        "protected_character_changes_total": (
            sum(protected_changes_values) if protected_changes_values else None
        ),
        "protected_character_changes_mean": _safe_mean(protected_changes_values),
    }


def _aggregate_rows(rows: list[dict[str, Any]], key_field: str, output_field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = str(row.get(key_field, "") or "").strip() or "unknown"
        grouped[key].append(row)

    output_rows: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        cer_values = [float(val) for val in (_safe_float(item.get("cer")) for item in items) if val is not None]
        wer_values = [float(val) for val in (_safe_float(item.get("wer")) for item in items) if val is not None]
        runtime_values = [
            float(val) for val in (_safe_float(item.get("runtime_ms")) for item in items) if val is not None
        ]
        failed_pages = sum(1 for item in items if str(item.get("status", "")).strip().lower() == "failed")
        timeout_pages = sum(1 for item in items if _safe_bool(item.get("timed_out")))
        empty_pages = sum(1 for item in items if _safe_bool(item.get("empty_output")))

        output_rows.append(
            {
                output_field: key,
                "pages": len(items),
                "cer_mean": _safe_mean(cer_values),
                "cer_median": _quantile(cer_values, 0.50),
                "cer_p90": _quantile(cer_values, 0.90),
                "wer_mean": _safe_mean(wer_values),
                "wer_median": _quantile(wer_values, 0.50),
                "wer_p90": _quantile(wer_values, 0.90),
                "runtime_ms_mean": _safe_mean(runtime_values),
                "runtime_ms_p90": _quantile(runtime_values, 0.90),
                "empty_page_rate": empty_pages / float(len(items)) if items else None,
                "timeout_rate": timeout_pages / float(len(items)) if items else None,
                "failed_page_rate": failed_pages / float(len(items)) if items else None,
            }
        )
    return output_rows


def _aggregate_multivalue_rows(rows: list[dict[str, Any]], key_field: str, output_field: str) -> list[dict[str, Any]]:
    expanded_rows: list[dict[str, Any]] = []
    for row in rows:
        values = _parse_list_field(row.get(key_field, ""))
        if not values:
            values = ["unknown"]
        for value in values:
            copy_row = dict(row)
            copy_row[output_field] = value
            expanded_rows.append(copy_row)

    return _aggregate_rows(expanded_rows, output_field, output_field)


def _summarize_group_changes(
    current_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    key_field: str,
) -> dict[str, list[dict[str, Any]]]:
    current_by_key = {str(row.get(key_field, "")).strip(): row for row in current_rows}
    baseline_by_key = {str(row.get(key_field, "")).strip(): row for row in baseline_rows}

    improved: list[dict[str, Any]] = []
    worsened: list[dict[str, Any]] = []

    for key in sorted(set(current_by_key) & set(baseline_by_key)):
        current = current_by_key[key]
        baseline = baseline_by_key[key]

        current_cer = _safe_float(current.get("cer_mean"))
        baseline_cer = _safe_float(baseline.get("cer_mean"))
        current_wer = _safe_float(current.get("wer_mean"))
        baseline_wer = _safe_float(baseline.get("wer_mean"))

        if current_cer is None or baseline_cer is None:
            continue

        delta_cer = baseline_cer - current_cer
        delta_wer = 0.0
        if current_wer is not None and baseline_wer is not None:
            delta_wer = baseline_wer - current_wer

        aggregate_delta = delta_cer + delta_wer
        entry = {
            key_field: key,
            "cer_delta": current_cer - baseline_cer,
            "wer_delta": (
                (current_wer - baseline_wer) if (current_wer is not None and baseline_wer is not None) else None
            ),
            "aggregate_improvement": aggregate_delta,
            "current_pages": _safe_int(current.get("pages")),
            "baseline_pages": _safe_int(baseline.get("pages")),
        }

        if aggregate_delta > 1e-9:
            improved.append(entry)
        elif aggregate_delta < -1e-9:
            worsened.append(entry)

    improved.sort(key=lambda item: float(item.get("aggregate_improvement") or 0.0), reverse=True)
    worsened.sort(key=lambda item: float(item.get("aggregate_improvement") or 0.0))

    return {"improved": improved, "worsened": worsened}


def _compare_overall_metrics(current_summary: dict[str, Any], baseline_summary: dict[str, Any]) -> dict[str, Any]:
    compared: list[dict[str, Any]] = []
    improved: list[dict[str, Any]] = []
    worsened: list[dict[str, Any]] = []

    metrics = [
        "cer_mean",
        "cer_median",
        "cer_p90",
        "wer_mean",
        "wer_median",
        "wer_p90",
        "empty_page_rate",
        "timeout_rate",
        "failed_page_rate",
        "runtime_ms_mean",
        "runtime_ms_p90",
        "runtime_ms_p95",
        "quality_score_mean",
    ]

    for metric in metrics:
        current_val = _safe_float(current_summary.get(metric))
        baseline_val = _safe_float(baseline_summary.get(metric))
        if current_val is None or baseline_val is None:
            continue

        delta = current_val - baseline_val
        if metric in LOWER_IS_BETTER_METRICS:
            directional_improvement = baseline_val - current_val
        else:
            directional_improvement = current_val - baseline_val

        # Relative deltas avoid scale dominance (for example runtime in ms vs CER in [0, 1]).
        denominator = max(abs(baseline_val), 1e-9)
        relative_change = directional_improvement / denominator
        weight = float(METRIC_WEIGHTS.get(metric, 1.0))
        weighted_score = relative_change * weight

        entry = {
            "metric": metric,
            "current": current_val,
            "baseline": baseline_val,
            "delta": delta,
            "directional_improvement": directional_improvement,
            "relative_change": relative_change,
            "weight": weight,
            "weighted_score": weighted_score,
        }
        compared.append(entry)

        if directional_improvement > 1e-9:
            improved.append(entry)
        elif directional_improvement < -1e-9:
            worsened.append(entry)

    improved.sort(key=lambda item: abs(float(item.get("weighted_score") or 0.0)), reverse=True)
    worsened.sort(key=lambda item: abs(float(item.get("weighted_score") or 0.0)), reverse=True)

    weight_sum = sum(float(item.get("weight") or 0.0) for item in compared) or 1.0
    score = sum(float(item.get("weighted_score") or 0.0) for item in compared) / weight_sum

    quality_score = _category_score(compared, QUALITY_METRICS)
    reliability_score = _category_score(compared, RELIABILITY_METRICS)
    runtime_score = _category_score(compared, RUNTIME_METRICS)

    quality_status = _status_from_score(quality_score)
    reliability_status = _status_from_score(reliability_score)
    runtime_status = _status_from_score(runtime_score)
    final_status = _final_status_from_components(
        quality_status=quality_status,
        runtime_status=runtime_status,
        reliability_status=reliability_status,
    )

    return {
        "overall": final_status,
        "final_status": final_status,
        "score": score,
        "quality_score": quality_score,
        "reliability_score": reliability_score,
        "runtime_score": runtime_score,
        "quality_status": quality_status,
        "reliability_status": reliability_status,
        "runtime_status": runtime_status,
        "compared_metrics": compared,
        "improved": improved,
        "worsened": worsened,
    }


def _page_regression_rows(
    current_pages: list[dict[str, Any]],
    baseline_pages: list[dict[str, Any]],
    top_n: int,
) -> dict[str, list[dict[str, Any]]]:
    baseline_by_key: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for row in baseline_pages:
        provenance = _page_key_provenance(row)
        join_key = str(provenance.get("normalized_page_key", "")).strip()
        if not join_key:
            continue
        if join_key not in baseline_by_key:
            baseline_by_key[join_key] = (row, provenance)

    deltas: list[dict[str, Any]] = []

    for row in current_pages:
        current_provenance = _page_key_provenance(row)
        normalized_page_key = str(current_provenance.get("normalized_page_key", "")).strip()
        if not normalized_page_key or normalized_page_key not in baseline_by_key:
            continue

        baseline_row, baseline_provenance = baseline_by_key[normalized_page_key]
        page_key = str(row.get("page_key", "")).strip() or normalized_page_key

        current_score = (_safe_float(row.get("normalized_cer")) or _safe_float(row.get("cer")) or 0.0) + (
            _safe_float(row.get("normalized_wer")) or _safe_float(row.get("wer")) or 0.0
        )
        baseline_score = (
            (_safe_float(baseline_row.get("normalized_cer")) or _safe_float(baseline_row.get("cer")) or 0.0)
            + (_safe_float(baseline_row.get("normalized_wer")) or _safe_float(baseline_row.get("wer")) or 0.0)
        )

        improvement = baseline_score - current_score
        deltas.append(
            {
                "page_key": page_key,
                "normalized_page_key": normalized_page_key,
                "baseline_page_key": str(baseline_row.get("page_key", "")).strip() or normalized_page_key,
                "pdf_name": row.get("pdf_name", ""),
                "page": row.get("page", ""),
                "language_primary": row.get("language_primary", ""),
                "layout_type": row.get("layout_type", ""),
                "engine_used": _derive_engine_name(row),
                "original_page_key": current_provenance.get("original_page_key", ""),
                "original_document_key": current_provenance.get("original_document_key", ""),
                "normalized_document_key": current_provenance.get("normalized_document_key", ""),
                "key_normalization_applied": bool(
                    current_provenance.get("key_normalization_applied", False)
                    or baseline_provenance.get("key_normalization_applied", False)
                ),
                "key_normalization_warnings": "|".join(
                    token
                    for token in [
                        str(current_provenance.get("key_normalization_warnings", "")).strip(),
                        str(baseline_provenance.get("key_normalization_warnings", "")).strip(),
                    ]
                    if token
                ),
                "improvement_score": improvement,
                "baseline_score": baseline_score,
                "current_score": current_score,
            }
        )

    deltas.sort(key=lambda item: float(item.get("improvement_score") or 0.0), reverse=True)
    n = max(1, int(top_n))
    improved_pages = deltas[:n]
    regressed_pages = list(reversed(deltas[-n:]))

    return {
        "improved_pages": improved_pages,
        "regressed_pages": regressed_pages,
    }


def _collect_failing_pages(
    current_pages: list[dict[str, Any]],
    regression_pages: dict[str, list[dict[str, Any]]],
    cer_threshold: float = 0.35,
    wer_threshold: float = 0.45,
) -> list[dict[str, Any]]:
    regression_map = {
        str(item.get("normalized_page_key", "") or item.get("page_key", "")).strip(): float(
            item.get("improvement_score") or 0.0
        )
        for item in regression_pages.get("regressed_pages", [])
    }

    output: list[dict[str, Any]] = []
    for row in current_pages:
        provenance = _page_key_provenance(row)
        page_key = str(row.get("page_key", "")).strip()
        normalized_page_key = str(provenance.get("normalized_page_key", "")).strip() or page_key
        pdf_name = str(row.get("pdf_name", "")).strip()
        page_number = _safe_int(row.get("page"))
        if not normalized_page_key or normalized_page_key.startswith("unknown") or not pdf_name or page_number is None:
            continue

        status = str(row.get("status", "")).strip().lower()
        cer = _safe_float(row.get("cer"))
        wer = _safe_float(row.get("wer"))

        reasons: list[str] = []
        if status in {"failed", "timed_out"}:
            reasons.append(f"status:{status}")
        if _safe_bool(row.get("empty_output")):
            reasons.append("empty_output")
        if cer is not None and cer >= cer_threshold:
            reasons.append(f"high_cer:{cer:.3f}")
        if wer is not None and wer >= wer_threshold:
            reasons.append(f"high_wer:{wer:.3f}")

        regression_delta = regression_map.get(normalized_page_key)
        if regression_delta is not None and regression_delta < 0:
            reasons.append(f"regressed:{regression_delta:.4f}")

        if not reasons:
            continue

        output.append(
            {
                "page_key": page_key or normalized_page_key,
                "normalized_page_key": normalized_page_key,
                "pdf_name": pdf_name,
                "page": page_number,
                "status": status,
                "failure_reason": row.get("failure_reason", ""),
                "cer": cer,
                "wer": wer,
                "runtime_ms": _safe_float(row.get("runtime_ms")),
                "language_primary": row.get("language_primary", ""),
                "languages_present": row.get("languages_present", ""),
                "script_type": row.get("script_type", ""),
                "document_type": row.get("document_type", ""),
                "layout_type": row.get("layout_type", ""),
                "scan_quality": row.get("scan_quality", ""),
                "difficulty_level": row.get("expected_difficulty", ""),
                "engine_used": _derive_engine_name(row),
                "original_page_key": provenance.get("original_page_key", ""),
                "original_document_key": provenance.get("original_document_key", ""),
                "normalized_document_key": provenance.get("normalized_document_key", ""),
                "key_normalization_applied": provenance.get("key_normalization_applied", False),
                "key_normalization_warnings": provenance.get("key_normalization_warnings", ""),
                "regression_delta": regression_delta,
                "reason": "|".join(reasons),
            }
        )

    output.sort(key=lambda item: (str(item.get("status", "")), -(float(item.get("cer") or 0.0))))
    return output


def _count_malformed_output_rows(per_page_rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in per_page_rows:
        provenance = _page_key_provenance(row)
        page_key = str(provenance.get("normalized_page_key", "") or row.get("page_key", "")).strip()
        pdf_name = str(row.get("pdf_name", "") or provenance.get("original_document_key", "")).strip()
        page_number = _safe_int(row.get("page"))
        if not page_key or page_key.startswith("unknown") or not pdf_name or page_number is None:
            count += 1
    return count


def _build_release_recommendation(
    *,
    summary: dict[str, Any],
    comparison: dict[str, Any] | None,
    malformed_output_rows: int,
) -> dict[str, Any]:
    reasons: list[str] = []
    run_family = _run_family(summary, comparison)

    failed_rate = _safe_float(summary.get("failed_page_rate")) or 0.0
    timeout_rate = _safe_float(summary.get("timeout_rate")) or 0.0
    cer_mean = _safe_float(summary.get("cer_mean")) or 0.0
    wer_mean = _safe_float(summary.get("wer_mean")) or 0.0

    quality_status = ""
    reliability_status = ""
    runtime_status = ""
    final_status = ""

    if malformed_output_rows > 0:
        reasons.append(f"malformed_output_rows:{malformed_output_rows}")

    if failed_rate > 0.05:
        reasons.append(f"failed_page_rate:{failed_rate:.4f}")
    if timeout_rate > 0.03:
        reasons.append(f"timeout_rate:{timeout_rate:.4f}")

    if comparison:
        quality_status = str(comparison.get("quality_status", "")).strip().lower()
        reliability_status = str(comparison.get("reliability_status", "")).strip().lower()
        runtime_status = str(comparison.get("runtime_status", "")).strip().lower()
        final_status = str(comparison.get("final_status", "")).strip().lower()

        if quality_status == "regressed":
            reasons.append("quality_regressed_vs_baseline")
        if reliability_status == "regressed":
            reasons.append("reliability_regressed_vs_baseline")
        if final_status == "faster_but_regressed":
            reasons.append("runtime_improved_but_quality_or_reliability_regressed")
        if runtime_status == "regressed":
            reasons.append("runtime_regressed_vs_baseline")

    if run_family == "ensemble" and (
        quality_status == "regressed"
        or reliability_status == "regressed"
        or final_status in {"regressed", "faster_but_regressed"}
    ):
        if "quality_regressed_vs_baseline" not in reasons:
            reasons.append("quality_regressed_vs_baseline")
        return {
            "status": "reject_ensemble_due_to_quality_regression",
            "reasons": reasons,
        }

    if run_family == "paddle_two_pass" and comparison and quality_status != "regressed" and reliability_status != "regressed":
        default_reasons = ["quality_reliability_non_regressing_vs_baseline"]
        if runtime_status == "regressed":
            default_reasons.append("runtime_regressed_accepted_for_quality")
        if failed_rate > 0.05:
            default_reasons.append(f"failed_page_rate_above_absolute_target:{failed_rate:.4f}")
        if timeout_rate > 0.03:
            default_reasons.append(f"timeout_rate_above_absolute_target:{timeout_rate:.4f}")
        return {
            "status": "use_two_pass_paddle_default",
            "reasons": default_reasons,
        }

    if reasons:
        if any("quality_regressed" in reason or "reliability_regressed" in reason for reason in reasons):
            status = "not_ready_quality_regression"
        elif any(reason.startswith("malformed_output_rows:") for reason in reasons):
            status = "not_ready_malformed_outputs"
        elif any(reason.startswith("failed_page_rate:") or reason.startswith("timeout_rate:") for reason in reasons):
            status = "not_ready_reliability"
        else:
            status = "needs_review"
        return {
            "status": status,
            "reasons": reasons,
        }

    if cer_mean <= 0.35 and wer_mean <= 0.45 and failed_rate <= 0.03 and timeout_rate <= 0.01:
        return {
            "status": "ready_for_internal_testing",
            "reasons": ["quality_and_reliability_within_targets"],
        }

    return {
        "status": "conditional_ready",
        "reasons": ["no_regression_detected_but_targets_not_fully_met"],
    }


def _build_recommendations(
    *,
    summary: dict[str, Any],
    by_scan_quality: list[dict[str, Any]],
    by_layout: list[dict[str, Any]],
    by_engine: list[dict[str, Any]],
    regression: dict[str, Any] | None,
    per_page_rows: list[dict[str, Any]],
    baseline_pages: list[dict[str, Any]] | None,
) -> list[str]:
    recommendations: list[str] = []

    cer_mean = _safe_float(summary.get("cer_mean")) or 0.0
    timeout_rate = _safe_float(summary.get("timeout_rate")) or 0.0
    failed_rate = _safe_float(summary.get("failed_page_rate")) or 0.0

    noisy_rows = [
        row for row in by_scan_quality if str(row.get("scan_quality", "")).strip().lower() in {"noisy", "low", "degraded"}
    ]
    if noisy_rows:
        worst_noisy = max(noisy_rows, key=lambda row: _safe_float(row.get("cer_mean")) or -1.0)
        noisy_cer = _safe_float(worst_noisy.get("cer_mean"))
        if noisy_cer is not None and noisy_cer > cer_mean + 0.05:
            recommendations.append(
                "Noisy scans still have high CER; prioritize preprocessing profile noisy_scan and review denoise/contrast settings."
            )

    two_column_rows = [
        row for row in by_layout if "two" in str(row.get("layout_type", "")).lower() or "multi" in str(row.get("layout_type", "")).lower()
    ]
    if regression and two_column_rows:
        worsened_layouts = regression.get("layouts_worsened", [])
        if worsened_layouts:
            recommendations.append(
                "Two-column pages regress after recent changes; inspect reading order reconstruction and region ordering confidence."
            )

    if timeout_rate > 0.03:
        recommendations.append(
            "Timeout rate is elevated; inspect per-engine timeouts and consider lowering expensive fallback fan-out on difficult pages."
        )

    if failed_rate > 0.05:
        recommendations.append(
            "Failed page rate is above target; audit failing_pages.csv and tighten engine readiness checks before launch."
        )

    translit_rows = [row for row in per_page_rows if "translit" in str(row.get("script_type", "")).lower()]
    if translit_rows:
        current_diacritic = _safe_mean(
            [
                float(val)
                for val in (_safe_float(row.get("diacritic_preservation_rate")) for row in translit_rows)
                if val is not None
            ]
        )
        baseline_diacritic = None
        if baseline_pages:
            baseline_translit = [row for row in baseline_pages if "translit" in str(row.get("script_type", "")).lower()]
            baseline_diacritic = _safe_mean(
                [
                    float(val)
                    for val in (
                        _safe_float(row.get("diacritic_preservation_rate")) for row in baseline_translit
                    )
                    if val is not None
                ]
            )

        if baseline_diacritic is not None and current_diacritic is not None and current_diacritic < baseline_diacritic - 0.02:
            recommendations.append(
                "Akkadian/transliteration CER improved but diacritic preservation regressed; review protected-character safeguards in postprocessing."
            )

    engines_with_runtime = [row for row in by_engine if _safe_float(row.get("runtime_ms_mean")) is not None]
    engines_with_quality = [row for row in by_engine if _safe_float(row.get("cer_mean")) is not None]
    if engines_with_runtime and engines_with_quality:
        fastest = min(engines_with_runtime, key=lambda row: _safe_float(row.get("runtime_ms_mean")) or float("inf"))
        best_quality = min(engines_with_quality, key=lambda row: _safe_float(row.get("cer_mean")) or float("inf"))
        fastest_engine = str(fastest.get("engine", "unknown"))
        best_engine = str(best_quality.get("engine", "unknown"))
        if fastest_engine != best_engine and fastest_engine != "unknown" and best_engine != "unknown":
            recommendations.append(
                f"{fastest_engine} is fastest on average, while {best_engine} has better OCR quality; route hard pages toward the quality-favored engine."
            )

    if not recommendations:
        recommendations.append(
            "No critical regressions detected; continue monitoring CER/WER by language and layout while expanding gold coverage."
        )

    return recommendations


def _markdown_metric(value: Any, precision: int = 4) -> str:
    as_float = _safe_float(value)
    if as_float is None:
        return "n/a"
    return f"{as_float:.{precision}f}"


def _build_metrics_summary_markdown(
    *,
    run_metadata: dict[str, Any],
    summary: dict[str, Any],
    quality_distribution: dict[str, Any],
    domain_metrics: dict[str, Any],
    recommendations: list[str],
) -> str:
    lines: list[str] = []
    lines.append("# OCR Metrics Summary")
    lines.append("")
    lines.append("## Run Metadata")
    lines.append("")
    lines.append(f"- run_id: {run_metadata.get('run_id', '')}")
    lines.append(f"- timestamp: {run_metadata.get('timestamp', '')}")
    lines.append(f"- git_commit: {run_metadata.get('git_commit', '')}")
    lines.append(f"- branch: {run_metadata.get('branch', '')}")
    lines.append(f"- config_file: {run_metadata.get('config_file', '')}")
    lines.append(f"- gold_set_version: {run_metadata.get('gold_set_version', '')}")
    lines.append("")
    lines.append("## Core Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| CER mean | {_markdown_metric(summary.get('cer_mean'))} |")
    lines.append(f"| CER median | {_markdown_metric(summary.get('cer_median'))} |")
    lines.append(f"| CER p90 | {_markdown_metric(summary.get('cer_p90'))} |")
    lines.append(f"| WER mean | {_markdown_metric(summary.get('wer_mean'))} |")
    lines.append(f"| WER median | {_markdown_metric(summary.get('wer_median'))} |")
    lines.append(f"| WER p90 | {_markdown_metric(summary.get('wer_p90'))} |")
    lines.append(f"| Empty page rate | {_markdown_metric(summary.get('empty_page_rate'))} |")
    lines.append(f"| Timeout rate | {_markdown_metric(summary.get('timeout_rate'))} |")
    lines.append(f"| Failed page rate | {_markdown_metric(summary.get('failed_page_rate'))} |")
    lines.append(f"| Runtime mean (ms) | {_markdown_metric(summary.get('runtime_ms_mean'), precision=2)} |")
    lines.append(f"| Runtime p90 (ms) | {_markdown_metric(summary.get('runtime_ms_p90'), precision=2)} |")
    lines.append(f"| Runtime p95 (ms) | {_markdown_metric(summary.get('runtime_ms_p95'), precision=2)} |")
    lines.append(f"| Aggregate score | {_markdown_metric(summary.get('aggregate_score'))} |")
    lines.append(f"| Quality score | {_markdown_metric(summary.get('quality_score'))} |")
    lines.append(f"| Runtime score | {_markdown_metric(summary.get('runtime_score'))} |")
    lines.append(f"| Reliability score | {_markdown_metric(summary.get('reliability_score'))} |")
    lines.append(f"| Quality status | {summary.get('quality_status', 'n/a')} |")
    lines.append(f"| Runtime status | {summary.get('runtime_status', 'n/a')} |")
    lines.append(f"| Reliability status | {summary.get('reliability_status', 'n/a')} |")
    lines.append(f"| Final status | {summary.get('final_status', 'n/a')} |")
    lines.append(f"| Malformed output rows | {summary.get('malformed_output_rows', 0)} |")
    lines.append(
        f"| Production recommendation | {summary.get('production_recommendation_status', 'n/a')} |"
    )
    lines.append("")
    lines.append("## Quality Score Distribution")
    lines.append("")
    lines.append(f"- class_counts: {_json_dumps(quality_distribution.get('classes', {}))}")
    lines.append(f"- score_bins: {_json_dumps(quality_distribution.get('score_bins', {}))}")
    lines.append(f"- score_mean: {_markdown_metric(quality_distribution.get('score_mean'))}")
    lines.append("")
    lines.append("## Domain-Specific Metrics")
    lines.append("")
    lines.append(f"- diacritic_preservation_mean: {_markdown_metric(domain_metrics.get('diacritic_preservation_mean'))}")
    lines.append(
        f"- transliteration_token_validity_mean: {_markdown_metric(domain_metrics.get('transliteration_token_validity_mean'))}"
    )
    lines.append(f"- lexicon_coverage_mean: {_markdown_metric(domain_metrics.get('lexicon_coverage_mean'))}")
    lines.append(
        f"- protected_character_changes_total: {_markdown_metric(domain_metrics.get('protected_character_changes_total'), precision=2)}"
    )
    lines.append("")
    lines.append("## Recommended Next Actions")
    lines.append("")
    for recommendation in recommendations:
        lines.append(f"- {recommendation}")
    lines.append("")
    return "\n".join(lines)


def _build_regression_markdown(
    *,
    comparison: dict[str, Any] | None,
    page_comparison: dict[str, Any],
    language_changes: dict[str, list[dict[str, Any]]],
    layout_changes: dict[str, list[dict[str, Any]]],
    engine_changes: dict[str, list[dict[str, Any]]],
    runtime_changes: list[dict[str, Any]],
    recommendations: list[str],
) -> str:
    lines: list[str] = []
    lines.append("# OCR Regression Report")
    lines.append("")

    if comparison is None:
        lines.append("No baseline comparison was available for this run.")
        lines.append("")
        lines.append("## Recommended Next Actions")
        lines.append("")
        for recommendation in recommendations:
            lines.append(f"- {recommendation}")
        lines.append("")
        return "\n".join(lines)

    lines.append("## Overall Improvement/Regression")
    lines.append("")
    lines.append(f"- status: {comparison.get('final_status', comparison.get('overall', 'unknown'))}")
    lines.append(f"- aggregate_score: {_markdown_metric(comparison.get('score'))}")
    lines.append(f"- quality_status: {comparison.get('quality_status', 'unknown')}")
    lines.append(f"- runtime_status: {comparison.get('runtime_status', 'unknown')}")
    lines.append(f"- reliability_status: {comparison.get('reliability_status', 'unknown')}")
    lines.append(f"- quality_score: {_markdown_metric(comparison.get('quality_score'))}")
    lines.append(f"- runtime_score: {_markdown_metric(comparison.get('runtime_score'))}")
    lines.append(f"- reliability_score: {_markdown_metric(comparison.get('reliability_score'))}")
    lines.append("")

    lines.append("## Pages That Improved Most")
    lines.append("")
    for item in page_comparison.get("improved_pages", [])[:10]:
        lines.append(
            f"- {item.get('page_key', '')}: improvement={_markdown_metric(item.get('improvement_score'))}, "
            f"baseline={_markdown_metric(item.get('baseline_score'))}, current={_markdown_metric(item.get('current_score'))}"
        )
    if not page_comparison.get("improved_pages"):
        lines.append("- none")
    lines.append("")

    lines.append("## Pages That Regressed Most")
    lines.append("")
    for item in page_comparison.get("regressed_pages", [])[:10]:
        lines.append(
            f"- {item.get('page_key', '')}: improvement={_markdown_metric(item.get('improvement_score'))}, "
            f"baseline={_markdown_metric(item.get('baseline_score'))}, current={_markdown_metric(item.get('current_score'))}"
        )
    if not page_comparison.get("regressed_pages"):
        lines.append("- none")
    lines.append("")

    lines.append("## Languages Improved")
    lines.append("")
    for item in language_changes.get("improved", [])[:10]:
        lines.append(f"- {item.get('language_primary', '')}: aggregate={_markdown_metric(item.get('aggregate_improvement'))}")
    if not language_changes.get("improved"):
        lines.append("- none")
    lines.append("")

    lines.append("## Languages Worsened")
    lines.append("")
    for item in language_changes.get("worsened", [])[:10]:
        lines.append(f"- {item.get('language_primary', '')}: aggregate={_markdown_metric(item.get('aggregate_improvement'))}")
    if not language_changes.get("worsened"):
        lines.append("- none")
    lines.append("")

    lines.append("## Layouts Improved")
    lines.append("")
    for item in layout_changes.get("improved", [])[:10]:
        lines.append(f"- {item.get('layout_type', '')}: aggregate={_markdown_metric(item.get('aggregate_improvement'))}")
    if not layout_changes.get("improved"):
        lines.append("- none")
    lines.append("")

    lines.append("## Layouts Worsened")
    lines.append("")
    for item in layout_changes.get("worsened", [])[:10]:
        lines.append(f"- {item.get('layout_type', '')}: aggregate={_markdown_metric(item.get('aggregate_improvement'))}")
    if not layout_changes.get("worsened"):
        lines.append("- none")
    lines.append("")

    lines.append("## Engines Improved/Worsened")
    lines.append("")
    if engine_changes.get("improved"):
        for item in engine_changes.get("improved", [])[:10]:
            lines.append(f"- improved: {item.get('engine', '')} aggregate={_markdown_metric(item.get('aggregate_improvement'))}")
    if engine_changes.get("worsened"):
        for item in engine_changes.get("worsened", [])[:10]:
            lines.append(f"- worsened: {item.get('engine', '')} aggregate={_markdown_metric(item.get('aggregate_improvement'))}")
    if not engine_changes.get("improved") and not engine_changes.get("worsened"):
        lines.append("- none")
    lines.append("")

    lines.append("## Runtime Changes")
    lines.append("")
    for item in runtime_changes:
        lines.append(
            f"- {item.get('metric', '')}: baseline={_markdown_metric(item.get('baseline'), 2)}, "
            f"current={_markdown_metric(item.get('current'), 2)}, delta={_markdown_metric(item.get('delta'), 2)}"
        )
    if not runtime_changes:
        lines.append("- none")
    lines.append("")

    lines.append("## Recommended Next Actions")
    lines.append("")
    for recommendation in recommendations:
        lines.append(f"- {recommendation}")
    lines.append("")

    return "\n".join(lines)


def _prepare_run_metadata(
    *,
    run_id: str | None,
    config_file: str | None,
    engine_versions_arg: str | None,
    model_versions_arg: str | None,
    gold_set_version: str | None,
    gold_csv_path: Path | None,
    ocr_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metadata_run_id = run_id or str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    preprocessing_profiles: set[str] = set()
    adapters: set[str] = set()

    for row in ocr_rows:
        for field in ("recommended_preprocessing_profile", "applied_preprocessing_profile", "preprocessing_profile"):
            value = str(row.get(field, "") or "").strip()
            if value:
                preprocessing_profiles.add(value)
        adapter = str(row.get("adapter_used", "") or "").strip()
        if adapter:
            adapters.add(adapter)

    if not gold_set_version and gold_csv_path and gold_csv_path.exists():
        digest = _file_sha256(gold_csv_path)
        gold_set_version = f"{gold_csv_path.name}:{digest[:12]}"

    env_info = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "hostname": socket.gethostname(),
    }
    cpu_gpu_info = {
        "cpu_logical_count": os.cpu_count(),
        "cpu_processor": platform.processor(),
        "architecture": platform.architecture()[0],
        **_detect_gpu_info(),
    }

    return {
        "run_id": metadata_run_id,
        "timestamp": timestamp,
        "git_commit": _git_value(["rev-parse", "HEAD"]),
        "branch": _git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        "config_file": str(config_file or ""),
        "engine_versions": _parse_json_or_file(engine_versions_arg),
        "model_versions": _parse_json_or_file(model_versions_arg),
        "preprocessing_profiles": sorted(preprocessing_profiles),
        "postprocessing_adapters": sorted(adapters),
        "gold_set_version": gold_set_version or "",
        "environment_info": env_info,
        "cpu_gpu_info": cpu_gpu_info,
    }


def _build_engine_metrics(per_page_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in per_page_rows:
        copy_row = dict(row)
        copy_row["engine"] = _derive_engine_name(row)
        rows.append(copy_row)
    return _aggregate_rows(rows, "engine", "engine")


def _extract_runtime_changes(comparison: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not comparison:
        return []
    runtime_metrics = {"runtime_ms_mean", "runtime_ms_p90", "runtime_ms_p95"}
    out: list[dict[str, Any]] = []
    for item in comparison.get("compared_metrics", []):
        metric = str(item.get("metric", ""))
        if metric in runtime_metrics:
            out.append(item)
    return out


def _build_tracking_payload(
    *,
    run_metadata: dict[str, Any],
    metrics_summary_row: dict[str, Any],
    quality_distribution: dict[str, Any],
    domain_metrics: dict[str, Any],
    language_rows: list[dict[str, Any]],
    layout_rows: list[dict[str, Any]],
    engine_rows: list[dict[str, Any]],
    script_rows: list[dict[str, Any]],
    document_type_rows: list[dict[str, Any]],
    scan_quality_rows: list[dict[str, Any]],
    difficulty_rows: list[dict[str, Any]],
    languages_present_rows: list[dict[str, Any]],
    regression_payload: dict[str, Any] | None,
    recommendations: list[str],
    production_recommendation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_metadata": run_metadata,
        "metrics_summary": metrics_summary_row,
        "quality_distribution": quality_distribution,
        "domain_specific_metrics": domain_metrics,
        "metrics_by_language": language_rows,
        "metrics_by_layout": layout_rows,
        "metrics_by_engine": engine_rows,
        "metrics_by_script": script_rows,
        "metrics_by_document_type": document_type_rows,
        "metrics_by_scan_quality": scan_quality_rows,
        "metrics_by_difficulty": difficulty_rows,
        "metrics_by_languages_present": languages_present_rows,
        "regression": regression_payload,
        "production_recommendation": production_recommendation,
        "recommended_next_actions": recommendations,
    }


def build_tracking_dashboard(
    *,
    output_dir: Path,
    per_page_rows: list[dict[str, Any]],
    evaluation_summary_row: dict[str, Any],
    ocr_csv_path: str | Path | None,
    gold_csv_path: str | Path | None,
    baseline_dir: str | Path | None = None,
    baseline_summary_csv: str | Path | None = None,
    baseline_per_page_csv: str | Path | None = None,
    top_n: int = 10,
    run_id: str | None = None,
    config_file: str | None = None,
    engine_versions_arg: str | None = None,
    model_versions_arg: str | None = None,
    gold_set_version: str | None = None,
    history_jsonl: str | Path | None = None,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ocr_path = Path(ocr_csv_path) if ocr_csv_path else None
    gold_path = Path(gold_csv_path) if gold_csv_path else None
    ocr_rows = _read_csv_rows(ocr_path)

    run_metadata = _prepare_run_metadata(
        run_id=run_id,
        config_file=config_file,
        engine_versions_arg=engine_versions_arg,
        model_versions_arg=model_versions_arg,
        gold_set_version=gold_set_version,
        gold_csv_path=gold_path,
        ocr_rows=ocr_rows,
    )

    core_metrics = _core_metrics_from_per_page(per_page_rows, evaluation_summary_row)
    quality_distribution = _quality_distribution(per_page_rows)
    domain_metrics = _domain_specific_metrics(per_page_rows)

    language_rows = _aggregate_rows(per_page_rows, "language_primary", "language_primary")
    layout_rows = _aggregate_rows(per_page_rows, "layout_type", "layout_type")
    engine_rows = _build_engine_metrics(per_page_rows)
    script_rows = _aggregate_rows(per_page_rows, "script_type", "script_type")
    document_type_rows = _aggregate_rows(per_page_rows, "document_type", "document_type")
    scan_quality_rows = _aggregate_rows(per_page_rows, "scan_quality", "scan_quality")
    difficulty_rows = _aggregate_rows(per_page_rows, "expected_difficulty", "difficulty_level")
    languages_present_rows = _aggregate_multivalue_rows(
        per_page_rows,
        "languages_present",
        "language_present",
    )

    metrics_summary_row = {
        **run_metadata,
        **core_metrics,
        **domain_metrics,
        "quality_distribution": _json_dumps(quality_distribution),
    }

    baseline_summary_path: Path | None = Path(baseline_summary_csv) if baseline_summary_csv else None
    baseline_per_page_path: Path | None = Path(baseline_per_page_csv) if baseline_per_page_csv else None

    if baseline_dir:
        base_dir = Path(baseline_dir)
        if baseline_summary_path is None:
            if (base_dir / "metrics_summary.csv").exists():
                baseline_summary_path = base_dir / "metrics_summary.csv"
            else:
                baseline_summary_path = base_dir / "evaluation_summary.csv"
        if baseline_per_page_path is None:
            baseline_per_page_path = base_dir / "per_page_metrics.csv"

    baseline_summary = _load_single_csv_row(baseline_summary_path)
    baseline_pages = _read_csv_rows(baseline_per_page_path)

    baseline_language_rows = []
    baseline_layout_rows = []
    baseline_engine_rows = []
    if baseline_pages:
        baseline_language_rows = _aggregate_rows(baseline_pages, "language_primary", "language_primary")
        baseline_layout_rows = _aggregate_rows(baseline_pages, "layout_type", "layout_type")
        baseline_engine_rows = _build_engine_metrics(baseline_pages)

    overall_comparison = (
        _compare_overall_metrics(metrics_summary_row, baseline_summary)
        if baseline_summary
        else None
    )

    if overall_comparison is not None:
        metrics_summary_row.update(
            {
                "aggregate_score": overall_comparison.get("score"),
                "quality_score": overall_comparison.get("quality_score"),
                "runtime_score": overall_comparison.get("runtime_score"),
                "reliability_score": overall_comparison.get("reliability_score"),
                "quality_status": overall_comparison.get("quality_status"),
                "runtime_status": overall_comparison.get("runtime_status"),
                "reliability_status": overall_comparison.get("reliability_status"),
                "final_status": overall_comparison.get("final_status", overall_comparison.get("overall")),
            }
        )
    else:
        metrics_summary_row.update(
            {
                "aggregate_score": "",
                "quality_score": "",
                "runtime_score": "",
                "reliability_score": "",
                "quality_status": "no_baseline",
                "runtime_status": "no_baseline",
                "reliability_status": "no_baseline",
                "final_status": "no_baseline",
            }
        )

    malformed_output_rows = _count_malformed_output_rows(per_page_rows)
    metrics_summary_row["malformed_output_rows"] = malformed_output_rows

    production_recommendation = _build_release_recommendation(
        summary=metrics_summary_row,
        comparison=overall_comparison,
        malformed_output_rows=malformed_output_rows,
    )
    metrics_summary_row["production_recommendation_status"] = production_recommendation.get("status", "")
    metrics_summary_row["production_recommendation_reasons"] = "|".join(
        production_recommendation.get("reasons", [])
    )

    page_comparison = (
        _page_regression_rows(per_page_rows, baseline_pages, top_n=top_n)
        if baseline_pages
        else {"improved_pages": [], "regressed_pages": []}
    )
    language_changes = (
        _summarize_group_changes(language_rows, baseline_language_rows, "language_primary")
        if baseline_language_rows
        else {"improved": [], "worsened": []}
    )
    layout_changes = (
        _summarize_group_changes(layout_rows, baseline_layout_rows, "layout_type")
        if baseline_layout_rows
        else {"improved": [], "worsened": []}
    )
    engine_changes = (
        _summarize_group_changes(engine_rows, baseline_engine_rows, "engine")
        if baseline_engine_rows
        else {"improved": [], "worsened": []}
    )

    runtime_changes = _extract_runtime_changes(overall_comparison)

    regression_payload: dict[str, Any] | None = None
    if overall_comparison is not None or baseline_pages:
        regression_payload = {
            "overall": overall_comparison,
            "pages": page_comparison,
            "languages_improved": language_changes.get("improved", []),
            "languages_worsened": language_changes.get("worsened", []),
            "layouts_improved": layout_changes.get("improved", []),
            "layouts_worsened": layout_changes.get("worsened", []),
            "engines_improved": engine_changes.get("improved", []),
            "engines_worsened": engine_changes.get("worsened", []),
            "runtime_changes": runtime_changes,
        }

    recommendations = _build_recommendations(
        summary=metrics_summary_row,
        by_scan_quality=scan_quality_rows,
        by_layout=layout_rows,
        by_engine=engine_rows,
        regression={
            "layouts_worsened": layout_changes.get("worsened", []),
        }
        if regression_payload is not None
        else None,
        per_page_rows=per_page_rows,
        baseline_pages=baseline_pages or None,
    )
    recommendations.insert(
        0,
        "production_recommendation_status="
        f"{production_recommendation.get('status', '')}; "
        "reasons="
        f"{'|'.join(production_recommendation.get('reasons', []))}",
    )

    failing_pages_rows = _collect_failing_pages(per_page_rows, page_comparison)

    summary_csv = output_dir / "metrics_summary.csv"
    summary_md = output_dir / "metrics_summary.md"
    language_csv = output_dir / "metrics_by_language.csv"
    layout_csv = output_dir / "metrics_by_layout.csv"
    engine_csv = output_dir / "metrics_by_engine.csv"
    script_csv = output_dir / "metrics_by_script.csv"
    document_type_csv = output_dir / "metrics_by_document_type.csv"
    scan_quality_csv = output_dir / "metrics_by_scan_quality.csv"
    difficulty_csv = output_dir / "metrics_by_difficulty.csv"
    languages_present_csv = output_dir / "metrics_by_languages_present.csv"
    regression_md = output_dir / "regression_report.md"
    failing_pages_csv = output_dir / "failing_pages.csv"
    metadata_json = output_dir / "experiment_run_metadata.json"
    payload_json = output_dir / "experiment_tracking.json"

    _write_csv(
        summary_csv,
        [metrics_summary_row],
        [
            "run_id",
            "timestamp",
            "git_commit",
            "branch",
            "config_file",
            "gold_set_version",
            "pages",
            "cer_mean",
            "cer_median",
            "cer_p90",
            "wer_mean",
            "wer_median",
            "wer_p90",
            "empty_page_rate",
            "timeout_rate",
            "failed_page_rate",
            "runtime_ms_mean",
            "runtime_ms_p90",
            "runtime_ms_p95",
            "aggregate_score",
            "quality_score",
            "runtime_score",
            "reliability_score",
            "quality_status",
            "runtime_status",
            "reliability_status",
            "final_status",
            "malformed_output_rows",
            "production_recommendation_status",
            "production_recommendation_reasons",
        ],
    )
    _write_csv(language_csv, language_rows, ["language_primary", "pages", "cer_mean", "wer_mean", "runtime_ms_mean"])
    _write_csv(layout_csv, layout_rows, ["layout_type", "pages", "cer_mean", "wer_mean", "runtime_ms_mean"])
    _write_csv(engine_csv, engine_rows, ["engine", "pages", "cer_mean", "wer_mean", "runtime_ms_mean"])
    _write_csv(script_csv, script_rows, ["script_type", "pages", "cer_mean", "wer_mean", "runtime_ms_mean"])
    _write_csv(
        document_type_csv,
        document_type_rows,
        ["document_type", "pages", "cer_mean", "wer_mean", "runtime_ms_mean"],
    )
    _write_csv(
        scan_quality_csv,
        scan_quality_rows,
        ["scan_quality", "pages", "cer_mean", "wer_mean", "runtime_ms_mean"],
    )
    _write_csv(
        difficulty_csv,
        difficulty_rows,
        ["difficulty_level", "pages", "cer_mean", "wer_mean", "runtime_ms_mean"],
    )
    _write_csv(
        languages_present_csv,
        languages_present_rows,
        ["language_present", "pages", "cer_mean", "wer_mean", "runtime_ms_mean"],
    )
    _write_csv(
        failing_pages_csv,
        failing_pages_rows,
        [
            "page_key",
            "pdf_name",
            "page",
            "status",
            "failure_reason",
            "cer",
            "wer",
            "runtime_ms",
            "language_primary",
            "script_type",
            "document_type",
            "layout_type",
            "scan_quality",
            "difficulty_level",
            "engine_used",
            "regression_delta",
            "reason",
        ],
    )

    summary_markdown = _build_metrics_summary_markdown(
        run_metadata=run_metadata,
        summary=metrics_summary_row,
        quality_distribution=quality_distribution,
        domain_metrics=domain_metrics,
        recommendations=recommendations,
    )
    _write_text(summary_md, summary_markdown)

    regression_markdown = _build_regression_markdown(
        comparison=overall_comparison,
        page_comparison=page_comparison,
        language_changes=language_changes,
        layout_changes=layout_changes,
        engine_changes=engine_changes,
        runtime_changes=runtime_changes,
        recommendations=recommendations,
    )
    _write_text(regression_md, regression_markdown)

    tracking_payload = _build_tracking_payload(
        run_metadata=run_metadata,
        metrics_summary_row=metrics_summary_row,
        quality_distribution=quality_distribution,
        domain_metrics=domain_metrics,
        language_rows=language_rows,
        layout_rows=layout_rows,
        engine_rows=engine_rows,
        script_rows=script_rows,
        document_type_rows=document_type_rows,
        scan_quality_rows=scan_quality_rows,
        difficulty_rows=difficulty_rows,
        languages_present_rows=languages_present_rows,
        regression_payload=regression_payload,
        recommendations=recommendations,
        production_recommendation=production_recommendation,
    )

    _write_json(metadata_json, run_metadata)
    _write_json(payload_json, tracking_payload)

    history_path = Path(history_jsonl) if history_jsonl else (output_dir / "experiment_history.jsonl")
    _append_jsonl(
        history_path,
        {
            "run_id": run_metadata.get("run_id", ""),
            "timestamp": run_metadata.get("timestamp", ""),
            "git_commit": run_metadata.get("git_commit", ""),
            "branch": run_metadata.get("branch", ""),
            "config_file": run_metadata.get("config_file", ""),
            "gold_set_version": run_metadata.get("gold_set_version", ""),
            "metrics": {
                "cer_mean": metrics_summary_row.get("cer_mean"),
                "wer_mean": metrics_summary_row.get("wer_mean"),
                "empty_page_rate": metrics_summary_row.get("empty_page_rate"),
                "timeout_rate": metrics_summary_row.get("timeout_rate"),
                "failed_page_rate": metrics_summary_row.get("failed_page_rate"),
                "runtime_ms_mean": metrics_summary_row.get("runtime_ms_mean"),
                "runtime_ms_p90": metrics_summary_row.get("runtime_ms_p90"),
                "runtime_ms_p95": metrics_summary_row.get("runtime_ms_p95"),
                "aggregate_score": metrics_summary_row.get("aggregate_score"),
                "quality_status": metrics_summary_row.get("quality_status"),
                "runtime_status": metrics_summary_row.get("runtime_status"),
                "reliability_status": metrics_summary_row.get("reliability_status"),
                "final_status": metrics_summary_row.get("final_status"),
                "production_recommendation_status": metrics_summary_row.get("production_recommendation_status"),
            },
        },
    )

    return {
        "metrics_summary_md": str(summary_md),
        "metrics_summary_csv": str(summary_csv),
        "metrics_by_language_csv": str(language_csv),
        "metrics_by_layout_csv": str(layout_csv),
        "metrics_by_engine_csv": str(engine_csv),
        "regression_report_md": str(regression_md),
        "failing_pages_csv": str(failing_pages_csv),
        "experiment_run_metadata_json": str(metadata_json),
        "experiment_tracking_json": str(payload_json),
        "experiment_history_jsonl": str(history_path),
    }


def compare_tracking_runs(
    *,
    current_dir: Path,
    baseline_dir: Path,
    output_dir: Path,
    top_n: int = 10,
) -> dict[str, str]:
    current_dir = Path(current_dir)
    baseline_dir = Path(baseline_dir)
    output_dir = Path(output_dir)

    current_summary = _load_single_csv_row(current_dir / "metrics_summary.csv")
    if current_summary is None:
        current_summary = _load_single_csv_row(current_dir / "evaluation_summary.csv")

    baseline_summary = _load_single_csv_row(baseline_dir / "metrics_summary.csv")
    if baseline_summary is None:
        baseline_summary = _load_single_csv_row(baseline_dir / "evaluation_summary.csv")

    current_pages = _read_csv_rows(current_dir / "per_page_metrics.csv")
    baseline_pages = _read_csv_rows(baseline_dir / "per_page_metrics.csv")

    current_language = _read_csv_rows(current_dir / "metrics_by_language.csv")
    if not current_language:
        current_language = _aggregate_rows(current_pages, "language_primary", "language_primary")

    current_layout = _read_csv_rows(current_dir / "metrics_by_layout.csv")
    if not current_layout:
        current_layout = _aggregate_rows(current_pages, "layout_type", "layout_type")

    current_engine = _read_csv_rows(current_dir / "metrics_by_engine.csv")
    if not current_engine:
        current_engine = _build_engine_metrics(current_pages)

    baseline_language = _read_csv_rows(baseline_dir / "metrics_by_language.csv")
    if not baseline_language:
        baseline_language = _aggregate_rows(baseline_pages, "language_primary", "language_primary")

    baseline_layout = _read_csv_rows(baseline_dir / "metrics_by_layout.csv")
    if not baseline_layout:
        baseline_layout = _aggregate_rows(baseline_pages, "layout_type", "layout_type")

    baseline_engine = _read_csv_rows(baseline_dir / "metrics_by_engine.csv")
    if not baseline_engine:
        baseline_engine = _build_engine_metrics(baseline_pages)

    comparison = (
        _compare_overall_metrics(current_summary or {}, baseline_summary or {})
        if current_summary and baseline_summary
        else None
    )
    page_comparison = _page_regression_rows(current_pages, baseline_pages, top_n=top_n)
    language_changes = _summarize_group_changes(current_language, baseline_language, "language_primary")
    layout_changes = _summarize_group_changes(current_layout, baseline_layout, "layout_type")
    engine_changes = _summarize_group_changes(current_engine, baseline_engine, "engine")
    runtime_changes = _extract_runtime_changes(comparison)

    recommendations = _build_recommendations(
        summary=current_summary or {},
        by_scan_quality=_aggregate_rows(current_pages, "scan_quality", "scan_quality"),
        by_layout=current_layout,
        by_engine=current_engine,
        regression={"layouts_worsened": layout_changes.get("worsened", [])},
        per_page_rows=current_pages,
        baseline_pages=baseline_pages,
    )

    failing_pages_rows = _collect_failing_pages(current_pages, page_comparison)

    regression_markdown = _build_regression_markdown(
        comparison=comparison,
        page_comparison=page_comparison,
        language_changes=language_changes,
        layout_changes=layout_changes,
        engine_changes=engine_changes,
        runtime_changes=runtime_changes,
        recommendations=recommendations,
    )

    regression_report_path = output_dir / "regression_report.md"
    failing_pages_path = output_dir / "failing_pages.csv"
    comparison_json_path = output_dir / "regression_comparison.json"

    _write_text(regression_report_path, regression_markdown)
    _write_csv(
        failing_pages_path,
        failing_pages_rows,
        [
            "page_key",
            "normalized_page_key",
            "pdf_name",
            "page",
            "status",
            "failure_reason",
            "cer",
            "wer",
            "runtime_ms",
            "language_primary",
            "layout_type",
            "engine_used",
            "key_normalization_applied",
            "key_normalization_warnings",
            "regression_delta",
            "reason",
        ],
    )
    _write_json(
        comparison_json_path,
        {
            "overall": comparison,
            "pages": page_comparison,
            "languages": language_changes,
            "layouts": layout_changes,
            "engines": engine_changes,
            "runtime_changes": runtime_changes,
            "recommended_next_actions": recommendations,
        },
    )

    return {
        "regression_report_md": str(regression_report_path),
        "failing_pages_csv": str(failing_pages_path),
        "regression_comparison_json": str(comparison_json_path),
    }
