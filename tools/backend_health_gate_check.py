#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STATUS_ELIGIBLE_DEFAULT = "eligible_default"
STATUS_ELIGIBLE_SPECIALIST = "eligible_specialist"
STATUS_EXPERIMENTAL_ONLY = "experimental_only"
STATUS_DISABLED_UNHEALTHY = "disabled_unhealthy"
STATUS_UNAVAILABLE_DEPENDENCY = "unavailable_dependency_error"

ALLOWED_STATUSES = {
    STATUS_ELIGIBLE_DEFAULT,
    STATUS_ELIGIBLE_SPECIALIST,
    STATUS_EXPERIMENTAL_ONLY,
    STATUS_DISABLED_UNHEALTHY,
    STATUS_UNAVAILABLE_DEPENDENCY,
}


@dataclass
class BackendHealthResult:
    backend: str
    import_ok: bool
    import_detail: str
    smoke_evidence: str
    pages: int
    failure_rate: float | None
    empty_rate: float | None
    runtime_p95_ms: float | None
    schema_compatible: bool
    document_model_conversion_rate: float | None
    eligibility_status: str
    status_reason: str

    def to_row(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "import_ok": self.import_ok,
            "import_detail": self.import_detail,
            "smoke_evidence": self.smoke_evidence,
            "pages": self.pages,
            "failure_rate": self.failure_rate,
            "empty_rate": self.empty_rate,
            "runtime_p95_ms": self.runtime_p95_ms,
            "schema_compatible": self.schema_compatible,
            "document_model_conversion_rate": self.document_model_conversion_rate,
            "eligibility_status": self.eligibility_status,
            "status_reason": self.status_reason,
        }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except Exception:
        return None


def _import_health(module_name: str) -> tuple[bool, str]:
    try:
        importlib.import_module(module_name)
        return True, "import_ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _aggregate_baseline_metrics(matrix_rows: list[dict[str, str]], backend: str) -> dict[str, Any]:
    rows = [r for r in matrix_rows if str(r.get("backend", "")).strip() == backend]
    failure = [_safe_float(r.get("failed_rate")) for r in rows]
    empty = [_safe_float(r.get("empty_rate")) for r in rows]
    p95 = [_safe_float(r.get("runtime_ms_p95")) for r in rows]

    return {
        "rows": rows,
        "pages": sum(int(float(str(r.get("pages", "0") or "0"))) for r in rows),
        "failure_rate": mean([v for v in failure if v is not None]) if any(v is not None for v in failure) else None,
        "empty_rate": mean([v for v in empty if v is not None]) if any(v is not None for v in empty) else None,
        "runtime_p95_ms": max([v for v in p95 if v is not None], default=None),
    }


def _load_baseline_matrix_rows(baseline_matrix_path: Path, routed_metrics_path: Path) -> list[dict[str, str]]:
    if baseline_matrix_path.exists():
        return _read_csv(baseline_matrix_path)

    if not routed_metrics_path.exists():
        return []

    # Fallback: routed experiment matrix includes overall rows for current/routed variants.
    rows = _read_csv(routed_metrics_path)
    fallback_rows: list[dict[str, str]] = []
    for row in rows:
        if str(row.get("dataset_id", "")).strip() != "ALL":
            continue
        if str(row.get("document_type", "")).strip() != "ALL":
            continue
        backend = str(row.get("pipeline_variant", "")).strip()
        if not backend:
            continue
        fallback_rows.append(
            {
                "backend": backend,
                "pages": str(row.get("pages", "0") or "0"),
                "failed_rate": str(row.get("failed_rate", "") or ""),
                "empty_rate": str(row.get("empty_rate", "") or ""),
                "runtime_ms_p95": str(row.get("runtime_ms_p95", "") or ""),
            }
        )
    return fallback_rows


def _aggregate_per_page_metrics(per_page_rows: list[dict[str, Any]], backend: str) -> dict[str, Any]:
    rows = [r for r in per_page_rows if str(r.get("backend", "")).strip() == backend]
    required_fields = {"backend", "status", "runtime_ms", "empty_output", "document_model_available"}
    schema_ok = all(required_fields.issubset(set(r.keys())) for r in rows) if rows else False

    model_values = [
        bool(r.get("document_model_available"))
        for r in rows
        if "document_model_available" in r
    ]
    model_rate = (sum(1 for v in model_values if v) / len(model_values)) if model_values else None

    return {
        "rows": rows,
        "schema_ok": schema_ok,
        "document_model_conversion_rate": model_rate,
    }


def _routed_health_from_phase8_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "pages": 0,
            "failure_rate": None,
            "empty_rate": None,
            "runtime_p95_ms": None,
            "non_regression": "missing",
        }

    rows = _read_csv(path)
    target = None
    for row in rows:
        if (
            row.get("benchmark_set") == "smoke_50_gate"
            and row.get("pipeline_variant") == "routed_document_type_pipeline"
            and row.get("aggregation_dimension") == "overall"
            and row.get("aggregation_value") == "ALL"
        ):
            target = row
            break

    if target is None:
        return {
            "pages": 0,
            "failure_rate": None,
            "empty_rate": None,
            "runtime_p95_ms": None,
            "non_regression": "missing",
        }

    return {
        "pages": int(float(str(target.get("pages", "0") or "0"))),
        "failure_rate": _safe_float(target.get("failed_rate")),
        "empty_rate": _safe_float(target.get("empty_output_rate")),
        "runtime_p95_ms": _safe_float(target.get("runtime_ms_p95")),
        "non_regression": str(target.get("non_regression_vs_current", "") or ""),
    }


def _classify_backend(
    *,
    backend: str,
    import_ok: bool,
    pages: int,
    failure_rate: float | None,
    empty_rate: float | None,
    schema_ok: bool,
    routed_non_regression: str | None = None,
) -> tuple[str, str]:
    if not import_ok:
        return STATUS_UNAVAILABLE_DEPENDENCY, "missing_or_broken_dependency"

    if not schema_ok:
        return STATUS_DISABLED_UNHEALTHY, "output_schema_incompatible"

    if pages <= 0:
        return STATUS_EXPERIMENTAL_ONLY, "missing_health_evidence"

    if failure_rate is not None and failure_rate >= 0.8:
        return STATUS_DISABLED_UNHEALTHY, "failure_rate_too_high"

    if empty_rate is not None and empty_rate >= 0.95:
        return STATUS_DISABLED_UNHEALTHY, "empty_rate_too_high"

    if backend == "current_pipeline":
        if (failure_rate or 0.0) <= 0.05 and (empty_rate or 0.0) <= 0.05:
            return STATUS_ELIGIBLE_DEFAULT, "stable_default_backend"
        return STATUS_ELIGIBLE_SPECIALIST, "usable_but_not_default_grade"

    if backend == "routed_document_type_pipeline":
        if routed_non_regression == "pass" and (failure_rate or 0.0) <= 0.05:
            return STATUS_ELIGIBLE_SPECIALIST, "smoke_non_regression_pass"
        return STATUS_EXPERIMENTAL_ONLY, "route_health_not_strong_enough"

    if backend == "paddleocr_ppstructure":
        if (failure_rate or 1.0) <= 0.1 and (empty_rate or 1.0) <= 0.4:
            return STATUS_ELIGIBLE_SPECIALIST, "specialist_backend_within_bounds"
        return STATUS_EXPERIMENTAL_ONLY, "quality_or_empty_rate_risk"

    if backend in {"docling", "marker", "surya"}:
        if (failure_rate or 1.0) >= 0.8:
            return STATUS_DISABLED_UNHEALTHY, "consistently_unhealthy"
        return STATUS_EXPERIMENTAL_ONLY, "not_proven_for_default_or_specialist"

    return STATUS_EXPERIMENTAL_ONLY, "default_experimental_classification"


def _build_backend_results(root: Path) -> list[BackendHealthResult]:
    baseline_matrix_path = root / "reports" / "industry_baseline_bakeoff_matrix.csv"
    routed_experiment_metrics_path = root / "reports" / "routed_pipeline_experiment_metrics.csv"
    per_page_path = root / "reports" / "industry_baseline_per_page_outputs.jsonl"
    phase8_metrics_path = root / "reports" / "private_beta_expanded_benchmark_metrics.csv"

    baseline_rows = _load_baseline_matrix_rows(baseline_matrix_path, routed_experiment_metrics_path)

    per_page_rows: list[dict[str, Any]] = []
    if per_page_path.exists():
        with per_page_path.open("r", encoding="utf-8") as fh:
            per_page_rows = [json.loads(line) for line in fh if line.strip()]

    routed_health = _routed_health_from_phase8_metrics(phase8_metrics_path)

    checks = [
        ("current_pipeline", "production.ensemble_ocr"),
        ("routed_document_type_pipeline", "tools.run_routed_document_pipeline_experiment"),
        ("paddleocr_ppstructure", "paddleocr"),
        ("surya", "surya"),
        ("docling", "docling"),
        ("marker", "marker"),
    ]

    results: list[BackendHealthResult] = []
    for backend, module_name in checks:
        import_ok, import_detail = _import_health(module_name)

        if backend == "routed_document_type_pipeline":
            pages = int(routed_health.get("pages") or 0)
            failure_rate = _safe_float(routed_health.get("failure_rate"))
            empty_rate = _safe_float(routed_health.get("empty_rate"))
            runtime_p95_ms = _safe_float(routed_health.get("runtime_p95_ms"))
            schema_ok = True
            model_rate = 1.0
            smoke_evidence = "phase8_smoke_gate"
        else:
            baseline = _aggregate_baseline_metrics(baseline_rows, backend)
            per_page = _aggregate_per_page_metrics(per_page_rows, backend)
            pages = int(baseline.get("pages") or len(per_page.get("rows", [])))
            failure_rate = baseline.get("failure_rate")
            empty_rate = baseline.get("empty_rate")
            runtime_p95_ms = baseline.get("runtime_p95_ms")
            schema_ok = bool(per_page.get("schema_ok")) if per_page.get("rows") else backend in {
                "current_pipeline",
                "paddleocr_ppstructure",
                "docling",
                "marker",
            }
            model_rate = per_page.get("document_model_conversion_rate")
            smoke_evidence = "industry_baseline_bakeoff"

        eligibility, reason = _classify_backend(
            backend=backend,
            import_ok=import_ok,
            pages=pages,
            failure_rate=failure_rate,
            empty_rate=empty_rate,
            schema_ok=schema_ok,
            routed_non_regression=str(routed_health.get("non_regression", "")),
        )
        if eligibility not in ALLOWED_STATUSES:
            raise RuntimeError(f"invalid eligibility status: {eligibility}")

        results.append(
            BackendHealthResult(
                backend=backend,
                import_ok=import_ok,
                import_detail=import_detail,
                smoke_evidence=smoke_evidence,
                pages=pages,
                failure_rate=failure_rate,
                empty_rate=empty_rate,
                runtime_p95_ms=runtime_p95_ms,
                schema_compatible=schema_ok,
                document_model_conversion_rate=model_rate,
                eligibility_status=eligibility,
                status_reason=reason,
            )
        )

    return results


def _write_csv(path: Path, rows: list[BackendHealthResult]) -> None:
    fieldnames = [
        "backend",
        "import_ok",
        "import_detail",
        "smoke_evidence",
        "pages",
        "failure_rate",
        "empty_rate",
        "runtime_p95_ms",
        "schema_compatible",
        "document_model_conversion_rate",
        "eligibility_status",
        "status_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for result in rows:
            writer.writerow(result.to_row())


def _write_report(path: Path, rows: list[BackendHealthResult]) -> None:
    today = date.today().isoformat()
    lines: list[str] = []
    lines.append("# Backend Health Gate")
    lines.append("")
    lines.append(f"Date: {today}")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Backends evaluated: current pipeline, routed pipeline, Paddle/PP-Structure, Surya, Docling, Marker.")
    lines.append("- Gate statuses: eligible_default, eligible_specialist, experimental_only, disabled_unhealthy, unavailable_dependency_error.")
    lines.append("")
    lines.append("## Results")
    for row in rows:
        lines.append(
            "- "
            + f"{row.backend}: status={row.eligibility_status} "
            + f"failure_rate={row.failure_rate} empty_rate={row.empty_rate} "
            + f"runtime_p95_ms={row.runtime_p95_ms} reason={row.status_reason}"
        )
    lines.append("")
    lines.append("## Routing Enforcement")
    lines.append("- Default routing must only select backends with status=eligible_default.")
    lines.append("- Specialist routing may only select backends with status=eligible_specialist.")
    lines.append("- Backends classified as experimental_only, disabled_unhealthy, or unavailable_dependency_error are blocked from default selection.")
    lines.append("")
    lines.append("## No Silent Failure Rule")
    lines.append("- Every backend row includes explicit status_reason and import_detail for operator visibility.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(root: Path, *, enforce: bool) -> int:
    rows = _build_backend_results(root)

    csv_path = root / "reports" / "backend_health_gate.csv"
    md_path = root / "reports" / "backend_health_gate.md"
    _write_csv(csv_path, rows)
    _write_report(md_path, rows)

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")

    if enforce:
        status_by_backend = {row.backend: row.eligibility_status for row in rows}
        if status_by_backend.get("current_pipeline") != STATUS_ELIGIBLE_DEFAULT:
            print("ERROR: current_pipeline failed backend health enforcement")
            return 1
        routed_status = status_by_backend.get("routed_document_type_pipeline")
        if routed_status not in {STATUS_ELIGIBLE_SPECIALIST, STATUS_ELIGIBLE_DEFAULT}:
            print("ERROR: routed_document_type_pipeline failed specialist health enforcement")
            return 1

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run backend health eligibility checks.")
    parser.add_argument(
        "--enforce",
        action="store_true",
        default=False,
        help="Exit non-zero when required health gates fail (for CI).",
    )
    args = parser.parse_args()
    return run(ROOT, enforce=bool(args.enforce))


if __name__ == "__main__":
    raise SystemExit(main())
