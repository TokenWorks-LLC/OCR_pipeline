#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import sys
import traceback
from dataclasses import dataclass
from datetime import date
from importlib import metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def _safe_distribution_version(*names: str) -> str:
    for name in names:
        try:
            return metadata.version(name)
        except Exception:
            continue
    return ""


def _module_file(module: Any) -> str:
    try:
        value = getattr(module, "__file__", "")
        if value is None:
            return ""
        return str(value)
    except Exception:
        return ""


def _classify(import_ok: bool, api_ok: bool, detail: str, expected_symbol: str) -> tuple[str, str]:
    text = str(detail or "").lower()
    if not import_ok:
        return "unavailable_dependency_error", "module_import_failed"
    if api_ok:
        return "ready", "api_contract_ok"
    if "cannot import" in text or "no module named" in text:
        if "huggingface_hub" in text or "is_offline_mode" in text:
            return "dependency_version_mismatch", "module_importable_but_transitive_dependency_incompatible"
        return "unavailable_dependency_error", "required_symbol_missing"
    if "has no attribute" in text and expected_symbol:
        return "adapter_api_mismatch", "module_present_but_expected_api_missing"
    return "adapter_api_mismatch", "api_contract_check_failed"


@dataclass
class BackendDependencyRow:
    backend: str
    module_name: str
    distribution_name: str
    distribution_version: str
    import_ok: bool
    module_path: str
    api_check: str
    api_ok: bool
    detail: str
    readiness_class: str
    status_reason: str
    recommended_action: str

    def to_row(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "module_name": self.module_name,
            "distribution_name": self.distribution_name,
            "distribution_version": self.distribution_version,
            "import_ok": self.import_ok,
            "module_path": self.module_path,
            "api_check": self.api_check,
            "api_ok": self.api_ok,
            "detail": self.detail,
            "readiness_class": self.readiness_class,
            "status_reason": self.status_reason,
            "recommended_action": self.recommended_action,
        }


def _check_paddle() -> BackendDependencyRow:
    detail = ""
    import_ok = False
    api_ok = False
    module_path = ""
    try:
        mod = importlib.import_module("paddleocr")
        import_ok = True
        module_path = _module_file(mod)
        api_ok = hasattr(mod, "PaddleOCR")
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"

    if not detail and not api_ok:
        detail = "module imported but PaddleOCR symbol not found"

    readiness_class, status_reason = _classify(import_ok, api_ok, detail, "PaddleOCR")
    if readiness_class == "ready":
        action = "no_change"
    else:
        action = "reinstall_or_repair_paddleocr_stack"

    return BackendDependencyRow(
        backend="paddleocr_ppstructure",
        module_name="paddleocr",
        distribution_name="paddleocr",
        distribution_version=_safe_distribution_version("paddleocr"),
        import_ok=import_ok,
        module_path=module_path,
        api_check="hasattr(paddleocr, 'PaddleOCR')",
        api_ok=api_ok,
        detail=detail,
        readiness_class=readiness_class,
        status_reason=status_reason,
        recommended_action=action,
    )


def _check_docling() -> BackendDependencyRow:
    detail = ""
    import_ok = False
    api_ok = False
    module_path = ""
    try:
        mod = importlib.import_module("docling")
        import_ok = True
        module_path = _module_file(mod)
        from docling.document_converter import DocumentConverter  # noqa: F401

        api_ok = True
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"

    readiness_class, status_reason = _classify(import_ok, api_ok, detail, "docling.document_converter.DocumentConverter")
    if readiness_class == "ready":
        action = "no_change"
    elif "is_offline_mode" in detail or "huggingface_hub" in detail:
        action = "align_huggingface_hub_with_docling_compatible_version_or_upgrade_docling_adapter"
    else:
        action = "repair_docling_install_or_adapter"

    return BackendDependencyRow(
        backend="docling",
        module_name="docling",
        distribution_name="docling",
        distribution_version=_safe_distribution_version("docling"),
        import_ok=import_ok,
        module_path=module_path,
        api_check="from docling.document_converter import DocumentConverter",
        api_ok=api_ok,
        detail=detail,
        readiness_class=readiness_class,
        status_reason=status_reason,
        recommended_action=action,
    )


def _check_marker() -> BackendDependencyRow:
    detail = ""
    import_ok = False
    api_ok = False
    module_path = ""
    try:
        mod = importlib.import_module("marker")
        import_ok = True
        module_path = _module_file(mod)
        api_ok = hasattr(mod, "convert")
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"

    if not detail and not api_ok:
        detail = "module imported but marker.convert was not found"

    readiness_class, status_reason = _classify(import_ok, api_ok, detail, "marker.convert")
    if readiness_class == "ready":
        action = "no_change"
    else:
        action = "update_marker_adapter_for_current_marker_pdf_entrypoint_or_pin_compatible_marker_release"

    return BackendDependencyRow(
        backend="marker",
        module_name="marker",
        distribution_name="marker-pdf",
        distribution_version=_safe_distribution_version("marker-pdf", "marker"),
        import_ok=import_ok,
        module_path=module_path,
        api_check="hasattr(marker, 'convert')",
        api_ok=api_ok,
        detail=detail,
        readiness_class=readiness_class,
        status_reason=status_reason,
        recommended_action=action,
    )


def _check_surya() -> BackendDependencyRow:
    detail = ""
    import_ok = False
    api_ok = False
    module_path = ""
    try:
        mod = importlib.import_module("surya")
        import_ok = True
        module_path = _module_file(mod)

        try:
            from surya.ocr import run_ocr  # type: ignore  # noqa: F401

            api_ok = True
        except Exception:
            from surya import run_ocr  # type: ignore  # noqa: F401

            api_ok = True
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        api_ok = False
    else:
        if not api_ok:
            detail = "surya imported but run_ocr entrypoint not found in surya.ocr or surya"

    readiness_class, status_reason = _classify(import_ok, api_ok, detail, "surya.ocr.run_ocr")
    if readiness_class == "ready":
        action = "no_change"
    else:
        action = "update_surya_adapter_to_current_api_or_pin_surya_ocr_release_with_run_ocr"

    return BackendDependencyRow(
        backend="surya",
        module_name="surya",
        distribution_name="surya-ocr",
        distribution_version=_safe_distribution_version("surya-ocr", "surya"),
        import_ok=import_ok,
        module_path=module_path,
        api_check="from surya.ocr import run_ocr OR from surya import run_ocr",
        api_ok=api_ok,
        detail=detail,
        readiness_class=readiness_class,
        status_reason=status_reason,
        recommended_action=action,
    )


def build_rows() -> list[BackendDependencyRow]:
    return [_check_paddle(), _check_docling(), _check_marker(), _check_surya()]


def write_csv(path: Path, rows: list[BackendDependencyRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "backend",
        "module_name",
        "distribution_name",
        "distribution_version",
        "import_ok",
        "module_path",
        "api_check",
        "api_ok",
        "detail",
        "readiness_class",
        "status_reason",
        "recommended_action",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_row())


def write_md(path: Path, rows: list[BackendDependencyRow]) -> None:
    today = date.today().isoformat()
    lines: list[str] = []
    lines.append("# Optional Backend Dependency Diagnostics")
    lines.append("")
    lines.append(f"Date: {today}")
    lines.append("")
    lines.append("## Results")
    for row in rows:
        lines.append(
            "- "
            + f"{row.backend}: readiness={row.readiness_class} "
            + f"api_ok={row.api_ok} "
            + f"dist={row.distribution_name}=={row.distribution_version or 'unknown'} "
            + f"reason={row.status_reason}"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("- This diagnostic checks adapter entrypoints, not OCR quality.")
    lines.append("- Use the phase health gate and bakeoff metrics before any routing decision.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(output_csv: Path, output_md: Path) -> int:
    try:
        rows = build_rows()
        write_csv(output_csv, rows)
        write_md(output_md, rows)
        print(f"Wrote {output_csv}")
        print(f"Wrote {output_md}")
        return 0
    except Exception:
        traceback.print_exc()
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose optional OCR backend dependency readiness.")
    parser.add_argument(
        "--output-csv",
        default=str(REPORTS / "phase19_optional_backend_dependency_diagnostics.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output-md",
        default=str(REPORTS / "phase19_optional_backend_dependency_diagnostics.md"),
        help="Output markdown path.",
    )
    args = parser.parse_args()

    return run(Path(args.output_csv), Path(args.output_md))


if __name__ == "__main__":
    raise SystemExit(main())
