#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from production.document_model import validate_document_model
from production.document_model_adapters import (
    from_current_pipeline,
    from_docling,
    from_marker,
    from_paddle,
    from_surya,
)

BENCHMARK_ROOT = ROOT / "reports" / "real_gold_eval_runs" / "smoke_50"
RUN_DIR = BENCHMARK_ROOT / "run"
INPUT_DIR = BENCHMARK_ROOT / "input_pdfs"
GOLD_CSV = BENCHMARK_ROOT / "gold.csv"
CURRENT_CSV = RUN_DIR / "client_page_text.csv"
CURRENT_LAYOUT_JSONL = RUN_DIR / "layout_regions.jsonl"

REPORTS = ROOT / "reports"
RAW_DIR = REPORTS / "industry_baseline_raw"
PER_PAGE_JSONL = REPORTS / "industry_baseline_per_page_outputs.jsonl"
MATRIX_CSV = REPORTS / "industry_baseline_bakeoff_matrix.csv"
REPORT_MD = REPORTS / "industry_baseline_bakeoff_report.md"
INSTALL_NOTES_MD = REPORTS / "industry_baseline_install_runtime_notes.md"


BACKENDS = [
    "current_pipeline",
    "paddleocr_ppstructure",
    "surya",
    "docling",
    "marker",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _safe_float(value: Any) -> float | None:
    try:
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _edit_distance(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = list(range(len(b) + 1))
    for i, av in enumerate(a, start=1):
        prev = dp[0]
        dp[0] = i
        for j, bv in enumerate(b, start=1):
            cost = 0 if av == bv else 1
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
    return dp[-1]


def _cer(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _edit_distance(list(reference), list(hypothesis)) / len(reference)


def _wer(reference: str, hypothesis: str) -> float:
    r = reference.split()
    h = hypothesis.split()
    if not r:
        return 0.0 if not h else 1.0
    return _edit_distance(r, h) / len(r)


def _line_order_similarity(reference: str, hypothesis: str) -> float:
    r = [x.strip() for x in reference.splitlines() if x.strip()]
    h = [x.strip() for x in hypothesis.splitlines() if x.strip()]
    if not r:
        return 1.0 if not h else 0.0
    if not h:
        return 0.0
    return max(0.0, 1.0 - (_edit_distance(r, h) / len(r)))


def _paragraph_order_similarity(reference: str, hypothesis: str) -> float:
    r = [x.strip() for x in re.split(r"\n\s*\n", reference) if x.strip()]
    h = [x.strip() for x in re.split(r"\n\s*\n", hypothesis) if x.strip()]
    if not r:
        return 1.0 if not h else 0.0
    if not h:
        return 0.0
    return max(0.0, 1.0 - (_edit_distance(r, h) / len(r)))


def _extract_kv(text: str) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for line in text.splitlines():
        s = line.strip()
        if ":" not in s:
            continue
        k, v = s.split(":", 1)
        k = k.strip().lower()
        v = v.strip().lower()
        if k and v:
            out.add((k, v))
    return out


def _f1(a: set[tuple[str, str]], b: set[tuple[str, str]]) -> float | None:
    if not a and not b:
        return None
    if not a or not b:
        return 0.0
    tp = len(a & b)
    p = tp / len(b)
    r = tp / len(a)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def _metric_family(dataset_id: str, document_type: str) -> str:
    ds = dataset_id.lower()
    dt = document_type.lower()
    if "cord" in ds or "receipt" in dt:
        return "structured_receipt"
    if "funsd" in ds or "form" in dt:
        return "structured_form"
    if "local_gold" in ds:
        return "local_scope_validated"
    return "plain_text_layout"


def _runtime_quantiles(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]
    ordered = sorted(values)
    q90_index = int(round((len(ordered) - 1) * 0.9))
    q95_index = int(round((len(ordered) - 1) * 0.95))
    return ordered[q90_index], ordered[q95_index]


def _run_cmd(cmd: list[str], timeout_s: int = 120) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    return proc.returncode, proc.stdout, proc.stderr


def _discover_backend_availability() -> dict[str, dict[str, Any]]:
    checks = {
        "paddleocr_ppstructure": {
            "import": "paddleocr",
            "install": [sys.executable, "-m", "pip", "install", "paddleocr", "paddlepaddle", "--quiet"],
        },
        "surya": {
            "import": "surya",
            "install": [sys.executable, "-m", "pip", "install", "surya-ocr", "--quiet"],
        },
        "docling": {
            "import": "docling",
            "install": [sys.executable, "-m", "pip", "install", "docling", "--quiet"],
        },
        "marker": {
            "import": "marker",
            "install": [sys.executable, "-m", "pip", "install", "marker-pdf", "--quiet"],
        },
    }
    availability: dict[str, dict[str, Any]] = {}
    for backend in BACKENDS:
        if backend == "current_pipeline":
            availability[backend] = {
                "status": "available",
                "install_attempted": False,
                "install_note": "in-repo baseline",
            }
            continue

        spec = checks.get(backend)
        if spec is None:
            availability[backend] = {"status": "skipped", "install_attempted": False, "install_note": "no checker"}
            continue

        code = f"import importlib.util;print('ok' if importlib.util.find_spec('{spec['import']}') else 'missing')"
        rc, out, err = _run_cmd([sys.executable, "-c", code], timeout_s=30)
        if rc == 0 and "ok" in out:
            availability[backend] = {
                "status": "available",
                "install_attempted": False,
                "install_note": "already installed",
            }
            continue

        install_note = "install not attempted"
        try:
            irc, iout, ierr = _run_cmd(spec["install"], timeout_s=600)
            if irc == 0:
                rr, rout, rerr = _run_cmd([sys.executable, "-c", code], timeout_s=30)
                if rr == 0 and "ok" in rout:
                    availability[backend] = {
                        "status": "available",
                        "install_attempted": True,
                        "install_note": "installed during bakeoff",
                    }
                    continue
                install_note = f"installed command ok but import unresolved: {rout.strip()} {rerr.strip()}"
            else:
                install_note = f"install failed rc={irc}: {(ierr or iout).strip()[:240]}"
        except Exception as exc:
            install_note = f"install error: {exc}"

        availability[backend] = {
            "status": "skipped",
            "install_attempted": True,
            "install_note": install_note,
        }
    return availability


def _load_gold_map() -> dict[tuple[str, int], dict[str, Any]]:
    rows = _read_csv(GOLD_CSV)
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        pdf_name = str(row.get("pdf_name", "")).strip()
        page = int(float(str(row.get("page", "1") or "1")))
        out[(pdf_name, page)] = {
            "gold_text": str(row.get("ground_truth_text", "") or ""),
            "dataset_id": str(row.get("dataset_id", "") or "unknown"),
            "document_type": str(row.get("document_type", "") or "unknown"),
            "layout_type": str(row.get("layout_type", "") or "unknown"),
            "language_primary": str(row.get("language_primary", "") or "unknown"),
            "page_reference": str(row.get("page_reference", "") or ""),
        }
    return out


def _load_current_outputs() -> tuple[dict[tuple[str, int], dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    ocr_rows = _read_csv(CURRENT_CSV)
    ocr_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in ocr_rows:
        key = (str(row.get("pdf_name", "")).strip(), int(float(str(row.get("page", "1") or "1")))
        )
        ocr_by_key[key] = row

    layout_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    if CURRENT_LAYOUT_JSONL.exists():
        with CURRENT_LAYOUT_JSONL.open("r", encoding="utf-8") as fh:
            for line in fh:
                payload = line.strip()
                if not payload:
                    continue
                obj = json.loads(payload)
                key = (str(obj.get("pdf_name", "")).strip(), int(obj.get("page", 1)))
                layout_by_key[key] = obj
    return ocr_by_key, layout_by_key


def _iter_pages() -> list[tuple[str, int, Path]]:
    pages: list[tuple[str, int, Path]] = []
    for pdf in sorted(INPUT_DIR.glob("*.pdf")):
        pages.append((pdf.name, 1, pdf))
    return pages


def _extract_with_paddle(pdf_path: Path) -> dict[str, Any]:
    from paddleocr import PaddleOCR
    import fitz

    started = time.perf_counter()
    ocr = PaddleOCR(use_angle_cls=True, lang="en")
    with fitz.open(str(pdf_path)) as doc:
        page = doc[0]
        pix = page.get_pixmap(alpha=False)
        width = float(pix.width)
        height = float(pix.height)
        image_bytes = pix.tobytes("png")

    import numpy as np
    import cv2

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    result = ocr.ocr(image, cls=True)
    lines: list[dict[str, Any]] = []
    texts: list[str] = []
    for block in (result or []):
        for item in (block or []):
            if not item or len(item) < 2:
                continue
            poly = item[0]
            txt = str(item[1][0])
            conf = float(item[1][1]) if item[1][1] is not None else None
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            bbox = [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
            lines.append({"text": txt, "bbox": bbox, "confidence": conf})
            texts.append(txt)

    runtime_ms = (time.perf_counter() - started) * 1000.0
    return {
        "text": "\n".join(texts),
        "lines": lines,
        "width": width,
        "height": height,
        "runtime_ms": runtime_ms,
    }


def _extract_with_docling(pdf_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    doc = result.document
    markdown = doc.export_to_markdown()
    structured = doc.export_to_dict()
    text = doc.export_to_text() if hasattr(doc, "export_to_text") else markdown
    runtime_ms = (time.perf_counter() - started) * 1000.0
    return {
        "text": text,
        "markdown": markdown,
        "structured": structured if isinstance(structured, dict) else {},
        "width": None,
        "height": None,
        "runtime_ms": runtime_ms,
    }


def _extract_with_marker(pdf_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    import marker

    if hasattr(marker, "convert"):
        rendered = marker.convert(str(pdf_path))
        if isinstance(rendered, dict):
            markdown = str(rendered.get("markdown", "") or "")
            text = str(rendered.get("text", "") or markdown)
        else:
            markdown = str(rendered)
            text = markdown
    else:
        raise RuntimeError("marker module does not expose convert API")

    runtime_ms = (time.perf_counter() - started) * 1000.0
    return {
        "text": text,
        "markdown": markdown,
        "width": None,
        "height": None,
        "runtime_ms": runtime_ms,
    }


def _extract_with_surya(pdf_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    # API surfaces vary across releases; support best-effort import and call patterns.
    try:
        from surya.ocr import run_ocr  # type: ignore
        result = run_ocr(str(pdf_path))
    except Exception:
        from surya import run_ocr  # type: ignore
        result = run_ocr(str(pdf_path))

    text = ""
    lines: list[dict[str, Any]] = []
    if isinstance(result, dict):
        text = str(result.get("text", "") or "")
        line_items = result.get("lines", []) if isinstance(result.get("lines", []), list) else []
        for item in line_items:
            if isinstance(item, dict):
                lines.append(
                    {
                        "text": str(item.get("text", "") or ""),
                        "bbox": item.get("bbox"),
                        "confidence": _safe_float(item.get("confidence")),
                    }
                )
    runtime_ms = (time.perf_counter() - started) * 1000.0
    return {
        "text": text,
        "lines": lines,
        "width": None,
        "height": None,
        "runtime_ms": runtime_ms,
    }


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    availability = _discover_backend_availability()

    gold = _load_gold_map()
    current_rows, current_layout = _load_current_outputs()
    pages = _iter_pages()

    per_page_rows: list[dict[str, Any]] = []
    backend_notes: list[dict[str, Any]] = []

    for backend in BACKENDS:
        backend_dir = RAW_DIR / backend
        backend_dir.mkdir(parents=True, exist_ok=True)

        available = availability.get(backend, {"status": "skipped", "install_note": "unknown"})
        if available.get("status") != "available":
            backend_notes.append(
                {
                    "backend": backend,
                    "status": "skipped",
                    "install_attempted": bool(available.get("install_attempted", False)),
                    "note": str(available.get("install_note", "")),
                }
            )
            continue

        backend_failures = 0
        backend_runtime: list[float] = []
        dependency_issues = 0
        memory_issues = 0

        for pdf_name, page, pdf_path in pages:
            key = (pdf_name, page)
            gold_meta = gold.get(key)
            if not gold_meta:
                continue

            started = time.perf_counter()
            result_text = ""
            model = None
            status = "success"
            issue = ""

            try:
                if backend == "current_pipeline":
                    row = current_rows.get(key, {})
                    if not row:
                        raise RuntimeError("missing current pipeline row")
                    layout = current_layout.get(key)
                    model = from_current_pipeline(row, structured_layout=(layout or {}).get("layout", {}))
                    result_text = str(row.get("page_text", "") or "")
                    runtime_ms = _safe_float(row.get("runtime_ms")) or ((time.perf_counter() - started) * 1000.0)
                elif backend == "paddleocr_ppstructure":
                    extracted = _extract_with_paddle(pdf_path)
                    result_text = extracted["text"]
                    runtime_ms = float(extracted["runtime_ms"])
                    model = from_paddle(
                        pdf_name=pdf_name,
                        page=page,
                        text=result_text,
                        lines=extracted["lines"],
                        width=extracted["width"],
                        height=extracted["height"],
                        runtime_ms=runtime_ms,
                    )
                elif backend == "surya":
                    extracted = _extract_with_surya(pdf_path)
                    result_text = extracted["text"]
                    runtime_ms = float(extracted["runtime_ms"])
                    model = from_surya(
                        pdf_name=pdf_name,
                        page=page,
                        text=result_text,
                        lines=extracted["lines"],
                        width=extracted["width"],
                        height=extracted["height"],
                    )
                elif backend == "docling":
                    extracted = _extract_with_docling(pdf_path)
                    result_text = extracted["text"]
                    runtime_ms = float(extracted["runtime_ms"])
                    model = from_docling(
                        pdf_name=pdf_name,
                        page=page,
                        text=result_text,
                        markdown=extracted["markdown"],
                        structured=extracted["structured"],
                        width=extracted["width"],
                        height=extracted["height"],
                    )
                elif backend == "marker":
                    extracted = _extract_with_marker(pdf_path)
                    result_text = extracted["text"]
                    runtime_ms = float(extracted["runtime_ms"])
                    model = from_marker(
                        pdf_name=pdf_name,
                        page=page,
                        text=result_text,
                        markdown=extracted["markdown"],
                        width=extracted["width"],
                        height=extracted["height"],
                    )
                else:
                    raise RuntimeError(f"unsupported backend {backend}")
            except MemoryError:
                status = "failed"
                issue = "memory_error"
                backend_failures += 1
                memory_issues += 1
                runtime_ms = (time.perf_counter() - started) * 1000.0
            except Exception as exc:
                status = "failed"
                issue = str(exc)
                backend_failures += 1
                runtime_ms = (time.perf_counter() - started) * 1000.0
                if "No module named" in issue or "cannot import" in issue.lower():
                    dependency_issues += 1
                if "memory" in issue.lower() or "oom" in issue.lower():
                    memory_issues += 1
                model = None

            backend_runtime.append(runtime_ms)

            gold_text = str(gold_meta.get("gold_text", "") or "")
            cer = _cer(gold_text, result_text) if status == "success" else None
            wer = _wer(gold_text, result_text) if status == "success" else None
            line_sim = _line_order_similarity(gold_text, result_text) if status == "success" else None
            para_sim = _paragraph_order_similarity(gold_text, result_text) if status == "success" else None
            kv_f1 = _f1(_extract_kv(gold_text), _extract_kv(result_text)) if status == "success" else None
            empty = (not result_text.strip()) if status == "success" else True
            family = _metric_family(str(gold_meta.get("dataset_id", "")), str(gold_meta.get("document_type", "")))

            if model is not None:
                errs = validate_document_model(model)
                if errs:
                    status = "failed"
                    issue = "document_model_validation_failed: " + "; ".join(errs[:3])
                    backend_failures += 1

            raw_payload = {
                "backend": backend,
                "pdf_name": pdf_name,
                "page": page,
                "status": status,
                "issue": issue,
                "runtime_ms": runtime_ms,
                "result_text": result_text,
                "document_model": model,
            }
            with (backend_dir / f"{Path(pdf_name).stem}_p{page}.json").open("w", encoding="utf-8") as fh:
                json.dump(raw_payload, fh, ensure_ascii=False, indent=2)

            per_page_rows.append(
                {
                    "backend": backend,
                    "pdf_name": pdf_name,
                    "page": page,
                    "dataset_id": gold_meta.get("dataset_id", "unknown"),
                    "document_type": gold_meta.get("document_type", "unknown"),
                    "layout_type": gold_meta.get("layout_type", "unknown"),
                    "metric_family": family,
                    "status": status,
                    "issue": issue,
                    "runtime_ms": runtime_ms,
                    "empty_output": empty,
                    "cer": cer,
                    "wer": wer,
                    "line_order_similarity": line_sim,
                    "paragraph_order_similarity": para_sim,
                    "key_value_f1": kv_f1,
                    "document_model_available": bool(model is not None),
                }
            )

        backend_notes.append(
            {
                "backend": backend,
                "status": "ran",
                "install_attempted": bool(available.get("install_attempted", False)),
                "note": str(available.get("install_note", "")),
                "pages": len(pages),
                "failed_pages": backend_failures,
                "dependency_issues": dependency_issues,
                "memory_issues": memory_issues,
            }
        )

    with PER_PAGE_JSONL.open("w", encoding="utf-8") as fh:
        for row in per_page_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Aggregate matrix.
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in per_page_rows:
        key = (str(row["backend"]), str(row["dataset_id"]), str(row["document_type"]))
        grouped.setdefault(key, []).append(row)

    matrix_rows: list[dict[str, Any]] = []
    for (backend, dataset_id, document_type), items in sorted(grouped.items()):
        cer_vals = [float(x["cer"]) for x in items if x["cer"] is not None]
        wer_vals = [float(x["wer"]) for x in items if x["wer"] is not None]
        line_vals = [float(x["line_order_similarity"]) for x in items if x["line_order_similarity"] is not None]
        para_vals = [float(x["paragraph_order_similarity"]) for x in items if x["paragraph_order_similarity"] is not None]
        kv_vals = [float(x["key_value_f1"]) for x in items if x["key_value_f1"] is not None]
        runtime_vals = [float(x["runtime_ms"]) for x in items]
        p90, p95 = _runtime_quantiles(runtime_vals)
        failed_rate = sum(1 for x in items if x["status"] != "success") / len(items)
        empty_rate = sum(1 for x in items if x["empty_output"]) / len(items)

        matrix_rows.append(
            {
                "backend": backend,
                "dataset_id": dataset_id,
                "document_type": document_type,
                "pages": len(items),
                "cer_mean": statistics.mean(cer_vals) if cer_vals else None,
                "wer_mean": statistics.mean(wer_vals) if wer_vals else None,
                "line_order_similarity_mean": statistics.mean(line_vals) if line_vals else None,
                "paragraph_order_similarity_mean": statistics.mean(para_vals) if para_vals else None,
                "key_value_f1_mean": statistics.mean(kv_vals) if kv_vals else None,
                "runtime_ms_p90": p90,
                "runtime_ms_p95": p95,
                "failed_rate": failed_rate,
                "empty_rate": empty_rate,
            }
        )

    with MATRIX_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "backend",
                "dataset_id",
                "document_type",
                "pages",
                "cer_mean",
                "wer_mean",
                "line_order_similarity_mean",
                "paragraph_order_similarity_mean",
                "key_value_f1_mean",
                "runtime_ms_p90",
                "runtime_ms_p95",
                "failed_rate",
                "empty_rate",
            ],
        )
        writer.writeheader()
        for row in matrix_rows:
            writer.writerow(row)

    with INSTALL_NOTES_MD.open("w", encoding="utf-8") as fh:
        fh.write("# Industry Baseline Install/Runtime Notes\n\n")
        for note in backend_notes:
            fh.write(f"- {note['backend']}: status={note['status']} install_attempted={note.get('install_attempted', False)}")
            if "pages" in note:
                fh.write(
                    f" pages={note.get('pages', 0)} failed_pages={note.get('failed_pages', 0)} dependency_issues={note.get('dependency_issues', 0)} memory_issues={note.get('memory_issues', 0)}"
                )
            fh.write(f" note={note.get('note', '')}\n")

    def _best_for(predicate):
        candidates = [r for r in matrix_rows if predicate(r) and r.get("cer_mean") is not None]
        if not candidates:
            return None
        return min(candidates, key=lambda x: (float(x["cer_mean"]), float(x.get("wer_mean") or 9999.0)))

    best_map = {
        "historical books": _best_for(lambda r: "ocrd" in str(r.get("dataset_id", "")).lower() or "book" in str(r.get("document_type", "")).lower()),
        "receipts": _best_for(lambda r: "receipt" in str(r.get("document_type", "")).lower() or "cord" in str(r.get("dataset_id", "")).lower()),
        "forms": _best_for(lambda r: "form" in str(r.get("document_type", "")).lower() or "funsd" in str(r.get("dataset_id", "")).lower()),
        "typewritten documents": _best_for(lambda r: "type" in str(r.get("document_type", "")).lower()),
        "local_gold / academic pages": _best_for(lambda r: "local_gold" in str(r.get("dataset_id", "")).lower() or "academic" in str(r.get("document_type", "")).lower()),
        "general documents": _best_for(lambda r: True),
    }

    with REPORT_MD.open("w", encoding="utf-8") as fh:
        fh.write("# Open-Source Industry Baseline Bakeoff\n\n")
        fh.write("## Scope\n")
        fh.write("- Verified subset: reports/real_gold_eval_runs/smoke_50\n")
        fh.write("- Baselines requested: current pipeline, PaddleOCR/PP-Structure-like, Surya, Docling, Marker\n")
        fh.write("- Outputs normalized to DocumentModel when conversion succeeded\n\n")

        fh.write("## Backend status\n")
        for note in backend_notes:
            fh.write(f"- {note['backend']}: {note['status']} ({note.get('note', '')})\n")

        fh.write("\n## Best backend by category (CER/WER-driven on available runs)\n")
        for category, row in best_map.items():
            if row is None:
                fh.write(f"- {category}: no successful comparable run\n")
            else:
                fh.write(
                    f"- {category}: {row['backend']} (dataset={row['dataset_id']}, document_type={row['document_type']}, CER={row['cer_mean']:.4f}, WER={row['wer_mean']:.4f})\n"
                )

        fh.write("\n## Notes\n")
        fh.write("- This bakeoff does not trigger global replacement decisions.\n")
        fh.write("- Failures/installation issues are recorded in reports/industry_baseline_install_runtime_notes.md.\n")
        fh.write("- Raw outputs are preserved under reports/industry_baseline_raw/<backend>/.\n")

    print(f"Wrote {PER_PAGE_JSONL}")
    print(f"Wrote {MATRIX_CSV}")
    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {INSTALL_NOTES_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
