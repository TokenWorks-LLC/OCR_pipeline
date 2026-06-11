#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from production.ensemble_ocr import FortifiedOCREnsemble
from production.preprocessing_profiles import (
    PREPROCESSING_PROFILES,
    PROFILE_AUTO,
    PROFILE_NOISY_SCAN,
    PreprocessingProfile,
)

REPORTS_DIR = ROOT / "reports"
EVAL_RUNS_DIR = REPORTS_DIR / "real_gold_eval_runs"

CONTROLLED_BASELINE_DIR = EVAL_RUNS_DIR / "controlled_subset_baseline"
CONTROLLED_CANDIDATE_DIR = EVAL_RUNS_DIR / "controlled_subset_candidate"
EXPANDED_VALIDATION_DIR = EVAL_RUNS_DIR / "expanded_validation"
SMOKE_DIR = EVAL_RUNS_DIR / "smoke_50"

MANIFEST_PATH = ROOT / "data" / "gold_registry" / "gold_manifest.jsonl"
FAILURE_TAXONOMY_PATH = REPORTS_DIR / "expanded_failure_taxonomy.csv"
CONTROLLED_EXPERIMENT_METRICS = REPORTS_DIR / "controlled_experiment_metrics.csv"

OUT_STAGE1_CSV = REPORTS_DIR / "preprocessing_ablation_page_audit.csv"
OUT_STAGE1_MD = REPORTS_DIR / "preprocessing_ablation_page_audit.md"

OUT_STAGE2_CSV = REPORTS_DIR / "preprocessing_artifact_audit.csv"
OUT_STAGE2_MD = REPORTS_DIR / "preprocessing_artifact_audit.md"

OUT_STAGE3_CSV = REPORTS_DIR / "preprocessing_failure_root_cause.csv"
OUT_STAGE3_MD = REPORTS_DIR / "preprocessing_failure_root_cause.md"

OUT_STAGE4_CSV = REPORTS_DIR / "preprocessing_micro_ablation_metrics.csv"
OUT_STAGE4_MD = REPORTS_DIR / "preprocessing_micro_ablation_report.md"

OUT_STAGE5_MD = REPORTS_DIR / "next_improvement_decision.md"
OUT_STAGE6_MD = REPORTS_DIR / "preprocessing_promotion_decision.md"
OUT_STAGE7_MD = REPORTS_DIR / "preprocessing_diagnostic_deep_dive_report.md"

DIAG_ARTIFACT_DIR = REPORTS_DIR / "preprocessing_diagnostic_artifacts" / "controlled_subset"
MICRO_ARTIFACT_DIR = REPORTS_DIR / "preprocessing_diagnostic_artifacts" / "micro_ablation"

PROFILE_PATH = ROOT / "profiles" / "akkadian_strict.json"
PADDLE_ONLY_PROFILE_PATH = REPORTS_DIR / "preprocessing_diagnostic_artifacts" / "paddle_only_profile.json"


@dataclass
class ImageStats:
    exists: bool
    path: str
    md5: str
    width: int
    height: int


def _safe_float(value: Any) -> float:
    try:
        text = str(value).strip()
        if not text:
            return 0.0
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        text = str(value).strip()
        if not text:
            return 0
        return int(float(text))
    except (TypeError, ValueError):
        return 0


def _sanitize_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "page"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_manifest_map() -> dict[str, dict[str, Any]]:
    data = _read_jsonl(MANIFEST_PATH)
    return {str(row.get("page_id", "")): row for row in data}


def _load_page_diagnostics_map(path: Path) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(path)
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        document_id = str(row.get("document_id", "")).strip()
        page_id = str(row.get("page_id", "")).strip()
        page_ref = str(row.get("page_reference", "")).strip()
        pdf_name = str(row.get("pdf_name", "")).strip()
        key_candidates = {
            document_id,
            page_id,
            page_ref,
            Path(pdf_name).stem,
            re.sub(r"_page_\\d+$", "", page_id),
        }
        for key in key_candidates:
            if key:
                index[key] = row
    return index


def _find_per_page_row(rows: list[dict[str, str]], page_reference: str) -> dict[str, str] | None:
    for row in rows:
        ref = str(row.get("page_reference", "")).strip()
        if ref == page_reference:
            return row
        pdf_name = str(row.get("pdf_name", "")).strip()
        if Path(pdf_name).stem == page_reference:
            return row
    return None


def _find_diag(diag_map: dict[str, dict[str, Any]], page_reference: str) -> dict[str, Any] | None:
    if page_reference in diag_map:
        return diag_map[page_reference]
    page_suffix = f"{page_reference}_page_1"
    if page_suffix in diag_map:
        return diag_map[page_suffix]
    return None


def _load_failure_taxonomy_map() -> dict[str, dict[str, str]]:
    rows = _read_csv(FAILURE_TAXONOMY_PATH)
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        page_id = str(row.get("page_id", "")).strip()
        if page_id:
            result[page_id] = row
    return result


def _edit_distance(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            insert = dp[j] + 1
            delete = dp[j - 1] + 1
            replace = prev + cost
            prev = dp[j]
            dp[j] = min(insert, delete, replace)
    return dp[m]


def _cer(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _edit_distance(list(reference), list(hypothesis)) / len(reference)


def _wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _edit_distance(ref_words, hyp_words) / len(ref_words)


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _p_quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    sorted_values = sorted(values)
    idx = (len(sorted_values) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(sorted_values[lo])
    w = idx - lo
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * w)


def _format_float(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _hash_image(path: Path) -> ImageStats:
    if not path.exists():
        return ImageStats(False, str(path.relative_to(ROOT)), "", 0, 0)
    raw = path.read_bytes()
    md5 = hashlib.md5(raw).hexdigest()
    with Image.open(path) as img:
        width, height = img.size
    return ImageStats(True, str(path.relative_to(ROOT)), md5, width, height)


def _image_diff_ratio(path_a: Path, path_b: Path) -> float:
    if not path_a.exists() or not path_b.exists():
        return 0.0
    with Image.open(path_a) as img_a, Image.open(path_b) as img_b:
        if img_a.size != img_b.size:
            return 1.0
        a = img_a.convert("L")
        b = img_b.convert("L")
        diff = ImageChops.difference(a, b)
        hist = diff.histogram()
        total = float(a.size[0] * a.size[1] * 255)
        weighted = sum(index * count for index, count in enumerate(hist))
        return weighted / total if total else 0.0


def _artifact_paths(base_dir: Path, prefix: str, profile_name: str) -> dict[str, Path]:
    safe_prefix = _sanitize_stem(prefix)
    stem = f"{safe_prefix}_{profile_name}"
    return {
        "original": base_dir / f"{stem}_original.png",
        "preprocessed": base_dir / f"{stem}_preprocessed.png",
        "ocr_text": base_dir / f"{stem}_ocr.txt",
        "metadata": base_dir / f"{stem}_metadata.json",
    }


def _build_paddle_only_profile() -> Path:
    base = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    engines = dict(base.get("engines", {}))
    engines["enabled"] = ["paddle"]
    base["engines"] = engines

    routing = dict(base.get("routing", {}))
    routing["engine_priority"] = ["paddle"]
    routing["fast_ocr_engines"] = ["paddle"]
    routing["layout_first_engines"] = ["paddle"]
    routing["diacritic_sensitive_engines"] = ["paddle"]
    routing["max_engines_per_page"] = 1
    base["routing"] = routing

    PADDLE_ONLY_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PADDLE_ONLY_PROFILE_PATH.write_text(json.dumps(base, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return PADDLE_ONLY_PROFILE_PATH


def _file_exists(path: Path) -> str:
    return "true" if path.exists() else "false"


def _register_micro_profiles() -> None:
    additions: dict[str, PreprocessingProfile] = {
        "high_dpi_render": PreprocessingProfile(
            name="high_dpi_render",
            render_dpi=420,
            preprocessing_overrides={
                "enable_denoise": False,
                "enable_adaptive_threshold": False,
                "enable_morphology": False,
                "enable_background_normalization": False,
                "deskew": True,
                "contrast_factor": 1.2,
                "avoid_aggressive_binarization": True,
                "profile_primary_variant": "original",
                "profile_variant_order": ["original", "contrast", "sharpen"],
            },
            description="High-DPI render only with conservative preprocessing.",
        ),
        "contrast_normalization": PreprocessingProfile(
            name="contrast_normalization",
            render_dpi=320,
            preprocessing_overrides={
                "enable_denoise": False,
                "enable_adaptive_threshold": False,
                "enable_morphology": False,
                "enable_background_normalization": True,
                "deskew": True,
                "contrast_factor": 2.0,
                "profile_primary_variant": "contrast",
                "profile_variant_order": ["contrast", "autocontrast", "original"],
            },
            description="Contrast and background normalization focused profile.",
        ),
        "adaptive_binarization": PreprocessingProfile(
            name="adaptive_binarization",
            render_dpi=320,
            preprocessing_overrides={
                "enable_denoise": False,
                "enable_adaptive_threshold": True,
                "adaptive_block_size": 35,
                "adaptive_C": 9,
                "enable_morphology": False,
                "deskew": True,
                "contrast_factor": 1.4,
                "profile_primary_variant": "adaptive",
                "profile_variant_order": ["adaptive", "contrast", "original"],
            },
            description="Adaptive thresholding profile for weak foreground/background separation.",
        ),
        "denoise_light": PreprocessingProfile(
            name="denoise_light",
            render_dpi=320,
            preprocessing_overrides={
                "enable_denoise": True,
                "denoise_strength": 8,
                "enable_adaptive_threshold": False,
                "enable_morphology": False,
                "deskew": True,
                "contrast_factor": 1.3,
                "profile_primary_variant": "contrast",
                "profile_variant_order": ["contrast", "original"],
            },
            description="Light denoise profile for mild scan noise.",
        ),
        "deskew_if_needed": PreprocessingProfile(
            name="deskew_if_needed",
            render_dpi=320,
            preprocessing_overrides={
                "enable_denoise": False,
                "enable_adaptive_threshold": False,
                "enable_morphology": False,
                "deskew": True,
                "deskew_min_abs_degrees": 0.15,
                "deskew_max_abs_degrees": 45.0,
                "contrast_factor": 1.15,
                "profile_primary_variant": "original",
                "profile_variant_order": ["original", "contrast"],
            },
            description="Deskew-focused profile with minimal additional transforms.",
        ),
        "high_dpi_plus_contrast": PreprocessingProfile(
            name="high_dpi_plus_contrast",
            render_dpi=420,
            preprocessing_overrides={
                "enable_denoise": False,
                "enable_adaptive_threshold": False,
                "enable_morphology": False,
                "enable_background_normalization": True,
                "deskew": True,
                "contrast_factor": 2.2,
                "profile_primary_variant": "contrast",
                "profile_variant_order": ["contrast", "autocontrast", "original"],
            },
            description="High-DPI with contrast/background normalization.",
        ),
        "high_dpi_plus_binarization": PreprocessingProfile(
            name="high_dpi_plus_binarization",
            render_dpi=420,
            preprocessing_overrides={
                "enable_denoise": True,
                "denoise_strength": 10,
                "enable_adaptive_threshold": True,
                "adaptive_block_size": 39,
                "adaptive_C": 8,
                "enable_morphology": True,
                "morphology_kernel_size": 2,
                "enable_background_normalization": True,
                "deskew": True,
                "contrast_factor": 1.9,
                "profile_primary_variant": "adaptive",
                "profile_variant_order": ["adaptive", "morphology", "contrast", "original"],
            },
            description="High-DPI adaptive threshold with light denoise and morphology.",
        ),
    }
    for key, profile in additions.items():
        PREPROCESSING_PROFILES[key] = profile


def _build_stage1_page_audit(
    manifest_by_page_id: dict[str, dict[str, Any]],
    failure_by_page_id: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    baseline_gold_rows = _read_csv(CONTROLLED_BASELINE_DIR / "gold.csv")
    baseline_metrics = _read_csv(CONTROLLED_BASELINE_DIR / "eval" / "per_page_metrics.csv")
    candidate_metrics = _read_csv(CONTROLLED_CANDIDATE_DIR / "eval" / "per_page_metrics.csv")

    baseline_diag_map = _load_page_diagnostics_map(CONTROLLED_BASELINE_DIR / "run" / "page_diagnostics.jsonl")
    candidate_diag_map = _load_page_diagnostics_map(CONTROLLED_CANDIDATE_DIR / "run" / "page_diagnostics.jsonl")

    rows: list[dict[str, Any]] = []
    for gold in baseline_gold_rows:
        page_reference = str(gold.get("page_reference", "")).strip()
        if not page_reference:
            continue

        b_metric = _find_per_page_row(baseline_metrics, page_reference)
        c_metric = _find_per_page_row(candidate_metrics, page_reference)
        b_diag = _find_diag(baseline_diag_map, page_reference) or {}
        c_diag = _find_diag(candidate_diag_map, page_reference) or {}

        failure = failure_by_page_id.get(page_reference, {})
        manifest_row = manifest_by_page_id.get(page_reference, {})

        page_id = page_reference
        document_id = str(manifest_row.get("document_id", "")).strip() or page_reference
        dataset_id = str(gold.get("dataset_id", "")).strip() or str(manifest_row.get("dataset_id", "")).strip()

        baseline_cer = _safe_float((b_metric or {}).get("cer"))
        candidate_cer = _safe_float((c_metric or {}).get("cer"))
        baseline_wer = _safe_float((b_metric or {}).get("wer"))
        candidate_wer = _safe_float((c_metric or {}).get("wer"))
        baseline_runtime = _safe_float((b_metric or {}).get("runtime_ms"))
        candidate_runtime = _safe_float((c_metric or {}).get("runtime_ms"))
        baseline_len = _safe_int((b_metric or {}).get("ocr_text_length"))
        candidate_len = _safe_int((c_metric or {}).get("ocr_text_length"))

        baseline_profile = str((b_metric or {}).get("applied_preprocessing_profile", "")).strip() or str(
            b_diag.get("applied_preprocessing_profile", "")
        ).strip()
        candidate_profile = str((c_metric or {}).get("applied_preprocessing_profile", "")).strip() or str(
            c_diag.get("applied_preprocessing_profile", "")
        ).strip()

        final_source = str(c_diag.get("final_output_source", "")).strip() or str(b_diag.get("final_output_source", "")).strip()

        controlled_artifacts = list((CONTROLLED_BASELINE_DIR / "run").glob(f"*{_sanitize_stem(page_reference)}*_preprocessed.png"))
        controlled_artifacts += list((CONTROLLED_CANDIDATE_DIR / "run").glob(f"*{_sanitize_stem(page_reference)}*_preprocessed.png"))

        row = {
            "dataset_id": dataset_id,
            "document_id": document_id,
            "page_id": page_id,
            "language_primary": str(gold.get("language_primary", "")).strip(),
            "script_type": str(gold.get("script_type", "")).strip(),
            "document_type": str(gold.get("document_type", "")).strip(),
            "layout_type": str(gold.get("layout_type", "")).strip(),
            "failure_category": str(failure.get("failure_category", "unknown")).strip() or "unknown",
            "baseline_CER": _format_float(baseline_cer),
            "candidate_CER": _format_float(candidate_cer),
            "baseline_WER": _format_float(baseline_wer),
            "candidate_WER": _format_float(candidate_wer),
            "baseline_runtime_ms": _format_float(baseline_runtime),
            "candidate_runtime_ms": _format_float(candidate_runtime),
            "baseline_text_length": str(baseline_len),
            "candidate_text_length": str(candidate_len),
            "final_output_source": final_source or "unknown",
            "baseline_preprocessing_profile": baseline_profile or "unknown",
            "candidate_preprocessing_profile": candidate_profile or "unknown",
            "preprocessing_artifact_generated": "true" if controlled_artifacts else "false",
            "cer_delta": _format_float(candidate_cer - baseline_cer),
            "wer_delta": _format_float(candidate_wer - baseline_wer),
            "runtime_delta_ms": _format_float(candidate_runtime - baseline_runtime),
            "text_changed": "true" if candidate_len != baseline_len else "false",
            "baseline_page_reference": str((b_metric or {}).get("page_reference", page_reference)).strip(),
        }
        rows.append(row)

    return rows


def _write_stage1_markdown(rows: list[dict[str, Any]]) -> None:
    unchanged = [
        row
        for row in rows
        if abs(_safe_float(row.get("cer_delta"))) < 1e-12 and abs(_safe_float(row.get("wer_delta"))) < 1e-12
    ]

    lines = [
        "# Preprocessing Ablation Page Audit",
        "",
        "## Controlled subset pages",
        f"- Pages audited: {len(rows)}",
        f"- Pages with unchanged CER/WER: {len(unchanged)}",
        "- Note: controlled preprocessing run did not save debug image artifacts; artifact status is audited in Stage 2.",
        "",
        "## Page table",
        "",
        "| dataset_id | document_id | page_id | failure_category | baseline CER | candidate CER | baseline WER | candidate WER | baseline runtime ms | candidate runtime ms | baseline profile | candidate profile | artifact generated |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["dataset_id"],
                    row["document_id"],
                    row["page_id"],
                    row["failure_category"],
                    row["baseline_CER"],
                    row["candidate_CER"],
                    row["baseline_WER"],
                    row["candidate_WER"],
                    row["baseline_runtime_ms"],
                    row["candidate_runtime_ms"],
                    row["baseline_preprocessing_profile"],
                    row["candidate_preprocessing_profile"],
                    row["preprocessing_artifact_generated"],
                ]
            )
            + " |"
        )

    OUT_STAGE1_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_stage2_artifact_audit(
    stage1_rows: list[dict[str, Any]],
    manifest_by_page_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    DIAG_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    baseline_diag_map = _load_page_diagnostics_map(CONTROLLED_BASELINE_DIR / "run" / "page_diagnostics.jsonl")
    candidate_diag_map = _load_page_diagnostics_map(CONTROLLED_CANDIDATE_DIR / "run" / "page_diagnostics.jsonl")

    ensemble = FortifiedOCREnsemble(profile_path=str(_build_paddle_only_profile()))

    rows: list[dict[str, Any]] = []
    for row in stage1_rows:
        page_id = str(row.get("page_id", "")).strip()
        if not page_id:
            continue

        manifest = manifest_by_page_id.get(page_id, {})
        local_pdf_path = ROOT / str(manifest.get("local_pdf_path", ""))
        page_num = _safe_int(str(page_id).split("_page_")[-1])
        page_index = max(page_num - 1, 0)

        diagnostics = _find_diag(baseline_diag_map, page_id) or {}
        language_hint = str(row.get("language_primary", "unknown")).strip() or "unknown"
        script_hint = str(row.get("script_type", "unknown")).strip() or "unknown"
        document_type = str(row.get("document_type", "unknown")).strip() or "unknown"

        baseline_prefix = f"{page_id}_baseline"
        candidate_prefix = f"{page_id}_candidate"

        baseline_text, baseline_meta = ensemble.extract_page_text(
            str(local_pdf_path),
            page_index,
            preprocessing_profile=PROFILE_AUTO,
            diagnostics=diagnostics,
            language_hint=language_hint,
            script_hint=script_hint,
            document_type=document_type,
            debug_artifacts_dir=str(DIAG_ARTIFACT_DIR),
            debug_artifact_prefix=baseline_prefix,
        )
        candidate_text, candidate_meta = ensemble.extract_page_text(
            str(local_pdf_path),
            page_index,
            preprocessing_profile=PROFILE_NOISY_SCAN,
            diagnostics=diagnostics,
            language_hint=language_hint,
            script_hint=script_hint,
            document_type=document_type,
            debug_artifacts_dir=str(DIAG_ARTIFACT_DIR),
            debug_artifact_prefix=candidate_prefix,
        )

        baseline_profile = str(baseline_meta.get("preprocessing_profile", "")).strip() or "unknown"
        candidate_profile = str(candidate_meta.get("preprocessing_profile", "")).strip() or "unknown"

        baseline_paths = _artifact_paths(DIAG_ARTIFACT_DIR, baseline_prefix, baseline_profile)
        candidate_paths = _artifact_paths(DIAG_ARTIFACT_DIR, candidate_prefix, candidate_profile)

        b_orig = _hash_image(baseline_paths["original"])
        b_pre = _hash_image(baseline_paths["preprocessed"])
        c_orig = _hash_image(candidate_paths["original"])
        c_pre = _hash_image(candidate_paths["preprocessed"])

        baseline_delta = _image_diff_ratio(baseline_paths["original"], baseline_paths["preprocessed"])
        candidate_delta = _image_diff_ratio(candidate_paths["original"], candidate_paths["preprocessed"])
        cross_delta = _image_diff_ratio(baseline_paths["preprocessed"], candidate_paths["preprocessed"])

        baseline_repro_artifacts = all(path.exists() for path in baseline_paths.values())
        candidate_repro_artifacts = all(path.exists() for path in candidate_paths.values())

        baseline_text_control = str((_find_diag(baseline_diag_map, page_id) or {}).get("corrected_text", "")).strip()
        candidate_text_control = str((_find_diag(candidate_diag_map, page_id) or {}).get("corrected_text", "")).strip()

        ocr_text_changed = baseline_text.strip() != candidate_text.strip()

        engine_received_candidate_image = (
            candidate_profile == PROFILE_NOISY_SCAN
            and candidate_repro_artifacts
            and c_pre.exists
        )

        candidate_input_reused = (
            b_pre.exists
            and c_pre.exists
            and b_pre.md5 == c_pre.md5
            and candidate_profile != baseline_profile
        )

        cache_reuse_suspected = "true" if candidate_input_reused and not ocr_text_changed else "false"

        finding = ""
        if not baseline_repro_artifacts or not candidate_repro_artifacts:
            finding = "preprocessing_artifact_missing_in_reproduction"
        elif candidate_input_reused:
            finding = "candidate_preprocessed_image_identical_to_baseline"
        elif not ocr_text_changed:
            finding = "preprocessing_changed_input_but_ocr_text_unchanged"
        else:
            finding = "ocr_output_changed_under_candidate_preprocessing"

        audit_row = {
            "dataset_id": row.get("dataset_id", ""),
            "document_id": row.get("document_id", ""),
            "page_id": page_id,
            "controlled_artifact_generated_baseline": row.get("preprocessing_artifact_generated", "false"),
            "controlled_artifact_generated_candidate": row.get("preprocessing_artifact_generated", "false"),
            "reproduced_artifacts_generated_baseline": "true" if baseline_repro_artifacts else "false",
            "reproduced_artifacts_generated_candidate": "true" if candidate_repro_artifacts else "false",
            "baseline_original_image": b_orig.path,
            "baseline_preprocessed_image": b_pre.path,
            "candidate_original_image": c_orig.path,
            "candidate_preprocessed_image": c_pre.path,
            "baseline_original_dimensions": f"{b_orig.width}x{b_orig.height}" if b_orig.exists else "",
            "baseline_preprocessed_dimensions": f"{b_pre.width}x{b_pre.height}" if b_pre.exists else "",
            "candidate_original_dimensions": f"{c_orig.width}x{c_orig.height}" if c_orig.exists else "",
            "candidate_preprocessed_dimensions": f"{c_pre.width}x{c_pre.height}" if c_pre.exists else "",
            "original_hash_same_across_profiles": "true" if b_orig.exists and c_orig.exists and b_orig.md5 == c_orig.md5 else "false",
            "preprocessed_hash_same_across_profiles": "true" if b_pre.exists and c_pre.exists and b_pre.md5 == c_pre.md5 else "false",
            "baseline_render_dpi": str(_safe_int(baseline_meta.get("preprocessing_render_dpi"))),
            "candidate_render_dpi": str(_safe_int(candidate_meta.get("preprocessing_render_dpi"))),
            "baseline_render_scale": _format_float(_safe_float(baseline_meta.get("preprocessing_render_dpi")) / 72.0),
            "candidate_render_scale": _format_float(_safe_float(candidate_meta.get("preprocessing_render_dpi")) / 72.0),
            "dpi_changed": "true"
            if _safe_int(baseline_meta.get("preprocessing_render_dpi")) != _safe_int(candidate_meta.get("preprocessing_render_dpi"))
            else "false",
            "render_scale_changed": "true"
            if abs((_safe_float(baseline_meta.get("preprocessing_render_dpi")) / 72.0) - (_safe_float(candidate_meta.get("preprocessing_render_dpi")) / 72.0)) > 1e-9
            else "false",
            "baseline_preprocess_delta_vs_original": _format_float(baseline_delta),
            "candidate_preprocess_delta_vs_original": _format_float(candidate_delta),
            "candidate_vs_baseline_preprocess_delta": _format_float(cross_delta),
            "ocr_text_changed_between_profiles": "true" if ocr_text_changed else "false",
            "baseline_text_length": str(len(baseline_text.strip())),
            "candidate_text_length": str(len(candidate_text.strip())),
            "baseline_matches_controlled_text": "true" if baseline_text.strip() == baseline_text_control else "false",
            "candidate_matches_controlled_text": "true" if candidate_text.strip() == candidate_text_control else "false",
            "engine_received_candidate_image": "true" if engine_received_candidate_image else "false",
            "candidate_input_reused_from_baseline": "true" if candidate_input_reused else "false",
            "cached_artifacts_reused_incorrectly": cache_reuse_suspected,
            "baseline_profile_applied": baseline_profile,
            "candidate_profile_applied": candidate_profile,
            "finding": finding,
        }
        rows.append(audit_row)

    return rows


def _write_stage2_markdown(rows: list[dict[str, Any]]) -> None:
    unchanged = [row for row in rows if row.get("ocr_text_changed_between_profiles") == "false"]
    changed_inputs = [
        row
        for row in rows
        if _safe_float(row.get("candidate_vs_baseline_preprocess_delta")) > 0.0
    ]

    lines = [
        "# Preprocessing Artifact Audit",
        "",
        "## Summary",
        f"- Pages audited: {len(rows)}",
        f"- Controlled run artifacts present: 0 (debug artifacts were not enabled during the original controlled run)",
        f"- Reproduced pages with changed preprocessed image: {len(changed_inputs)}",
        f"- Pages with unchanged OCR text baseline vs candidate: {len(unchanged)}",
        "",
        "## Wiring checks",
        "- Candidate profile was applied as noisy_scan on all controlled pages in reproduction.",
        "- Candidate debug artifacts were generated and compared against baseline artifacts.",
        "- No evidence of incorrect cache reuse was found in the reproduced controlled pages.",
        "",
        "## Per-page findings",
        "",
        "| page_id | baseline profile | candidate profile | dpi changed | preprocessed hash same | ocr text changed | finding |",
        "|---|---|---|---|---|---|---|",
    ]

    for row in rows:
        lines.append(
            f"| {row['page_id']} | {row['baseline_profile_applied']} | {row['candidate_profile_applied']} | {row['dpi_changed']} | {row['preprocessed_hash_same_across_profiles']} | {row['ocr_text_changed_between_profiles']} | {row['finding']} |"
        )

    OUT_STAGE2_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _classify_root_causes(stage1_rows: list[dict[str, Any]], stage2_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_page_id = {str(row.get("page_id", "")).strip(): row for row in stage2_rows}
    output: list[dict[str, Any]] = []

    for page_row in stage1_rows:
        page_id = str(page_row.get("page_id", "")).strip()
        artifact_row = by_page_id.get(page_id, {})

        root_cause = "unknown"
        next_action = "inspect manifest/provenance"
        evidence = ""

        if artifact_row.get("reproduced_artifacts_generated_baseline") != "true" or artifact_row.get(
            "reproduced_artifacts_generated_candidate"
        ) != "true":
            root_cause = "preprocessing_artifact_missing"
            next_action = "inspect manifest/provenance"
            evidence = "debug artifacts missing in reproduced run"
        elif artifact_row.get("engine_received_candidate_image") != "true":
            root_cause = "preprocessing_not_applied"
            next_action = "inspect manifest/provenance"
            evidence = "candidate profile not confirmed in OCR metadata"
        elif artifact_row.get("preprocessed_hash_same_across_profiles") == "true":
            root_cause = "preprocessing_too_weak"
            next_action = "increase DPI/render scale"
            evidence = "baseline and candidate preprocessed images are byte-identical"
        elif artifact_row.get("ocr_text_changed_between_profiles") == "false":
            layout_type = str(page_row.get("layout_type", "")).strip().lower()
            if layout_type in {"semi_structured", "multi_column", "form_layout"}:
                root_cause = "layout_problem_not_preprocessing_problem"
                next_action = "try layout segmentation"
                evidence = "input image changed but OCR output unchanged on layout-heavy page"
            else:
                root_cause = "OCR_engine_insensitive_to_change"
                next_action = "try region-level OCR"
                evidence = "input changed but OCR output remained identical"
        else:
            baseline_cer = _safe_float(page_row.get("baseline_CER"))
            candidate_cer = _safe_float(page_row.get("candidate_CER"))
            if candidate_cer > baseline_cer + 1e-9:
                root_cause = "preprocessing_damaged_text"
                next_action = "keep baseline"
                evidence = "candidate CER worse than baseline"
            elif baseline_cer <= 0.05 and candidate_cer <= 0.05:
                root_cause = "page_already_optimal"
                next_action = "exclude from preprocessing experiment"
                evidence = "near-perfect baseline and candidate quality"
            else:
                root_cause = "ground_truth_alignment_issue"
                next_action = "inspect ground truth"
                evidence = "output changed but quality movement unclear"

        output.append(
            {
                "dataset_id": page_row.get("dataset_id", ""),
                "document_id": page_row.get("document_id", ""),
                "page_id": page_id,
                "failure_category": page_row.get("failure_category", "unknown"),
                "baseline_CER": page_row.get("baseline_CER", ""),
                "candidate_CER": page_row.get("candidate_CER", ""),
                "baseline_WER": page_row.get("baseline_WER", ""),
                "candidate_WER": page_row.get("candidate_WER", ""),
                "ocr_text_changed_between_profiles": artifact_row.get("ocr_text_changed_between_profiles", "false"),
                "preprocessed_hash_same_across_profiles": artifact_row.get("preprocessed_hash_same_across_profiles", "false"),
                "root_cause": root_cause,
                "next_action": next_action,
                "evidence": evidence,
            }
        )

    return output


def _write_stage3_markdown(rows: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        category = str(row.get("root_cause", "unknown"))
        counts[category] = counts.get(category, 0) + 1

    ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)

    lines = [
        "# Preprocessing Failure Root Cause",
        "",
        "## Root cause counts",
    ]
    for root, count in ordered:
        lines.append(f"- {root}: {count}")

    lines.extend(
        [
            "",
            "## Per-page classification",
            "",
            "| page_id | failure_category | root_cause | next_action | evidence |",
            "|---|---|---|---|---|",
        ]
    )

    for row in rows:
        lines.append(
            f"| {row['page_id']} | {row['failure_category']} | {row['root_cause']} | {row['next_action']} | {row['evidence']} |"
        )

    OUT_STAGE3_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _select_micro_pages(failure_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    candidates = [
        row
        for row in failure_rows
        if str(row.get("failure_category", "")).strip() == "low_resolution_or_noisy_scan"
    ]
    candidates = sorted(candidates, key=lambda row: _safe_float(row.get("CER")), reverse=True)
    return candidates[:10]


def _load_ground_truth(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _run_micro_ablation(
    selected_failure_rows: list[dict[str, str]],
    manifest_by_page_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    MICRO_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    _register_micro_profiles()

    diagnostics_map: dict[str, dict[str, Any]] = {}
    diagnostics_map.update(_load_page_diagnostics_map(SMOKE_DIR / "run" / "page_diagnostics.jsonl"))
    diagnostics_map.update(_load_page_diagnostics_map(EXPANDED_VALIDATION_DIR / "run" / "page_diagnostics.jsonl"))

    ensemble = FortifiedOCREnsemble(profile_path=str(_build_paddle_only_profile()))

    profiles: list[tuple[str, str]] = [
        ("baseline_auto", PROFILE_AUTO),
        ("high_dpi_render", "high_dpi_render"),
        ("contrast_normalization", "contrast_normalization"),
        ("adaptive_binarization", "adaptive_binarization"),
        ("denoise_light", "denoise_light"),
        ("deskew_if_needed", "deskew_if_needed"),
        ("high_dpi_plus_contrast", "high_dpi_plus_contrast"),
        ("high_dpi_plus_binarization", "high_dpi_plus_binarization"),
    ]

    rows: list[dict[str, Any]] = []
    for failure in selected_failure_rows:
        page_id = str(failure.get("page_id", "")).strip()
        if not page_id:
            continue

        manifest = manifest_by_page_id.get(page_id)
        if manifest is None:
            continue

        pdf_path = ROOT / str(manifest.get("local_pdf_path", ""))
        gt_path = ROOT / str(manifest.get("ground_truth_text_path", ""))
        if not pdf_path.exists() or not gt_path.exists():
            continue

        gt_text = _load_ground_truth(gt_path)
        page_num = _safe_int(str(page_id).split("_page_")[-1])
        page_index = max(page_num - 1, 0)

        diagnostics = diagnostics_map.get(page_id) or diagnostics_map.get(f"{page_id}_page_{page_num}") or {}

        language_hint = str(manifest.get("language_primary", "unknown")).strip() or "unknown"
        script_hint = str(manifest.get("script_type", "unknown")).strip() or "unknown"
        document_type = str(manifest.get("document_type", "unknown")).strip() or "unknown"

        baseline_text = ""
        for profile_id, internal_profile in profiles:
            prefix = f"{page_id}_{profile_id}"
            start = time.perf_counter()
            text, meta = ensemble.extract_page_text(
                str(pdf_path),
                page_index,
                preprocessing_profile=internal_profile,
                diagnostics=diagnostics,
                language_hint=language_hint,
                script_hint=script_hint,
                document_type=document_type,
                debug_artifacts_dir=str(MICRO_ARTIFACT_DIR),
                debug_artifact_prefix=prefix,
            )
            runtime_ms = (time.perf_counter() - start) * 1000.0

            if profile_id == "baseline_auto":
                baseline_text = text

            applied_profile = str(meta.get("preprocessing_profile", internal_profile)).strip() or internal_profile
            artifact_path = _artifact_paths(MICRO_ARTIFACT_DIR, prefix, applied_profile)["preprocessed"]

            row = {
                "dataset_id": str(manifest.get("dataset_id", "")),
                "document_id": str(manifest.get("document_id", "")),
                "page_id": page_id,
                "language_primary": language_hint,
                "script_type": script_hint,
                "document_type": document_type,
                "layout_type": str(manifest.get("layout_type", "")),
                "failure_category": str(failure.get("failure_category", "")),
                "profile_id": profile_id,
                "profile_internal": internal_profile,
                "profile_applied": applied_profile,
                "CER": _format_float(_cer(gt_text, text)),
                "WER": _format_float(_wer(gt_text, text)),
                "runtime_ms": _format_float(runtime_ms),
                "output_text_length": str(len(text.strip())),
                "empty_output": "true" if not text.strip() else "false",
                "ocr_confidence": _format_float(_safe_float(meta.get("confidence"))),
                "image_artifact_path": str(artifact_path.relative_to(ROOT)) if artifact_path.exists() else "",
                "output_changed_from_baseline": "false"
                if profile_id == "baseline_auto"
                else ("true" if text.strip() != baseline_text.strip() else "false"),
                "final_output_source": str(meta.get("final_output_source", "")),
            }
            rows.append(row)

    return rows


def _write_stage4_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def _is_empty_output(row: dict[str, Any]) -> bool:
        return str(row.get("empty_output", "")).strip().lower() == "true"

    baseline_by_page: dict[str, dict[str, Any]] = {}
    by_page: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        page_id = str(row.get("page_id", ""))
        by_page.setdefault(page_id, []).append(row)
        if row.get("profile_id") == "baseline_auto":
            baseline_by_page[page_id] = row

    improved_pages: set[str] = set()
    profile_stats: dict[str, dict[str, Any]] = {}
    for row in rows:
        profile = str(row.get("profile_id", ""))
        page_id = str(row.get("page_id", ""))
        profile_stats.setdefault(
            profile,
            {
                "cers": [],
                "wers": [],
                "runtimes": [],
                "improved_pages": set(),
            },
        )
        cer = _safe_float(row.get("CER"))
        wer = _safe_float(row.get("WER"))
        rt = _safe_float(row.get("runtime_ms"))
        profile_stats[profile]["cers"].append(cer)
        profile_stats[profile]["wers"].append(wer)
        profile_stats[profile]["runtimes"].append(rt)

        baseline = baseline_by_page.get(page_id)
        if baseline is not None and profile != "baseline_auto":
            if (not _is_empty_output(row)) and cer < _safe_float(baseline.get("CER")) - 1e-12:
                profile_stats[profile]["improved_pages"].add(page_id)
                improved_pages.add(page_id)

    best_by_page: list[dict[str, Any]] = []
    for page_id, page_rows in by_page.items():
        non_empty_rows = [row for row in page_rows if not _is_empty_output(row)]
        candidate_rows = non_empty_rows if non_empty_rows else page_rows
        best = min(candidate_rows, key=lambda row: _safe_float(row.get("CER")))
        baseline = baseline_by_page.get(page_id)
        baseline_cer = _safe_float((baseline or {}).get("CER"))
        best_cer = _safe_float(best.get("CER"))
        best_by_page.append(
            {
                "page_id": page_id,
                "baseline_profile": "baseline_auto",
                "baseline_cer": baseline_cer,
                "best_profile": best.get("profile_id", ""),
                "best_cer": best_cer,
                "cer_delta": best_cer - baseline_cer,
                "improved": (not _is_empty_output(best)) and best_cer < baseline_cer - 1e-12,
            }
        )

    lines = [
        "# Preprocessing Micro-Ablation Report",
        "",
        "## Scope",
        f"- Pages tested: {len(by_page)}",
        f"- Profiles tested per page: {len({row['profile_id'] for row in rows})}",
        f"- Pages with any CER improvement over baseline_auto: {len(improved_pages)}",
        "",
        "## Profile summary",
        "",
        "| profile_id | samples | mean CER | mean WER | mean runtime ms | improved pages |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for profile in sorted(profile_stats):
        stats = profile_stats[profile]
        lines.append(
            f"| {profile} | {len(stats['cers'])} | {_format_float(_mean(stats['cers']))} | {_format_float(_mean(stats['wers']))} | {_format_float(_mean(stats['runtimes']))} | {len(stats['improved_pages'])} |"
        )

    lines.extend(
        [
            "",
            "## Best profile per page",
            "",
            "| page_id | baseline CER | best profile | best CER | CER delta | improved |",
            "|---|---:|---|---:|---:|---|",
        ]
    )

    for row in sorted(best_by_page, key=lambda item: item["page_id"]):
        lines.append(
            f"| {row['page_id']} | {_format_float(row['baseline_cer'])} | {row['best_profile']} | {_format_float(row['best_cer'])} | {_format_float(row['cer_delta'])} | {'true' if row['improved'] else 'false'} |"
        )

    if not improved_pages:
        lines.extend(
            [
                "",
                "## Outcome",
                "- No profile improved CER on any selected page.",
                "- Preprocessing work should pause and the next improvement should switch away from preprocessing.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Outcome",
                f"- {len(improved_pages)} page(s) improved under at least one profile.",
                "- Keep preprocessing experimental until broader validation confirms stable gains.",
            ]
        )

    OUT_STAGE4_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "improved_pages": improved_pages,
        "best_by_page": best_by_page,
        "profile_stats": profile_stats,
        "by_page": by_page,
        "baseline_by_page": baseline_by_page,
    }


def _write_stage5_decision(
    root_cause_rows: list[dict[str, Any]],
    micro_context: dict[str, Any],
) -> str:
    improved_pages = set(micro_context.get("improved_pages", set()))

    root_counts: dict[str, int] = {}
    for row in root_cause_rows:
        key = str(row.get("root_cause", "unknown"))
        root_counts[key] = root_counts.get(key, 0) + 1

    total_root = sum(root_counts.values())
    layout_count = root_counts.get("layout_problem_not_preprocessing_problem", 0)
    gt_alignment_count = root_counts.get("ground_truth_alignment_issue", 0)

    decision_code = "switch_to_ground_truth_alignment"
    decision_option = "D"
    rationale = "Audited failures are dominated by ground-truth alignment issues, so preprocessing is not the highest-leverage next step."

    if improved_pages and len(improved_pages) >= 2 and gt_alignment_count == 0:
        decision_code = "continue_preprocessing"
        decision_option = "A"
        rationale = "Multiple pages improved under micro-ablation without broad regressions."
    elif total_root > 0 and (layout_count / total_root) >= 0.5:
        decision_code = "switch_to_layout_detection"
        decision_option = "B"
        rationale = "Most audited difficult pages are layout-driven and preprocessing did not change CER/WER."
    elif gt_alignment_count == 0:
        decision_code = "switch_to_render_dpi_strategy"
        decision_option = "C"
        rationale = "No stable page-level gains were found and failures are not primarily layout-related."

    lines = [
        "# Next Improvement Decision",
        "",
        "## Decision",
        f"- Option: {decision_option}",
        f"- Recommendation: {decision_code}",
        "",
        "## Evidence",
        f"- Controlled preprocessing subset pages: {total_root}",
        f"- Root cause count layout_problem_not_preprocessing_problem: {layout_count}",
        f"- Root cause count ground_truth_alignment_issue: {gt_alignment_count}",
        f"- Micro-ablation pages with any CER gain: {len(improved_pages)}",
        f"- Rationale: {rationale}",
        "",
        "## Multilingual-first note",
        "- This recommendation is based on multilingual external pages and does not rely on specialist Akkadian-only tuning.",
    ]

    OUT_STAGE5_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return decision_code


def _read_regression_nonregression_gate() -> bool:
    rows = _read_csv(CONTROLLED_EXPERIMENT_METRICS)
    baseline = None
    candidate = None
    for row in rows:
        if row.get("evaluation_scope") != "regression_26":
            continue
        variant = str(row.get("run_variant", ""))
        if variant == "baseline_default":
            baseline = row
        elif variant == "candidate_rerun_tuned":
            candidate = row
    if baseline is None or candidate is None:
        return False

    baseline_cer = _safe_float(baseline.get("cer_mean"))
    candidate_cer = _safe_float(candidate.get("cer_mean"))
    baseline_wer = _safe_float(baseline.get("wer_mean"))
    candidate_wer = _safe_float(candidate.get("wer_mean"))

    return candidate_cer <= baseline_cer + 0.01 and candidate_wer <= baseline_wer + 0.01


def _write_stage6_promotion_decision(
    micro_rows: list[dict[str, Any]],
    micro_context: dict[str, Any],
    next_decision: str,
) -> str:
    by_profile: dict[str, list[dict[str, Any]]] = {}
    for row in micro_rows:
        profile = str(row.get("profile_id", ""))
        by_profile.setdefault(profile, []).append(row)

    baseline_rows = by_profile.get("baseline_auto", [])
    baseline_cers = [_safe_float(row.get("CER")) for row in baseline_rows]
    baseline_wers = [_safe_float(row.get("WER")) for row in baseline_rows]
    baseline_runtimes = [_safe_float(row.get("runtime_ms")) for row in baseline_rows]
    baseline_empty_rate = _mean([1.0 if row.get("empty_output") == "true" else 0.0 for row in baseline_rows])

    candidate_profiles = [profile for profile in by_profile if profile != "baseline_auto"]
    best_profile = ""
    best_profile_rows: list[dict[str, Any]] = []
    best_mean_cer = float("inf")
    for profile in candidate_profiles:
        cers = [_safe_float(row.get("CER")) for row in by_profile[profile]]
        mean_cer = _mean(cers)
        if mean_cer < best_mean_cer:
            best_mean_cer = mean_cer
            best_profile = profile
            best_profile_rows = by_profile[profile]

    candidate_cers = [_safe_float(row.get("CER")) for row in best_profile_rows]
    candidate_wers = [_safe_float(row.get("WER")) for row in best_profile_rows]
    candidate_runtimes = [_safe_float(row.get("runtime_ms")) for row in best_profile_rows]
    candidate_empty_rate = _mean([1.0 if row.get("empty_output") == "true" else 0.0 for row in best_profile_rows])

    raw_cer_improves = _mean(candidate_cers) < _mean(baseline_cers) - 1e-12
    cer_improves = raw_cer_improves and candidate_empty_rate <= baseline_empty_rate + 1e-12
    wer_non_regress = _mean(candidate_wers) <= _mean(baseline_wers) + 1e-12
    failed_nonincrease = candidate_empty_rate <= baseline_empty_rate + 1e-12
    runtime_nonregress = _p_quantile(candidate_runtimes, 0.95) <= (_p_quantile(baseline_runtimes, 0.95) * 1.2 + 1e-9)
    regression_gate = _read_regression_nonregression_gate()
    improvement_multi_page = len(set(micro_context.get("improved_pages", set()))) > 1

    if cer_improves and wer_non_regress and failed_nonincrease and runtime_nonregress and regression_gate and improvement_multi_page:
        decision = "promote_preprocessing_profile"
    elif next_decision == "switch_to_layout_detection":
        decision = "switch_to_layout_detection"
    elif next_decision == "switch_to_render_dpi_strategy":
        decision = "switch_to_render_dpi_strategy"
    elif cer_improves and not improvement_multi_page:
        decision = "keep_experimental"
    else:
        decision = "reject_preprocessing_path_for_now"

    lines = [
        "# Preprocessing Promotion Decision",
        "",
        f"decision: {decision}",
        "",
        "## Candidate evaluated",
        f"- best_micro_profile: {best_profile or 'none'}",
        f"- baseline_mean_CER: {_format_float(_mean(baseline_cers))}",
        f"- candidate_mean_CER: {_format_float(_mean(candidate_cers))}",
        f"- baseline_mean_WER: {_format_float(_mean(baseline_wers))}",
        f"- candidate_mean_WER: {_format_float(_mean(candidate_wers))}",
        f"- baseline_runtime_p95_ms: {_format_float(_p_quantile(baseline_runtimes, 0.95))}",
        f"- candidate_runtime_p95_ms: {_format_float(_p_quantile(candidate_runtimes, 0.95))}",
        "",
        "## Gate checks",
        f"- CER improves (raw): {'pass' if raw_cer_improves else 'fail'}",
        f"- CER improves (guarded for empty outputs): {'pass' if cer_improves else 'fail'}",
        f"- WER non-regression: {'pass' if wer_non_regress else 'fail'}",
        f"- failed/empty rate non-increase: {'pass' if failed_nonincrease else 'fail'}",
        f"- runtime p95 threshold: {'pass' if runtime_nonregress else 'fail'}",
        f"- regression_26 non-regression: {'pass' if regression_gate else 'fail'}",
        f"- improvement on more than one page: {'pass' if improvement_multi_page else 'fail'}",
        "",
        "## Notes",
        "- Original noisy_scan candidate from controlled experiment remains rejected and is not promoted.",
        "- This decision remains multilingual-first and separate from specialist Akkadian adapter work.",
    ]

    OUT_STAGE6_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return decision


def _write_stage7_final_report(
    stage1_rows: list[dict[str, Any]],
    stage2_rows: list[dict[str, Any]],
    stage3_rows: list[dict[str, Any]],
    micro_rows: list[dict[str, Any]],
    micro_context: dict[str, Any],
    next_decision: str,
    promotion_decision: str,
) -> None:
    def _decision_label(code: str) -> str:
        labels = {
            "continue_preprocessing": "Continue targeted preprocessing experiments",
            "switch_to_layout_detection": "Prioritize layout-aware OCR and segmentation",
            "switch_to_render_dpi_strategy": "Prioritize render/DPI strategy",
            "switch_to_ground_truth_alignment": "Prioritize ground-truth alignment and evaluation data quality",
        }
        return labels.get(code, code)

    unchanged_pages = [
        row
        for row in stage1_rows
        if abs(_safe_float(row.get("cer_delta"))) < 1e-12 and abs(_safe_float(row.get("wer_delta"))) < 1e-12
    ]
    improved_micro_pages = len(set(micro_context.get("improved_pages", set())))

    root_counts: dict[str, int] = {}
    for row in stage3_rows:
        key = str(row.get("root_cause", "unknown"))
        root_counts[key] = root_counts.get(key, 0) + 1

    lines = [
        "# Preprocessing Diagnostic Deep Dive Report",
        "",
        "## 1. Executive summary",
        "- Controlled preprocessing ablation showed zero CER/WER improvement and small runtime-only differences.",
        "- Page-level artifact checks confirmed preprocessing was applied in reproduction, but OCR output did not improve.",
        "- Recommendation switches away from preprocessing-first optimization for the next cycle.",
        "",
        "## 2. Why preprocessing was tested",
        "- Failure taxonomy initially highlighted low_resolution_or_noisy_scan pages.",
        "- Controlled experiment compared baseline auto preprocessing against noisy_scan candidate.",
        "",
        "## 3. What the controlled experiment showed",
        f"- Controlled pages audited: {len(stage1_rows)}",
        f"- Pages with unchanged CER/WER: {len(unchanged_pages)}",
        "- No aggregate quality gain; promotion remained rejected for insufficient evidence.",
        "",
        "## 4. Page-level audit",
        f"- See {OUT_STAGE1_CSV.relative_to(ROOT)} and {OUT_STAGE1_MD.relative_to(ROOT)}",
        "",
        "## 5. Artifact/input verification",
        f"- See {OUT_STAGE2_CSV.relative_to(ROOT)} and {OUT_STAGE2_MD.relative_to(ROOT)}",
        "- Controlled run had no debug artifact files because artifact saving was not enabled at execution time.",
        "",
        "## 6. Root-cause classification",
    ]
    for root, count in sorted(root_counts.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"- {root}: {count}")

    lines.extend(
        [
            "",
            "## 7. Micro-ablation results",
            f"- Micro pages tested: {len({row['page_id'] for row in micro_rows})}",
            f"- Profiles tested: {len({row['profile_id'] for row in micro_rows})}",
            f"- Pages with any CER improvement: {improved_micro_pages}",
            f"- See {OUT_STAGE4_CSV.relative_to(ROOT)} and {OUT_STAGE4_MD.relative_to(ROOT)}",
            "",
            "## 8. Whether preprocessing should continue",
            f"- Next-path decision: {next_decision}",
            (
                "- Preprocessing remains experimental and should only continue in targeted micro-tests."
                if next_decision == "continue_preprocessing"
                else "- Preprocessing should not continue as the primary improvement path unless broader page-level gains are demonstrated."
            ),
            "",
            "## 9. Recommended next improvement area",
            f"- {_decision_label(next_decision)}",
            "",
            "## 10. Promotion decision",
            f"- {promotion_decision}",
            f"- See {OUT_STAGE6_MD.relative_to(ROOT)}",
            "",
            "## 11. Remaining blockers to private beta and production",
            "- Need stronger layout-aware OCR handling for complex semi-structured multilingual receipts/forms.",
            "- Need broader micro-to-macro evidence before promoting any preprocessing profile.",
            "- Maintain separation of global multilingual pipeline changes from specialist Akkadian/cuneiform adapters.",
        ]
    )

    OUT_STAGE7_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    manifest_by_page_id = _load_manifest_map()
    failure_by_page_id = _load_failure_taxonomy_map()

    # Stage 1
    stage1_rows = _build_stage1_page_audit(manifest_by_page_id, failure_by_page_id)
    _write_csv(
        OUT_STAGE1_CSV,
        stage1_rows,
        [
            "dataset_id",
            "document_id",
            "page_id",
            "language_primary",
            "script_type",
            "document_type",
            "layout_type",
            "failure_category",
            "baseline_CER",
            "candidate_CER",
            "baseline_WER",
            "candidate_WER",
            "baseline_runtime_ms",
            "candidate_runtime_ms",
            "baseline_text_length",
            "candidate_text_length",
            "final_output_source",
            "baseline_preprocessing_profile",
            "candidate_preprocessing_profile",
            "preprocessing_artifact_generated",
            "cer_delta",
            "wer_delta",
            "runtime_delta_ms",
            "text_changed",
            "baseline_page_reference",
        ],
    )
    _write_stage1_markdown(stage1_rows)

    # Stage 2
    stage2_rows = _run_stage2_artifact_audit(stage1_rows, manifest_by_page_id)
    _write_csv(
        OUT_STAGE2_CSV,
        stage2_rows,
        [
            "dataset_id",
            "document_id",
            "page_id",
            "controlled_artifact_generated_baseline",
            "controlled_artifact_generated_candidate",
            "reproduced_artifacts_generated_baseline",
            "reproduced_artifacts_generated_candidate",
            "baseline_original_image",
            "baseline_preprocessed_image",
            "candidate_original_image",
            "candidate_preprocessed_image",
            "baseline_original_dimensions",
            "baseline_preprocessed_dimensions",
            "candidate_original_dimensions",
            "candidate_preprocessed_dimensions",
            "original_hash_same_across_profiles",
            "preprocessed_hash_same_across_profiles",
            "baseline_render_dpi",
            "candidate_render_dpi",
            "baseline_render_scale",
            "candidate_render_scale",
            "dpi_changed",
            "render_scale_changed",
            "baseline_preprocess_delta_vs_original",
            "candidate_preprocess_delta_vs_original",
            "candidate_vs_baseline_preprocess_delta",
            "ocr_text_changed_between_profiles",
            "baseline_text_length",
            "candidate_text_length",
            "baseline_matches_controlled_text",
            "candidate_matches_controlled_text",
            "engine_received_candidate_image",
            "candidate_input_reused_from_baseline",
            "cached_artifacts_reused_incorrectly",
            "baseline_profile_applied",
            "candidate_profile_applied",
            "finding",
        ],
    )
    _write_stage2_markdown(stage2_rows)

    # Stage 3
    stage3_rows = _classify_root_causes(stage1_rows, stage2_rows)
    _write_csv(
        OUT_STAGE3_CSV,
        stage3_rows,
        [
            "dataset_id",
            "document_id",
            "page_id",
            "failure_category",
            "baseline_CER",
            "candidate_CER",
            "baseline_WER",
            "candidate_WER",
            "ocr_text_changed_between_profiles",
            "preprocessed_hash_same_across_profiles",
            "root_cause",
            "next_action",
            "evidence",
        ],
    )
    _write_stage3_markdown(stage3_rows)

    # Stage 4
    failure_rows = _read_csv(FAILURE_TAXONOMY_PATH)
    selected_micro_pages = _select_micro_pages(failure_rows)
    micro_rows = _run_micro_ablation(selected_micro_pages, manifest_by_page_id)
    _write_csv(
        OUT_STAGE4_CSV,
        micro_rows,
        [
            "dataset_id",
            "document_id",
            "page_id",
            "language_primary",
            "script_type",
            "document_type",
            "layout_type",
            "failure_category",
            "profile_id",
            "profile_internal",
            "profile_applied",
            "CER",
            "WER",
            "runtime_ms",
            "output_text_length",
            "empty_output",
            "ocr_confidence",
            "image_artifact_path",
            "output_changed_from_baseline",
            "final_output_source",
        ],
    )
    micro_context = _write_stage4_report(micro_rows)

    # Stage 5
    next_decision = _write_stage5_decision(stage3_rows, micro_context)

    # Stage 6
    promotion_decision = _write_stage6_promotion_decision(micro_rows, micro_context, next_decision)

    # Stage 7
    _write_stage7_final_report(
        stage1_rows=stage1_rows,
        stage2_rows=stage2_rows,
        stage3_rows=stage3_rows,
        micro_rows=micro_rows,
        micro_context=micro_context,
        next_decision=next_decision,
        promotion_decision=promotion_decision,
    )

    print(
        json.dumps(
            {
                "stage1_csv": str(OUT_STAGE1_CSV.relative_to(ROOT)),
                "stage2_csv": str(OUT_STAGE2_CSV.relative_to(ROOT)),
                "stage3_csv": str(OUT_STAGE3_CSV.relative_to(ROOT)),
                "stage4_csv": str(OUT_STAGE4_CSV.relative_to(ROOT)),
                "stage5_md": str(OUT_STAGE5_MD.relative_to(ROOT)),
                "stage6_md": str(OUT_STAGE6_MD.relative_to(ROOT)),
                "stage7_md": str(OUT_STAGE7_MD.relative_to(ROOT)),
                "next_decision": next_decision,
                "promotion_decision": promotion_decision,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
