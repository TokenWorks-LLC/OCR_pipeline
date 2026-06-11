#!/usr/bin/env python3
"""
Page-level text extraction with Akkadian detection.

Extracts page-level text from PDFs (with optional OCR fallback), applies
preprocessing and multi-engine consensus for difficult pages when ensemble OCR
is enabled, detects Akkadian content per page, and outputs CSV rows with
structured status metadata.

Usage:
    python tools/run_page_text.py \\
        --manifest data\\gold\\manifest.txt \\
        --output-root reports\\page_text_20251009 \\
        --prefer-text-layer \\
        --ocr-fallback ensemble \
        --status-bar \\
        --progress-csv reports\\page_text_20251009\\progress.csv
        
    OR:
    
    python tools/run_page_text.py \\
        --inputs "G:\\Shared drives\\Secondary Sources" \\
        --output-root reports\\page_text_20251009 \\
        --prefer-text-layer
"""
import argparse
import csv
import json
import logging
import os
import re
import sys
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from production.page_diagnostics import PageDiagnosticsAnalyzer
from production.preprocessing_profiles import (
    PROFILE_AUTO,
    PROFILE_UNKNOWN_SAFE_DEFAULT,
    available_preprocessing_profiles,
    resolve_preprocessing_profile,
)
from production.ocr_strategy import OCRStrategySelector
from production.postprocessing import PostprocessingPipeline
from production.quality_scoring import OCRQualityScorer, QualityClassThresholds
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production.ensemble_ocr import FortifiedOCREnsemble

# Core dependencies
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


ENGINE_STATUS_AVAILABLE = "available"
ENGINE_STATUS_UNAVAILABLE_DEPENDENCY = "unavailable_dependency_error"
ENGINE_STATUS_AVAILABLE_UNHEALTHY = "available_but_unhealthy"
ENGINE_STATUS_DISABLED_BY_CONFIG = "disabled_by_config"
ENGINE_STATUS_TIMED_OUT = "timed_out"
ENGINE_STATUS_FAILED_ON_PAGE = "failed_on_page"

PAGE_STATUS_SUCCESS = "success"
PAGE_STATUS_PARTIAL_SUCCESS = "partial_success"
PAGE_STATUS_FAILED = "failed"
PAGE_STATUS_SKIPPED = "skipped"
PAGE_STATUS_TIMED_OUT = "timed_out"


def _run_with_timeout(func, timeout_s: float | None):
    """Best-effort timeout wrapper using daemon threads."""
    if not timeout_s or timeout_s <= 0:
        return func(), False

    result: Dict[str, Any] = {}
    error: Dict[str, BaseException] = {}
    done = threading.Event()

    def _target() -> None:
        try:
            result["value"] = func()
        except BaseException as exc:  # pragma: no cover - passthrough path
            error["value"] = exc
        finally:
            done.set()

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    if not done.wait(float(timeout_s)):
        return None, True

    if "value" in error:
        raise error["value"]
    return result.get("value"), False


def _json_string(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _parse_domain_paths(values: List[str] | None) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for raw in values or []:
        text = str(raw or "").strip()
        if not text:
            continue

        if "=" in text:
            domain, path = text.split("=", 1)
            key = str(domain or "general").strip().lower() or "general"
            value = str(path or "").strip()
        else:
            key = "general"
            value = text

        if not value:
            continue
        parsed[key] = value
    return parsed


def _load_quality_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    file_path = Path(path)
    if not file_path.exists():
        return {}
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


DETECTION_OUTPUT_FIELDS = [
    "detected_orientation_angle",
    "detected_orientation_class",
    "detected_rotation_base_angle",
    "detected_skew_angle",
    "detected_layout_type",
    "detected_column_count",
    "detected_has_columns",
]

LAYOUT_ANALYSIS_FIELDS = [
    "detected_page_layout_mode",
    "detected_region_count",
    "detected_has_footnotes",
    "detected_has_table_interruptions",
    "reading_order_confidence",
    "reading_order_source",
    "ordering_source",
    "region_ocr_used",
    "region_ocr_attempted",
]

ROUTING_FIELDS = [
    "selected_strategy",
    "strategy_mode",
    "strategy_use_text_layer",
    "strategy_use_full_page_ocr",
    "strategy_use_region_ocr",
    "strategy_primary_engine",
    "strategy_fallback_engines",
    "strategy_ensemble_needed",
    "strategy_max_engines_per_page",
    "engines_attempted",
    "engines_skipped",
    "engine_skip_reasons",
    "fallback_path",
    "final_output_source",
    "runtime_per_engine_ms",
    "confidence_per_engine",
]

ENSEMBLE_ANALYSIS_FIELDS = [
    "consensus_used",
    "consensus_winner_engine",
    "char_disagreement_rate",
    "token_disagreement_rate",
    "line_disagreement_rate",
    "engine_agreement_score",
    "consensus_entropy",
    "ensemble_uncertain",
    "human_review_recommended",
    "low_quality_ensemble",
]

POSTPROCESSING_FIELDS = [
    "raw_text",
    "cleaned_text",
    "corrected_text",
    "adapter_used",
    "corrections_applied",
    "correction_diff",
    "correction_confidence",
    "lexicon_coverage",
    "unknown_token_rate",
    "protected_character_changes",
    "needs_human_review",
    "postprocess_quality_score",
    "postprocess_quality_metrics",
    "model_correction_reason",
]

QUALITY_FIELDS = [
    "page_quality_score",
    "document_quality_score",
    "quality_class",
    "quality_reasons",
    "failed_gate",
    "gate_reason",
]

PAGE_DIAGNOSTIC_FIELDS = [
    "page_number",
    "input_file",
    "input_pdf_path",
    "document_id",
    "page_id",
    "engine_used",
    "runtime_ms",
    "output_text_length",
    "pass_number",
    "first_pass_status",
    "second_pass_status",
    "final_status",
    "fallback_reason",
    "fallback_engine",
    "first_pass_runtime_ms",
    "second_pass_runtime_ms",
    "total_page_runtime_ms",
    "width",
    "height",
    "dpi",
    "render_scale",
    "is_born_digital",
    "text_layer_char_count",
    "text_layer_word_count",
    "text_density",
    "foreground_ratio",
    "estimated_skew_degrees",
    "blur_score",
    "contrast_score",
    "noise_score",
    "connected_component_count",
    "estimated_column_count",
    "has_large_images",
    "has_tables_estimate",
    "layout_complexity_score",
    "recommended_preprocessing_profile",
    "applied_preprocessing_profile",
    "recommended_ocr_strategy",
    "language_hint",
    "page_image_rendered",
    "render_failure_reason",
    "page_has_usable_text_layer",
]

TEXT_LAYER_DIAGNOSTIC_FIELDS = [
    "text_layer_usable",
    "text_layer_accepted",
    "text_layer_rejected_reason",
    "text_layer_quality_score",
    "text_layer_suspicious_patterns",
    "text_layer_suspicious_reasons",
    "text_layer_acceptance",
    "is_mostly_blank",
]

OUTPUT_METADATA_FIELDS = (
    DETECTION_OUTPUT_FIELDS
    + LAYOUT_ANALYSIS_FIELDS
    + ROUTING_FIELDS
    + ENSEMBLE_ANALYSIS_FIELDS
    + POSTPROCESSING_FIELDS
    + QUALITY_FIELDS
    + PAGE_DIAGNOSTIC_FIELDS
    + TEXT_LAYER_DIAGNOSTIC_FIELDS
)


def _format_optional_number(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if number.is_integer():
        return str(int(number))
    return f"{number:.4f}".rstrip("0").rstrip(".")


def _format_optional_int(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return ""


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _format_optional_bool(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value in (None, ""):
        return ""

    text = str(value).strip().lower()
    if not text:
        return ""
    if text in {"1", "true", "yes", "y", "on", "t"}:
        return "true"
    if text in {"0", "false", "no", "n", "off", "f"}:
        return "false"
    return ""


def _format_optional_text(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, list):
        return "|".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def _format_optional_json(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, (dict, list)):
        return _json_string(value)
    return _format_optional_text(value)


def _normalize_output_metadata(meta: Dict[str, Any] | None) -> Dict[str, str]:
    meta = meta or {}
    normalized: Dict[str, str] = {
        "detected_orientation_angle": _format_optional_number(meta.get("rotation_angle", "")),
        "detected_orientation_class": str(meta.get("orientation_class", "") or "").strip(),
        "detected_rotation_base_angle": _format_optional_number(meta.get("rotation_base_angle", "")),
        "detected_skew_angle": _format_optional_number(meta.get("deskew_angle", "")),
        "detected_layout_type": str(meta.get("detected_layout_type", "") or "").strip(),
        "detected_column_count": _format_optional_int(meta.get("detected_column_count", "")),
        "detected_has_columns": _format_optional_bool(meta.get("detected_has_columns", "")),
        "detected_page_layout_mode": str(meta.get("detected_layout_type", "") or "").strip(),
        "detected_region_count": _format_optional_int(meta.get("detected_region_count", "")),
        "detected_has_footnotes": _format_optional_bool(meta.get("detected_has_footnotes", "")),
        "detected_has_table_interruptions": _format_optional_bool(meta.get("detected_has_table_interruptions", "")),
        "reading_order_confidence": _format_optional_number(meta.get("reading_order_confidence", "")),
        "reading_order_source": _format_optional_text(meta.get("reading_order_source", "")),
        "ordering_source": _format_optional_text(meta.get("ordering_source", "")),
        "region_ocr_used": _format_optional_bool(meta.get("region_ocr_used", "")),
        "region_ocr_attempted": _format_optional_bool(meta.get("region_ocr_attempted", "")),
        "selected_strategy": _format_optional_text(meta.get("selected_strategy", "")),
        "strategy_mode": _format_optional_text(meta.get("strategy_mode", "")),
        "strategy_use_text_layer": _format_optional_bool(meta.get("strategy_use_text_layer", "")),
        "strategy_use_full_page_ocr": _format_optional_bool(meta.get("strategy_use_full_page_ocr", "")),
        "strategy_use_region_ocr": _format_optional_bool(meta.get("strategy_use_region_ocr", "")),
        "strategy_primary_engine": _format_optional_text(meta.get("strategy_primary_engine", "")),
        "strategy_fallback_engines": _format_optional_text(meta.get("strategy_fallback_engines", "")),
        "strategy_ensemble_needed": _format_optional_bool(meta.get("strategy_ensemble_needed", "")),
        "strategy_max_engines_per_page": _format_optional_int(meta.get("strategy_max_engines_per_page", "")),
        "engines_attempted": _format_optional_text(meta.get("engines_attempted", "")),
        "engines_skipped": _format_optional_json(meta.get("engines_skipped", "")),
        "engine_skip_reasons": _format_optional_text(meta.get("engine_skip_reasons", "")),
        "fallback_path": _format_optional_text(meta.get("fallback_path", "")),
        "final_output_source": _format_optional_text(meta.get("final_output_source", "")),
        "runtime_per_engine_ms": _format_optional_json(meta.get("runtime_per_engine_ms", "")),
        "confidence_per_engine": _format_optional_json(meta.get("confidence_per_engine", "")),
        "consensus_used": _format_optional_bool(meta.get("consensus_used", "")),
        "consensus_winner_engine": _format_optional_text(meta.get("consensus_winner_engine", "")),
        "char_disagreement_rate": _format_optional_number(meta.get("char_disagreement_rate", "")),
        "token_disagreement_rate": _format_optional_number(meta.get("token_disagreement_rate", "")),
        "line_disagreement_rate": _format_optional_number(meta.get("line_disagreement_rate", "")),
        "engine_agreement_score": _format_optional_number(meta.get("engine_agreement_score", "")),
        "consensus_entropy": _format_optional_number(meta.get("consensus_entropy", "")),
        "ensemble_uncertain": _format_optional_bool(meta.get("ensemble_uncertain", "")),
        "human_review_recommended": _format_optional_bool(meta.get("human_review_recommended", "")),
        "low_quality_ensemble": _format_optional_bool(meta.get("low_quality_ensemble", "")),
        "raw_text": _format_optional_text(meta.get("raw_text", "")),
        "cleaned_text": _format_optional_text(meta.get("cleaned_text", "")),
        "corrected_text": _format_optional_text(meta.get("corrected_text", "")),
        "adapter_used": _format_optional_text(meta.get("adapter_used", "")),
        "corrections_applied": _format_optional_json(meta.get("corrections_applied", "")),
        "correction_diff": _format_optional_text(meta.get("correction_diff", "")),
        "correction_confidence": _format_optional_number(meta.get("correction_confidence", "")),
        "lexicon_coverage": _format_optional_number(meta.get("lexicon_coverage", "")),
        "unknown_token_rate": _format_optional_number(meta.get("unknown_token_rate", "")),
        "protected_character_changes": _format_optional_int(meta.get("protected_character_changes", "")),
        "needs_human_review": _format_optional_bool(meta.get("needs_human_review", "")),
        "postprocess_quality_score": _format_optional_number(meta.get("postprocess_quality_score", "")),
        "postprocess_quality_metrics": _format_optional_json(meta.get("postprocess_quality_metrics", "")),
        "model_correction_reason": _format_optional_text(meta.get("model_correction_reason", "")),
        "page_quality_score": _format_optional_number(meta.get("page_quality_score", "")),
        "document_quality_score": _format_optional_number(meta.get("document_quality_score", "")),
        "quality_class": _format_optional_text(meta.get("quality_class", "")),
        "quality_reasons": _format_optional_text(meta.get("quality_reasons", "")),
        "failed_gate": _format_optional_text(meta.get("failed_gate", "")),
        "gate_reason": _format_optional_text(meta.get("gate_reason", "")),
    }

    if not normalized["detected_has_columns"] and normalized["detected_column_count"]:
        normalized["detected_has_columns"] = "true" if int(normalized["detected_column_count"]) > 1 else "false"

    if not normalized["detected_column_count"]:
        normalized["detected_column_count"] = _format_optional_int(meta.get("estimated_column_count", ""))
        if normalized["detected_column_count"] and not normalized["detected_has_columns"]:
            normalized["detected_has_columns"] = "true" if int(normalized["detected_column_count"]) > 1 else "false"

    if not normalized["detected_layout_type"]:
        if normalized["detected_has_columns"] == "true":
            normalized["detected_layout_type"] = "multi_column"
        elif normalized["detected_has_columns"] == "false":
            normalized["detected_layout_type"] = "single_column"

    if not normalized["detected_page_layout_mode"]:
        normalized["detected_page_layout_mode"] = normalized.get("detected_layout_type", "")

    for field in PAGE_DIAGNOSTIC_FIELDS:
        if field in {
            "width",
            "height",
            "render_scale",
            "text_density",
            "foreground_ratio",
            "estimated_skew_degrees",
            "blur_score",
            "contrast_score",
            "noise_score",
            "layout_complexity_score",
            "text_layer_quality_score",
            "runtime_ms",
            "first_pass_runtime_ms",
            "second_pass_runtime_ms",
            "total_page_runtime_ms",
        }:
            normalized[field] = _format_optional_number(meta.get(field, ""))
        elif field in {
            "page_number",
            "dpi",
            "text_layer_char_count",
            "text_layer_word_count",
            "connected_component_count",
            "estimated_column_count",
            "output_text_length",
            "pass_number",
        }:
            normalized[field] = _format_optional_int(meta.get(field, ""))
        elif field in {"is_born_digital", "has_large_images", "has_tables_estimate", "page_image_rendered", "page_has_usable_text_layer"}:
            normalized[field] = _format_optional_bool(meta.get(field, ""))
        else:
            normalized[field] = _format_optional_text(meta.get(field, ""))

    for field in TEXT_LAYER_DIAGNOSTIC_FIELDS:
        if field in {"text_layer_usable", "text_layer_accepted", "is_mostly_blank"}:
            normalized[field] = _format_optional_bool(meta.get(field, ""))
        elif field in {"text_layer_quality_score"}:
            normalized[field] = _format_optional_number(meta.get(field, ""))
        else:
            normalized[field] = _format_optional_text(meta.get(field, ""))

    if not normalized.get("language_hint"):
        normalized["language_hint"] = "unknown"

    return normalized


def _merge_non_empty_metadata(base: Dict[str, str], updates: Dict[str, str]) -> Dict[str, str]:
    merged = dict(base)
    for key, value in updates.items():
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        merged[key] = text
    return merged


def _slugify_identifier(value: str, default: str = "unknown") -> str:
    text = str(value or "").strip().lower()
    if not text:
        return default
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or default


def _apply_row_integrity_metadata(
    *,
    output_meta: Dict[str, str],
    pdf_path: str,
    pdf_name: str,
    page: int,
    extraction_method: str,
    status: str,
    failure_reason: str,
    text: str,
    runtime_ms: int,
) -> None:
    output_meta["page_number"] = output_meta.get("page_number") or str(page)
    output_meta["input_file"] = output_meta.get("input_file") or pdf_name

    if pdf_path:
        output_meta["input_pdf_path"] = output_meta.get("input_pdf_path") or str(Path(pdf_path).resolve())

    document_id = output_meta.get("document_id") or _slugify_identifier(Path(pdf_name).stem)
    output_meta["document_id"] = document_id
    output_meta["page_id"] = output_meta.get("page_id") or f"{document_id}_page_{page}"

    output_meta["engine_used"] = extraction_method or output_meta.get("engine_used", "")
    output_meta["runtime_ms"] = _format_optional_int(runtime_ms)
    output_meta["output_text_length"] = _format_optional_int(len(text or ""))

    pass_number = output_meta.get("pass_number") or "1"
    if pass_number not in {"1", "2"}:
        pass_number = "1"
    output_meta["pass_number"] = pass_number

    if pass_number == "1":
        output_meta["first_pass_status"] = status
        output_meta["first_pass_runtime_ms"] = _format_optional_int(runtime_ms)
    else:
        output_meta["second_pass_status"] = status
        output_meta["second_pass_runtime_ms"] = _format_optional_int(runtime_ms)

    first_runtime = _safe_int(output_meta.get("first_pass_runtime_ms", ""))
    second_runtime = _safe_int(output_meta.get("second_pass_runtime_ms", ""))
    if first_runtime is not None or second_runtime is not None:
        output_meta["total_page_runtime_ms"] = _format_optional_int((first_runtime or 0) + (second_runtime or 0))
    else:
        output_meta["total_page_runtime_ms"] = _format_optional_int(runtime_ms)

    output_meta["final_status"] = status
    output_meta["fallback_reason"] = output_meta.get("fallback_reason") or failure_reason
    output_meta["fallback_engine"] = output_meta.get("fallback_engine") or extraction_method


class AkkadianDetector:
    """Page-level Akkadian detection with any-line aggregation."""
    
    def __init__(self, profile_path: Optional[str] = None):
        """Initialize detector with profile configuration."""
        self.config = self._load_profile(profile_path)
        self.akkadian_lm = None
        
        # Try to load char LM if available
        try:
            from python_char_lm import PythonCharLM
            lm_path = os.environ.get('AKKADIAN_LM_PATH', 'models/akkadian_char_lm.json')
            if os.path.exists(lm_path):
                self.akkadian_lm = PythonCharLM()
                self.akkadian_lm.load(lm_path)
                logger.info(f"Loaded Akkadian char LM from {lm_path}")
        except (ImportError, Exception) as e:
            logger.debug(f"Akkadian char LM not available: {e}")
    
    def _load_profile(self, profile_path: Optional[str]) -> Dict:
        """Load detection profile from JSON."""
        default_config = {
            "threshold": 0.25,
            "require_diacritic_or_marker": True,
            "min_diacritics_per_line": 1,
            "min_syllabic_tokens": 3,
            "min_syllabic_ratio": 0.25,
            "aggregation_mode": "any-line",
            "aggregation_qual_lines_min": 3,
            "aggregation_qual_ratio_min": 0.25,
            "markers_strict": True,
            "ppl_boosts": {"lt20": 0.3, "lt40": 0.1},
            "negative_lexicon": [
                "der", "die", "das", "und", "den", "des", "dem", "im", "vom", "zum", "zur",
                "für", "mit", "nach", "bei", "über", "auf", "aus", "nicht", "auch", "nur", "sich",
                "ve", "ile", "için", "bu", "bir", "veya", "de", "da", "olarak", "gibi", "ki", "mi",
                "the", "and", "of", "to", "in", "a", "is", "was", "are", "were", "been",
                "being", "have", "has", "had", "do", "does", "did", "will", "would", "should",
                "could", "may", "might", "must", "can"
            ],
            "neg_penalty_cap": 0.15
        }
        
        if profile_path and os.path.exists(profile_path):
            try:
                with open(profile_path, 'r', encoding='utf-8') as f:
                    profile = json.load(f)
                    if 'akkadian_detection' in profile:
                        default_config.update(profile['akkadian_detection'])
                        logger.info(f"Loaded profile from {profile_path}")
            except Exception as e:
                logger.warning(f"Failed to load profile {profile_path}: {e}, using defaults")
        
        return default_config
    
    def detect_line(self, line: str) -> Tuple[bool, float]:
        """Detect if a single line contains Akkadian transliteration."""
        if not line or len(line.strip()) < 3:
            return False, 0.0
        
        cfg = self.config
        
        # Akkadian markers (determinatives, logograms)
        STRICT_MARKERS = {
            "DUMU", "LUGAL", "KÙ.BABBAR", "KUBABBAR", "KU.BABBAR", "URU", 
            "É", "É.GAL", "EGAL", "KUR", "LU₂", "LÚ", "MUNUS", "MÍ",
            "GIŠ", "ᵈ", "ᵐ", "ᶠ"
        }
        
        # Diacritic chars
        DIACRITIC_CHARS = "āēīūŠšṢṣṬṭḪḫáéíóúàèìù"
        
        # Count diacritics
        num_diacritics = sum(1 for char in line if char in DIACRITIC_CHARS)
        has_diacritic = num_diacritics > 0
        
        # Check for markers
        line_upper = line.upper()
        has_marker = any(m in line_upper for m in STRICT_MARKERS)
        
        # Check for syllabic pattern (hyphenated transliteration)
        syllabic_pattern = re.compile(
            r'\b[a-zšṣṭḫāēīū]{1,4}(?:[-—][a-zšṣṭḫāēīū]{1,4}){2,}\b', 
            re.IGNORECASE
        )
        syllabic_matches = syllabic_pattern.findall(line)
        
        # Count tokens
        all_tokens = re.findall(
            r'\b[a-zšṣṭḫāēīū]+(?:[-—][a-zšṣṭḫāēīū]+)*\b', 
            line, 
            re.IGNORECASE
        )
        total_tokens = len(all_tokens)
        syllabic_token_count = len(syllabic_matches)
        syllabic_ratio = syllabic_token_count / total_tokens if total_tokens > 0 else 0.0
        
        # Apply density gates
        min_syllabic_tokens = cfg.get("min_syllabic_tokens", 3)
        min_syllabic_ratio = cfg.get("min_syllabic_ratio", 0.25)
        
        has_syllabic = (
            syllabic_token_count >= min_syllabic_tokens and 
            syllabic_ratio >= min_syllabic_ratio
        )
        
        # Require diacritics or markers when syllabic pattern present
        if cfg.get("require_diacritic_or_marker", True) and has_syllabic:
            min_diac = cfg.get("min_diacritics_per_line", 1)
            has_syllabic = (num_diacritics >= min_diac) or has_marker
        
        # Negative lexicon penalty
        neg_lexicon = set(cfg.get("negative_lexicon", []))
        tokens_lower = [t.lower() for t in all_tokens]
        neg_count = sum(1 for t in tokens_lower if t in neg_lexicon)
        neg_penalty_cap = cfg.get("neg_penalty_cap", 0.15)
        neg_penalty = min(neg_penalty_cap, 0.03 * neg_count)
        
        # Calculate score
        score = 0.0
        if has_syllabic:
            score += 0.45
        if has_diacritic:
            score += 0.20
        if has_marker:
            score += 0.15
        
        # Char LM boost
        if self.akkadian_lm:
            try:
                ppl = self.akkadian_lm.perplexity(line)
                ppl_boosts = cfg.get("ppl_boosts", {})
                if ppl < 20:
                    score += ppl_boosts.get("lt20", 0.3)
                elif ppl < 40:
                    score += ppl_boosts.get("lt40", 0.1)
            except Exception:
                pass
        
        score -= neg_penalty
        score = max(0.0, min(1.0, score))
        
        threshold = cfg.get("threshold", 0.25)
        is_akkadian = score >= threshold
        
        return is_akkadian, score
    
    def detect_page(self, text: str) -> Tuple[bool, Dict]:
        """
        Detect Akkadian at page level using any-line aggregation.
        
        Returns:
            Tuple of (has_akkadian, metadata_dict)
        """
        if not text or len(text.strip()) < 10:
            return False, {"qualified_lines": 0, "total_lines": 0, "ratio": 0.0}
        
        lines = text.split('\n')
        qualified_lines = 0
        line_scores = []
        
        for line in lines:
            line = line.strip()
            if len(line) < 3:
                continue
            
            is_akk, score = self.detect_line(line)
            line_scores.append(score)
            if is_akk:
                qualified_lines += 1
        
        total_lines = len([l for l in lines if len(l.strip()) >= 3])
        qual_ratio = qualified_lines / total_lines if total_lines > 0 else 0.0
        
        # Any-line aggregation
        min_qual_lines = self.config.get("aggregation_qual_lines_min", 3)
        min_qual_ratio = self.config.get("aggregation_qual_ratio_min", 0.25)
        
        has_akkadian = (qualified_lines >= min_qual_lines) or (qual_ratio >= min_qual_ratio)
        
        metadata = {
            "qualified_lines": qualified_lines,
            "total_lines": total_lines,
            "ratio": qual_ratio,
            "max_score": max(line_scores) if line_scores else 0.0
        }
        
        return has_akkadian, metadata


class NoopDetector:
    """Language-agnostic detector adapter for multilingual-first runtime."""

    def detect_page(self, text: str) -> Tuple[bool, Dict]:
        return False, {
            "qualified_lines": 0,
            "total_lines": 0,
            "ratio": 0.0,
            "mode": "disabled",
        }


class PDFTextExtractor:
    """Extract text from PDFs with optional OCR fallback."""
    
    def __init__(
        self,
        prefer_text_layer: bool = True,
        ocr_fallback: Optional[str] = None,
        force_ocr: bool = False,
        profile_path: Optional[str] = None,
        per_engine_timeout_s: Optional[float] = None,
        language_hint: str = "unknown",
        preprocessing_profile: str = PROFILE_AUTO,
        debug_preprocessing_dir: Optional[Path] = None,
    ):
        """
        Initialize PDF text extractor.
        
        Args:
            prefer_text_layer: Try text layer extraction first
            ocr_fallback: OCR engine to use for fallback ('paddle' or None)
        """
        self.prefer_text_layer = prefer_text_layer
        self.ocr_fallback = ocr_fallback
        self.force_ocr = force_ocr
        self.ocr_engine = None
        self.ensemble = None
        self.profile_path = profile_path
        self.per_engine_timeout_s = per_engine_timeout_s
        self.language_hint = (language_hint or "unknown").strip() or "unknown"
        self.preprocessing_profile = (preprocessing_profile or PROFILE_AUTO).strip().lower() or PROFILE_AUTO
        if self.preprocessing_profile != PROFILE_AUTO and self.preprocessing_profile not in available_preprocessing_profiles():
            self.preprocessing_profile = PROFILE_UNKNOWN_SAFE_DEFAULT
        self.debug_preprocessing_dir = debug_preprocessing_dir
        self._paddle_v3 = False
        self.engine_readiness: Dict[str, Dict[str, str]] = {}
        self.diagnostics = PageDiagnosticsAnalyzer.from_profile(profile_path, default_dpi=300)
        self.routing_config = self._load_routing_config(profile_path)
        self.strategy_selector = OCRStrategySelector(self.routing_config)
        
        if ocr_fallback == 'paddle':
            self._init_paddle_ocr()
        elif ocr_fallback == 'ensemble':
            self._init_ensemble()

    def get_engine_readiness(self) -> Dict[str, Dict[str, str]]:
        if self.ocr_fallback == 'ensemble' and self.ensemble is not None:
            if hasattr(self.ensemble, "get_engine_readiness"):
                self.engine_readiness = self.ensemble.get_engine_readiness()
            elif not self.engine_readiness:
                self.engine_readiness = {
                    "ensemble": {
                        "status": ENGINE_STATUS_AVAILABLE,
                        "reason": "stub_without_readiness_api",
                    }
                }
        return {name: dict(info) for name, info in self.engine_readiness.items()}

    def has_usable_ocr_engine(self) -> bool:
        if not self.ocr_fallback:
            return True

        readiness = self.get_engine_readiness()
        if self.ocr_fallback == 'ensemble':
            if self.ensemble is None:
                return False
            if hasattr(self.ensemble, "has_usable_backend"):
                return bool(self.ensemble.has_usable_backend())
            return True

        if self.ocr_fallback == 'paddle':
            status = readiness.get('paddle', {}).get('status')
            return status in {ENGINE_STATUS_AVAILABLE, ENGINE_STATUS_AVAILABLE_UNHEALTHY}

        return False

    def strict_readiness_ok(self) -> bool:
        if not self.ocr_fallback:
            return True

        readiness = self.get_engine_readiness()
        if self.ocr_fallback == 'ensemble':
            if self.ensemble is None:
                return False
            if hasattr(self.ensemble, "strict_readiness_ok"):
                return bool(self.ensemble.strict_readiness_ok())
            return True
        if self.ocr_fallback == 'paddle':
            return readiness.get('paddle', {}).get('status') == ENGINE_STATUS_AVAILABLE
        return False
    
    def _init_paddle_ocr(self):
        """Initialize PaddleOCR engine (supports both v2.x and v3.x)."""
        try:
            from paddleocr import PaddleOCR
            try:
                import paddleocr
                self._paddle_v3 = int(paddleocr.__version__.split(".")[0]) >= 3
            except Exception:
                self._paddle_v3 = False
            self.ocr_engine = PaddleOCR(lang='en')
            self.engine_readiness['paddle'] = {"status": ENGINE_STATUS_AVAILABLE, "reason": ""}
            logger.info("Initialized PaddleOCR v%s for fallback", "3.x" if self._paddle_v3 else "2.x")
        except Exception as e:
            logger.warning(f"Failed to initialize PaddleOCR: {e}")
            self.ocr_engine = None
            self.engine_readiness['paddle'] = {
                "status": ENGINE_STATUS_UNAVAILABLE_DEPENDENCY,
                "reason": str(e),
            }

    def _init_ensemble(self):
        """Initialize the fortified OCR ensemble lazily."""
        try:
            self.ensemble = FortifiedOCREnsemble(
                self.profile_path,
                per_engine_timeout_s=self.per_engine_timeout_s,
            )
            self.engine_readiness = self.ensemble.get_engine_readiness()
            logger.info("Initialized fortified OCR ensemble fallback")
        except Exception as e:
            logger.warning(f"Failed to initialize OCR ensemble: {e}")
            self.ensemble = None
            self.engine_readiness = {
                "ensemble": {
                    "status": ENGINE_STATUS_UNAVAILABLE_DEPENDENCY,
                    "reason": str(e),
                }
            }

    @staticmethod
    def _load_routing_config(profile_path: Optional[str]) -> Dict[str, Any]:
        if not profile_path:
            return {}
        profile = Path(profile_path)
        if not profile.exists():
            return {}
        try:
            with profile.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            routing = payload.get("routing", {}) if isinstance(payload, dict) else {}
            return routing if isinstance(routing, dict) else {}
        except Exception:
            return {}
    
    def extract_page_text(self, pdf_path: str, page_num: int) -> Tuple[str, bool, Dict]:
        """
        Extract text from a PDF page.
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-based)
        
        Returns:
            Tuple of (text, used_text_layer, metadata)
        """
        metadata = {
            "method": "unknown",
            "char_count": 0,
            "engine_page_statuses": {},
            "failure_reason": "",
            "rotation_angle": "",
            "rotation_base_angle": "",
            "deskew_angle": "",
            "orientation_class": "",
            "detected_layout_type": "",
            "detected_column_count": "",
            "detected_has_columns": "",
            "language_hint": self.language_hint,
            "text_layer_acceptance": "not_attempted",
            "text_layer_accepted": "false",
            "text_layer_rejected_reason": "",
            "text_layer_quality_score": "",
            "text_layer_suspicious_patterns": "",
            "page_image_rendered": "true",
            "render_failure_reason": "",
            "page_has_usable_text_layer": "false",
        }
        metadata.update({field: "" for field in OUTPUT_METADATA_FIELDS if field not in metadata})
        metadata["language_hint"] = self.language_hint

        text_layer_candidate = ""
        try:
            diagnostics, text_layer_candidate = self.diagnostics.inspect_page(
                pdf_path,
                page_num,
                language_hint=self.language_hint,
            )
            metadata.update(diagnostics.to_pipeline_metadata())
        except Exception as diag_exc:
            logger.debug("Diagnostics failed for %s page %s: %s", pdf_path, page_num + 1, diag_exc)
            metadata["text_layer_suspicious_reasons"] = "diagnostics_failed"
            metadata["text_layer_suspicious_patterns"] = "diagnostics_failed"
            metadata["text_layer_rejected_reason"] = "diagnostics_failed"
            metadata["text_layer_accepted"] = "false"
            metadata["page_image_rendered"] = "false"
            metadata["render_failure_reason"] = f"diagnostics_failed:{type(diag_exc).__name__}"

        selected_profile = resolve_preprocessing_profile(
            diagnostics=metadata,
            language_hint=self.language_hint,
            requested_profile=self.preprocessing_profile,
        )

        readiness_snapshot = self.get_engine_readiness()
        enabled_engines = list(readiness_snapshot.keys())
        previous_performance: Dict[str, Dict[str, Any]] = {}
        if self.ensemble is not None:
            try:
                previous_performance = self.ensemble.get_engine_performance_summary()
            except Exception:
                previous_performance = {}

        strategy = self.strategy_selector.select(
            diagnostics=metadata,
            engine_readiness=readiness_snapshot,
            enabled_engines=enabled_engines,
            language_hint=self.language_hint,
            script_hint=str(metadata.get("script_hint", "unknown") or "unknown"),
            document_type=str(metadata.get("document_type", "unknown") or "unknown"),
            requested_profile=selected_profile,
            default_profile=PROFILE_UNKNOWN_SAFE_DEFAULT,
            previous_engine_performance=previous_performance,
            timeout_config={"per_engine_timeout_s": self.per_engine_timeout_s or 0.0},
            quality_thresholds=self.routing_config.get("quality_thresholds", {}),
            force_ocr=self.force_ocr,
            prefer_text_layer=self.prefer_text_layer,
        )
        metadata.update(strategy.to_metadata())
        selected_profile = str(strategy.preprocessing_profile or selected_profile)
        metadata["applied_preprocessing_profile"] = selected_profile

        if self.force_ocr and self.ocr_fallback:
            metadata["text_layer_acceptance"] = "forced_ocr"
            text, success, ocr_meta = self._extract_via_ocr(
                pdf_path,
                page_num,
                preprocessing_profile=selected_profile,
                diagnostics_meta=metadata,
            )
            if success:
                metadata["method"] = f"ocr_{self.ocr_fallback}"
                metadata["char_count"] = len(text)
                metadata.update(ocr_meta)
                return text, False, metadata
            metadata.update(ocr_meta)
        
        # Try text layer first, or as fallback when forced OCR fails.
        should_try_text_layer = self.prefer_text_layer and (
            (not self.force_ocr) or bool(metadata.get("failure_reason"))
        )
        if should_try_text_layer:
            text = text_layer_candidate
            success = bool(text.strip())
            if not success:
                text, success = self._extract_text_layer(pdf_path, page_num)

            has_usable_signal = metadata.get("text_layer_usable", "") not in ("", None)
            text_layer_usable = str(metadata.get("text_layer_usable", "")).strip().lower() == "true"
            if not has_usable_signal:
                text_layer_usable = success and len(text.strip()) >= 16
                metadata["text_layer_usable"] = "true" if text_layer_usable else "false"

            if not metadata.get("text_layer_suspicious_patterns") and metadata.get("text_layer_suspicious_reasons"):
                metadata["text_layer_suspicious_patterns"] = metadata.get("text_layer_suspicious_reasons", "")

            force_ocr_failed = bool(self.force_ocr and metadata.get("failure_reason"))

            if success and text_layer_usable and (strategy.use_text_layer or force_ocr_failed):
                metadata["method"] = "text_layer"
                metadata["char_count"] = len(text)
                metadata["text_layer_acceptance"] = "accepted_after_ocr_failure" if force_ocr_failed else "accepted"
                metadata["text_layer_accepted"] = "true"
                metadata["page_has_usable_text_layer"] = "true"
                metadata["text_layer_rejected_reason"] = ""
                metadata["final_output_source"] = "text_layer"
                return text, True, metadata
            if success and not text_layer_usable:
                metadata["text_layer_acceptance"] = "rejected_quality"
                metadata["text_layer_accepted"] = "false"
                if not metadata.get("text_layer_rejected_reason"):
                    metadata["text_layer_rejected_reason"] = "quality_rejected"
            elif success and not strategy.use_text_layer:
                metadata["text_layer_acceptance"] = "rejected_strategy"
                metadata["text_layer_accepted"] = "false"
                if not metadata.get("text_layer_rejected_reason"):
                    metadata["text_layer_rejected_reason"] = "strategy_rejected"
            elif not success and metadata.get("text_layer_acceptance") == "not_attempted":
                metadata["text_layer_acceptance"] = "unavailable"
                metadata["text_layer_accepted"] = "false"
                if not metadata.get("text_layer_rejected_reason"):
                    metadata["text_layer_rejected_reason"] = "text_layer_unavailable"
        
        # Fallback to OCR if enabled
        if self.ocr_fallback:
            text, success, ocr_meta = self._extract_via_ocr(
                pdf_path,
                page_num,
                preprocessing_profile=selected_profile,
                diagnostics_meta=metadata,
            )
            if success:
                metadata["method"] = f"ocr_{self.ocr_fallback}"
                metadata["char_count"] = len(text)
                metadata.update(ocr_meta)
                return text, False, metadata
            metadata.update(ocr_meta)
        
        # Return empty if all methods failed
        metadata["method"] = "failed"
        if not metadata.get("failure_reason"):
            if str(metadata.get("text_layer_acceptance", "")).startswith("rejected") and not self.ocr_fallback:
                metadata["failure_reason"] = "text_layer_rejected"
            else:
                metadata["failure_reason"] = "all_extraction_methods_failed"
        return "", False, metadata
    
    def _extract_text_layer(self, pdf_path: str, page_num: int) -> Tuple[str, bool]:
        """Extract text from PDF text layer using PyMuPDF."""
        if not PYMUPDF_AVAILABLE:
            return "", False
        
        try:
            doc = fitz.open(pdf_path)
            if page_num >= len(doc):
                return "", False
            
            page = doc[page_num]
            text = page.get_text()
            doc.close()
            
            # Normalize whitespace
            text = self._normalize_whitespace(text)
            
            return text, True
        except Exception as e:
            logger.debug(f"Text layer extraction failed for {pdf_path} page {page_num}: {e}")
            return "", False
    
    def _extract_via_ocr(
        self,
        pdf_path: str,
        page_num: int,
        preprocessing_profile: str = PROFILE_AUTO,
        diagnostics_meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, bool, Dict[str, Any]]:
        """Extract text via OCR fallback or ensemble."""
        if self.ocr_fallback == 'ensemble' and self.ensemble is not None:
            debug_prefix = f"{Path(pdf_path).stem}_p{page_num + 1:04d}"
            timeout_config = {"per_engine_timeout_s": self.per_engine_timeout_s or 0.0}
            quality_thresholds = self.routing_config.get("quality_thresholds", {})
            try:
                text, meta = self.ensemble.extract_page_text(
                    pdf_path,
                    page_num,
                    preprocessing_profile=preprocessing_profile,
                    diagnostics=diagnostics_meta or {},
                    language_hint=self.language_hint,
                    script_hint=str((diagnostics_meta or {}).get("script_hint", "unknown") or "unknown"),
                    document_type=str((diagnostics_meta or {}).get("document_type", "unknown") or "unknown"),
                    timeout_config=timeout_config,
                    quality_thresholds=quality_thresholds,
                    debug_artifacts_dir=str(self.debug_preprocessing_dir) if self.debug_preprocessing_dir else None,
                    debug_artifact_prefix=debug_prefix,
                )
            except TypeError:
                # Backward compatibility for stubs/mocks using legacy signature.
                text, meta = self.ensemble.extract_page_text(pdf_path, page_num)
            if text:
                logger.debug("Ensemble OCR used engines: %s", meta.get("engines_used", []))
                return text, True, {
                    "engine_page_statuses": meta.get("engine_page_statuses", {}),
                    "engines_used": meta.get("engines_used", []),
                    "errors": meta.get("errors", {}),
                    "confidence": meta.get("confidence", ""),
                    "rotation_angle": meta.get("rotation_angle", ""),
                    "rotation_base_angle": meta.get("rotation_base_angle", ""),
                    "deskew_angle": meta.get("deskew_angle", ""),
                    "orientation_class": meta.get("orientation_class", ""),
                    "detected_layout_type": meta.get("detected_layout_type", ""),
                    "detected_column_count": meta.get("detected_column_count", ""),
                    "detected_has_columns": meta.get("detected_has_columns", ""),
                    "detected_region_count": meta.get("detected_region_count", ""),
                    "detected_has_footnotes": meta.get("detected_has_footnotes", ""),
                    "detected_has_table_interruptions": meta.get("detected_has_table_interruptions", ""),
                    "reading_order_confidence": meta.get("reading_order_confidence", ""),
                    "reading_order_source": meta.get("reading_order_source", ""),
                    "ordering_source": meta.get("ordering_source", ""),
                    "region_ocr_used": meta.get("region_ocr_used", ""),
                    "region_ocr_attempted": meta.get("region_ocr_attempted", ""),
                    "selected_strategy": meta.get("selected_strategy", ""),
                    "strategy_mode": meta.get("strategy_mode", ""),
                    "strategy_use_text_layer": meta.get("strategy_use_text_layer", ""),
                    "strategy_use_full_page_ocr": meta.get("strategy_use_full_page_ocr", ""),
                    "strategy_use_region_ocr": meta.get("strategy_use_region_ocr", ""),
                    "strategy_primary_engine": meta.get("strategy_primary_engine", ""),
                    "strategy_fallback_engines": meta.get("strategy_fallback_engines", []),
                    "strategy_ensemble_needed": meta.get("strategy_ensemble_needed", ""),
                    "strategy_max_engines_per_page": meta.get("strategy_max_engines_per_page", ""),
                    "engines_attempted": meta.get("engines_attempted", []),
                    "engines_skipped": meta.get("engines_skipped", {}),
                    "engine_skip_reasons": meta.get("engine_skip_reasons", []),
                    "fallback_path": meta.get("fallback_path", []),
                    "final_output_source": meta.get("final_output_source", ""),
                    "runtime_per_engine_ms": meta.get("runtime_per_engine_ms", {}),
                    "confidence_per_engine": meta.get("confidence_per_engine", {}),
                    "structured_layout": meta.get("structured_layout", {}),
                    "plain_text_reconstruction": meta.get("plain_text_reconstruction", text),
                    "applied_preprocessing_profile": meta.get("preprocessing_profile", preprocessing_profile),
                }
            return "", False, {
                "engine_page_statuses": meta.get("engine_page_statuses", {}),
                "engines_used": meta.get("engines_used", []),
                "errors": meta.get("errors", {}),
                "failure_reason": meta.get("failure_reason", "ocr_ensemble_empty"),
                "confidence": meta.get("confidence", ""),
                "rotation_angle": meta.get("rotation_angle", ""),
                "rotation_base_angle": meta.get("rotation_base_angle", ""),
                "deskew_angle": meta.get("deskew_angle", ""),
                "orientation_class": meta.get("orientation_class", ""),
                "detected_layout_type": meta.get("detected_layout_type", ""),
                "detected_column_count": meta.get("detected_column_count", ""),
                "detected_has_columns": meta.get("detected_has_columns", ""),
                "detected_region_count": meta.get("detected_region_count", ""),
                "detected_has_footnotes": meta.get("detected_has_footnotes", ""),
                "detected_has_table_interruptions": meta.get("detected_has_table_interruptions", ""),
                "reading_order_confidence": meta.get("reading_order_confidence", ""),
                "reading_order_source": meta.get("reading_order_source", ""),
                "ordering_source": meta.get("ordering_source", ""),
                "region_ocr_used": meta.get("region_ocr_used", ""),
                "region_ocr_attempted": meta.get("region_ocr_attempted", ""),
                "selected_strategy": meta.get("selected_strategy", ""),
                "strategy_mode": meta.get("strategy_mode", ""),
                "strategy_use_text_layer": meta.get("strategy_use_text_layer", ""),
                "strategy_use_full_page_ocr": meta.get("strategy_use_full_page_ocr", ""),
                "strategy_use_region_ocr": meta.get("strategy_use_region_ocr", ""),
                "strategy_primary_engine": meta.get("strategy_primary_engine", ""),
                "strategy_fallback_engines": meta.get("strategy_fallback_engines", []),
                "strategy_ensemble_needed": meta.get("strategy_ensemble_needed", ""),
                "strategy_max_engines_per_page": meta.get("strategy_max_engines_per_page", ""),
                "engines_attempted": meta.get("engines_attempted", []),
                "engines_skipped": meta.get("engines_skipped", {}),
                "engine_skip_reasons": meta.get("engine_skip_reasons", []),
                "fallback_path": meta.get("fallback_path", []),
                "final_output_source": meta.get("final_output_source", ""),
                "runtime_per_engine_ms": meta.get("runtime_per_engine_ms", {}),
                "confidence_per_engine": meta.get("confidence_per_engine", {}),
                "structured_layout": meta.get("structured_layout", {}),
                "plain_text_reconstruction": meta.get("plain_text_reconstruction", ""),
                "applied_preprocessing_profile": meta.get("preprocessing_profile", preprocessing_profile),
            }

        if self.ocr_fallback == 'ensemble' and self.ensemble is None:
            return "", False, {
                "engine_page_statuses": self.get_engine_readiness(),
                "failure_reason": "ensemble_not_initialized",
            }

        if not self.ocr_engine:
            return "", False, {
                "engine_page_statuses": self.get_engine_readiness(),
                "failure_reason": "paddle_not_initialized",
            }
        
        try:
            # Render page to image at 300 DPI
            doc = fitz.open(pdf_path)
            if page_num >= len(doc):
                doc.close()
                return "", False, {
                    "engine_page_statuses": {"paddle": {"status": ENGINE_STATUS_FAILED_ON_PAGE, "reason": "page_out_of_range"}},
                    "failure_reason": "page_out_of_range",
                }
            
            page = doc[page_num]
            # 300 DPI: matrix scale = 300/72 = 4.166...
            mat = fitz.Matrix(4.166, 4.166)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to numpy array
            import numpy as np
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            
            # Run OCR (version-aware)
            if getattr(self, '_paddle_v3', False) and hasattr(self.ocr_engine, 'predict'):
                results = self.ocr_engine.predict(img)
                lines = []
                for res in results:
                    rec_texts = getattr(res, 'rec_texts', None)
                    if rec_texts is None and isinstance(res, dict):
                        rec_texts = res.get('rec_texts', [])
                    for txt in (rec_texts or []):
                        t = str(txt).strip()
                        if t:
                            lines.append(t)
            else:
                result = self.ocr_engine.ocr(img, cls=True)
                lines = []
                if result and result[0]:
                    for line in result[0]:
                        if line and len(line) >= 2:
                            text = line[1][0]
                            lines.append(text)
            
            text = '\n'.join(lines)
            text = self._normalize_whitespace(text)
            
            doc.close()
            if not text.strip():
                return "", False, {
                    "engine_page_statuses": {"paddle": {"status": ENGINE_STATUS_FAILED_ON_PAGE, "reason": "no_text_candidate"}},
                    "failure_reason": "ocr_empty_output",
                    "engines_attempted": ["paddle"],
                    "final_output_source": "none",
                }
            return text, True, {
                "engine_page_statuses": {"paddle": {"status": ENGINE_STATUS_AVAILABLE, "reason": ""}},
                "engines_used": ["paddle"],
                "engines_attempted": ["paddle"],
                "final_output_source": "paddle",
            }
            
        except Exception as e:
            logger.debug(f"OCR extraction failed for {pdf_path} page {page_num}: {e}")
            return "", False, {
                "engine_page_statuses": {"paddle": {"status": ENGINE_STATUS_FAILED_ON_PAGE, "reason": str(e)}},
                "failure_reason": "ocr_exception",
            }
    
    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Normalize whitespace while preserving newlines."""
        # Collapse >2 spaces to single space on each line
        lines = text.split('\n')
        normalized_lines = []
        for line in lines:
            # Collapse multiple spaces
            line = re.sub(r'  +', ' ', line)
            normalized_lines.append(line.strip())
        
        # Remove empty lines but keep paragraph breaks
        result = '\n'.join(normalized_lines)
        # Collapse >2 consecutive newlines to 2
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        return result.strip()


class PageTextPipeline:
    """Main pipeline for page-level text extraction."""
    
    def __init__(self, args):
        """Initialize pipeline with CLI arguments."""
        self.args = args
        self.output_root = Path(args.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.debug_artifacts_dir = self.output_root / "debug_artifacts" if args.debug else None
        if self.debug_artifacts_dir is not None:
            (self.debug_artifacts_dir / "page_diagnostics").mkdir(parents=True, exist_ok=True)
            (self.debug_artifacts_dir / "preprocessing").mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.extractor = PDFTextExtractor(
            prefer_text_layer=args.prefer_text_layer,
            ocr_fallback=args.ocr_fallback if args.ocr_fallback != 'none' else None,
            force_ocr=args.force_ocr,
            profile_path=args.profile or 'profiles/akkadian_strict.json',
            per_engine_timeout_s=(args.engine_timeout_ms / 1000.0) if args.engine_timeout_ms else None,
            language_hint=args.language_hint,
            preprocessing_profile=args.preprocessing_profile,
            debug_preprocessing_dir=(self.debug_artifacts_dir / "preprocessing") if self.debug_artifacts_dir else None,
        )
        
        profile_path = args.profile or 'profiles/akkadian_strict.json'
        if args.language_detector == 'none':
            self.detector = NoopDetector()
        else:
            self.detector = AkkadianDetector(profile_path)

        self.postprocess_enabled = not bool(args.disable_postprocessing)
        self.postprocess_adapter_hint = str(args.postprocess_adapter or "").strip().lower()
        self.postprocessor: PostprocessingPipeline | None = None
        if self.postprocess_enabled:
            lexicon_paths = _parse_domain_paths(args.postprocess_lexicon)
            if args.akkadian_lexicon_path:
                lexicon_paths["akkadian"] = str(args.akkadian_lexicon_path)
            oracc_lexicon = os.getenv("ORACC_LEXICON_PATH", "").strip()
            if oracc_lexicon and "akkadian" not in lexicon_paths:
                lexicon_paths["akkadian"] = oracc_lexicon

            self.postprocessor = PostprocessingPipeline.from_optional_lexicon_paths(
                lexicon_paths=lexicon_paths,
                enable_rule_corrections=not bool(args.disable_rule_corrections),
                enable_model_correction=bool(args.enable_model_correction),
            )

        quality_config = _load_quality_config(args.quality_config)
        class_cfg = quality_config.get("class_thresholds", {}) if isinstance(quality_config.get("class_thresholds", {}), dict) else {}
        class_thresholds = QualityClassThresholds(
            production_quality_min=float(
                args.quality_threshold_production
                if args.quality_threshold_production is not None
                else class_cfg.get("production_quality_min", 0.85)
            ),
            usable_with_review_min=float(
                args.quality_threshold_usable
                if args.quality_threshold_usable is not None
                else class_cfg.get("usable_with_review_min", 0.70)
            ),
            weak_ocr_min=float(
                args.quality_threshold_weak
                if args.quality_threshold_weak is not None
                else class_cfg.get("weak_ocr_min", 0.50)
            ),
        )

        gate_overrides: Dict[str, Any] = {}
        gate_cfg = quality_config.get("gate_thresholds", {}) if isinstance(quality_config.get("gate_thresholds", {}), dict) else {}
        if gate_cfg:
            mode_cfg = gate_cfg.get(args.launch_gate_mode, {}) if isinstance(gate_cfg.get(args.launch_gate_mode, {}), dict) else {}
            if mode_cfg:
                gate_overrides.update(mode_cfg)
            else:
                gate_overrides.update(gate_cfg)

        if args.gate_max_empty_rate is not None:
            gate_overrides["max_empty_rate"] = float(args.gate_max_empty_rate)
        if args.gate_max_timeout_rate is not None:
            gate_overrides["max_timeout_rate"] = float(args.gate_max_timeout_rate)
        if args.gate_max_failed_rate is not None:
            gate_overrides["max_failed_rate"] = float(args.gate_max_failed_rate)
        if args.gate_min_avg_quality is not None:
            gate_overrides["min_avg_quality"] = float(args.gate_min_avg_quality)
        if args.gate_max_review_rate is not None:
            gate_overrides["max_review_rate"] = float(args.gate_max_review_rate)

        self.quality_scorer = OCRQualityScorer(
            class_thresholds=class_thresholds,
            gate_mode=args.launch_gate_mode,
            gate_overrides=gate_overrides,
        )
        self.launch_gate_mode = str(args.launch_gate_mode or "internal")
        
        # Output CSV paths
        self.output_csv = self.output_root / "client_page_text.csv"
        self.progress_csv = args.progress_csv or (self.output_root / "progress.csv")
        self.page_diagnostics_jsonl = args.page_diagnostics_jsonl or (self.output_root / "page_diagnostics.jsonl")
        self.layout_jsonl = args.layout_jsonl or (self.output_root / "layout_regions.jsonl")
        self.ensemble_output_jsonl = args.ensemble_output_jsonl or (self.output_root / "ensemble_output.jsonl")
        self.per_engine_output_jsonl = args.per_engine_output_jsonl or (self.output_root / "per_engine_output.jsonl")
        self.disagreement_report_json = args.disagreement_report_json or (self.output_root / "disagreement_report.json")
        self.confusion_matrix_json = args.confusion_matrix_json or (self.output_root / "confusion_matrix.json")
        self.final_output_json = args.final_output_json or (self.output_root / "client_page_text.json")
        self.document_quality_jsonl = args.document_quality_jsonl or (self.output_root / "document_quality.jsonl")
        self.run_quality_json = args.run_quality_json or (self.output_root / "run_quality.json")
        self.page_timeout_s = (args.page_timeout_ms / 1000.0) if args.page_timeout_ms else 0.0

        self.final_output_records: List[Dict[str, Any]] = []
        self.progress_records: List[Dict[str, Any]] = []
        self.page_diagnostic_records: List[Dict[str, Any]] = []
        self.layout_records: List[Dict[str, Any]] = []
        self.disagreement_pages: List[Dict[str, Any]] = []
        self.confusion_counter: Counter[str] = Counter()
        self.page_quality_records: List[Dict[str, Any]] = []
        self.document_page_records: Dict[str, List[Dict[str, Any]]] = {}
        self.document_quality_summaries: Dict[str, Dict[str, Any]] = {}
        self.run_quality_summary: Dict[str, Any] = {}
        self.launch_gate_result: Dict[str, Any] = {
            "mode": self.launch_gate_mode,
            "should_fail_run": False,
            "failed_gate": "",
            "gate_reason": "",
            "reasons": [],
            "review_required": False,
        }
        
        # Stats
        self.stats = {
            "pages_processed": 0,
            "total_pages": 0,
            "pages_with_akkadian": 0,
            "success_pages": 0,
            "partial_success_pages": 0,
            "failed_pages": 0,
            "timed_out_pages": 0,
            "skipped_pages": 0,
            "empty_pages": 0,
            "text_layer_used": 0,
            "ocr_used": 0,
            "human_review_pages": 0,
            "documents_processed": 0,
            "avg_page_quality_score": 0.0,
            "run_quality_score": 0.0,
            "errors": 0
        }
    
    def run(self) -> int:
        """Run the pipeline."""
        logger.info("Starting page text extraction pipeline")
        logger.info(f"Output: {self.output_csv}")
        logger.info("Engine readiness: %s", _json_string(self.extractor.get_engine_readiness()))
        
        # Get PDF pages to process
        pages = self._collect_pages()
        self.stats["total_pages"] = len(pages)
        logger.info(f"Found {len(pages)} pages to process")
        
        if not pages:
            logger.warning("No pages to process")
            return 1
        
        # Initialize output CSVs
        self._init_output_csv()
        self._init_progress_csv()
        self._init_page_diagnostics_jsonl()
        self._init_layout_jsonl()
        self._init_ensemble_output_jsonl()
        self._init_per_engine_output_jsonl()
        self._init_document_quality_jsonl()
        self._init_run_quality_json()
        
        # Process pages
        iterator = tqdm(pages, desc="Processing pages") if TQDM_AVAILABLE and self.args.status_bar else pages
        
        for pdf_path, page_num in iterator:
            self._process_page(pdf_path, page_num)

        self._finalize_quality_artifacts()
        self._finalize_ensemble_artifacts()
        
        # Report stats and return final exit code
        exit_code, failure_reasons = self._evaluate_exit_code()
        self._report_stats(exit_code, failure_reasons)
        return exit_code
    
    def _collect_pages(self) -> List[Tuple[str, int]]:
        """Collect PDF pages from manifest or inputs directory."""
        pages = []
        
        if self.args.manifest:
            # Read manifest TSV
            with open(self.args.manifest, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Skip header line
                    if i == 0 and line.startswith('pdf_path'):
                        continue
                    
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        pdf_path = parts[0]
                        try:
                            page_num = int(parts[1]) - 1  # Convert to 0-based
                            # Trust manifest - don't check existence (too slow for large manifests)
                            pages.append((pdf_path, page_num))
                        except ValueError:
                            # Skip lines with non-numeric page numbers (like headers)
                            continue
        
        elif self.args.inputs:
            # Recursively find PDFs
            input_path = Path(self.args.inputs)
            pdf_files = [p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"]
            
            for pdf_path in pdf_files:
                # Get page count
                try:
                    if PYMUPDF_AVAILABLE:
                        doc = fitz.open(str(pdf_path))
                        page_count = len(doc)
                        doc.close()
                        
                        for page_num in range(page_count):
                            pages.append((str(pdf_path), page_num))
                except Exception as e:
                    logger.error(f"Failed to open {pdf_path}: {e}")
        
        return pages
    
    def _init_output_csv(self):
        """Initialize output CSV with UTF-8 BOM."""
        with open(self.output_csv, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'pdf_name',
                'page',
                'page_text',
                'has_akkadian',
                'status',
                'failure_reason',
                'extraction_method',
                'engine_statuses',
                *OUTPUT_METADATA_FIELDS,
            ])
    
    def _init_progress_csv(self):
        """Initialize progress CSV."""
        with open(self.progress_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'pdf_name',
                'page',
                'ms',
                'used_text_layer',
                'has_akkadian',
                'status',
                'failure_reason',
                'extraction_method',
                'engine_statuses',
                *OUTPUT_METADATA_FIELDS,
                'timestamp',
            ])

    def _init_page_diagnostics_jsonl(self):
        """Initialize per-page diagnostics JSONL artifact."""
        with open(self.page_diagnostics_jsonl, 'w', encoding='utf-8') as f:
            f.write("")

    def _init_layout_jsonl(self):
        """Initialize structured layout JSONL artifact."""
        with open(self.layout_jsonl, 'w', encoding='utf-8') as f:
            f.write("")

    def _init_ensemble_output_jsonl(self):
        """Initialize ensemble-level output JSONL artifact."""
        with open(self.ensemble_output_jsonl, 'w', encoding='utf-8') as f:
            f.write("")

    def _init_per_engine_output_jsonl(self):
        """Initialize per-engine output JSONL artifact."""
        with open(self.per_engine_output_jsonl, 'w', encoding='utf-8') as f:
            f.write("")

    def _init_document_quality_jsonl(self):
        """Initialize per-document quality summary JSONL artifact."""
        with open(self.document_quality_jsonl, 'w', encoding='utf-8') as f:
            f.write("")

    def _init_run_quality_json(self):
        """Initialize run-level quality summary JSON artifact."""
        with open(self.run_quality_json, 'w', encoding='utf-8') as f:
            json.dump({}, f)

    def _score_and_attach_quality(
        self,
        *,
        pdf_name: str,
        page: int,
        status: str,
        failure_reason: str,
        text: str,
        extraction_method: str,
        engine_statuses: Dict[str, Dict[str, str]],
        extract_meta_raw: Dict[str, Any],
        output_meta: Dict[str, str],
        runtime_ms: int,
    ) -> None:
        language_hint = str(output_meta.get("language_hint", self.args.language_hint or "unknown") or "unknown")
        script_hint = str(extract_meta_raw.get("script_hint", output_meta.get("script_hint", "unknown")) or "unknown")

        quality = self.quality_scorer.score_page(
            text=text,
            status=status,
            failure_reason=failure_reason,
            metadata=extract_meta_raw,
            engine_statuses=engine_statuses,
            language_hint=language_hint,
            script_hint=script_hint,
        )

        existing_review = str(output_meta.get("needs_human_review", "")).strip().lower() == "true"
        needs_review = bool(existing_review or quality.get("needs_human_review", False))
        quality_reasons = list(quality.get("quality_reasons", []))

        output_meta["page_quality_score"] = _format_optional_number(quality.get("page_quality_score", ""))
        output_meta["quality_class"] = str(quality.get("quality_class", "") or "")
        output_meta["quality_reasons"] = _format_optional_text(quality_reasons)
        output_meta["needs_human_review"] = "true" if needs_review else "false"
        output_meta["failed_gate"] = _format_optional_text(quality.get("failed_gate", ""))
        output_meta["gate_reason"] = _format_optional_text(quality.get("gate_reason", ""))

        extract_meta_raw["page_quality_score"] = float(quality.get("page_quality_score", 0.0) or 0.0)
        extract_meta_raw["quality_class"] = str(quality.get("quality_class", "") or "")
        extract_meta_raw["quality_reasons"] = quality_reasons
        extract_meta_raw["needs_human_review"] = bool(needs_review)
        extract_meta_raw["failed_gate"] = str(quality.get("failed_gate", "") or "")
        extract_meta_raw["gate_reason"] = str(quality.get("gate_reason", "") or "")
        extract_meta_raw["quality_signal_breakdown"] = quality.get("signal_breakdown", {})

        page_quality_record = {
            "pdf_name": pdf_name,
            "page": int(page),
            "status": status,
            "failure_reason": failure_reason,
            "page_quality_score": float(quality.get("page_quality_score", 0.0) or 0.0),
            "quality_class": str(quality.get("quality_class", "") or ""),
            "needs_human_review": bool(needs_review),
            "runtime_ms": int(runtime_ms),
            "is_empty": not bool(str(text or "").strip()),
            "quality_reasons": quality_reasons,
            "extraction_method": extraction_method,
        }
        self.page_quality_records.append(page_quality_record)
        self.document_page_records.setdefault(pdf_name, []).append(page_quality_record)

        doc_summary = self.quality_scorer.aggregate_document_quality(self.document_page_records.get(pdf_name, []))
        output_meta["document_quality_score"] = _format_optional_number(doc_summary.get("document_quality_score", ""))
        extract_meta_raw["document_quality_score"] = float(doc_summary.get("document_quality_score", 0.0) or 0.0)

        if needs_review:
            self.stats["human_review_pages"] += 1

    def _finalize_quality_artifacts(self) -> None:
        """Write document/run quality summaries and propagate final gate status to outputs."""
        document_summaries: Dict[str, Dict[str, Any]] = {}
        for pdf_name, records in self.document_page_records.items():
            summary = self.quality_scorer.aggregate_document_quality(records)
            payload = {"pdf_name": pdf_name, **summary}
            document_summaries[pdf_name] = payload

        self.document_quality_summaries = document_summaries
        self.stats["documents_processed"] = len(document_summaries)

        with open(self.document_quality_jsonl, 'w', encoding='utf-8') as f:
            for pdf_name in sorted(document_summaries.keys()):
                f.write(json.dumps(document_summaries[pdf_name], ensure_ascii=False, sort_keys=True) + "\n")

        run_summary = self.quality_scorer.aggregate_run_quality(
            document_summaries=document_summaries,
            page_records=self.page_quality_records,
        )
        launch_gate = self.quality_scorer.evaluate_launch_gates(
            run_summary=run_summary,
            has_usable_engine=self.extractor.has_usable_ocr_engine(),
            strict_readiness_ok=self.extractor.strict_readiness_ok(),
        )

        self.run_quality_summary = dict(run_summary)
        self.launch_gate_result = {
            "mode": launch_gate.mode,
            "should_fail_run": bool(launch_gate.should_fail_run),
            "failed_gate": launch_gate.failed_gate,
            "gate_reason": launch_gate.gate_reason,
            "reasons": list(launch_gate.reasons),
            "review_required": bool(launch_gate.review_required),
        }

        self.stats["avg_page_quality_score"] = float(run_summary.get("average_quality_score", 0.0) or 0.0)
        self.stats["run_quality_score"] = float(run_summary.get("run_quality_score", 0.0) or 0.0)

        run_payload = {
            "generated_at": datetime.now().isoformat(),
            "launch_gate_mode": self.launch_gate_mode,
            "run_quality": run_summary,
            "launch_gate": self.launch_gate_result,
            "quality_thresholds": {
                "production_quality_min": self.quality_scorer.class_thresholds.production_quality_min,
                "usable_with_review_min": self.quality_scorer.class_thresholds.usable_with_review_min,
                "weak_ocr_min": self.quality_scorer.class_thresholds.weak_ocr_min,
            },
        }
        with open(self.run_quality_json, 'w', encoding='utf-8') as f:
            json.dump(run_payload, f, ensure_ascii=False, indent=2, sort_keys=True)

        for record in self.final_output_records:
            doc_summary = document_summaries.get(str(record.get("pdf_name", "")), {})
            if doc_summary:
                record["document_quality_score"] = doc_summary.get("document_quality_score", record.get("document_quality_score", ""))

            if launch_gate.failed_gate and not str(record.get("failed_gate", "")).strip():
                record["failed_gate"] = launch_gate.failed_gate
                record["gate_reason"] = launch_gate.gate_reason

        quality_by_page: Dict[Tuple[str, int], Dict[str, Any]] = {
            (str(item.get("pdf_name", "")), int(item.get("page", 0) or 0)): item
            for item in self.final_output_records
        }

        for progress_record in self.progress_records:
            key = (str(progress_record.get("pdf_name", "")), int(progress_record.get("page", 0) or 0))
            page_meta = quality_by_page.get(key)
            if page_meta is None:
                continue
            for field in QUALITY_FIELDS:
                if field in page_meta:
                    progress_record.setdefault("output_meta", {})[field] = page_meta.get(field, "")

        for diag_record in self.page_diagnostic_records:
            key = (str(diag_record.get("pdf_name", "")), int(diag_record.get("page", 0) or 0))
            page_meta = quality_by_page.get(key)
            if page_meta is None:
                continue
            for field in QUALITY_FIELDS:
                diag_record[field] = page_meta.get(field, diag_record.get(field, ""))

        for layout_record in self.layout_records:
            key = (str(layout_record.get("pdf_name", "")), int(layout_record.get("page", 0) or 0))
            page_meta = quality_by_page.get(key)
            if page_meta is None:
                continue
            metadata = layout_record.setdefault("metadata", {})
            for field in QUALITY_FIELDS:
                metadata[field] = page_meta.get(field, metadata.get(field, ""))

        # Rewrite streaming artifacts so document/run-level quality fields reflect final aggregates.
        self._init_output_csv()
        for row in self.final_output_records:
            with open(self.output_csv, 'a', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    row.get("pdf_name", ""),
                    row.get("page", ""),
                    str(row.get("page_text", "") or "").replace('\n', '\\n'),
                    'true' if bool(row.get("has_akkadian", False)) else 'false',
                    row.get("status", ""),
                    row.get("failure_reason", ""),
                    row.get("extraction_method", ""),
                    _json_string(row.get("engine_statuses", {})),
                    *(row.get(field, "") for field in OUTPUT_METADATA_FIELDS),
                ])

        self._init_progress_csv()
        for row in self.progress_records:
            meta = dict(row.get("output_meta", {}))
            with open(self.progress_csv, 'a', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    row.get("pdf_name", ""),
                    row.get("page", ""),
                    row.get("ms", ""),
                    row.get("used_text_layer", False),
                    row.get("has_akkadian", False),
                    row.get("status", ""),
                    row.get("failure_reason", ""),
                    row.get("extraction_method", ""),
                    _json_string(row.get("engine_statuses", {})),
                    *(meta.get(field, "") for field in OUTPUT_METADATA_FIELDS),
                    row.get("timestamp", ""),
                ])

        with open(self.page_diagnostics_jsonl, 'w', encoding='utf-8') as f:
            for record in self.page_diagnostic_records:
                f.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")

        with open(self.layout_jsonl, 'w', encoding='utf-8') as f:
            for record in self.layout_records:
                f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def _append_ensemble_artifacts(
        self,
        pdf_name: str,
        page: int,
        status: str,
        failure_reason: str,
        extraction_method: str,
        text: str,
        extract_meta: Dict[str, Any],
        output_meta: Dict[str, str],
    ) -> None:
        """Append page-level and per-engine ensemble analysis artifacts."""
        extract_meta = extract_meta or {}
        per_engine_outputs = extract_meta.get("per_engine_outputs", []) if isinstance(extract_meta, dict) else []
        if not isinstance(per_engine_outputs, list):
            per_engine_outputs = []

        confusion_counts = extract_meta.get("confusion_counts", {}) if isinstance(extract_meta, dict) else {}
        if isinstance(confusion_counts, dict):
            for key, value in confusion_counts.items():
                try:
                    self.confusion_counter[str(key)] += int(value)
                except (TypeError, ValueError):
                    continue

        char_disagreement = float(_format_optional_number(output_meta.get("char_disagreement_rate", "")) or 0.0)
        token_disagreement = float(_format_optional_number(output_meta.get("token_disagreement_rate", "")) or 0.0)
        line_disagreement = float(_format_optional_number(output_meta.get("line_disagreement_rate", "")) or 0.0)
        uncertain = output_meta.get("ensemble_uncertain", "") == "true"
        low_quality = output_meta.get("low_quality_ensemble", "") == "true"

        page_record: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "pdf_name": pdf_name,
            "page": int(page),
            "status": status,
            "failure_reason": failure_reason,
            "extraction_method": extraction_method,
            "text": text or "",
            "consensus_explanation": extract_meta.get("consensus_explanation", {}),
            "alignment_metrics": extract_meta.get("alignment_metrics", {}),
            "confusion_counts": confusion_counts if isinstance(confusion_counts, dict) else {},
            "per_engine_outputs": per_engine_outputs,
            "metadata": {field: output_meta.get(field, "") for field in OUTPUT_METADATA_FIELDS},
        }
        with open(self.ensemble_output_jsonl, 'a', encoding='utf-8') as f:
            f.write(json.dumps(page_record, ensure_ascii=False, sort_keys=True) + "\n")
            f.flush()

        if not per_engine_outputs:
            per_engine_outputs = [
                {
                    "engine": "none",
                    "text": "",
                    "confidence": 0.0,
                    "runtime_ms": 0.0,
                    "status": "not_applicable",
                    "error": "",
                    "timed_out": False,
                }
            ]

        for output in per_engine_outputs:
            if not isinstance(output, dict):
                continue
            engine_record = {
                "timestamp": datetime.now().isoformat(),
                "pdf_name": pdf_name,
                "page": int(page),
                "status": status,
                "failure_reason": failure_reason,
                "extraction_method": extraction_method,
                "consensus_winner_engine": output_meta.get("consensus_winner_engine", ""),
                "ensemble_uncertain": uncertain,
                "human_review_recommended": output_meta.get("human_review_recommended", "") == "true",
                **output,
            }
            with open(self.per_engine_output_jsonl, 'a', encoding='utf-8') as f:
                f.write(json.dumps(engine_record, ensure_ascii=False, sort_keys=True) + "\n")
                f.flush()

        self.disagreement_pages.append(
            {
                "pdf_name": pdf_name,
                "page": int(page),
                "status": status,
                "failure_reason": failure_reason,
                "char_disagreement_rate": round(char_disagreement, 6),
                "token_disagreement_rate": round(token_disagreement, 6),
                "line_disagreement_rate": round(line_disagreement, 6),
                "ensemble_uncertain": uncertain,
                "low_quality_ensemble": low_quality,
            }
        )

    def _finalize_ensemble_artifacts(self) -> None:
        """Write end-of-run aggregate disagreement/confusion/final JSON artifacts."""
        page_count = len(self.disagreement_pages)
        char_mean = (
            sum(item.get("char_disagreement_rate", 0.0) for item in self.disagreement_pages) / float(page_count)
            if page_count
            else 0.0
        )
        token_mean = (
            sum(item.get("token_disagreement_rate", 0.0) for item in self.disagreement_pages) / float(page_count)
            if page_count
            else 0.0
        )
        line_mean = (
            sum(item.get("line_disagreement_rate", 0.0) for item in self.disagreement_pages) / float(page_count)
            if page_count
            else 0.0
        )
        uncertain_pages = [item for item in self.disagreement_pages if item.get("ensemble_uncertain", False)]
        low_quality_pages = [item for item in self.disagreement_pages if item.get("low_quality_ensemble", False)]

        disagreement_report = {
            "generated_at": datetime.now().isoformat(),
            "pages_analyzed": page_count,
            "uncertain_pages": len(uncertain_pages),
            "low_quality_pages": len(low_quality_pages),
            "mean_char_disagreement_rate": round(float(char_mean), 6),
            "mean_token_disagreement_rate": round(float(token_mean), 6),
            "mean_line_disagreement_rate": round(float(line_mean), 6),
            "top_disagreement_pages": sorted(
                self.disagreement_pages,
                key=lambda item: float(item.get("char_disagreement_rate", 0.0)),
                reverse=True,
            )[:50],
        }

        with open(self.disagreement_report_json, 'w', encoding='utf-8') as f:
            json.dump(disagreement_report, f, ensure_ascii=False, indent=2, sort_keys=True)

        confusion_matrix_payload = {
            "generated_at": datetime.now().isoformat(),
            "pair_counts": dict(self.confusion_counter),
        }
        with open(self.confusion_matrix_json, 'w', encoding='utf-8') as f:
            json.dump(confusion_matrix_payload, f, ensure_ascii=False, indent=2, sort_keys=True)

        with open(self.final_output_json, 'w', encoding='utf-8') as f:
            json.dump(self.final_output_records, f, ensure_ascii=False, indent=2, sort_keys=True)
    
    def _process_page(self, pdf_path: str, page_num: int):
        """Process a single PDF page."""
        start_time = time.time()
        pdf_name = os.path.basename(pdf_path)
        page_1based = page_num + 1
        text = ""
        has_akkadian = False
        used_text_layer = False
        extraction_method = "unknown"
        engine_statuses: Dict[str, Dict[str, str]] = {}
        extract_meta_raw: Dict[str, Any] = {}
        output_meta: Dict[str, str] = {field: "" for field in OUTPUT_METADATA_FIELDS}
        output_meta["page_number"] = str(page_1based)
        output_meta["input_file"] = pdf_name
        output_meta["language_hint"] = str(self.args.language_hint or "unknown")
        status = PAGE_STATUS_FAILED
        failure_reason = ""
        
        try:
            # Extract text with optional per-page timeout guardrail
            if self.page_timeout_s > 0:
                extraction_result, timed_out = _run_with_timeout(
                    lambda: self.extractor.extract_page_text(pdf_path, page_num),
                    self.page_timeout_s,
                )
                if timed_out:
                    status = PAGE_STATUS_TIMED_OUT
                    failure_reason = f"page_timeout_{self.args.page_timeout_ms}ms"
                    self.stats["errors"] += 1
                    self.stats["empty_pages"] += 1
                    self.stats["pages_processed"] += 1
                    self._record_page_status(status)
                    elapsed_ms = int((time.time() - start_time) * 1000)
                    self._score_and_attach_quality(
                        pdf_name=pdf_name,
                        page=page_1based,
                        status=status,
                        failure_reason=failure_reason,
                        text="",
                        extraction_method="timeout",
                        engine_statuses={},
                        extract_meta_raw=extract_meta_raw,
                        output_meta=output_meta,
                        runtime_ms=elapsed_ms,
                    )
                    self._append_output(
                        pdf_name,
                        page_1based,
                        "",
                        False,
                        status,
                        failure_reason,
                        "timeout",
                        {},
                        output_meta,
                    )
                    self._append_progress(
                        pdf_name,
                        page_1based,
                        elapsed_ms,
                        False,
                        False,
                        status,
                        failure_reason,
                        "timeout",
                        {},
                        output_meta,
                    )
                    self._append_page_diagnostics(
                        pdf_name,
                        page_1based,
                        status,
                        failure_reason,
                        "timeout",
                        False,
                        False,
                        output_meta,
                    )
                    self._append_layout_regions(
                        pdf_name,
                        page_1based,
                        status,
                        failure_reason,
                        "timeout",
                        "",
                        engine_statuses,
                        extract_meta_raw,
                        output_meta,
                    )
                    self._append_ensemble_artifacts(
                        pdf_name,
                        page_1based,
                        status,
                        failure_reason,
                        "timeout",
                        "",
                        extract_meta_raw,
                        output_meta,
                    )
                    return
                text, used_text_layer, extract_meta = extraction_result
            else:
                text, used_text_layer, extract_meta = self.extractor.extract_page_text(pdf_path, page_num)

            extract_meta_raw = dict(extract_meta or {})
            extraction_method = extract_meta.get("method", "unknown")
            engine_statuses = extract_meta.get("engine_page_statuses") or {}
            output_meta = _normalize_output_metadata(extract_meta)
            output_meta["page_number"] = output_meta.get("page_number") or str(page_1based)
            output_meta["input_file"] = output_meta.get("input_file") or pdf_name
            if not output_meta.get("language_hint"):
                output_meta["language_hint"] = str(self.args.language_hint or "unknown")

            if self.postprocessor is not None:
                post_result = self.postprocessor.process(
                    text,
                    language_hint=str(output_meta.get("language_hint", self.args.language_hint or "unknown") or "unknown"),
                    script_hint=str(extract_meta_raw.get("script_hint", "unknown") or "unknown"),
                    adapter_hint=self.postprocess_adapter_hint or None,
                    page_number=page_1based,
                )
                text = post_result.corrected_text
                post_meta = post_result.to_metadata()
                extract_meta_raw.update(post_meta)
                output_meta = _merge_non_empty_metadata(output_meta, _normalize_output_metadata(post_meta))
            
            if not text:
                logger.warning(
                    "No text extracted from %s page %s (method=%s)",
                    pdf_name,
                    page_1based,
                    extraction_method,
                )
                is_mostly_blank = str(output_meta.get("is_mostly_blank", "")).strip().lower() == "true"
                page_image_rendered = str(output_meta.get("page_image_rendered", "true")).strip().lower() != "false"
                render_failure_reason = str(output_meta.get("render_failure_reason", "") or "").strip()

                if is_mostly_blank:
                    status = PAGE_STATUS_SKIPPED
                    failure_reason = "blank_page_detected"
                elif (not page_image_rendered) or render_failure_reason:
                    status = PAGE_STATUS_FAILED
                    failure_reason = f"render_failure:{render_failure_reason or 'unknown'}"
                    self.stats["errors"] += 1
                else:
                    status = PAGE_STATUS_FAILED
                    failure_reason = extract_meta.get("failure_reason", "empty_extraction")
                    self.stats["errors"] += 1

                self.stats["empty_pages"] += 1
                self.stats["pages_processed"] += 1
                self._record_page_status(status)
                elapsed_ms = int((time.time() - start_time) * 1000)
                _apply_row_integrity_metadata(
                    output_meta=output_meta,
                    pdf_path=pdf_path,
                    pdf_name=pdf_name,
                    page=page_1based,
                    extraction_method=extraction_method,
                    status=status,
                    failure_reason=failure_reason,
                    text="",
                    runtime_ms=elapsed_ms,
                )
                self._score_and_attach_quality(
                    pdf_name=pdf_name,
                    page=page_1based,
                    status=status,
                    failure_reason=failure_reason,
                    text="",
                    extraction_method=extraction_method,
                    engine_statuses=engine_statuses,
                    extract_meta_raw=extract_meta_raw,
                    output_meta=output_meta,
                    runtime_ms=elapsed_ms,
                )
                # Preserve one-row-per-page behavior even when extraction fails.
                self._append_output(
                    pdf_name,
                    page_1based,
                    "",
                    False,
                    status,
                    failure_reason,
                    extraction_method,
                    engine_statuses,
                    output_meta,
                )
                self._append_progress(
                    pdf_name,
                    page_1based,
                    elapsed_ms,
                    used_text_layer,
                    False,
                    status,
                    failure_reason,
                    extraction_method,
                    engine_statuses,
                    output_meta,
                )
                self._append_page_diagnostics(
                    pdf_name,
                    page_1based,
                    status,
                    failure_reason,
                    extraction_method,
                    used_text_layer,
                    False,
                    output_meta,
                )
                self._append_layout_regions(
                    pdf_name,
                    page_1based,
                    status,
                    failure_reason,
                    extraction_method,
                    "",
                    engine_statuses,
                    extract_meta_raw,
                    output_meta,
                )
                self._append_ensemble_artifacts(
                    pdf_name,
                    page_1based,
                    status,
                    failure_reason,
                    extraction_method,
                    "",
                    extract_meta_raw,
                    output_meta,
                )
                return
            
            # Detect Akkadian
            try:
                has_akkadian, _ = self.detector.detect_page(text)
            except Exception as detect_exc:
                has_akkadian = False
                status = PAGE_STATUS_PARTIAL_SUCCESS
                failure_reason = f"detection_error:{type(detect_exc).__name__}"
                self.stats["errors"] += 1
                logger.warning(
                    "Detector failure for %s page %s: %s",
                    pdf_name,
                    page_1based,
                    detect_exc,
                )

            if has_akkadian and output_meta.get("language_hint", "unknown") in {"", "unknown"}:
                output_meta["language_hint"] = "Akkadian transliteration"

            if status != PAGE_STATUS_PARTIAL_SUCCESS:
                degraded_engine_path = any(
                    info.get("status") in {
                        ENGINE_STATUS_TIMED_OUT,
                        ENGINE_STATUS_FAILED_ON_PAGE,
                        ENGINE_STATUS_UNAVAILABLE_DEPENDENCY,
                        ENGINE_STATUS_AVAILABLE_UNHEALTHY,
                    }
                    for info in engine_statuses.values()
                )
                if degraded_engine_path and extraction_method.startswith("ocr_"):
                    status = PAGE_STATUS_PARTIAL_SUCCESS
                    failure_reason = "degraded_engine_path"
                else:
                    status = PAGE_STATUS_SUCCESS
                    failure_reason = ""

            elapsed_ms = int((time.time() - start_time) * 1000)
            _apply_row_integrity_metadata(
                output_meta=output_meta,
                pdf_path=pdf_path,
                pdf_name=pdf_name,
                page=page_1based,
                extraction_method=extraction_method,
                status=status,
                failure_reason=failure_reason,
                text=text,
                runtime_ms=elapsed_ms,
            )
            self._score_and_attach_quality(
                pdf_name=pdf_name,
                page=page_1based,
                status=status,
                failure_reason=failure_reason,
                text=text,
                extraction_method=extraction_method,
                engine_statuses=engine_statuses,
                extract_meta_raw=extract_meta_raw,
                output_meta=output_meta,
                runtime_ms=elapsed_ms,
            )
            
            # Write to output CSV
            self._append_output(
                pdf_name,
                page_1based,
                text,
                has_akkadian,
                status,
                failure_reason,
                extraction_method,
                engine_statuses,
                output_meta,
            )
            
            # Update stats
            self.stats["pages_processed"] += 1
            self._record_page_status(status)
            if has_akkadian:
                self.stats["pages_with_akkadian"] += 1
            if used_text_layer:
                self.stats["text_layer_used"] += 1
            else:
                self.stats["ocr_used"] += 1
            
            # Write progress
            self._append_progress(
                pdf_name,
                page_1based,
                elapsed_ms,
                used_text_layer,
                has_akkadian,
                status,
                failure_reason,
                extraction_method,
                engine_statuses,
                output_meta,
            )
            self._append_page_diagnostics(
                pdf_name,
                page_1based,
                status,
                failure_reason,
                extraction_method,
                used_text_layer,
                has_akkadian,
                output_meta,
            )
            self._append_layout_regions(
                pdf_name,
                page_1based,
                status,
                failure_reason,
                extraction_method,
                text,
                engine_statuses,
                extract_meta_raw,
                output_meta,
            )
            self._append_ensemble_artifacts(
                pdf_name,
                page_1based,
                status,
                failure_reason,
                extraction_method,
                text,
                extract_meta_raw,
                output_meta,
            )
            
        except Exception as e:
            logger.error(f"Error processing {pdf_name} page {page_1based}: {e}")
            self.stats["errors"] += 1
            self.stats["empty_pages"] += 1
            self.stats["pages_processed"] += 1
            self._record_page_status(PAGE_STATUS_FAILED)
            elapsed_ms = int((time.time() - start_time) * 1000)
            failure_reason = f"unhandled_exception:{type(e).__name__}"
            _apply_row_integrity_metadata(
                output_meta=output_meta,
                pdf_path=pdf_path,
                pdf_name=pdf_name,
                page=page_1based,
                extraction_method=extraction_method,
                status=PAGE_STATUS_FAILED,
                failure_reason=failure_reason,
                text="",
                runtime_ms=elapsed_ms,
            )
            self._score_and_attach_quality(
                pdf_name=pdf_name,
                page=page_1based,
                status=PAGE_STATUS_FAILED,
                failure_reason=failure_reason,
                text="",
                extraction_method=extraction_method,
                engine_statuses=engine_statuses,
                extract_meta_raw=extract_meta_raw,
                output_meta=output_meta,
                runtime_ms=elapsed_ms,
            )
            self._append_output(
                pdf_name,
                page_1based,
                "",
                False,
                PAGE_STATUS_FAILED,
                failure_reason,
                extraction_method,
                engine_statuses,
                output_meta,
            )
            self._append_progress(
                pdf_name,
                page_1based,
                elapsed_ms,
                used_text_layer,
                False,
                PAGE_STATUS_FAILED,
                failure_reason,
                extraction_method,
                engine_statuses,
                output_meta,
            )
            self._append_page_diagnostics(
                pdf_name,
                page_1based,
                PAGE_STATUS_FAILED,
                failure_reason,
                extraction_method,
                used_text_layer,
                False,
                output_meta,
            )
            self._append_layout_regions(
                pdf_name,
                page_1based,
                PAGE_STATUS_FAILED,
                failure_reason,
                extraction_method,
                "",
                engine_statuses,
                extract_meta_raw,
                output_meta,
            )
            self._append_ensemble_artifacts(
                pdf_name,
                page_1based,
                PAGE_STATUS_FAILED,
                failure_reason,
                extraction_method,
                "",
                extract_meta_raw,
                output_meta,
            )

    def _append_output(
        self,
        pdf_name: str,
        page: int,
        text: str,
        has_akkadian: bool,
        status: str,
        failure_reason: str,
        extraction_method: str,
        engine_statuses: Dict[str, Dict[str, str]],
        output_meta: Dict[str, str],
    ):
        """Append row to output CSV."""
        record = {
            "pdf_name": pdf_name,
            "page": int(page),
            "page_text": text or "",
            "has_akkadian": bool(has_akkadian),
            "status": status,
            "failure_reason": failure_reason,
            "extraction_method": extraction_method,
            "engine_statuses": engine_statuses,
            **{field: output_meta.get(field, "") for field in OUTPUT_METADATA_FIELDS},
        }
        self.final_output_records.append(record)

        with open(self.output_csv, 'a', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                pdf_name,
                page,
                (text or "").replace('\n', '\\n'),  # Escape newlines for CSV
                'true' if has_akkadian else 'false',
                status,
                failure_reason,
                extraction_method,
                _json_string(engine_statuses),
                *(output_meta.get(field, "") for field in OUTPUT_METADATA_FIELDS),
            ])
            f.flush()
    
    def _append_progress(
        self,
        pdf_name: str,
        page: int,
        ms: int,
        used_text_layer: bool,
        has_akkadian: bool,
        status: str,
        failure_reason: str,
        extraction_method: str,
        engine_statuses: Dict[str, Dict[str, str]],
        output_meta: Dict[str, str],
    ):
        """Append row to progress CSV."""
        timestamp = datetime.now().isoformat()
        self.progress_records.append(
            {
                "pdf_name": pdf_name,
                "page": int(page),
                "ms": int(ms),
                "used_text_layer": bool(used_text_layer),
                "has_akkadian": bool(has_akkadian),
                "status": status,
                "failure_reason": failure_reason,
                "extraction_method": extraction_method,
                "engine_statuses": engine_statuses,
                "output_meta": dict(output_meta),
                "timestamp": timestamp,
            }
        )

        with open(self.progress_csv, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                pdf_name,
                page,
                ms,
                used_text_layer,
                has_akkadian,
                status,
                failure_reason,
                extraction_method,
                _json_string(engine_statuses),
                *(output_meta.get(field, "") for field in OUTPUT_METADATA_FIELDS),
                timestamp,
            ])
            f.flush()

    def _append_page_diagnostics(
        self,
        pdf_name: str,
        page: int,
        status: str,
        failure_reason: str,
        extraction_method: str,
        used_text_layer: bool,
        has_akkadian: bool,
        output_meta: Dict[str, str],
    ):
        """Append structured per-page diagnostics JSONL and optional debug artifact."""
        record: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "pdf_name": pdf_name,
            "page": page,
            "status": status,
            "failure_reason": failure_reason,
            "extraction_method": extraction_method,
            "used_text_layer": bool(used_text_layer),
            "has_akkadian": bool(has_akkadian),
        }
        for field in OUTPUT_METADATA_FIELDS:
            record[field] = output_meta.get(field, "")

        self.page_diagnostic_records.append(dict(record))

        with open(self.page_diagnostics_jsonl, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
            f.flush()

        if self.debug_artifacts_dir is None:
            return

        diagnostics_dir = self.debug_artifacts_dir / "page_diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        safe_pdf = re.sub(r'[^A-Za-z0-9._-]+', '_', pdf_name).strip('_') or "page"
        debug_path = diagnostics_dir / f"{safe_pdf}_p{page:04d}.json"
        with open(debug_path, 'w', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=True, indent=2, sort_keys=True)

    def _append_layout_regions(
        self,
        pdf_name: str,
        page: int,
        status: str,
        failure_reason: str,
        extraction_method: str,
        text: str,
        engine_statuses: Dict[str, Dict[str, str]],
        extract_meta: Dict[str, Any],
        output_meta: Dict[str, str],
    ):
        """Append structured layout/region reconstruction artifact."""
        structured_layout = extract_meta.get("structured_layout") if isinstance(extract_meta, dict) else None
        if not isinstance(structured_layout, dict):
            structured_layout = {
                "page": int(page),
                "column_count": int(_format_optional_int(output_meta.get("detected_column_count", "")) or 0),
                "column_mode": output_meta.get("detected_page_layout_mode", "") or output_meta.get("detected_layout_type", ""),
                "has_footnotes": output_meta.get("detected_has_footnotes", "") == "true",
                "has_table_interruptions": output_meta.get("detected_has_table_interruptions", "") == "true",
                "reading_order_confidence": float(_format_optional_number(output_meta.get("reading_order_confidence", "")) or 0.0),
                "ordering_source": output_meta.get("reading_order_source", "") or output_meta.get("ordering_source", "") or "unknown",
                "text_direction": "unknown",
                "regions": [
                    {
                        "region_id": f"p{int(page)}_r1",
                        "type": "unknown",
                        "bbox": [],
                        "reading_order": 1,
                        "confidence": 0.0,
                        "ordering_source": "unknown",
                        "line_reading_confidence": 0.0,
                        "text": text or "",
                        "line_ordering": [],
                        "engine": extraction_method,
                    }
                ],
            }

        plain_text = ""
        if isinstance(extract_meta, dict):
            plain_text = str(extract_meta.get("plain_text_reconstruction", "") or "")
        if not plain_text:
            plain_text = text or ""

        if isinstance(structured_layout, dict) and self.postprocessor is not None:
            regions = structured_layout.get("regions", [])
            if isinstance(regions, list):
                for region in regions:
                    if not isinstance(region, dict):
                        continue
                    region_text = str(region.get("text", "") or "")

                    region_result = self.postprocessor.process(
                        region_text,
                        language_hint=str(output_meta.get("language_hint", "unknown") or "unknown"),
                        script_hint=str((extract_meta or {}).get("script_hint", "unknown") or "unknown"),
                        adapter_hint=self.postprocess_adapter_hint or None,
                        page_number=int(page),
                        region_id=str(region.get("region_id", "") or ""),
                    )
                    region["postprocessing"] = region_result.to_metadata()

        page_quality_score = float(_format_optional_number(output_meta.get("page_quality_score", "")) or 0.0)
        if isinstance(structured_layout, dict):
            regions = structured_layout.get("regions", [])
            if isinstance(regions, list):
                for region in regions:
                    if not isinstance(region, dict):
                        continue
                    region_quality = self.quality_scorer.score_region(region, page_quality_score=page_quality_score)
                    region["region_quality_score"] = region_quality.get("region_quality_score", 0.0)
                    region["region_type"] = region_quality.get("region_type", region.get("type", "unknown"))
                    region["confidence"] = region_quality.get("confidence", region.get("confidence", 0.0))
                    region["text_length"] = region_quality.get("text_length", len(str(region.get("text", "") or "")))
                    region["engine_used"] = region_quality.get("engine_used", region.get("engine", "unknown"))
                    region["needs_review"] = bool(region_quality.get("needs_review", False))
                    region["quality_reasons"] = region_quality.get("quality_reasons", [])
                    region["quality_class"] = region_quality.get("quality_class", "usable_with_review")

        layout_record: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "pdf_name": pdf_name,
            "page": int(page),
            "status": status,
            "failure_reason": failure_reason,
            "extraction_method": extraction_method,
            "engine_statuses": engine_statuses,
            "layout": structured_layout,
            "plain_text_reconstruction": plain_text,
            "metadata": {field: output_meta.get(field, "") for field in OUTPUT_METADATA_FIELDS},
        }

        self.layout_records.append(dict(layout_record))

        with open(self.layout_jsonl, 'a', encoding='utf-8') as f:
            f.write(json.dumps(layout_record, ensure_ascii=False, sort_keys=True) + "\n")
            f.flush()
    
    def _record_page_status(self, status: str):
        if status == PAGE_STATUS_SUCCESS:
            self.stats["success_pages"] += 1
        elif status == PAGE_STATUS_PARTIAL_SUCCESS:
            self.stats["partial_success_pages"] += 1
            self.stats["success_pages"] += 1
        elif status == PAGE_STATUS_TIMED_OUT:
            self.stats["timed_out_pages"] += 1
            self.stats["failed_pages"] += 1
        elif status == PAGE_STATUS_SKIPPED:
            self.stats["skipped_pages"] += 1
        else:
            self.stats["failed_pages"] += 1

    def _evaluate_exit_code(self) -> Tuple[int, List[str]]:
        failure_reasons: List[str] = []

        total_pages = int(self.stats.get("total_pages", 0))
        if total_pages > 0 and self.stats.get("success_pages", 0) == 0:
            failure_reasons.append("all_pages_failed")

        if self.args.ocr_fallback != 'none' and not self.extractor.has_usable_ocr_engine():
            failure_reasons.append("no_usable_ocr_engine")

        if self.args.strict_readiness and not self.extractor.strict_readiness_ok():
            failure_reasons.append("strict_readiness_failed")

        if total_pages > 0:
            empty_rate = self.stats.get("empty_pages", 0) / float(total_pages)
            if empty_rate > self.args.max_empty_rate:
                failure_reasons.append(
                    f"empty_extraction_rate_exceeded({empty_rate:.3f}>{self.args.max_empty_rate:.3f})"
                )

        if bool(self.launch_gate_result.get("should_fail_run", False)):
            failed_gate = str(self.launch_gate_result.get("failed_gate", "") or "launch_gate")
            gate_reason = str(self.launch_gate_result.get("gate_reason", "") or "launch gate failed")
            failure_reasons.append(f"launch_gate_failed:{failed_gate}:{gate_reason}")

        for artifact_path in (
            Path(self.output_csv),
            Path(self.progress_csv),
            Path(self.page_diagnostics_jsonl),
            Path(self.layout_jsonl),
            Path(self.ensemble_output_jsonl),
            Path(self.per_engine_output_jsonl),
            Path(self.disagreement_report_json),
            Path(self.confusion_matrix_json),
            Path(self.final_output_json),
            Path(self.document_quality_jsonl),
            Path(self.run_quality_json),
        ):
            if not artifact_path.exists() or artifact_path.stat().st_size == 0:
                failure_reasons.append(f"missing_output:{artifact_path.name}")

        return (1, failure_reasons) if failure_reasons else (0, [])

    def _report_stats(self, exit_code: int, failure_reasons: List[str]):
        """Report final statistics."""
        total_pages = max(1, int(self.stats.get("total_pages", 0)))
        empty_rate = self.stats.get("empty_pages", 0) / float(total_pages)
        logger.info("=" * 60)
        logger.info("Pipeline completed")
        logger.info(f"Pages processed: {self.stats['pages_processed']}")
        logger.info(
            "Page statuses: success=%s partial_success=%s failed=%s timed_out=%s skipped=%s",
            self.stats['success_pages'],
            self.stats['partial_success_pages'],
            self.stats['failed_pages'],
            self.stats['timed_out_pages'],
            self.stats['skipped_pages'],
        )
        logger.info(f"Pages with Akkadian: {self.stats['pages_with_akkadian']}")
        logger.info(f"Text layer used: {self.stats['text_layer_used']}")
        logger.info(f"OCR used: {self.stats['ocr_used']}")
        logger.info("Postprocessing human-review pages: %s", self.stats.get("human_review_pages", 0))
        logger.info("Documents processed: %s", self.stats.get("documents_processed", 0))
        logger.info("Average page quality score: %.3f", float(self.stats.get("avg_page_quality_score", 0.0) or 0.0))
        logger.info("Run quality score: %.3f", float(self.stats.get("run_quality_score", 0.0) or 0.0))
        logger.info("Launch gate mode: %s", self.launch_gate_mode)
        logger.info("Launch gate result: fail=%s gate=%s", self.launch_gate_result.get("should_fail_run", False), self.launch_gate_result.get("failed_gate", ""))
        logger.info("Empty extraction rate: %.3f", empty_rate)
        logger.info(f"Errors: {self.stats['errors']}")
        logger.info(f"Output CSV: {self.output_csv}")
        logger.info(f"Final output JSON: {self.final_output_json}")
        logger.info(f"Progress CSV: {self.progress_csv}")
        logger.info(f"Page diagnostics JSONL: {self.page_diagnostics_jsonl}")
        logger.info(f"Layout regions JSONL: {self.layout_jsonl}")
        logger.info(f"Ensemble output JSONL: {self.ensemble_output_jsonl}")
        logger.info(f"Per-engine output JSONL: {self.per_engine_output_jsonl}")
        logger.info(f"Disagreement report JSON: {self.disagreement_report_json}")
        logger.info(f"Confusion matrix JSON: {self.confusion_matrix_json}")
        logger.info(f"Document quality JSONL: {self.document_quality_jsonl}")
        logger.info(f"Run quality JSON: {self.run_quality_json}")
        if failure_reasons:
            logger.error("Launch-blocking failures: %s", ", ".join(failure_reasons))
        logger.info("Exit code: %s", exit_code)
        logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Page-level text extraction with Akkadian detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--manifest', type=str,
                            help='Path to manifest TSV (pdf_path<TAB>page_no)')
    input_group.add_argument('--inputs', type=str,
                            help='Directory to scan for PDFs recursively')
    
    # Output
    parser.add_argument('--output-root', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--progress-csv', type=str,
                       help='Path to progress CSV (default: output-root/progress.csv)')
    parser.add_argument('--page-diagnostics-jsonl', type=str,
                       help='Path to per-page diagnostics JSONL (default: output-root/page_diagnostics.jsonl)')
    parser.add_argument('--layout-jsonl', type=str,
                       help='Path to per-page structured layout JSONL (default: output-root/layout_regions.jsonl)')
    parser.add_argument('--ensemble-output-jsonl', type=str,
                       help='Path to page-level ensemble output JSONL (default: output-root/ensemble_output.jsonl)')
    parser.add_argument('--per-engine-output-jsonl', type=str,
                       help='Path to per-engine output JSONL (default: output-root/per_engine_output.jsonl)')
    parser.add_argument('--disagreement-report-json', type=str,
                       help='Path to aggregate disagreement report JSON (default: output-root/disagreement_report.json)')
    parser.add_argument('--confusion-matrix-json', type=str,
                       help='Path to aggregate confusion matrix JSON (default: output-root/confusion_matrix.json)')
    parser.add_argument('--final-output-json', type=str,
                       help='Path to final client output JSON (default: output-root/client_page_text.json)')
    parser.add_argument('--document-quality-jsonl', type=str,
                       help='Path to per-document quality summary JSONL (default: output-root/document_quality.jsonl)')
    parser.add_argument('--run-quality-json', type=str,
                       help='Path to run-level quality summary JSON (default: output-root/run_quality.json)')

    # Postprocessing
    parser.add_argument('--disable-postprocessing', action='store_true', default=False,
                       help='Disable postprocessing pipeline and keep raw extracted text only')
    parser.add_argument('--disable-rule-corrections', action='store_true', default=False,
                       help='Disable lexicon/rule-based corrections while retaining cleanup and quality scoring')
    parser.add_argument('--enable-model-correction', action='store_true', default=False,
                       help='Enable optional guarded model correction stage (no model is configured by default)')
    parser.add_argument('--postprocess-adapter', type=str,
                       help='Force a postprocessing adapter (e.g., default_latin, german, akkadian_transliteration)')
    parser.add_argument('--postprocess-lexicon', action='append', default=[],
                       help='Optional lexicon path mapping in form domain=/path/to/words.txt; may be repeated')
    parser.add_argument('--akkadian-lexicon-path', type=str,
                       help='Optional domain lexicon path for akkadian/transliteration adapter')
    
    # Text extraction
    parser.add_argument('--prefer-text-layer', action='store_true', default=False,
                       help='Prefer PDF text layer extraction')
    parser.add_argument('--ocr-fallback', type=str, choices=['paddle', 'ensemble', 'none'], default='none',
                       help='OCR engine for fallback when text layer fails')
    parser.add_argument('--force-ocr', action='store_true', default=False,
                       help='Run OCR even when a text layer is present')
    
    # Akkadian detection
    parser.add_argument('--profile', type=str,
                       help='Path to detection profile JSON (default: profiles/akkadian_strict.json)')
    parser.add_argument('--language-detector', type=str, choices=['akkadian', 'none'], default='akkadian',
                       help='Language/domain detector adapter to apply after extraction')
    parser.add_argument('--language-hint', type=str, default='unknown',
                       help='Optional language/domain hint to attach to diagnostics (no hardcoded behavior)')
    parser.add_argument(
        '--preprocessing-profile',
        type=str,
        default=PROFILE_AUTO,
        choices=[PROFILE_AUTO, *available_preprocessing_profiles()],
        help='Preprocessing profile override for OCR pages (default: auto diagnostics-driven selection)',
    )
    
    # UI
    parser.add_argument('--status-bar', action='store_true', default=False,
                       help='Show progress bar')
    parser.add_argument('--debug', action='store_true', default=False,
                       help='Enable debug-level logging')

    # Quality scoring and launch gates
    parser.add_argument('--launch-gate-mode', type=str, choices=['internal', 'beta', 'production', 'strict'], default='internal',
                       help='Launch gate mode (internal/beta/production/strict)')
    parser.add_argument('--quality-config', type=str,
                       help='Optional JSON config file for quality thresholds and gate overrides')
    parser.add_argument('--quality-threshold-production', type=float,
                       help='Lower bound for production_quality class (default 0.85)')
    parser.add_argument('--quality-threshold-usable', type=float,
                       help='Lower bound for usable_with_review class (default 0.70)')
    parser.add_argument('--quality-threshold-weak', type=float,
                       help='Lower bound for weak_ocr class (default 0.50)')
    parser.add_argument('--gate-max-empty-rate', type=float,
                       help='Override launch gate max empty page rate for selected mode')
    parser.add_argument('--gate-max-timeout-rate', type=float,
                       help='Override launch gate max timeout rate for selected mode')
    parser.add_argument('--gate-max-failed-rate', type=float,
                       help='Override launch gate max failed page rate for selected mode')
    parser.add_argument('--gate-min-avg-quality', type=float,
                       help='Override launch gate minimum average quality for selected mode')
    parser.add_argument('--gate-max-review-rate', type=float,
                       help='Override launch gate maximum review-needed rate for selected mode')

    # Runtime guardrails and failure semantics
    parser.add_argument('--strict-readiness', action='store_true', default=False,
                       help='Fail run when any enabled OCR engine is not fully ready')
    parser.add_argument('--max-empty-rate', type=float, default=1.0,
                       help='Fail when empty extraction rate exceeds this threshold (0.0-1.0)')
    parser.add_argument('--page-timeout-ms', type=int, default=0,
                       help='Best-effort timeout per page in milliseconds (0 disables)')
    parser.add_argument('--engine-timeout-ms', type=int, default=0,
                       help='Best-effort timeout per OCR backend call in milliseconds (0 uses profile/default)')
    
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if not (0.0 <= args.max_empty_rate <= 1.0):
        logger.error("--max-empty-rate must be between 0.0 and 1.0")
        sys.exit(2)

    if args.page_timeout_ms < 0 or args.engine_timeout_ms < 0:
        logger.error("--page-timeout-ms and --engine-timeout-ms must be >= 0")
        sys.exit(2)

    quality_threshold_values = [
        args.quality_threshold_production,
        args.quality_threshold_usable,
        args.quality_threshold_weak,
        args.gate_max_empty_rate,
        args.gate_max_timeout_rate,
        args.gate_max_failed_rate,
        args.gate_min_avg_quality,
        args.gate_max_review_rate,
    ]
    for candidate in quality_threshold_values:
        if candidate is None:
            continue
        if not (0.0 <= float(candidate) <= 1.0):
            logger.error("Quality thresholds and gate rates must be between 0.0 and 1.0")
            sys.exit(2)

    if not args.prefer_text_layer and args.ocr_fallback == 'none':
        logger.error(
            "No extraction method enabled. Use --prefer-text-layer and/or set --ocr-fallback to paddle or ensemble."
        )
        sys.exit(2)

    if args.force_ocr and args.ocr_fallback == 'none':
        logger.error("--force-ocr requires --ocr-fallback paddle or ensemble")
        sys.exit(2)
    
    # Validate dependencies
    if not PYMUPDF_AVAILABLE:
        logger.error("PyMuPDF (fitz) is required. Install: pip install PyMuPDF")
        sys.exit(1)
    
    if args.ocr_fallback == 'paddle':
        try:
            import paddleocr
        except ImportError:
            logger.error("PaddleOCR is required for OCR fallback. Install: pip install paddleocr")
            sys.exit(1)

    if args.ocr_fallback == 'ensemble':
        try:
            import PIL  # noqa: F401
        except ImportError:
            logger.error("Pillow is required for ensemble fallback. Install: pip install Pillow")
            sys.exit(1)
    
    # Run pipeline
    pipeline = PageTextPipeline(args)
    exit_code = pipeline.run()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
