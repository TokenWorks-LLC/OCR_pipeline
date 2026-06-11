#!/usr/bin/env python3
"""Active OCR ensemble support for the page-text pipeline.

This module provides four layers for hard OCR pages:

1. Aggressive page preprocessing with denoise, contrast, adaptive thresholding,
   and morphology-aware image variants.
2. Best-effort OCR backend adapters for PaddleOCR, docTR, MMOCR, and Kraken.
3. Orientation search across right-angle rotations plus fine-angle deskew correction.
4. A diacritic-aware fusion stage with multi-column reading-order reconstruction.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import json
import logging
import os
from pathlib import Path
import re
from statistics import median
import threading
import time
import unicodedata
from typing import Any, Iterable

import fitz

from production.ensemble_analysis import aggregate_confusion, analyze_alignment, explain_consensus
from production.layout_analysis import LayoutAnalyzer, LayoutPageResult
from production.ocr_strategy import OCRRoutingStrategy, OCRStrategySelector
from production.preprocessing_profiles import (
    PROFILE_AUTO,
    PROFILE_UNKNOWN_SAFE_DEFAULT,
    available_preprocessing_profiles,
    merge_profile_with_base,
    resolve_preprocessing_profile,
)

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False

try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps

    PIL_AVAILABLE = True
except ImportError:
    Image = None
    ImageEnhance = None
    ImageFilter = None
    ImageOps = None
    PIL_AVAILABLE = False


logger = logging.getLogger(__name__)

ARABIC_RANGES = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)
DEFAULT_PREPROCESSING = {
    "rotation_search_degrees": [0, 90, 180, 270],
    "enable_adaptive_threshold": True,
    "enable_denoise": True,
    "enable_morphology": True,
    "enable_background_normalization": False,
    "preserve_diacritics": False,
    "avoid_aggressive_binarization": False,
    "speckle_removal": False,
    "adaptive_block_size": 35,
    "adaptive_C": 11,
    "denoise_strength": 12,
    "morphology_kernel_size": 2,
    "profile_primary_variant": "adaptive",
    "profile_variant_order": [],
    "enable_region_ocr": True,
    "region_ocr_max_regions": 24,
    "region_ocr_min_chars": 18,
    "contrast_factor": 1.8,
    "sharpen_radius": 1.5,
    "sharpen_percent": 180,
    "deskew": True,
    "deskew_min_abs_degrees": 0.2,
    "deskew_max_abs_degrees": 45.0,
}
DEFAULT_LAYOUT = {
    "detect_multi_column": True,
    "min_lines_for_columns": 4,
    "column_gap_ratio": 0.14,
    "column_overlap_ratio": 0.55,
}

ENGINE_STATUS_AVAILABLE = "available"
ENGINE_STATUS_UNAVAILABLE_DEPENDENCY = "unavailable_dependency_error"
ENGINE_STATUS_AVAILABLE_UNHEALTHY = "available_but_unhealthy"
ENGINE_STATUS_DISABLED_BY_CONFIG = "disabled_by_config"
ENGINE_STATUS_TIMED_OUT = "timed_out"
ENGINE_STATUS_FAILED_ON_PAGE = "failed_on_page"


def _run_callable_with_timeout(func, timeout_s: float | None):
    """Run a callable with a best-effort timeout.

    The worker thread is daemonized so timed-out calls do not block batch
    completion. This gives the pipeline a bounded wait for slow pages/backends.
    """
    if not timeout_s or timeout_s <= 0:
        return func(), False

    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}
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


def _normalize_whitespace(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _strip_diacritics(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _comparison_key(text: str) -> str:
    simplified = _strip_diacritics(text).casefold()
    return re.sub(r"\s+", " ", simplified).strip()


_AKKADIAN_DIACRITICS = frozenset("šṣṭḫāēīūŠṢṬḪĀĒĪŪ")
_AKKADIAN_MARKERS = frozenset(["LUGAL", "DUMU", "DINGIR", "KÙ.BABBAR"])

# CuReD emits "{X" as shorthand for a single-character superscript (per its
# README: "{dAMAR.UTU" → "{d}AMAR.UTU"). The other engines and the akkadian
# profile's preserve_chars use real Unicode superscripts. Without this
# normalization CuReD's correct tokens never match anyone else's in the
# fuser, so the ROVER voting silently drops every superscript token it read.
_CURED_SUPERSCRIPT_MAP = {
    "d": "ᵈ",  # DINGIR / deity determinative
    "m": "ᵐ",  # male personal name
    "f": "ᶠ",  # female personal name
}
_CURED_OPEN_BRACE_RE = re.compile(r"\{(.)")
_CURED_CLOSED_SUPER_RE = re.compile(r"\{(.)\}")


def _postprocess_cured_text(text: str) -> str:
    """Apply CuReD's documented superscript closure and map to Unicode.

    Step 1: close every "{X" as "{X}" (straight from CuReD README).
    Step 2: for X in {d, m, f}, replace "{X}" with the Unicode superscript
    (ᵈ/ᵐ/ᶠ) the rest of the pipeline uses. Other "{X}" forms are left alone
    so they're still legible if they appear.
    """
    closed = _CURED_OPEN_BRACE_RE.sub(r"{\1}", text)
    return _CURED_CLOSED_SUPER_RE.sub(
        lambda m: _CURED_SUPERSCRIPT_MAP.get(m.group(1), m.group(0)), closed
    )
# Short-token hyphenated words: 1-3 char tokens joined by 2+ hyphens.
# Akkadian transliteration is dominated by these (a-na-ku, qi-bi-ma, be-li-ia).
# English hyphenated words almost always have ≥1 token >3 chars
# (step-by-step, mother-in-law, out-of-date), so this pattern misses them.
_SHORT_HYPHEN_RE = re.compile(r'\b[a-z]{1,3}(?:-[a-z]{1,3}){2,}\b', re.IGNORECASE)
# Numbered verse lines common in transliteration editions: "1. a-na ..."
_LINE_NUMBER_RE = re.compile(r'(?:^|\n)\s*\d{1,3}[.\)]\s+\S', re.MULTILINE)


def _is_akkadian_transliteration(text: str) -> bool:
    """Return True if text looks like Akkadian transliteration.

    Uses signals that survive diacritic-loss so the check works on output from
    general-purpose OCR engines (Paddle/docTR/MMOCR) which can't produce š/ṭ/ḫ.
    """
    if any(ch in _AKKADIAN_DIACRITICS for ch in text):
        return True
    upper = text.upper()
    if any(marker in upper for marker in _AKKADIAN_MARKERS):
        return True
    short_hyphen_count = len(_SHORT_HYPHEN_RE.findall(text))
    if short_hyphen_count >= 2:
        return True
    # One short-hyphen token is weak alone (e.g. "tic-tac-toe"); require a
    # numbered-line marker as corroboration.
    return short_hyphen_count >= 1 and bool(_LINE_NUMBER_RE.search(text))


def _word_looks_transliteration(word: str) -> bool:
    """Per-word version of the transliteration check.

    Line-level signals (numbered lines) don't apply to isolated words, so this
    relies on intrinsic per-word cues: diacritics, Sumerogram markers, or the
    short-hyphen shape (matching the whole word).
    """
    if not word:
        return False
    if any(ch in _AKKADIAN_DIACRITICS for ch in word):
        return True
    if word.upper() in _AKKADIAN_MARKERS:
        return True
    return bool(_SHORT_HYPHEN_RE.fullmatch(word))


def _normalize_for_enrichment(word: str) -> str:
    """Collapse word to letters only (no diacritics, no hyphens, casefolded).

    Used to detect when CuReD's output is the same word as another engine's
    output except for diacritics (and optional hyphenation), e.g. "be-lí-ia"
    vs "be-li-ia" or "belia". Hyphens are dropped because Paddle/docTR often
    run hyphenated transliteration tokens together.
    """
    stripped = _strip_diacritics(word).casefold()
    return re.sub(r"[-—\s]", "", stripped)


def _contains_arabic(text: str) -> bool:
    for ch in text:
        codepoint = ord(ch)
        if any(start <= codepoint <= end for start, end in ARABIC_RANGES):
            return True
    return False


def _diacritic_richness(text: str) -> float:
    if not text:
        return 0.0

    score = 0
    for ch in text:
        if unicodedata.combining(ch):
            score += 1
            continue

        name = unicodedata.name(ch, "")
        if ord(ch) > 127 and ch.isalpha():
            score += 1
        elif "ARABIC" in name:
            score += 2
    return score / max(len(text), 1)


def _normalize_angle_degrees(angle: float) -> float:
    normalized = float(angle) % 360.0
    return normalized if normalized >= 0 else normalized + 360.0


def _signed_angle_delta(angle: float, reference: float) -> float:
    return ((float(angle) - float(reference) + 180.0) % 360.0) - 180.0


def _orientation_class(angle: float, skew_threshold: float = 1.0) -> tuple[str, int, float]:
    normalized = _normalize_angle_degrees(angle)
    right_angles = (0, 90, 180, 270)
    nearest = min(right_angles, key=lambda item: abs(_signed_angle_delta(normalized, item)))
    skew = _signed_angle_delta(normalized, nearest)

    labels = {
        0: "upright",
        90: "rotated_90_cw",
        180: "upside_down",
        270: "rotated_90_ccw",
    }
    label = labels.get(nearest, "unknown")
    if abs(skew) >= float(skew_threshold):
        label = f"{label}_skewed"
    return label, nearest, skew


def _bbox_from_polygon(points: Any) -> tuple[float, float, float, float] | None:
    if not points:
        return None
    try:
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
    except Exception:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _geometry_to_bbox(geometry: Any, width: float, height: float) -> tuple[float, float, float, float] | None:
    if not geometry:
        return None
    try:
        if len(geometry) == 2 and len(geometry[0]) == 2 and len(geometry[1]) == 2:
            (x0, y0), (x1, y1) = geometry
            return (float(x0) * width, float(y0) * height, float(x1) * width, float(y1) * height)

        xs = [float(point[0]) * width for point in geometry]
        ys = [float(point[1]) * height for point in geometry]
        return (min(xs), min(ys), max(xs), max(ys))
    except Exception:
        return None


@dataclass
class OCRLine:
    text: str
    bbox: tuple[float, float, float, float] | None = None
    confidence: float = 0.0
    source: str = ""


@dataclass
class OCRCandidate:
    engine: str
    text: str
    confidence: float = 0.0
    variant: str = "original"
    meta: dict[str, Any] = field(default_factory=dict)
    lines: list[OCRLine] = field(default_factory=list)


class PageLayoutAnalyzer:
    """Reconstruct reading order for pages that look multi-column."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = {**DEFAULT_LAYOUT, **(config or {})}

    def reorder(
        self,
        lines: list[OCRLine],
        page_size: tuple[int, int],
        return_meta: bool = False,
    ) -> list[OCRLine] | tuple[list[OCRLine], dict[str, Any]]:
        visible_lines = [line for line in lines if line.text.strip()]
        if not visible_lines:
            ordered_empty: list[OCRLine] = []
            layout_meta = self._layout_meta(0)
            return (ordered_empty, layout_meta) if return_meta else ordered_empty

        boxed_lines = [line for line in visible_lines if line.bbox is not None]
        if len(boxed_lines) < max(2, int(self.config["min_lines_for_columns"])):
            ordered_single = sorted(visible_lines, key=self._single_column_key)
            layout_meta = self._layout_meta(1)
            return (ordered_single, layout_meta) if return_meta else ordered_single

        columns = self._split_columns(boxed_lines, page_size)
        if len(columns) <= 1:
            ordered_single = sorted(visible_lines, key=self._single_column_key)
            layout_meta = self._layout_meta(1)
            return (ordered_single, layout_meta) if return_meta else ordered_single

        ordered: list[OCRLine] = []
        for column in columns:
            ordered.extend(sorted(column, key=self._single_column_key))

        seen = {id(line) for line in ordered}
        remainder = [line for line in visible_lines if id(line) not in seen]
        ordered.extend(sorted(remainder, key=self._single_column_key))
        layout_meta = self._layout_meta(len(columns))
        return (ordered, layout_meta) if return_meta else ordered

    @staticmethod
    def _layout_meta(column_count: int) -> dict[str, Any]:
        normalized_count = max(int(column_count), 0)
        has_columns = normalized_count > 1
        if normalized_count <= 0:
            layout_type = "unknown"
        elif has_columns:
            layout_type = "multi_column"
        else:
            layout_type = "single_column"

        return {
            "detected_layout_type": layout_type,
            "detected_column_count": normalized_count,
            "detected_has_columns": has_columns,
        }

    @staticmethod
    def _single_column_key(line: OCRLine) -> tuple[float, float]:
        if line.bbox is None:
            return (float("inf"), float("inf"))
        x0, y0, _, _ = line.bbox
        return (y0, x0)

    def _split_columns(self, lines: list[OCRLine], page_size: tuple[int, int]) -> list[list[OCRLine]]:
        if not self.config.get("detect_multi_column", True):
            return [lines]

        page_width = max(page_size[0], 1)
        x_centers = sorted(((line.bbox[0] + line.bbox[2]) / 2.0) for line in lines if line.bbox)
        if len(x_centers) < 4:
            return [lines]

        widest_gap = 0.0
        split_index = -1
        for index in range(len(x_centers) - 1):
            gap = x_centers[index + 1] - x_centers[index]
            if gap > widest_gap:
                widest_gap = gap
                split_index = index

        min_gap = max(page_width * float(self.config["column_gap_ratio"]), 24.0)
        if widest_gap < min_gap or split_index < 0:
            return [lines]

        threshold = (x_centers[split_index] + x_centers[split_index + 1]) / 2.0
        left = [line for line in lines if line.bbox and ((line.bbox[0] + line.bbox[2]) / 2.0) <= threshold]
        right = [line for line in lines if line.bbox and ((line.bbox[0] + line.bbox[2]) / 2.0) > threshold]
        if len(left) < 2 or len(right) < 2:
            return [lines]

        left_x1 = max(line.bbox[2] for line in left if line.bbox)
        right_x0 = min(line.bbox[0] for line in right if line.bbox)
        if right_x0 <= left_x1:
            overlap = (left_x1 - right_x0) / page_width
            if overlap > float(self.config["column_overlap_ratio"]):
                return [lines]

        return sorted([left, right], key=lambda column: min(line.bbox[0] for line in column if line.bbox))


class PageImageVariants:
    """Lazily converted page variants used by OCR backends."""

    def __init__(
        self,
        base_image: Any,
        preprocessing_config: dict[str, Any] | None = None,
        rotation_angle: float = 0.0,
    ):
        if not PIL_AVAILABLE:
            raise RuntimeError("Pillow is required for the ensemble preprocessor")

        self.preprocessing_config = {**DEFAULT_PREPROCESSING, **(preprocessing_config or {})}
        self.rotation_angle = float(rotation_angle)
        self.base_image = base_image.convert("RGB")
        self.page_size = self.base_image.size
        self._pil_variants = self._build_variants(self.base_image)
        self._numpy_cache: dict[str, Any] = {}

    @classmethod
    def from_pixmap(
        cls,
        pixmap: fitz.Pixmap,
        preprocessing_config: dict[str, Any] | None = None,
    ) -> "PageImageVariants":
        if not PIL_AVAILABLE:
            raise RuntimeError("Pillow is required for the ensemble preprocessor")

        if pixmap.alpha:
            pixmap = fitz.Pixmap(fitz.csRGB, pixmap)
        image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
        return cls(image, preprocessing_config=preprocessing_config)

    def rotated(self, angle: float) -> "PageImageVariants":
        normalized_angle = float(angle) % 360.0
        if abs(normalized_angle) < 1e-6:
            return self

        rotated_image = self.base_image.rotate(normalized_angle, expand=True, fillcolor=(255, 255, 255))
        return PageImageVariants(
            rotated_image,
            preprocessing_config=self.preprocessing_config,
            rotation_angle=_normalize_angle_degrees(self.rotation_angle + normalized_angle),
        )

    def estimate_skew_angle(self, variant_name: str = "adaptive") -> float | None:
        if not self.preprocessing_config.get("deskew", False):
            return None
        if not (CV2_AVAILABLE and NUMPY_AVAILABLE):
            return None

        try:
            candidate = self.get_pil(variant_name) if variant_name in self._pil_variants else self.base_image
            gray_array = np.array(candidate.convert("L"))
            if gray_array.size == 0:
                return None

            _, thresholded = cv2.threshold(gray_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            points = np.column_stack(np.where(thresholded > 0))
            if points.shape[0] < 250:
                return None

            rect = cv2.minAreaRect(points.astype(np.float32))
            raw_angle = float(rect[-1])
            if raw_angle < -45.0:
                skew = -(90.0 + raw_angle)
            else:
                skew = -raw_angle

            if not np.isfinite(skew):
                return None

            min_abs = float(self.preprocessing_config.get("deskew_min_abs_degrees", 0.2))
            max_abs = float(self.preprocessing_config.get("deskew_max_abs_degrees", 45.0))
            if abs(skew) < min_abs:
                return 0.0
            if abs(skew) > max_abs:
                return None
            return float(skew)
        except Exception:
            return None

    def get_pil(self, name: str) -> Any:
        return self._pil_variants[name]

    def get_numpy(self, name: str) -> Any:
        if not NUMPY_AVAILABLE:
            raise RuntimeError("numpy is required for this OCR backend")
        if name not in self._numpy_cache:
            self._numpy_cache[name] = np.array(self._pil_variants[name])
        return self._numpy_cache[name]

    def variant_names(self) -> tuple[str, ...]:
        return tuple(self._pil_variants.keys())

    def _build_variants(self, base_rgb: Any) -> dict[str, Any]:
        gray = ImageOps.grayscale(base_rgb)
        normalized = self._background_normalize(gray)
        denoised = self._denoise(normalized)
        autocontrast = ImageOps.autocontrast(denoised)
        boosted = ImageEnhance.Contrast(autocontrast).enhance(float(self.preprocessing_config["contrast_factor"]))
        sharpened = boosted.filter(
            ImageFilter.UnsharpMask(
                radius=float(self.preprocessing_config["sharpen_radius"]),
                percent=int(self.preprocessing_config["sharpen_percent"]),
                threshold=3,
            )
        )

        preserve_diacritics = bool(self.preprocessing_config.get("preserve_diacritics", False))
        binary_source = autocontrast if preserve_diacritics else sharpened
        binary = self._binary_threshold(binary_source, preserve_diacritics=preserve_diacritics)
        adaptive = self._adaptive_threshold(sharpened, preserve_diacritics=preserve_diacritics)
        if self.preprocessing_config.get("speckle_removal", False):
            adaptive = self._remove_speckles(adaptive)
        morphology_input = binary if preserve_diacritics else adaptive
        morphology = self._morphology(morphology_input, preserve_diacritics=preserve_diacritics)

        return {
            "original": base_rgb,
            "grayscale": gray.convert("RGB"),
            "background_normalized": normalized.convert("RGB"),
            "denoise": denoised.convert("RGB"),
            "autocontrast": autocontrast.convert("RGB"),
            "contrast": boosted.convert("RGB"),
            "sharpen": sharpened.convert("RGB"),
            "binary": binary.convert("RGB"),
            "adaptive": adaptive.convert("RGB"),
            "morphology": morphology.convert("RGB"),
        }

    def _background_normalize(self, gray: Any) -> Any:
        if not self.preprocessing_config.get("enable_background_normalization", False):
            return gray

        if CV2_AVAILABLE and NUMPY_AVAILABLE:
            gray_array = np.array(gray)
            background = cv2.GaussianBlur(gray_array, (0, 0), sigmaX=25, sigmaY=25)
            normalized = cv2.divide(gray_array, background, scale=255)
            return Image.fromarray(normalized)

        return ImageOps.autocontrast(gray)

    def _denoise(self, gray: Any) -> Any:
        if not self.preprocessing_config.get("enable_denoise", True):
            return gray

        if CV2_AVAILABLE and NUMPY_AVAILABLE:
            gray_array = np.array(gray)
            strength = float(self.preprocessing_config.get("denoise_strength", 12))
            denoised = cv2.fastNlMeansDenoising(gray_array, None, strength, 7, 21)
            return Image.fromarray(denoised)
        return gray.filter(ImageFilter.MedianFilter(size=3))

    def _binary_threshold(self, gray: Any, preserve_diacritics: bool = False) -> Any:
        if preserve_diacritics and self.preprocessing_config.get("avoid_aggressive_binarization", False):
            return ImageOps.autocontrast(gray)

        if CV2_AVAILABLE and NUMPY_AVAILABLE:
            gray_array = np.array(gray)
            _, thresholded = cv2.threshold(gray_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return Image.fromarray(thresholded)

        if NUMPY_AVAILABLE:
            threshold = int(np.array(gray).mean())
        else:
            threshold = 160
        return gray.point(lambda px: 255 if px >= threshold else 0)

    def _adaptive_threshold(self, gray: Any, preserve_diacritics: bool = False) -> Any:
        if preserve_diacritics and self.preprocessing_config.get("avoid_aggressive_binarization", False):
            return ImageOps.autocontrast(gray)

        if self.preprocessing_config.get("enable_adaptive_threshold", True) and CV2_AVAILABLE and NUMPY_AVAILABLE:
            gray_array = np.array(gray)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray_array)
            block_size = int(self.preprocessing_config.get("adaptive_block_size", 35))
            if block_size < 3:
                block_size = 3
            if block_size % 2 == 0:
                block_size += 1
            adaptive_c = float(self.preprocessing_config.get("adaptive_C", 11))
            adaptive = cv2.adaptiveThreshold(
                clahe,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,
                adaptive_c,
            )
            return Image.fromarray(adaptive)

        return ImageOps.autocontrast(self._binary_threshold(gray))

    def _remove_speckles(self, binary: Any) -> Any:
        if CV2_AVAILABLE and NUMPY_AVAILABLE:
            binary_array = np.array(binary)
            kernel = np.ones((2, 2), np.uint8)
            opened = cv2.morphologyEx(binary_array, cv2.MORPH_OPEN, kernel)
            return Image.fromarray(opened)
        return binary.filter(ImageFilter.MedianFilter(size=3))

    def _morphology(self, binary: Any, preserve_diacritics: bool = False) -> Any:
        if preserve_diacritics or self.preprocessing_config.get("avoid_aggressive_binarization", False):
            return binary

        if self.preprocessing_config.get("enable_morphology", True) and CV2_AVAILABLE and NUMPY_AVAILABLE:
            binary_array = np.array(binary)
            kernel_size = int(self.preprocessing_config.get("morphology_kernel_size", 2))
            kernel_size = max(kernel_size, 1)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            morphed = cv2.morphologyEx(binary_array, cv2.MORPH_CLOSE, kernel)
            return Image.fromarray(morphed)

        return binary.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))


class OCRBackendBase:
    name = "base"
    preferred_variants: tuple[str, ...] = ("adaptive", "morphology", "contrast", "sharpen", "original")

    def __init__(self, config: dict[str, Any], layout_config: dict[str, Any] | None = None):
        self.config = config
        self._engine = None
        self._failed_reason: str | None = None
        self._compat_shim_applied: bool = False
        self.layout_analyzer = PageLayoutAnalyzer(layout_config)

    def infer(self, variants: PageImageVariants) -> OCRCandidate | None:
        if self._failed_reason:
            return None

        try:
            self._ensure_loaded()
        except Exception as exc:
            self._failed_reason = str(exc)
            logger.warning("Failed to initialize %s backend: %s", self.name, exc)
            return None

        best: OCRCandidate | None = None
        profile_order = variants.preprocessing_config.get("profile_variant_order")
        variant_order = (
            tuple(profile_order)
            if isinstance(profile_order, (list, tuple)) and profile_order
            else self.preferred_variants
        )
        for variant in variant_order:
            if variant not in variants.variant_names():
                continue
            try:
                candidate = self._infer_variant(variants, variant)
            except Exception as exc:
                logger.debug("%s failed on %s variant: %s", self.name, variant, exc)
                continue

            if candidate and candidate.text.strip():
                candidate = self._finalize_candidate(candidate, variants)
                if best is None or self._candidate_quality(candidate) > self._candidate_quality(best):
                    best = candidate
        return best

    def _candidate_quality(self, candidate: OCRCandidate) -> float:
        return len(candidate.text.strip()) + (candidate.confidence * 100.0) + (_diacritic_richness(candidate.text) * 50.0)

    def _finalize_candidate(self, candidate: OCRCandidate, variants: PageImageVariants) -> OCRCandidate:
        layout_meta = self.layout_analyzer._layout_meta(0)
        ordered_lines: list[OCRLine] = []
        if candidate.lines:
            ordered_lines, layout_meta = self.layout_analyzer.reorder(
                candidate.lines,
                variants.page_size,
                return_meta=True,
            )
        if ordered_lines:
            candidate.lines = ordered_lines
            candidate.text = _normalize_whitespace("\n".join(line.text for line in ordered_lines))
        else:
            candidate.text = _normalize_whitespace(candidate.text)

        candidate.meta.setdefault("rotation_angle", variants.rotation_angle)
        candidate.meta.setdefault("page_size", variants.page_size)
        candidate.meta.setdefault("detected_layout_type", layout_meta.get("detected_layout_type", "unknown"))
        candidate.meta.setdefault("detected_column_count", layout_meta.get("detected_column_count", 0))
        candidate.meta.setdefault("detected_has_columns", layout_meta.get("detected_has_columns", False))
        candidate.meta.setdefault("line_count", len(candidate.lines) or len(TextEnsembleFuser._lines(candidate.text)))
        return candidate

    def _ensure_loaded(self) -> None:
        raise NotImplementedError

    def _infer_variant(self, variants: PageImageVariants, variant: str) -> OCRCandidate | None:
        raise NotImplementedError


class PaddleBackend(OCRBackendBase):
    name = "paddle"
    _is_v3: bool = False

    def _ensure_loaded(self) -> None:
        if self._engine is not None:
            return
        from paddleocr import PaddleOCR

        language_hint = self.config.get("paddle_lang") or self.config.get("lang") or "en"

        try:
            import paddleocr
            self._is_v3 = int(paddleocr.__version__.split(".")[0]) >= 3
        except Exception:
            self._is_v3 = False

        if self._is_v3:
            self._engine = PaddleOCR(lang=language_hint)
        else:
            self._engine = PaddleOCR(lang=language_hint, use_textline_orientation=True)

    def _infer_variant(self, variants: PageImageVariants, variant: str) -> OCRCandidate | None:
        image = variants.get_numpy(variant)

        if self._is_v3 and hasattr(self._engine, "predict"):
            return self._infer_v3(image, variant)
        return self._infer_v2(image, variant)

    def _infer_v2(self, image, variant: str) -> OCRCandidate | None:
        """PaddleOCR v2.x: .ocr() returns [[[polygon, (text, conf)], ...]]."""
        result = self._engine.ocr(image, cls=True)

        lines: list[OCRLine] = []
        confidences: list[float] = []
        if result and result[0]:
            for item in result[0]:
                if item and len(item) >= 2:
                    text = str(item[1][0]).strip()
                    if not text:
                        continue

                    confidence = 0.0
                    try:
                        confidence = float(item[1][1])
                        confidences.append(confidence)
                    except Exception:
                        pass

                    lines.append(
                        OCRLine(
                            text=text,
                            bbox=_bbox_from_polygon(item[0]),
                            confidence=confidence,
                            source=self.name,
                        )
                    )

        text = _normalize_whitespace("\n".join(line.text for line in lines))
        if not text:
            return None

        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return OCRCandidate(self.name, text, confidence=confidence, variant=variant, lines=lines)

    def _infer_v3(self, image, variant: str) -> OCRCandidate | None:
        """PaddleOCR v3.x: .predict() returns result objects with rec_texts/rec_scores/dt_polys."""
        results = self._engine.predict(image)

        lines: list[OCRLine] = []
        confidences: list[float] = []
        for result in results:
            rec_texts = getattr(result, "rec_texts", None)
            if rec_texts is None and isinstance(result, dict):
                rec_texts = result.get("rec_texts", [])
            rec_texts = rec_texts or []

            rec_scores = getattr(result, "rec_scores", None)
            if rec_scores is None and isinstance(result, dict):
                rec_scores = result.get("rec_scores", [])
            rec_scores = rec_scores or []

            dt_polys = getattr(result, "dt_polys", None)
            if dt_polys is None and isinstance(result, dict):
                dt_polys = result.get("dt_polys", [])
            dt_polys = dt_polys or []

            for i, txt in enumerate(rec_texts):
                text = str(txt).strip()
                if not text:
                    continue

                confidence = 0.0
                if i < len(rec_scores):
                    try:
                        confidence = float(rec_scores[i])
                        confidences.append(confidence)
                    except Exception:
                        pass

                bbox = None
                if i < len(dt_polys):
                    bbox = _bbox_from_polygon(dt_polys[i])

                lines.append(
                    OCRLine(text=text, bbox=bbox, confidence=confidence, source=self.name)
                )

        text = _normalize_whitespace("\n".join(line.text for line in lines))
        if not text:
            return None

        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return OCRCandidate(self.name, text, confidence=confidence, variant=variant, lines=lines)


class DocTRBackend(OCRBackendBase):
    name = "doctr"

    @staticmethod
    def _apply_torch_dynamo_compat_shim() -> bool:
        """Provide minimal torch._dynamo surface for older torch builds.

        Some docTR releases reference torch._dynamo members even when running in
        inference-only mode. This shim keeps runtime initialization stable when
        those members are absent.
        """
        try:
            import types
            import torch
        except Exception:
            return False

        shim_applied = False
        dynamo_obj = getattr(torch, "_dynamo", None)
        if dynamo_obj is None:
            dynamo_obj = types.SimpleNamespace()
            shim_applied = True

        if not hasattr(dynamo_obj, "is_compiling"):
            dynamo_obj.is_compiling = lambda: False
            shim_applied = True

        if not hasattr(dynamo_obj, "disable"):
            dynamo_obj.disable = lambda fn=None, recursive=True: (fn if fn else (lambda f: f))
            shim_applied = True

        eval_frame = getattr(dynamo_obj, "eval_frame", None)
        if eval_frame is None:
            eval_frame = types.SimpleNamespace()
            shim_applied = True

        if not hasattr(eval_frame, "OptimizedModule"):
            class OptimizedModule(torch.nn.Module):
                pass

            eval_frame.OptimizedModule = OptimizedModule
            shim_applied = True

        dynamo_obj.eval_frame = eval_frame
        torch._dynamo = dynamo_obj
        return shim_applied

    def _ensure_loaded(self) -> None:
        if self._engine is not None:
            return
        self._compat_shim_applied = self._apply_torch_dynamo_compat_shim()
        from doctr.models import ocr_predictor

        self._engine = ocr_predictor(pretrained=True)

    def _infer_variant(self, variants: PageImageVariants, variant: str) -> OCRCandidate | None:
        image = variants.get_numpy(variant)
        result = self._engine([image])
        page = result.pages[0]
        page_width, page_height = variants.page_size

        lines: list[OCRLine] = []
        confidences: list[float] = []
        for block in page.blocks:
            for line in block.lines:
                words = []
                line_word_boxes = []
                for word in line.words:
                    value = str(getattr(word, "value", "")).strip()
                    if value:
                        words.append(value)
                        word_bbox = _geometry_to_bbox(getattr(word, "geometry", None), page_width, page_height)
                        if word_bbox:
                            line_word_boxes.append(word_bbox)
                        conf = getattr(word, "confidence", None)
                        if isinstance(conf, (int, float)):
                            confidences.append(float(conf))

                if not words:
                    continue

                line_bbox = _geometry_to_bbox(getattr(line, "geometry", None), page_width, page_height)
                if line_bbox is None and line_word_boxes:
                    xs0 = [box[0] for box in line_word_boxes]
                    ys0 = [box[1] for box in line_word_boxes]
                    xs1 = [box[2] for box in line_word_boxes]
                    ys1 = [box[3] for box in line_word_boxes]
                    line_bbox = (min(xs0), min(ys0), max(xs1), max(ys1))

                line_text = " ".join(words)
                line_conf = sum(confidences[-len(words) :]) / len(words) if words else 0.0
                lines.append(OCRLine(text=line_text, bbox=line_bbox, confidence=line_conf, source=self.name))

        text = _normalize_whitespace("\n".join(line.text for line in lines))
        if not text:
            return None

        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return OCRCandidate(self.name, text, confidence=confidence, variant=variant, lines=lines)


class MMOCRBackend(OCRBackendBase):
    name = "mmocr"

    def _ensure_loaded(self) -> None:
        if self._engine is not None:
            return
        from mmocr.apis import MMOCRInferencer

        self._engine = MMOCRInferencer(det="DBNet", rec="CRNN")

    def _infer_variant(self, variants: PageImageVariants, variant: str) -> OCRCandidate | None:
        image = variants.get_numpy(variant)
        result = self._engine(image, return_vis=False)
        predictions = result.get("predictions") or []
        prediction = predictions[0] if predictions else {}

        texts = prediction.get("rec_texts") or prediction.get("texts") or []
        scores = prediction.get("rec_scores") or prediction.get("scores") or []
        polygons = prediction.get("det_polygons") or prediction.get("polygons") or prediction.get("bboxes") or []

        if not texts and isinstance(prediction, list):
            texts = [str(item.get("text", "")).strip() for item in prediction if item.get("text")]
            scores = [item.get("score") for item in prediction if item.get("text")]
            polygons = [item.get("box") or item.get("bbox") or item.get("polygon") for item in prediction if item.get("text")]

        lines: list[OCRLine] = []
        numeric_scores: list[float] = []
        for index, value in enumerate(texts):
            text = str(value).strip()
            if not text:
                continue

            confidence = 0.0
            score = scores[index] if index < len(scores) else None
            if isinstance(score, (int, float)):
                confidence = float(score)
                numeric_scores.append(confidence)

            polygon = polygons[index] if index < len(polygons) else None
            lines.append(
                OCRLine(
                    text=text,
                    bbox=_bbox_from_polygon(polygon),
                    confidence=confidence,
                    source=self.name,
                )
            )

        text = _normalize_whitespace("\n".join(line.text for line in lines))
        if not text:
            return None

        confidence = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0.0
        return OCRCandidate(self.name, text, confidence=confidence, variant=variant, lines=lines)


class KrakenBackend(OCRBackendBase):
    name = "kraken"
    preferred_variants = ("morphology", "adaptive", "binary", "sharpen", "contrast")

    def _ensure_loaded(self) -> None:
        if self._engine is not None:
            return

        model_path = self.config.get("kraken_model_path") or os.getenv("KRAKEN_MODEL_PATH")
        if not model_path:
            raise RuntimeError("KRAKEN_MODEL_PATH is not configured")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Kraken model not found: {model_path}")

        from kraken.lib import models

        self._engine = models.load_any(model_path)

    def _infer_variant(self, variants: PageImageVariants, variant: str) -> OCRCandidate | None:
        from kraken import binarization, pageseg, rpred

        image = variants.get_pil(variant).convert("L")
        binary = binarization.nlbin(image)
        segments = pageseg.segment(binary)
        predictions = rpred.rpred(self._engine, binary, segments)

        lines: list[OCRLine] = []
        confidences: list[float] = []
        for record in predictions:
            text = str(getattr(record, "prediction", "")).strip()
            if not text:
                continue

            confidence = 0.0
            conf = getattr(record, "confidence", None)
            if isinstance(conf, (int, float)):
                confidence = float(conf)
                confidences.append(confidence)

            bbox = getattr(record, "bbox", None)
            lines.append(OCRLine(text=text, bbox=bbox, confidence=confidence, source=self.name))

        text = _normalize_whitespace("\n".join(line.text for line in lines))
        if not text:
            return None

        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return OCRCandidate(self.name, text, confidence=confidence, variant=variant, lines=lines)

class CuReDBackend(OCRBackendBase):
    """Kraken-based backend using the CuReD cuneiform recognition model."""
 
    name = "cured"
    preferred_variants = ("adaptive", "morphology", "binary", "contrast", "sharpen")
 
    _SEARCH_PATHS = (
        "models/CuReD.mlmodel",
        "models/cured.mlmodel",
        "models/latest.mlmodel",
    )
 
    def _ensure_loaded(self) -> None:
        if self._engine is not None:
            return
 
        model_path = self.config.get("cured_model_path") or os.getenv("CURED_MODEL_PATH")
        if not model_path:
            for candidate in self._SEARCH_PATHS:
                if Path(candidate).exists():
                    model_path = candidate
                    break
        if not model_path:
            raise RuntimeError(
                "CuReD model not found. Set cured_model_path in config, "
                "CURED_MODEL_PATH env var, or place the .mlmodel in models/CuReD.mlmodel"
            )
        if not Path(model_path).exists():
            raise FileNotFoundError(f"CuReD model not found: {model_path}")
 
        from kraken.lib import models
 
        self._engine = models.load_any(model_path)
 
    def _infer_variant(self, variants: PageImageVariants, variant: str) -> OCRCandidate | None:
        import warnings
 
        from kraken import binarization, pageseg, rpred
 
        image = variants.get_pil(variant).convert("L")
        binary = binarization.nlbin(image)
        segments = pageseg.segment(binary)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Using legacy polygon extractor")
            predictions = list(rpred.rpred(self._engine, binary, segments))
 
        lines: list[OCRLine] = []
        confidences: list[float] = []
        for record in predictions:
            text = str(getattr(record, "prediction", "")).strip()
            if not text:
                continue
            text = _postprocess_cured_text(text)

            confidence = 0.0
            conf = getattr(record, "confidence", None)
            if isinstance(conf, (int, float)):
                confidence = float(conf)
                confidences.append(confidence)

            bbox = getattr(record, "bbox", None)
            lines.append(OCRLine(text=text, bbox=bbox, confidence=confidence, source=self.name))

        text = _normalize_whitespace("\n".join(line.text for line in lines))
        if not text:
            return None

        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return OCRCandidate(self.name, text, confidence=confidence, variant=variant, lines=lines)

class TextEnsembleFuser:
    """Consensus-based text fusion that tries hard not to lose diacritics."""

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or {}

    def fuse(self, candidates: Iterable[OCRCandidate]) -> OCRCandidate:
        viable = [candidate for candidate in candidates if candidate.text.strip()]
        if not viable:
            raise ValueError("No OCR candidates available for fusion")
        if len(viable) == 1:
            return viable[0]

        # CuReD's line count reflects its hallucinated segmentation and can't
        # be trusted as page structure. When engines disagree on line counts
        # we pick the best non-CuReD candidate as the whole-text fallback;
        # otherwise CuReD wins _whole_score via weight + hallucinated diacritic
        # richness and its entire garbage output is returned verbatim.
        structural_pool = [c for c in viable if c.engine != "cured"] or viable
        best_whole = max(structural_pool, key=lambda candidate: self._whole_score(candidate, viable))
        line_counts = {len(self._lines(candidate.text)) for candidate in structural_pool}
        if len(line_counts) != 1:
            return OCRCandidate(
                engine="ensemble",
                text=best_whole.text,
                confidence=best_whole.confidence,
                variant=best_whole.variant,
                meta={
                    "method": "whole_text_consensus",
                    "winner": best_whole.engine,
                    "rotation_angle": best_whole.meta.get("rotation_angle", 0),
                },
                lines=best_whole.lines,
            )

        line_total = line_counts.pop()
        # Only include candidates whose own line count matches line_total;
        # otherwise `candidate_lines[line_index]` pairs unrelated lines across
        # engines (e.g. CuReD's hallucinated line 39 vs Paddle's line 39).
        line_fusion_pool = [c for c in viable if len(self._lines(c.text)) == line_total]
        per_engine_counts = ", ".join(
            f"{c.engine}={len(self._lines(c.text))}" for c in viable
        )
        pool_engines = ", ".join(c.engine for c in line_fusion_pool) or "<empty>"
        excluded = [c.engine for c in viable if c not in line_fusion_pool]
        logger.debug(
            "fusion line counts (target=%d): %s | pool=%s | excluded=%s",
            line_total, per_engine_counts, pool_engines, excluded or "none",
        )
        fused_lines: list[str] = []
        winner_counts: dict[str, int] = {}
        for line_index in range(line_total):
            line_options = []
            for candidate in line_fusion_pool:
                candidate_lines = self._lines(candidate.text)
                line_options.append((candidate, candidate_lines[line_index]))

            line_text, line_winners = self._fuse_line_words(line_options)
            fused_lines.append(line_text)
            line_tally: dict[str, int] = {}
            for engine in line_winners:
                winner_counts[engine] = winner_counts.get(engine, 0) + 1
                line_tally[engine] = line_tally.get(engine, 0) + 1
            if line_tally:
                tally_str = ", ".join(f"{eng}={count}" for eng, count in sorted(line_tally.items(), key=lambda kv: -kv[1]))
                logger.debug("fusion line %d winners: %s | %s", line_index, tally_str, line_text)

        if winner_counts:
            summary = ", ".join(f"{eng}={count}" for eng, count in sorted(winner_counts.items(), key=lambda kv: -kv[1]))
            logger.debug("fusion page winners (word-level): %s", summary)

        fused_text = _normalize_whitespace("\n".join(fused_lines))
        confidence = sum(candidate.confidence for candidate in viable) / len(viable)
        return OCRCandidate(
            engine="ensemble",
            text=fused_text,
            confidence=confidence,
            variant="fused",
            meta={
                "method": "line_consensus",
                "sources": [candidate.engine for candidate in viable],
                "rotation_angle": best_whole.meta.get("rotation_angle", 0),
                "word_winners": winner_counts,
            },
        )

    @staticmethod
    def _lines(text: str) -> list[str]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines or [text.strip()]

    def _whole_score(self, candidate: OCRCandidate, population: list[OCRCandidate]) -> float:
        weight = self.weights.get(candidate.engine, 1.0)
        similarities = []
        candidate_key = _comparison_key(candidate.text)
        non_cured_transliteration = False
        for other in population:
            if other.engine == candidate.engine:
                continue
            similarities.append(SequenceMatcher(None, candidate_key, _comparison_key(other.text)).ratio())
            if other.engine != "cured" and _is_akkadian_transliteration(other.text):
                non_cured_transliteration = True

        agreement = sum(similarities) / len(similarities) if similarities else 0.0
        richness = _diacritic_richness(candidate.text)
        completeness = min(len(candidate.text.strip()) / 300.0, 1.0)
        confidence = max(candidate.confidence, 0.0)
        line_bonus = min(candidate.meta.get("line_count", 0) / 12.0, 1.0) * 0.12
        transliteration_bonus = 0.20 if (candidate.engine == "cured" and non_cured_transliteration) else 0.0
        return (weight * 0.35) + agreement + (richness * 0.4) + (completeness * 0.1) + (confidence * 0.15) + line_bonus + transliteration_bonus

    @staticmethod
    def _tokenize(line: str) -> list[str]:
        return line.split()

    @staticmethod
    def _align_words(skeleton: list[str], other: list[str]) -> list[str | None]:
        """Align `other` onto `skeleton` positions via word-level diff.

        Returns a list the same length as `skeleton`; each slot holds the word
        from `other` that aligns to that skeleton position, or None if no
        word aligns (e.g. the other candidate dropped that word).
        """
        aligned: list[str | None] = [None] * len(skeleton)
        matcher = SequenceMatcher(a=[w.casefold() for w in skeleton],
                                  b=[w.casefold() for w in other], autojunk=False)
        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if op == "equal" or op == "replace":
                span = min(i2 - i1, j2 - j1)
                for k in range(span):
                    aligned[i1 + k] = other[j1 + k]
        return aligned

    def _fuse_line_words(self, line_options: list[tuple[OCRCandidate, str]]) -> tuple[str, list[str]]:
        """Word-level fusion across candidate lines (ROVER-style).

        Returns `(fused_line, winning_engine_per_word)`. Picks a skeleton
        candidate (highest-weight non-CuReD non-empty line), aligns every other
        candidate's words onto it, and for each word position chooses the best
        word using `_word_score`.
        """
        tokenized = [(cand, self._tokenize(line)) for cand, line in line_options]
        viable = [(c, w) for c, w in tokenized if w]
        if not viable:
            return "", []
        if len(viable) == 1:
            cand, words = viable[0]
            return " ".join(words), [cand.engine] * len(words)

        # Skeleton must be a general-purpose engine, not CuReD. CuReD's word
        # count reflects its hallucinated glyph segmentation on non-cuneiform
        # input, not the actual text structure. If we use CuReD as the skeleton,
        # every "extra" word it invented becomes a fused-output position where
        # only CuReD has a candidate, so CuReD wins by default.
        general_candidates = [cw for cw in viable if cw[0].engine != "cured"]
        pool = general_candidates or viable
        skeleton_cand, skeleton_words = max(
            pool, key=lambda cw: (self.weights.get(cw[0].engine, 1.0), len(cw[1]))
        )

        aligned_per_engine: list[tuple[OCRCandidate, list[str | None]]] = []
        for cand, words in viable:
            if cand.engine == skeleton_cand.engine:
                aligned_per_engine.append((cand, list(skeleton_words)))
            else:
                aligned_per_engine.append((cand, self._align_words(skeleton_words, words)))

        fused_words: list[str] = []
        word_winners: list[str] = []
        for idx in range(len(skeleton_words)):
            word_options: list[tuple[OCRCandidate, str]] = []
            for cand, aligned in aligned_per_engine:
                word = aligned[idx]
                if word:
                    word_options.append((cand, word))
            if not word_options:
                fused_words.append(skeleton_words[idx])
                word_winners.append(skeleton_cand.engine)
                continue
            # Tiebreak in favour of the skeleton engine so that when all
            # engines produced the same word, the skeleton (not whoever happens
            # to appear first in word_options) gets credited in the tally.
            best = max(
                word_options,
                key=lambda opt: (
                    self._word_score(opt[0], opt[1], word_options, line_options),
                    opt[0].engine == skeleton_cand.engine,
                ),
            )
            fused_words.append(best[1])
            word_winners.append(best[0].engine)

        return " ".join(fused_words), word_winners

    def _word_score(
        self,
        candidate: OCRCandidate,
        word: str,
        word_options: list[tuple[OCRCandidate, str]],
        line_options: list[tuple[OCRCandidate, str]],
    ) -> float:
        # Word-level agreement: case-insensitive match with other engines at this position.
        word_key = word.casefold()
        other_count = 0
        agree_count = 0
        for other_cand, other_word in word_options:
            if other_cand.engine == candidate.engine:
                continue
            other_count += 1
            if other_word.casefold() == word_key:
                agree_count += 1
        agreement = (agree_count / other_count) if other_count else 0.0

        # Transliteration evidence must come from another engine's aligned word
        # at THIS position (not line-level), otherwise CuReD wins every word on
        # any page that contains a transliteration somewhere.
        word_has_translit_evidence = any(
            other_cand.engine != "cured" and _word_looks_transliteration(other_word)
            for other_cand, other_word in word_options
            if other_cand.engine != candidate.engine
        )

        # Decisive case: CuReD's word looks Akkadian AND a non-CuReD engine
        # saw this *line* as transliteration-shaped. The line-level gate is
        # what separates "CuReD enriching a real Akkadian token" from "CuReD
        # hallucinating Akkadian-shaped garbage on plain prose" - CuReD is a
        # cuneiform model, so its output always looks transliterated even on
        # non-Akkadian input. _is_akkadian_transliteration triggers on cues
        # general engines can reproduce (multiple short-hyphen tokens,
        # LUGAL/DUMU markers, numbered-verse prefixes), so requiring one of
        # them to see the line gives us grounded evidence that the line is
        # really transliteration before we let CuReD override per word.
        if candidate.engine == "cured" and _word_looks_transliteration(word):
            line_is_transliteration = any(
                other_cand.engine != "cured" and _is_akkadian_transliteration(other_line)
                for other_cand, other_line in line_options
            )
            if line_is_transliteration:
                return 10.0 + _diacritic_richness(word)

        # Gate CuReD's weight boost AND its diacritic richness on per-word evidence.
        # Without this, CuReD's profile weight (1.8) and hallucinated diacritics
        # stack to overpower multi-engine agreement even on plain English words.
        profile_weight = self.weights.get(candidate.engine, 1.0)
        if candidate.engine == "cured" and not word_has_translit_evidence:
            effective_weight = 1.0
            richness = 0.0
        else:
            effective_weight = profile_weight
            richness = _diacritic_richness(word)

        arabic_bonus = 0.15 if _contains_arabic(word) else 0.0
        transliteration_bonus = 0.25 if (candidate.engine == "cured" and word_has_translit_evidence) else 0.0

        # Per-candidate confidence is an average over the whole line/page, so at
        # the word level it's just a constant engine bias - whichever engine
        # happens to have the highest overall confidence would win every tie.
        # Drop it; ties are resolved by the skeleton-preference tiebreaker in
        # `_fuse_line_words`, which makes log tallies meaningful.
        return (effective_weight * 0.35) + agreement + (richness * 0.45) + arabic_bonus + transliteration_bonus


class FortifiedOCREnsemble:
    """Multi-engine OCR ensemble used by the active page-text pipeline."""

    BACKEND_TYPES = {
        "paddle": PaddleBackend,
        "doctr": DocTRBackend,
        "mmocr": MMOCRBackend,
        "kraken": KrakenBackend,
        "cured": CuReDBackend
    }

    def __init__(self, profile_path: str | None = None, per_engine_timeout_s: float | None = None):
        self.profile = self._load_profile(profile_path)
        self.render_dpi = int(self.profile.get("rendering", {}).get("dpi", 300))
        engines_cfg = self.profile.get("engines", {})
        if "enabled" in engines_cfg:
            configured_engines = engines_cfg.get("enabled")
            self.enabled_engines = list(configured_engines) if isinstance(configured_engines, (list, tuple)) else []
        else:
            self.enabled_engines = ["paddle", "doctr", "mmocr", "kraken"]
        self.weights = self.profile.get("fusion", {}).get("weights", {})
        self.preprocessing_config = {**DEFAULT_PREPROCESSING, **self.profile.get("preprocessing", {})}
        configured_default = str(self.preprocessing_config.get("default_profile", PROFILE_UNKNOWN_SAFE_DEFAULT)).strip().lower()
        self.default_preprocessing_profile = (
            configured_default
            if configured_default in available_preprocessing_profiles()
            else PROFILE_UNKNOWN_SAFE_DEFAULT
        )
        self.layout_config = {**DEFAULT_LAYOUT, **self.profile.get("layout", {})}
        self.layout_detector = LayoutAnalyzer(self.profile.get("layout_detection", {}))
        configured_timeout = engines_cfg.get("per_engine_timeout_s")
        timeout_value = per_engine_timeout_s if per_engine_timeout_s is not None else configured_timeout
        self.per_engine_timeout_s = float(timeout_value) if timeout_value not in (None, "") else 0.0
        self.rotation_search = tuple(
            int(angle) % 360 for angle in self.preprocessing_config.get("rotation_search_degrees", [0, 90, 180, 270])
        )
        self.backends: list[OCRBackendBase] = [
            self.BACKEND_TYPES[name](engines_cfg, self.layout_config)
            for name in self.enabled_engines
            if name in self.BACKEND_TYPES
        ]
        self.backends_by_name: dict[str, OCRBackendBase] = {backend.name: backend for backend in self.backends}

        routing_cfg = self.profile.get("routing", {}) if isinstance(self.profile, dict) else {}
        if not isinstance(routing_cfg, dict):
            routing_cfg = {}
        routing_timeouts = routing_cfg.get("timeouts")
        if not isinstance(routing_timeouts, dict):
            routing_timeouts = {}
        if "per_engine_timeout_s" not in routing_timeouts and self.per_engine_timeout_s > 0:
            routing_timeouts["per_engine_timeout_s"] = float(self.per_engine_timeout_s)
        routing_cfg["timeouts"] = routing_timeouts
        self.strategy_selector = OCRStrategySelector(routing_cfg)

        self.fuser = TextEnsembleFuser(self.weights)
        self.engine_readiness = self._init_engine_readiness(self.enabled_engines)
        self.engine_runtime_state = self._init_engine_runtime_state(self.enabled_engines)
        self._probe_backend_readiness()

    @staticmethod
    def _load_profile(profile_path: str | None) -> dict[str, Any]:
        if not profile_path or not Path(profile_path).exists():
            return {}
        with Path(profile_path).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _init_engine_readiness(self, enabled_engines: list[str]) -> dict[str, dict[str, str]]:
        readiness: dict[str, dict[str, str]] = {}
        enabled_set = set(enabled_engines)
        for name in self.BACKEND_TYPES:
            if name in enabled_set:
                readiness[name] = {"status": ENGINE_STATUS_AVAILABLE_UNHEALTHY, "reason": "initialization_pending"}
            else:
                readiness[name] = {"status": ENGINE_STATUS_DISABLED_BY_CONFIG, "reason": "disabled_by_profile"}
        return readiness

    def _init_engine_runtime_state(self, enabled_engines: list[str]) -> dict[str, dict[str, Any]]:
        runtime_state: dict[str, dict[str, Any]] = {}
        for name in self.BACKEND_TYPES:
            runtime_state[name] = {
                "attempts": 0,
                "successes": 0,
                "failures": 0,
                "timeouts": 0,
                "consecutive_failures": 0,
                "consecutive_timeouts": 0,
                "avg_runtime_ms": 0.0,
                "avg_confidence": 0.0,
                "last_runtime_ms": 0.0,
                "last_confidence": 0.0,
            }
        for name in enabled_engines:
            runtime_state.setdefault(
                name,
                {
                    "attempts": 0,
                    "successes": 0,
                    "failures": 0,
                    "timeouts": 0,
                    "consecutive_failures": 0,
                    "consecutive_timeouts": 0,
                    "avg_runtime_ms": 0.0,
                    "avg_confidence": 0.0,
                    "last_runtime_ms": 0.0,
                    "last_confidence": 0.0,
                },
            )
        return runtime_state

    def _set_engine_readiness(self, name: str, status: str, reason: str = "") -> None:
        self.engine_readiness[name] = {"status": status, "reason": reason}

    def _record_engine_observation(
        self,
        engine: str,
        status: str,
        runtime_ms: float,
        confidence: float | None = None,
    ) -> None:
        entry = self.engine_runtime_state.setdefault(
            engine,
            {
                "attempts": 0,
                "successes": 0,
                "failures": 0,
                "timeouts": 0,
                "consecutive_failures": 0,
                "consecutive_timeouts": 0,
                "avg_runtime_ms": 0.0,
                "avg_confidence": 0.0,
                "last_runtime_ms": 0.0,
                "last_confidence": 0.0,
            },
        )

        entry["attempts"] = int(entry.get("attempts", 0)) + 1
        prev_attempts = max(int(entry.get("attempts", 1)) - 1, 0)
        prev_avg_runtime = float(entry.get("avg_runtime_ms", 0.0) or 0.0)
        runtime_ms = max(float(runtime_ms or 0.0), 0.0)
        entry["avg_runtime_ms"] = ((prev_avg_runtime * prev_attempts) + runtime_ms) / max(prev_attempts + 1, 1)
        entry["last_runtime_ms"] = runtime_ms

        if confidence is not None:
            prev_successes = max(int(entry.get("successes", 0)), 0)
            prev_avg_conf = float(entry.get("avg_confidence", 0.0) or 0.0)
            conf_val = max(float(confidence), 0.0)
            conf_count = prev_successes + 1 if status == ENGINE_STATUS_AVAILABLE else prev_successes
            if conf_count > 0 and status == ENGINE_STATUS_AVAILABLE:
                entry["avg_confidence"] = ((prev_avg_conf * prev_successes) + conf_val) / conf_count
                entry["last_confidence"] = conf_val

        if status == ENGINE_STATUS_AVAILABLE:
            entry["successes"] = int(entry.get("successes", 0)) + 1
            entry["consecutive_failures"] = 0
            entry["consecutive_timeouts"] = 0
        elif status == ENGINE_STATUS_TIMED_OUT:
            entry["timeouts"] = int(entry.get("timeouts", 0)) + 1
            entry["failures"] = int(entry.get("failures", 0)) + 1
            entry["consecutive_timeouts"] = int(entry.get("consecutive_timeouts", 0)) + 1
            entry["consecutive_failures"] = int(entry.get("consecutive_failures", 0)) + 1
        else:
            entry["failures"] = int(entry.get("failures", 0)) + 1
            entry["consecutive_failures"] = int(entry.get("consecutive_failures", 0)) + 1
            entry["consecutive_timeouts"] = 0

    def _probe_backend_readiness(self) -> None:
        for backend in self.backends:
            try:
                _, timed_out = _run_callable_with_timeout(
                    backend._ensure_loaded,
                    self.per_engine_timeout_s,
                )
                if timed_out:
                    reason = f"initialization timed out after {self.per_engine_timeout_s:.2f}s"
                    backend._failed_reason = reason
                    self._set_engine_readiness(backend.name, ENGINE_STATUS_TIMED_OUT, reason)
                    continue
            except Exception as exc:
                reason = str(exc)
                backend._failed_reason = reason
                self._set_engine_readiness(backend.name, ENGINE_STATUS_UNAVAILABLE_DEPENDENCY, reason)
                continue

            if getattr(backend, "_compat_shim_applied", False):
                self._set_engine_readiness(
                    backend.name,
                    ENGINE_STATUS_AVAILABLE_UNHEALTHY,
                    "loaded_with_runtime_compat_shim",
                )
            else:
                self._set_engine_readiness(backend.name, ENGINE_STATUS_AVAILABLE, "")

    def get_engine_readiness(self) -> dict[str, dict[str, str]]:
        return {name: dict(info) for name, info in self.engine_readiness.items()}

    def get_engine_performance_summary(self) -> dict[str, dict[str, Any]]:
        summary: dict[str, dict[str, Any]] = {}
        for engine, entry in self.engine_runtime_state.items():
            attempts = max(int(entry.get("attempts", 0)), 0)
            successes = max(int(entry.get("successes", 0)), 0)
            summary[engine] = {
                "attempts": attempts,
                "successes": successes,
                "failures": max(int(entry.get("failures", 0)), 0),
                "timeouts": max(int(entry.get("timeouts", 0)), 0),
                "success_rate": (successes / attempts) if attempts else 0.0,
                "consecutive_failures": max(int(entry.get("consecutive_failures", 0)), 0),
                "consecutive_timeouts": max(int(entry.get("consecutive_timeouts", 0)), 0),
                "avg_runtime_ms": float(entry.get("avg_runtime_ms", 0.0) or 0.0),
                "avg_confidence": float(entry.get("avg_confidence", 0.0) or 0.0),
                "last_runtime_ms": float(entry.get("last_runtime_ms", 0.0) or 0.0),
                "last_confidence": float(entry.get("last_confidence", 0.0) or 0.0),
            }
        return summary

    def select_strategy(
        self,
        diagnostics: dict[str, Any] | None,
        language_hint: str = "unknown",
        script_hint: str = "unknown",
        document_type: str = "unknown",
        requested_profile: str = PROFILE_AUTO,
        timeout_config: dict[str, Any] | None = None,
        quality_thresholds: dict[str, Any] | None = None,
        force_ocr: bool = True,
        prefer_text_layer: bool = False,
    ) -> OCRRoutingStrategy:
        return self.strategy_selector.select(
            diagnostics=diagnostics,
            engine_readiness=self.get_engine_readiness(),
            enabled_engines=list(self.enabled_engines),
            language_hint=language_hint,
            script_hint=script_hint,
            document_type=document_type,
            requested_profile=requested_profile,
            default_profile=self.default_preprocessing_profile,
            previous_engine_performance=self.get_engine_performance_summary(),
            timeout_config=timeout_config,
            quality_thresholds=quality_thresholds,
            force_ocr=force_ocr,
            prefer_text_layer=prefer_text_layer,
        )

    def has_usable_backend(self) -> bool:
        usable_states = {ENGINE_STATUS_AVAILABLE, ENGINE_STATUS_AVAILABLE_UNHEALTHY}
        return any(
            info.get("status") in usable_states
            for name, info in self.engine_readiness.items()
            if name in self.enabled_engines
        )

    def strict_readiness_ok(self) -> bool:
        for name in self.enabled_engines:
            if name not in self.BACKEND_TYPES:
                return False
            info = self.engine_readiness.get(name, {})
            if info.get("status") != ENGINE_STATUS_AVAILABLE:
                return False
        return True

    def _resolve_profile_config(
        self,
        preprocessing_profile: str | None,
        diagnostics: dict[str, Any] | None,
        language_hint: str,
    ) -> tuple[str, dict[str, Any], int]:
        requested = (preprocessing_profile or PROFILE_AUTO).strip().lower() or PROFILE_AUTO
        selected_profile = resolve_preprocessing_profile(
            diagnostics=diagnostics,
            language_hint=language_hint,
            requested_profile=requested,
        )
        if requested == PROFILE_AUTO and selected_profile == PROFILE_UNKNOWN_SAFE_DEFAULT:
            selected_profile = self.default_preprocessing_profile
        effective_preprocessing, effective_dpi = merge_profile_with_base(self.preprocessing_config, selected_profile)
        effective_render_dpi = int(effective_dpi or self.render_dpi)
        return selected_profile, effective_preprocessing, effective_render_dpi

    @staticmethod
    def _write_preprocessing_debug_artifacts(
        debug_artifacts_dir: str,
        debug_artifact_prefix: str,
        profile_name: str,
        original_image: Any,
        preprocessed_image: Any,
        ocr_text: str,
        meta: dict[str, Any],
    ) -> None:
        target_dir = Path(debug_artifacts_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", debug_artifact_prefix).strip("_") or "page"
        stem = f"{safe_prefix}_{profile_name}"

        try:
            original_image.save(target_dir / f"{stem}_original.png")
            preprocessed_image.save(target_dir / f"{stem}_preprocessed.png")
            with (target_dir / f"{stem}_ocr.txt").open("w", encoding="utf-8") as handle:
                handle.write(ocr_text or "")

            payload = {
                "profile": profile_name,
                "meta": meta,
            }
            with (target_dir / f"{stem}_metadata.json").open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.debug("Failed to write preprocessing debug artifacts: %s", exc)

    def _extract_from_variants(
        self,
        base_variants: PageImageVariants,
        rotation_search: tuple[int, ...],
        effective_preprocessing: dict[str, Any],
        routing_strategy: OCRRoutingStrategy | None = None,
        context: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any], PageImageVariants | None]:
        orientation_results: list[dict[str, Any]] = []
        all_errors: dict[str, str] = {}
        last_engine_page_statuses: dict[str, dict[str, str]] = {}
        deskew_enabled = bool(effective_preprocessing.get("deskew", False))
        routing_meta = routing_strategy.to_metadata() if routing_strategy is not None else {}
        context = context or {}

        overall_deadline: float | None = None
        if routing_strategy is not None and routing_strategy.per_page_timeout_s > 0:
            overall_deadline = time.perf_counter() + float(routing_strategy.per_page_timeout_s)

        for angle in rotation_search:
            if overall_deadline is not None and time.perf_counter() >= overall_deadline:
                break

            base_angle = int(angle) % 360
            variants = base_variants.rotated(base_angle)
            deskew_angle = 0.0
            if deskew_enabled:
                estimated_skew = variants.estimate_skew_angle()
                if estimated_skew is not None:
                    deskew_angle = float(estimated_skew)
                    if abs(deskew_angle) >= float(effective_preprocessing.get("deskew_min_abs_degrees", 0.2)):
                        variants = variants.rotated(deskew_angle)
                    else:
                        deskew_angle = 0.0

            effective_angle = _normalize_angle_degrees(variants.rotation_angle)
            candidates, backend_errors, engine_page_statuses, route_trace = self._collect_candidates(
                variants,
                routing_strategy=routing_strategy,
                deadline_s=overall_deadline,
                context=context,
            )
            last_engine_page_statuses = engine_page_statuses
            all_errors.update({f"{name}@{base_angle}": error for name, error in backend_errors.items()})
            if not candidates:
                if any(
                    bool(route_trace.get(key))
                    for key in ("fallback_path", "engines_attempted", "engines_skipped", "runtime_per_engine_ms")
                ):
                    orientation_results.append(
                        {
                            "angle": effective_angle,
                            "base_angle": base_angle,
                            "deskew_angle": deskew_angle,
                            "orientation_class": "no_candidate",
                            "score": -1.0,
                            "fused": None,
                            "candidates": [],
                            "errors": backend_errors,
                            "engine_page_statuses": engine_page_statuses,
                            "variants": variants,
                            "route_trace": route_trace,
                        }
                    )
                continue

            fused = self.fuser.fuse(candidates)
            orientation_label, _, skew_delta = _orientation_class(effective_angle)
            fused.meta["rotation_angle"] = round(effective_angle, 4)
            fused.meta["rotation_base_angle"] = base_angle
            fused.meta["deskew_angle"] = round(deskew_angle, 4)
            fused.meta["orientation_class"] = orientation_label
            fused.meta["orientation_residual_skew"] = round(skew_delta, 4)
            score = self._orientation_score(fused, candidates)
            orientation_results.append(
                {
                    "angle": effective_angle,
                    "base_angle": base_angle,
                    "deskew_angle": deskew_angle,
                    "orientation_class": orientation_label,
                    "score": score,
                    "fused": fused,
                    "candidates": candidates,
                    "errors": backend_errors,
                    "engine_page_statuses": engine_page_statuses,
                    "variants": variants,
                    "route_trace": route_trace,
                }
            )

        viable_orientations = [item for item in orientation_results if item.get("fused") is not None]
        if not viable_orientations:
            attempted: list[str] = []
            skipped: dict[str, str] = {}
            fallback_path: list[str] = []
            runtime_ms: dict[str, float] = {}
            confidence_by_engine: dict[str, float] = {}
            per_engine_outputs: list[dict[str, Any]] = []
            for item in orientation_results:
                trace = item.get("route_trace", {}) if isinstance(item.get("route_trace", {}), dict) else {}
                for engine in trace.get("engines_attempted", []):
                    name = str(engine).strip()
                    if name and name not in attempted:
                        attempted.append(name)
                for engine_name, reason in trace.get("engines_skipped", {}).items():
                    key = str(engine_name).strip()
                    if key and key not in skipped:
                        skipped[key] = str(reason)
                for step in trace.get("fallback_path", []):
                    token = str(step).strip()
                    if token:
                        fallback_path.append(token)
                for engine_name, value in trace.get("runtime_per_engine_ms", {}).items():
                    key = str(engine_name).strip()
                    if key:
                        runtime_ms[key] = runtime_ms.get(key, 0.0) + float(value or 0.0)
                for engine_name, value in trace.get("confidence_per_engine", {}).items():
                    key = str(engine_name).strip()
                    if key and key not in confidence_by_engine:
                        confidence_by_engine[key] = float(value or 0.0)
                for output in trace.get("per_engine_outputs", []):
                    if isinstance(output, dict):
                        per_engine_outputs.append(dict(output))

            no_result_meta = {
                "method": "ensemble",
                "engines_used": [],
                "errors": all_errors,
                "failure_reason": "no_candidate_text",
                "engine_page_statuses": last_engine_page_statuses,
                "rotation_angle": "",
                "rotation_base_angle": "",
                "deskew_angle": "",
                "orientation_class": "unknown",
                "detected_layout_type": "unknown",
                "detected_column_count": 0,
                "detected_has_columns": False,
                "confidence": 0.0,
                "final_output_source": "none",
                "engines_attempted": attempted,
                "engines_skipped": skipped,
                "engine_skip_reasons": sorted(set(skipped.values())),
                "fallback_path": fallback_path[:80],
                "runtime_per_engine_ms": {engine: round(runtime, 3) for engine, runtime in runtime_ms.items()},
                "confidence_per_engine": {engine: round(conf, 6) for engine, conf in confidence_by_engine.items()},
                "per_engine_outputs": per_engine_outputs,
                "alignment_metrics": {
                    "char_disagreement_rate": 1.0,
                    "token_disagreement_rate": 1.0,
                    "line_disagreement_rate": 1.0,
                    "engine_agreement_score": 0.0,
                    "consensus_entropy": 1.0,
                    "engine_agreement": {},
                    "disagreement_positions": [],
                },
                "char_disagreement_rate": 1.0,
                "token_disagreement_rate": 1.0,
                "line_disagreement_rate": 1.0,
                "engine_agreement_score": 0.0,
                "consensus_entropy": 1.0,
                "consensus_explanation": {
                    "winner_engine": "",
                    "consensus_used": False,
                    "disagreement_summary": "no_candidate_text",
                    "high_confidence": False,
                    "low_confidence": True,
                    "uncertain": True,
                    "human_review_recommended": True,
                    "low_quality_all_engines": True,
                    "reason_codes": ["no_candidate_text"],
                    "engine_scores": {},
                    "confidence_band": "low",
                },
                "consensus_used": False,
                "consensus_winner_engine": "",
                "ensemble_uncertain": True,
                "human_review_recommended": True,
                "low_quality_ensemble": True,
                "confusion_counts": {},
            }
            if routing_meta:
                selector_skipped = dict(routing_meta.get("engines_skipped", {}))
                selector_skip_reasons = list(routing_meta.get("engine_skip_reasons", []))
                for key, value in routing_meta.items():
                    if key in {"engines_skipped", "engine_skip_reasons"}:
                        continue
                    no_result_meta[key] = value
                combined_skips = {**selector_skipped, **no_result_meta.get("engines_skipped", {})}
                no_result_meta["engines_skipped"] = combined_skips
                no_result_meta["engine_skip_reasons"] = sorted(
                    set(selector_skip_reasons) | set(no_result_meta.get("engine_skip_reasons", []))
                )
            return "", no_result_meta, None

        best = max(viable_orientations, key=lambda item: item["score"])
        fused = best["fused"]
        fused_meta = fused.meta if isinstance(fused.meta, dict) else {}
        word_winners = fused_meta.get("word_winners", {})
        engine_page_statuses = best.get("engine_page_statuses", {})
        route_trace = best.get("route_trace", {}) if isinstance(best.get("route_trace", {}), dict) else {}
        chosen_angle = float(best.get("angle", 0.0))
        orientation_label, nearest_right_angle, residual_skew = _orientation_class(chosen_angle)
        detected_layout_type = str(fused_meta.get("detected_layout_type", "unknown") or "unknown")
        try:
            detected_column_count = int(fused_meta.get("detected_column_count", 0) or 0)
        except Exception:
            detected_column_count = 0
        detected_has_columns = bool(fused_meta.get("detected_has_columns", detected_column_count > 1))
        confidence = float(fused.confidence if fused.confidence is not None else 0.0)
        meta = {
            "method": "ensemble",
            "engines_used": [candidate.engine for candidate in best["candidates"]],
            "winner": fused_meta.get("winner", "ensemble"),
            "word_winners": word_winners,
            "errors": all_errors,
            "confidence": round(confidence, 6),
            "rotation_angle": round(chosen_angle, 4),
            "rotation_base_angle": int(best.get("base_angle", nearest_right_angle)),
            "deskew_angle": round(float(best.get("deskew_angle", 0.0)), 4),
            "orientation_class": best.get("orientation_class", orientation_label),
            "orientation_residual_skew": round(float(residual_skew), 4),
            "orientation_scores": {
                str(item["base_angle"]): round(float(item["score"]), 4)
                for item in orientation_results
            },
            "orientation_scores_effective": {
                f"{float(item['angle']):.4f}": round(float(item["score"]), 4)
                for item in orientation_results
            },
            "detected_layout_type": detected_layout_type,
            "detected_column_count": detected_column_count,
            "detected_has_columns": detected_has_columns,
            "engine_page_statuses": engine_page_statuses,
            "engines_attempted": list(route_trace.get("engines_attempted", [])),
            "engines_skipped": dict(route_trace.get("engines_skipped", {})),
            "engine_skip_reasons": sorted(set(route_trace.get("engines_skipped", {}).values())),
            "fallback_path": list(route_trace.get("fallback_path", [])),
            "runtime_per_engine_ms": dict(route_trace.get("runtime_per_engine_ms", {})),
            "confidence_per_engine": dict(route_trace.get("confidence_per_engine", {})),
            "final_output_source": "ensemble_fused" if fused.engine == "ensemble" else str(fused_meta.get("winner", fused.engine) or fused.engine),
        }

        per_engine_outputs = [dict(item) for item in route_trace.get("per_engine_outputs", []) if isinstance(item, dict)]
        if not per_engine_outputs:
            for candidate in best["candidates"]:
                per_engine_outputs.append(
                    {
                        "engine": candidate.engine,
                        "text": candidate.text,
                        "confidence": float(candidate.confidence if candidate.confidence is not None else 0.0),
                        "runtime_ms": float(meta.get("runtime_per_engine_ms", {}).get(candidate.engine, 0.0) or 0.0),
                        "status": "available",
                        "error": "",
                        "timed_out": False,
                        "preprocessing_profile": str(context.get("preprocessing_profile", "")),
                        "region_id": str(context.get("region_id", "")),
                        "language_hint": str(context.get("language_hint", "unknown")),
                        "script_hint": str(context.get("script_hint", "unknown")),
                    }
                )

        engine_texts = {
            str(item.get("engine", "")): str(item.get("text", ""))
            for item in per_engine_outputs
            if str(item.get("engine", "")).strip()
        }
        alignment_metrics = analyze_alignment(
            engine_texts=engine_texts,
            consensus_text=fused.text,
            language_hint=str(context.get("language_hint", "unknown")),
            script_hint=str(context.get("script_hint", "unknown")),
            preprocessing_profile=str(context.get("preprocessing_profile", "")),
        )
        consensus_explanation = explain_consensus(
            per_engine_outputs=per_engine_outputs,
            consensus_text=fused.text,
            alignment_metrics=alignment_metrics,
            historical_reliability=self.get_engine_performance_summary(),
            language_hint=str(context.get("language_hint", "unknown")),
            script_hint=str(context.get("script_hint", "unknown")),
            quality_thresholds=(routing_strategy.quality_thresholds if routing_strategy is not None else {}),
        )
        confusion_counts = aggregate_confusion(
            consensus_text=fused.text,
            engine_texts=engine_texts,
        )

        meta["per_engine_outputs"] = per_engine_outputs
        meta["alignment_metrics"] = alignment_metrics
        meta["char_disagreement_rate"] = float(alignment_metrics.get("char_disagreement_rate", 0.0) or 0.0)
        meta["token_disagreement_rate"] = float(alignment_metrics.get("token_disagreement_rate", 0.0) or 0.0)
        meta["line_disagreement_rate"] = float(alignment_metrics.get("line_disagreement_rate", 0.0) or 0.0)
        meta["engine_agreement_score"] = float(alignment_metrics.get("engine_agreement_score", 0.0) or 0.0)
        meta["consensus_entropy"] = float(alignment_metrics.get("consensus_entropy", 0.0) or 0.0)
        meta["consensus_explanation"] = consensus_explanation
        meta["consensus_used"] = bool(consensus_explanation.get("consensus_used", False))
        meta["consensus_winner_engine"] = str(consensus_explanation.get("winner_engine", "") or "")
        meta["ensemble_uncertain"] = bool(consensus_explanation.get("uncertain", False))
        meta["human_review_recommended"] = bool(consensus_explanation.get("human_review_recommended", False))
        meta["low_quality_ensemble"] = bool(consensus_explanation.get("low_quality_all_engines", False))
        meta["confusion_counts"] = confusion_counts

        if meta["low_quality_ensemble"]:
            meta["failure_reason"] = "low_quality_ensemble"
            meta["final_output_source"] = "none_low_quality"
            if routing_meta:
                selector_skipped = dict(routing_meta.get("engines_skipped", {}))
                selector_skip_reasons = list(routing_meta.get("engine_skip_reasons", []))
                for key, value in routing_meta.items():
                    if key in {"engines_skipped", "engine_skip_reasons"}:
                        continue
                    meta[key] = value
                combined_skips = {**selector_skipped, **meta.get("engines_skipped", {})}
                meta["engines_skipped"] = combined_skips
                meta["engine_skip_reasons"] = sorted(
                    set(selector_skip_reasons) | set(meta.get("engine_skip_reasons", []))
                )
            return "", meta, best.get("variants")

        if routing_meta:
            selector_skipped = dict(routing_meta.get("engines_skipped", {}))
            selector_skip_reasons = list(routing_meta.get("engine_skip_reasons", []))
            for key, value in routing_meta.items():
                if key in {"engines_skipped", "engine_skip_reasons"}:
                    continue
                meta[key] = value
            combined_skips = {**selector_skipped, **meta.get("engines_skipped", {})}
            meta["engines_skipped"] = combined_skips
            meta["engine_skip_reasons"] = sorted(
                set(selector_skip_reasons) | set(meta.get("engine_skip_reasons", []))
            )
        if not fused.text.strip():
            meta["failure_reason"] = "empty_fused_text"

        return fused.text, meta, best.get("variants")

    @staticmethod
    def _merge_engine_page_statuses(
        base_statuses: dict[str, dict[str, str]],
        incoming_statuses: dict[str, dict[str, str]],
    ) -> dict[str, dict[str, str]]:
        severity_rank = {
            ENGINE_STATUS_AVAILABLE: 0,
            ENGINE_STATUS_AVAILABLE_UNHEALTHY: 1,
            ENGINE_STATUS_DISABLED_BY_CONFIG: 1,
            ENGINE_STATUS_FAILED_ON_PAGE: 2,
            ENGINE_STATUS_TIMED_OUT: 3,
            ENGINE_STATUS_UNAVAILABLE_DEPENDENCY: 4,
        }
        merged = {name: dict(info) for name, info in base_statuses.items()}
        for engine, info in incoming_statuses.items():
            new_status = str(info.get("status", ""))
            if engine not in merged:
                merged[engine] = dict(info)
                continue
            current = merged[engine]
            current_status = str(current.get("status", ""))
            if severity_rank.get(new_status, 99) >= severity_rank.get(current_status, 99):
                merged[engine] = dict(info)
        return merged

    def _extract_regions_with_layout(
        self,
        page_image: Any,
        layout_result: LayoutPageResult,
        effective_preprocessing: dict[str, Any],
        rotation_search: tuple[int, ...],
        routing_strategy: OCRRoutingStrategy | None = None,
        context: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any], dict[str, dict[str, Any]]]:
        structured_layout = layout_result.to_structured_output()
        region_records = {item["region_id"]: item for item in structured_layout.get("regions", [])}
        selected_regions = self.layout_detector.regions_for_ocr(layout_result)
        max_regions = int(effective_preprocessing.get("region_ocr_max_regions", 24))
        selected_regions = selected_regions[: max(1, max_regions)]

        page_width, page_height = layout_result.page_size
        scale_x = float(page_image.width) / max(page_width, 1.0)
        scale_y = float(page_image.height) / max(page_height, 1.0)

        region_outputs: dict[str, dict[str, Any]] = {}
        engines_used: set[str] = set()
        merged_errors: dict[str, str] = {}
        merged_statuses: dict[str, dict[str, str]] = {}
        text_fragments: list[tuple[int, str]] = []
        confidence_values: list[float] = []
        attempted_engines: list[str] = []
        skipped_engines: dict[str, str] = {}
        fallback_path: list[str] = []
        runtime_per_engine_ms: dict[str, float] = {}
        confidence_samples: dict[str, list[float]] = {}
        per_engine_outputs: list[dict[str, Any]] = []
        confusion_counts_accumulator: Counter[str] = Counter()
        char_disagreements: list[float] = []
        token_disagreements: list[float] = []
        line_disagreements: list[float] = []
        agreement_scores: list[float] = []
        entropy_values: list[float] = []
        uncertain_regions = 0
        low_quality_regions = 0
        routing_meta = routing_strategy.to_metadata() if routing_strategy is not None else {}
        context = context or {}

        for region in selected_regions:
            x0, y0, x1, y1 = region.bbox
            crop_bbox = (
                int(max(0, round(x0 * scale_x))),
                int(max(0, round(y0 * scale_y))),
                int(min(page_image.width, round(x1 * scale_x))),
                int(min(page_image.height, round(y1 * scale_y))),
            )
            if crop_bbox[2] <= crop_bbox[0] or crop_bbox[3] <= crop_bbox[1]:
                continue

            crop_image = page_image.crop(crop_bbox)
            if crop_image.width < 16 or crop_image.height < 16:
                continue

            region_variants = PageImageVariants(crop_image.convert("RGB"), preprocessing_config=effective_preprocessing)
            region_text, region_meta, _ = self._extract_from_variants(
                region_variants,
                rotation_search,
                effective_preprocessing,
                routing_strategy=routing_strategy,
                context={
                    "region_id": region.region_id,
                    "page_number": layout_result.page,
                    "preprocessing_profile": str(context.get("preprocessing_profile", "")),
                    "language_hint": str(context.get("language_hint", "unknown")) if isinstance(context, dict) else "unknown",
                    "script_hint": str(context.get("script_hint", "unknown")) if isinstance(context, dict) else "unknown",
                },
            )
            region_confidence = float(region_meta.get("confidence", 0.0) or 0.0)
            winner = str(region_meta.get("winner", "ensemble") or "ensemble")

            engines_used.update(str(engine) for engine in region_meta.get("engines_used", []) if str(engine).strip())
            merged_statuses = self._merge_engine_page_statuses(merged_statuses, region_meta.get("engine_page_statuses", {}))
            for error_key, error_value in region_meta.get("errors", {}).items():
                merged_errors[f"{region.region_id}:{error_key}"] = str(error_value)

            for engine in region_meta.get("engines_attempted", []):
                engine_name = str(engine).strip()
                if engine_name and engine_name not in attempted_engines:
                    attempted_engines.append(engine_name)
            for engine_name, reason in region_meta.get("engines_skipped", {}).items():
                key = str(engine_name).strip()
                if key and key not in skipped_engines:
                    skipped_engines[key] = str(reason)
            for step in region_meta.get("fallback_path", []):
                normalized_step = str(step).strip()
                if normalized_step:
                    fallback_path.append(f"{region.region_id}:{normalized_step}")

            for engine_name, runtime_val in region_meta.get("runtime_per_engine_ms", {}).items():
                key = str(engine_name).strip()
                if not key:
                    continue
                runtime_per_engine_ms[key] = runtime_per_engine_ms.get(key, 0.0) + float(runtime_val or 0.0)

            for engine_name, conf_val in region_meta.get("confidence_per_engine", {}).items():
                key = str(engine_name).strip()
                if not key:
                    continue
                confidence_samples.setdefault(key, []).append(float(conf_val or 0.0))

            for output in region_meta.get("per_engine_outputs", []):
                if isinstance(output, dict):
                    normalized = dict(output)
                    if not normalized.get("region_id"):
                        normalized["region_id"] = region.region_id
                    per_engine_outputs.append(normalized)

            for pair_label, pair_count in region_meta.get("confusion_counts", {}).items():
                confusion_counts_accumulator[str(pair_label)] += int(pair_count)

            char_disagreements.append(float(region_meta.get("char_disagreement_rate", 0.0) or 0.0))
            token_disagreements.append(float(region_meta.get("token_disagreement_rate", 0.0) or 0.0))
            line_disagreements.append(float(region_meta.get("line_disagreement_rate", 0.0) or 0.0))
            agreement_scores.append(float(region_meta.get("engine_agreement_score", 0.0) or 0.0))
            entropy_values.append(float(region_meta.get("consensus_entropy", 0.0) or 0.0))
            if bool(region_meta.get("ensemble_uncertain", False)):
                uncertain_regions += 1
            if bool(region_meta.get("low_quality_ensemble", False)):
                low_quality_regions += 1

            region_payload = {
                "text": region_text,
                "engine": winner,
                "confidence": round(region_confidence, 6),
            }
            region_outputs[region.region_id] = region_payload

            if region.region_id in region_records:
                region_records[region.region_id]["text"] = region_text
                region_records[region.region_id]["engine"] = winner
                region_records[region.region_id]["confidence"] = round(region_confidence, 6)

            if region_text.strip():
                text_fragments.append((int(region.reading_order), region_text.strip()))
                confidence_values.append(region_confidence)

        reconstructed = "\n".join(text for _, text in sorted(text_fragments, key=lambda item: item[0]))
        avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        if not merged_statuses:
            merged_statuses = {name: dict(info) for name, info in self.engine_readiness.items()}

        confidence_per_engine = {
            engine: round(sum(values) / len(values), 6)
            for engine, values in confidence_samples.items()
            if values
        }
        mean_char_disagreement = sum(char_disagreements) / len(char_disagreements) if char_disagreements else 0.0
        mean_token_disagreement = sum(token_disagreements) / len(token_disagreements) if token_disagreements else 0.0
        mean_line_disagreement = sum(line_disagreements) / len(line_disagreements) if line_disagreements else 0.0
        mean_agreement_score = sum(agreement_scores) / len(agreement_scores) if agreement_scores else 0.0
        mean_entropy = sum(entropy_values) / len(entropy_values) if entropy_values else 0.0

        region_meta = {
            "method": "ensemble_region",
            "engines_used": sorted(engines_used),
            "errors": merged_errors,
            "engine_page_statuses": merged_statuses,
            "winner": "region_ensemble",
            "confidence": round(float(avg_confidence), 6),
            "rotation_angle": "",
            "rotation_base_angle": "",
            "deskew_angle": "",
            "orientation_class": "inferred_by_layout",
            "detected_layout_type": str(layout_result.column_mode),
            "detected_column_count": int(layout_result.column_count),
            "detected_has_columns": bool(layout_result.column_count > 1),
            "detected_has_footnotes": bool(layout_result.has_footnotes),
            "detected_has_table_interruptions": bool(layout_result.has_table_interruptions),
            "detected_region_count": len(structured_layout.get("regions", [])),
            "reading_order_confidence": round(float(layout_result.reading_order_confidence), 6),
            "reading_order_source": str(layout_result.ordering_source),
            "ordering_source": str(layout_result.ordering_source),
            "region_ocr_used": True,
            "structured_layout": structured_layout,
            "plain_text_reconstruction": reconstructed,
            "engines_attempted": attempted_engines,
            "engines_skipped": skipped_engines,
            "engine_skip_reasons": sorted(set(skipped_engines.values())),
            "fallback_path": fallback_path[:80],
            "runtime_per_engine_ms": {engine: round(runtime, 3) for engine, runtime in runtime_per_engine_ms.items()},
            "confidence_per_engine": confidence_per_engine,
            "final_output_source": "region_ocr",
            "per_engine_outputs": per_engine_outputs,
            "char_disagreement_rate": round(float(mean_char_disagreement), 6),
            "token_disagreement_rate": round(float(mean_token_disagreement), 6),
            "line_disagreement_rate": round(float(mean_line_disagreement), 6),
            "engine_agreement_score": round(float(mean_agreement_score), 6),
            "consensus_entropy": round(float(mean_entropy), 6),
            "ensemble_uncertain": bool(uncertain_regions > 0),
            "human_review_recommended": bool(uncertain_regions > 0),
            "low_quality_ensemble": bool(low_quality_regions > 0),
            "confusion_counts": dict(confusion_counts_accumulator),
            "alignment_metrics": {
                "char_disagreement_rate": round(float(mean_char_disagreement), 6),
                "token_disagreement_rate": round(float(mean_token_disagreement), 6),
                "line_disagreement_rate": round(float(mean_line_disagreement), 6),
                "engine_agreement_score": round(float(mean_agreement_score), 6),
                "consensus_entropy": round(float(mean_entropy), 6),
                "engine_agreement": {},
                "disagreement_positions": [],
            },
        }
        if routing_meta:
            selector_skipped = dict(routing_meta.get("engines_skipped", {}))
            selector_skip_reasons = list(routing_meta.get("engine_skip_reasons", []))
            for key, value in routing_meta.items():
                if key in {"engines_skipped", "engine_skip_reasons"}:
                    continue
                region_meta[key] = value
            combined_skips = {**selector_skipped, **region_meta.get("engines_skipped", {})}
            region_meta["engines_skipped"] = combined_skips
            region_meta["engine_skip_reasons"] = sorted(
                set(selector_skip_reasons) | set(region_meta.get("engine_skip_reasons", []))
            )
        if not reconstructed.strip():
            region_meta["failure_reason"] = "region_ocr_empty"

        return reconstructed, region_meta, region_outputs

    def extract_page_text(
        self,
        pdf_path: str,
        page_num: int,
        preprocessing_profile: str | None = None,
        diagnostics: dict[str, Any] | None = None,
        language_hint: str = "unknown",
        script_hint: str = "unknown",
        document_type: str = "unknown",
        timeout_config: dict[str, Any] | None = None,
        quality_thresholds: dict[str, Any] | None = None,
        debug_artifacts_dir: str | None = None,
        debug_artifact_prefix: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        selected_profile, effective_preprocessing, effective_render_dpi = self._resolve_profile_config(
            preprocessing_profile=preprocessing_profile,
            diagnostics=diagnostics,
            language_hint=language_hint,
        )
        rotation_search = tuple(
            int(angle) % 360
            for angle in effective_preprocessing.get("rotation_search_degrees", self.rotation_search or (0,))
        ) or (0,)

        with fitz.open(pdf_path) as document:
            if page_num >= len(document):
                return "", {
                    "method": "ensemble",
                    "error": "page_out_of_range",
                    "failure_reason": "page_out_of_range",
                    "engine_page_statuses": {},
                    "preprocessing_profile": selected_profile,
                    "preprocessing_render_dpi": effective_render_dpi,
                }

            page = document[page_num]
            scale = effective_render_dpi / 72.0
            pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            layout_result = self.layout_detector.analyze_page(
                page,
                page_number=page_num + 1,
                language_hint=language_hint,
            )

        routing_diagnostics = dict(diagnostics or {})
        routing_diagnostics.setdefault("language_hint", language_hint)
        routing_diagnostics["detected_column_count"] = int(layout_result.column_count)
        routing_diagnostics["detected_has_footnotes"] = bool(layout_result.has_footnotes)
        routing_diagnostics["detected_has_table_interruptions"] = bool(layout_result.has_table_interruptions)
        routing_diagnostics.setdefault(
            "layout_complexity_score",
            0.75 if self.layout_detector.is_complex_layout(layout_result) else 0.25,
        )

        routing_strategy = self.select_strategy(
            diagnostics=routing_diagnostics,
            language_hint=language_hint,
            script_hint=script_hint,
            document_type=document_type,
            requested_profile=selected_profile,
            timeout_config=timeout_config,
            quality_thresholds=quality_thresholds,
            force_ocr=True,
            prefer_text_layer=False,
        )
        routing_meta = routing_strategy.to_metadata()

        base_variants = PageImageVariants.from_pixmap(pixmap, preprocessing_config=effective_preprocessing)
        full_text = ""
        full_meta: dict[str, Any] = {}
        full_best_variants: PageImageVariants | None = None
        if routing_strategy.use_full_page_ocr:
            full_text, full_meta, full_best_variants = self._extract_from_variants(
                base_variants,
                rotation_search,
                effective_preprocessing,
                routing_strategy=routing_strategy,
                context={
                    "page_number": page_num + 1,
                    "preprocessing_profile": selected_profile,
                    "language_hint": language_hint,
                    "script_hint": script_hint,
                },
            )

        chosen_text = full_text
        chosen_meta = dict(full_meta)
        region_outputs: dict[str, dict[str, Any]] = {}
        region_ocr_attempted = False

        should_try_region_ocr = (
            bool(effective_preprocessing.get("enable_region_ocr", True))
            and bool(routing_strategy.use_region_ocr)
            and self.layout_detector.is_complex_layout(layout_result)
        )
        if should_try_region_ocr and PIL_AVAILABLE:
            region_ocr_attempted = True
            if pixmap.alpha:
                rgb_pixmap = fitz.Pixmap(fitz.csRGB, pixmap)
            else:
                rgb_pixmap = pixmap
            page_image = Image.frombytes("RGB", [rgb_pixmap.width, rgb_pixmap.height], rgb_pixmap.samples)
            region_text, region_meta, region_outputs = self._extract_regions_with_layout(
                page_image,
                layout_result,
                effective_preprocessing,
                rotation_search,
                routing_strategy=routing_strategy,
                context={
                    "page_number": page_num + 1,
                    "preprocessing_profile": selected_profile,
                    "language_hint": language_hint,
                    "script_hint": script_hint,
                },
            )
            min_region_chars = int(effective_preprocessing.get("region_ocr_min_chars", 18))
            if region_text.strip() and len(region_text.strip()) >= max(1, min_region_chars):
                chosen_text = region_text
                chosen_meta = region_meta
            elif not chosen_text.strip() and region_text.strip():
                chosen_text = region_text
                chosen_meta = region_meta

        # Preserve full-page fallback behavior even when layout-first is selected.
        if region_ocr_attempted and not chosen_text.strip():
            if not full_meta:
                full_text, full_meta, full_best_variants = self._extract_from_variants(
                    base_variants,
                    rotation_search,
                    effective_preprocessing,
                    routing_strategy=routing_strategy,
                    context={
                        "page_number": page_num + 1,
                        "preprocessing_profile": selected_profile,
                        "language_hint": language_hint,
                        "script_hint": script_hint,
                    },
                )
            if full_text.strip():
                chosen_text = full_text
                chosen_meta = dict(full_meta)
                fallback_steps = list(chosen_meta.get("fallback_path", []))
                fallback_steps.append("region_ocr_empty->full_page_fallback")
                chosen_meta["fallback_path"] = fallback_steps
                chosen_meta["final_output_source"] = "fallback_full_page_ocr"

        if not chosen_meta:
            chosen_meta = {
                "method": "ensemble",
                "engines_used": [],
                "errors": {},
                "engine_page_statuses": {},
                "confidence": 0.0,
            }

        if routing_meta:
            selector_skipped = dict(routing_meta.get("engines_skipped", {}))
            selector_skip_reasons = list(routing_meta.get("engine_skip_reasons", []))
            for key, value in routing_meta.items():
                if key in {"engines_skipped", "engine_skip_reasons"}:
                    continue
                chosen_meta[key] = value
            if "engines_skipped" not in chosen_meta:
                chosen_meta["engines_skipped"] = {}
            combined_skips = {**selector_skipped, **dict(chosen_meta.get("engines_skipped", {}))}
            chosen_meta["engines_skipped"] = combined_skips
            existing_skip_reasons = list(chosen_meta.get("engine_skip_reasons", []))
            chosen_meta["engine_skip_reasons"] = sorted(set(selector_skip_reasons) | set(existing_skip_reasons))

        structured_layout = chosen_meta.get("structured_layout")
        if not isinstance(structured_layout, dict):
            structured_layout = layout_result.to_structured_output()

        chosen_meta["structured_layout"] = structured_layout
        chosen_meta["plain_text_reconstruction"] = chosen_text
        chosen_meta["detected_layout_type"] = str(chosen_meta.get("detected_layout_type", layout_result.column_mode) or layout_result.column_mode)
        chosen_meta["detected_column_count"] = int(chosen_meta.get("detected_column_count", layout_result.column_count) or layout_result.column_count)
        chosen_meta["detected_has_columns"] = bool(chosen_meta.get("detected_has_columns", layout_result.column_count > 1))
        chosen_meta["detected_has_footnotes"] = bool(chosen_meta.get("detected_has_footnotes", layout_result.has_footnotes))
        chosen_meta["detected_has_table_interruptions"] = bool(
            chosen_meta.get("detected_has_table_interruptions", layout_result.has_table_interruptions)
        )
        chosen_meta["detected_region_count"] = int(chosen_meta.get("detected_region_count", len(layout_result.regions)) or len(layout_result.regions))
        chosen_meta["reading_order_confidence"] = round(
            float(chosen_meta.get("reading_order_confidence", layout_result.reading_order_confidence) or layout_result.reading_order_confidence),
            6,
        )
        chosen_meta["reading_order_source"] = str(chosen_meta.get("reading_order_source", layout_result.ordering_source) or layout_result.ordering_source)
        chosen_meta["ordering_source"] = str(chosen_meta.get("ordering_source", layout_result.ordering_source) or layout_result.ordering_source)
        chosen_meta["region_ocr_used"] = bool(chosen_meta.get("region_ocr_used", False))
        chosen_meta["region_ocr_attempted"] = bool(region_ocr_attempted)
        chosen_meta["preprocessing_profile"] = selected_profile
        chosen_meta["preprocessing_render_dpi"] = effective_render_dpi
        chosen_meta["profile_primary_variant"] = str(effective_preprocessing.get("profile_primary_variant", "adaptive"))
        if not chosen_meta.get("final_output_source"):
            if chosen_meta.get("region_ocr_used"):
                chosen_meta["final_output_source"] = "region_ocr"
            elif chosen_text.strip():
                chosen_meta["final_output_source"] = "full_page_ocr"
            else:
                chosen_meta["final_output_source"] = "none"

        if debug_artifacts_dir:
            debug_prefix = debug_artifact_prefix or f"{Path(pdf_path).stem}_p{page_num + 1:04d}"
            try:
                self.layout_detector.save_debug_artifacts(
                    pixmap,
                    layout_result,
                    output_dir=debug_artifacts_dir,
                    prefix=f"{debug_prefix}_layout",
                    region_outputs=region_outputs if region_outputs else None,
                )
            except Exception as exc:
                logger.debug("Failed to write layout debug artifacts: %s", exc)

            if isinstance(full_best_variants, PageImageVariants):
                primary_variant = str(effective_preprocessing.get("profile_primary_variant", "adaptive"))
                if primary_variant not in full_best_variants.variant_names():
                    primary_variant = "adaptive" if "adaptive" in full_best_variants.variant_names() else "original"
                self._write_preprocessing_debug_artifacts(
                    debug_artifacts_dir=debug_artifacts_dir,
                    debug_artifact_prefix=debug_prefix,
                    profile_name=selected_profile,
                    original_image=full_best_variants.get_pil("original"),
                    preprocessed_image=full_best_variants.get_pil(primary_variant),
                    ocr_text=chosen_text,
                    meta=chosen_meta,
                )

        if not chosen_text.strip() and not chosen_meta.get("failure_reason"):
            chosen_meta["failure_reason"] = "empty_fused_text"

        return chosen_text, chosen_meta

    def _collect_candidates(
        self,
        variants: PageImageVariants,
        routing_strategy: OCRRoutingStrategy | None = None,
        deadline_s: float | None = None,
        context: dict[str, Any] | None = None,
    ) -> tuple[list[OCRCandidate], dict[str, str], dict[str, dict[str, str]], dict[str, Any]]:
        candidates: list[OCRCandidate] = []
        backend_errors: dict[str, str] = {}
        engine_page_statuses: dict[str, dict[str, str]] = {}
        route_trace: dict[str, Any] = {
            "engines_attempted": [],
            "engines_skipped": {},
            "fallback_path": [],
            "runtime_per_engine_ms": {},
            "confidence_per_engine": {},
            "per_engine_outputs": [],
        }
        context = context or {}
        context_region_id = str(context.get("region_id", "") or "")
        context_profile = str(context.get("preprocessing_profile", "") or "")
        context_language_hint = str(context.get("language_hint", "unknown") or "unknown")
        context_script_hint = str(context.get("script_hint", "unknown") or "unknown")
        context_page_number = int(context.get("page_number", 0) or 0)

        def _record_engine_output(
            engine: str,
            status: str,
            text: str = "",
            confidence: float = 0.0,
            runtime_ms: float = 0.0,
            error: str = "",
            timed_out: bool = False,
        ) -> None:
            route_trace["per_engine_outputs"].append(
                {
                    "engine": str(engine),
                    "text": str(text or ""),
                    "confidence": round(float(confidence or 0.0), 6),
                    "runtime_ms": round(float(runtime_ms or 0.0), 3),
                    "status": str(status),
                    "error": str(error or ""),
                    "timed_out": bool(timed_out),
                    "preprocessing_profile": context_profile,
                    "region_id": context_region_id,
                    "language_hint": context_language_hint,
                    "script_hint": context_script_hint,
                    "page_number": context_page_number,
                }
            )

        quality_thresholds = routing_strategy.quality_thresholds if routing_strategy is not None else {}
        min_confidence = float(quality_thresholds.get("min_confidence", 0.0) or 0.0)
        min_text_chars = int(float(quality_thresholds.get("min_text_chars", 0) or 0))
        min_alnum_ratio = float(quality_thresholds.get("min_alnum_ratio", 0.0) or 0.0)

        planned_order = routing_strategy.engine_plan() if routing_strategy is not None else []
        max_engines = int(routing_strategy.max_engines_per_page) if routing_strategy is not None else 0
        if max_engines <= 0:
            max_engines = len(self.backends)

        per_engine_timeout_s = self.per_engine_timeout_s
        if routing_strategy is not None and routing_strategy.per_engine_timeout_s > 0:
            per_engine_timeout_s = float(routing_strategy.per_engine_timeout_s)

        for name, info in self.engine_readiness.items():
            if info.get("status") == ENGINE_STATUS_DISABLED_BY_CONFIG:
                engine_page_statuses[name] = {
                    "status": ENGINE_STATUS_DISABLED_BY_CONFIG,
                    "reason": info.get("reason", ""),
                }
                _record_engine_output(
                    name,
                    ENGINE_STATUS_DISABLED_BY_CONFIG,
                    error=str(info.get("reason", "")),
                    timed_out=False,
                )

        active_backends_by_name: dict[str, OCRBackendBase] = {backend.name: backend for backend in self.backends}
        backend_sequence: list[OCRBackendBase] = []
        if planned_order:
            for engine_name in planned_order:
                backend = active_backends_by_name.get(engine_name)
                if backend is not None and backend not in backend_sequence:
                    backend_sequence.append(backend)

            # Compatibility fallback: if the selected plan does not map to any
            # active backend (common in tests that inject stub backends), use
            # currently attached backends as the executable sequence.
            if not backend_sequence and self.backends:
                backend_sequence = list(self.backends)
            else:
                for backend in self.backends:
                    if backend.name in planned_order:
                        continue
                    route_trace["engines_skipped"].setdefault(backend.name, "not_in_strategy_plan")
        else:
            backend_sequence = list(self.backends)

        attempts = 0
        for backend in backend_sequence:
            if attempts >= max_engines:
                route_trace["engines_skipped"].setdefault(backend.name, "max_engines_per_page_reached")
                engine_page_statuses[backend.name] = {
                    "status": ENGINE_STATUS_DISABLED_BY_CONFIG,
                    "reason": "max_engines_per_page_reached",
                }
                _record_engine_output(
                    backend.name,
                    ENGINE_STATUS_DISABLED_BY_CONFIG,
                    error="max_engines_per_page_reached",
                )
                continue

            if deadline_s is not None and time.perf_counter() >= deadline_s:
                route_trace["engines_skipped"].setdefault(backend.name, "per_page_timeout_budget_exceeded")
                engine_page_statuses[backend.name] = {
                    "status": ENGINE_STATUS_TIMED_OUT,
                    "reason": "per_page_timeout_budget_exceeded",
                }
                _record_engine_output(
                    backend.name,
                    ENGINE_STATUS_TIMED_OUT,
                    error="per_page_timeout_budget_exceeded",
                    timed_out=True,
                )
                continue

            readiness = self.engine_readiness.get(backend.name, {})
            if backend.name in route_trace["engines_skipped"]:
                reason = str(route_trace["engines_skipped"][backend.name])
                engine_page_statuses.setdefault(
                    backend.name,
                    {
                        "status": ENGINE_STATUS_DISABLED_BY_CONFIG,
                        "reason": reason,
                    },
                )
                _record_engine_output(
                    backend.name,
                    ENGINE_STATUS_DISABLED_BY_CONFIG,
                    error=reason,
                )
                continue

            if readiness.get("status") in {ENGINE_STATUS_UNAVAILABLE_DEPENDENCY, ENGINE_STATUS_TIMED_OUT, ENGINE_STATUS_DISABLED_BY_CONFIG}:
                reason = readiness.get("reason", backend._failed_reason or "engine_unavailable")
                backend_errors[backend.name] = reason
                engine_page_statuses[backend.name] = {
                    "status": readiness.get("status", ENGINE_STATUS_UNAVAILABLE_DEPENDENCY),
                    "reason": reason,
                }
                route_trace["engines_skipped"][backend.name] = str(reason or "engine_unavailable")
                _record_engine_output(
                    backend.name,
                    str(readiness.get("status", ENGINE_STATUS_UNAVAILABLE_DEPENDENCY)),
                    error=str(reason or "engine_unavailable"),
                    timed_out=bool(readiness.get("status") == ENGINE_STATUS_TIMED_OUT),
                )
                continue

            attempts += 1
            route_trace["engines_attempted"].append(backend.name)
            started_at = time.perf_counter()
            try:
                candidate, timed_out = _run_callable_with_timeout(
                    lambda: backend.infer(variants),
                    per_engine_timeout_s,
                )
                elapsed_ms = (time.perf_counter() - started_at) * 1000.0
                route_trace["runtime_per_engine_ms"][backend.name] = round(float(elapsed_ms), 3)
                if timed_out:
                    reason = f"inference timed out after {per_engine_timeout_s:.2f}s"
                    backend_errors[backend.name] = reason
                    engine_page_statuses[backend.name] = {
                        "status": ENGINE_STATUS_TIMED_OUT,
                        "reason": reason,
                    }
                    route_trace["fallback_path"].append(f"{backend.name}:timed_out")
                    self._record_engine_observation(backend.name, ENGINE_STATUS_TIMED_OUT, elapsed_ms, confidence=None)
                    _record_engine_output(
                        backend.name,
                        ENGINE_STATUS_TIMED_OUT,
                        runtime_ms=elapsed_ms,
                        error=reason,
                        timed_out=True,
                    )
                    continue
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - started_at) * 1000.0
                route_trace["runtime_per_engine_ms"][backend.name] = round(float(elapsed_ms), 3)
                reason = str(exc)
                backend_errors[backend.name] = reason
                engine_page_statuses[backend.name] = {
                    "status": ENGINE_STATUS_FAILED_ON_PAGE,
                    "reason": reason,
                }
                route_trace["fallback_path"].append(f"{backend.name}:exception")
                self._record_engine_observation(backend.name, ENGINE_STATUS_FAILED_ON_PAGE, elapsed_ms, confidence=None)
                _record_engine_output(
                    backend.name,
                    ENGINE_STATUS_FAILED_ON_PAGE,
                    runtime_ms=elapsed_ms,
                    error=reason,
                )
                continue

            if candidate:
                text = str(candidate.text or "")
                confidence = float(candidate.confidence if candidate.confidence is not None else 0.0)
                alnum_ratio = (sum(1 for ch in text if ch.isalnum()) / max(len(text), 1)) if text else 0.0
                route_trace["confidence_per_engine"][backend.name] = round(confidence, 6)

                quality_reason = ""
                if min_confidence > 0.0 and confidence < min_confidence:
                    quality_reason = "below_confidence_threshold"
                elif min_text_chars > 0 and len(text.strip()) < min_text_chars:
                    quality_reason = "below_text_length_threshold"
                elif min_alnum_ratio > 0.0 and alnum_ratio < min_alnum_ratio:
                    quality_reason = "below_alnum_ratio_threshold"

                if quality_reason:
                    backend_errors[backend.name] = quality_reason
                    engine_page_statuses[backend.name] = {
                        "status": ENGINE_STATUS_FAILED_ON_PAGE,
                        "reason": quality_reason,
                    }
                    route_trace["fallback_path"].append(f"{backend.name}:{quality_reason}")
                    self._record_engine_observation(
                        backend.name,
                        ENGINE_STATUS_FAILED_ON_PAGE,
                        float(route_trace["runtime_per_engine_ms"].get(backend.name, 0.0)),
                        confidence=confidence,
                    )
                    _record_engine_output(
                        backend.name,
                        ENGINE_STATUS_FAILED_ON_PAGE,
                        text=text,
                        confidence=confidence,
                        runtime_ms=float(route_trace["runtime_per_engine_ms"].get(backend.name, 0.0)),
                        error=quality_reason,
                    )
                    continue

                candidates.append(candidate)
                engine_page_statuses[backend.name] = {"status": ENGINE_STATUS_AVAILABLE, "reason": ""}
                route_trace["fallback_path"].append(f"{backend.name}:accepted")
                self._record_engine_observation(
                    backend.name,
                    ENGINE_STATUS_AVAILABLE,
                    float(route_trace["runtime_per_engine_ms"].get(backend.name, 0.0)),
                    confidence=confidence,
                )
                _record_engine_output(
                    backend.name,
                    ENGINE_STATUS_AVAILABLE,
                    text=text,
                    confidence=confidence,
                    runtime_ms=float(route_trace["runtime_per_engine_ms"].get(backend.name, 0.0)),
                    error="",
                )

                if routing_strategy is not None and not routing_strategy.use_ensemble:
                    break
            else:
                reason = backend._failed_reason or "no_text_candidate"
                status = ENGINE_STATUS_FAILED_ON_PAGE
                if backend._failed_reason and readiness.get("status") == ENGINE_STATUS_UNAVAILABLE_DEPENDENCY:
                    status = ENGINE_STATUS_UNAVAILABLE_DEPENDENCY
                backend_errors[backend.name] = reason
                engine_page_statuses[backend.name] = {
                    "status": status,
                    "reason": reason,
                }
                route_trace["fallback_path"].append(f"{backend.name}:{reason}")
                self._record_engine_observation(
                    backend.name,
                    status,
                    float(route_trace["runtime_per_engine_ms"].get(backend.name, 0.0)),
                    confidence=None,
                )
                _record_engine_output(
                    backend.name,
                    status,
                    runtime_ms=float(route_trace["runtime_per_engine_ms"].get(backend.name, 0.0)),
                    error=reason,
                )

        route_trace["engine_skip_reasons"] = sorted(set(route_trace.get("engines_skipped", {}).values()))
        return candidates, backend_errors, engine_page_statuses, route_trace

    def _orientation_score(self, fused: OCRCandidate, candidates: list[OCRCandidate]) -> float:
        text = fused.text.strip()
        if not text:
            return -1.0

        alnum_count = sum(1 for char in text if char.isalnum())
        alnum_ratio = alnum_count / max(len(text), 1)
        avg_confidence = sum(candidate.confidence for candidate in candidates) / len(candidates)
        agreement = 0.0
        if len(candidates) > 1:
            comparisons = []
            for index, candidate in enumerate(candidates):
                for other in candidates[index + 1 :]:
                    comparisons.append(
                        SequenceMatcher(None, _comparison_key(candidate.text), _comparison_key(other.text)).ratio()
                    )
            agreement = sum(comparisons) / len(comparisons) if comparisons else 0.0

        line_count = len(TextEnsembleFuser._lines(text))
        return (
            (len(text) / 120.0)
            + (avg_confidence * 0.8)
            + (agreement * 0.7)
            + (alnum_ratio * 0.4)
            + (_diacritic_richness(text) * 2.0)
            + min(line_count / 12.0, 1.0) * 0.35
        )
