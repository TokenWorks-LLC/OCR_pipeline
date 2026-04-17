#!/usr/bin/env python3
"""Active OCR ensemble support for the page-text pipeline.

This module provides four layers for hard OCR pages:

1. Aggressive page preprocessing with denoise, contrast, adaptive thresholding,
   and morphology-aware image variants.
2. Best-effort OCR backend adapters for PaddleOCR, docTR, MMOCR, and Kraken.
3. Orientation search across right-angle rotations for upside-down and sideways pages.
4. A diacritic-aware fusion stage with multi-column reading-order reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
import json
import logging
import os
from pathlib import Path
import re
from statistics import median
import unicodedata
from typing import Any, Iterable

import fitz

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
    "contrast_factor": 1.8,
    "sharpen_radius": 1.5,
    "sharpen_percent": 180,
    "deskew": False,
}
DEFAULT_LAYOUT = {
    "detect_multi_column": True,
    "min_lines_for_columns": 4,
    "column_gap_ratio": 0.14,
    "column_overlap_ratio": 0.55,
}


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

    def reorder(self, lines: list[OCRLine], page_size: tuple[int, int]) -> list[OCRLine]:
        visible_lines = [line for line in lines if line.text.strip()]
        if not visible_lines:
            return []

        boxed_lines = [line for line in visible_lines if line.bbox is not None]
        if len(boxed_lines) < max(2, int(self.config["min_lines_for_columns"])):
            return sorted(visible_lines, key=self._single_column_key)

        columns = self._split_columns(boxed_lines, page_size)
        if len(columns) <= 1:
            return sorted(visible_lines, key=self._single_column_key)

        ordered: list[OCRLine] = []
        for column in columns:
            ordered.extend(sorted(column, key=self._single_column_key))

        seen = {id(line) for line in ordered}
        remainder = [line for line in visible_lines if id(line) not in seen]
        ordered.extend(sorted(remainder, key=self._single_column_key))
        return ordered

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
        rotation_angle: int = 0,
    ):
        if not PIL_AVAILABLE:
            raise RuntimeError("Pillow is required for the ensemble preprocessor")

        self.preprocessing_config = {**DEFAULT_PREPROCESSING, **(preprocessing_config or {})}
        self.rotation_angle = int(rotation_angle)
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

    def rotated(self, angle: int) -> "PageImageVariants":
        normalized_angle = int(angle) % 360
        if normalized_angle == 0:
            return self

        rotated_image = self.base_image.rotate(normalized_angle, expand=True, fillcolor=(255, 255, 255))
        return PageImageVariants(
            rotated_image,
            preprocessing_config=self.preprocessing_config,
            rotation_angle=normalized_angle,
        )

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
        denoised = self._denoise(gray)
        autocontrast = ImageOps.autocontrast(denoised)
        boosted = ImageEnhance.Contrast(autocontrast).enhance(float(self.preprocessing_config["contrast_factor"]))
        sharpened = boosted.filter(
            ImageFilter.UnsharpMask(
                radius=float(self.preprocessing_config["sharpen_radius"]),
                percent=int(self.preprocessing_config["sharpen_percent"]),
                threshold=3,
            )
        )

        binary = self._binary_threshold(sharpened)
        adaptive = self._adaptive_threshold(sharpened)
        morphology = self._morphology(adaptive)

        return {
            "original": base_rgb,
            "grayscale": gray.convert("RGB"),
            "denoise": denoised.convert("RGB"),
            "autocontrast": autocontrast.convert("RGB"),
            "contrast": boosted.convert("RGB"),
            "sharpen": sharpened.convert("RGB"),
            "binary": binary.convert("RGB"),
            "adaptive": adaptive.convert("RGB"),
            "morphology": morphology.convert("RGB"),
        }

    def _denoise(self, gray: Any) -> Any:
        if self.preprocessing_config.get("enable_denoise", True) and CV2_AVAILABLE and NUMPY_AVAILABLE:
            gray_array = np.array(gray)
            denoised = cv2.fastNlMeansDenoising(gray_array, None, 12, 7, 21)
            return Image.fromarray(denoised)
        return gray.filter(ImageFilter.MedianFilter(size=3))

    def _binary_threshold(self, gray: Any) -> Any:
        if CV2_AVAILABLE and NUMPY_AVAILABLE:
            gray_array = np.array(gray)
            _, thresholded = cv2.threshold(gray_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return Image.fromarray(thresholded)

        if NUMPY_AVAILABLE:
            threshold = int(np.array(gray).mean())
        else:
            threshold = 160
        return gray.point(lambda px: 255 if px >= threshold else 0)

    def _adaptive_threshold(self, gray: Any) -> Any:
        if self.preprocessing_config.get("enable_adaptive_threshold", True) and CV2_AVAILABLE and NUMPY_AVAILABLE:
            gray_array = np.array(gray)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray_array)
            adaptive = cv2.adaptiveThreshold(
                clahe,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                35,
                11,
            )
            return Image.fromarray(adaptive)

        return ImageOps.autocontrast(self._binary_threshold(gray))

    def _morphology(self, binary: Any) -> Any:
        if self.preprocessing_config.get("enable_morphology", True) and CV2_AVAILABLE and NUMPY_AVAILABLE:
            binary_array = np.array(binary)
            kernel = np.ones((2, 2), np.uint8)
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
        for variant in self.preferred_variants:
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
        ordered_lines = self.layout_analyzer.reorder(candidate.lines, variants.page_size) if candidate.lines else []
        if ordered_lines:
            candidate.lines = ordered_lines
            candidate.text = _normalize_whitespace("\n".join(line.text for line in ordered_lines))
        else:
            candidate.text = _normalize_whitespace(candidate.text)

        candidate.meta.setdefault("rotation_angle", variants.rotation_angle)
        candidate.meta.setdefault("page_size", variants.page_size)
        candidate.meta.setdefault("line_count", len(candidate.lines) or len(TextEnsembleFuser._lines(candidate.text)))
        return candidate

    def _ensure_loaded(self) -> None:
        raise NotImplementedError

    def _infer_variant(self, variants: PageImageVariants, variant: str) -> OCRCandidate | None:
        raise NotImplementedError


class PaddleBackend(OCRBackendBase):
    name = "paddle"

    def _ensure_loaded(self) -> None:
        if self._engine is not None:
            return
        from paddleocr import PaddleOCR

        language_hint = self.config.get("paddle_lang") or self.config.get("lang") or "en"
        self._engine = PaddleOCR(lang=language_hint, use_textline_orientation=True)

    def _infer_variant(self, variants: PageImageVariants, variant: str) -> OCRCandidate | None:
        image = variants.get_numpy(variant)
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


class DocTRBackend(OCRBackendBase):
    name = "doctr"

    def _ensure_loaded(self) -> None:
        if self._engine is not None:
            return
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
        logger.info(
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
                logger.info("fusion line %d winners: %s | %s", line_index, tally_str, line_text)

        if winner_counts:
            summary = ", ".join(f"{eng}={count}" for eng, count in sorted(winner_counts.items(), key=lambda kv: -kv[1]))
            logger.info("fusion page winners (word-level): %s", summary)

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

    def __init__(self, profile_path: str | None = None):
        self.profile = self._load_profile(profile_path)
        self.render_dpi = int(self.profile.get("rendering", {}).get("dpi", 300))
        engines_cfg = self.profile.get("engines", {})
        enabled_engines = engines_cfg.get("enabled") or ["paddle", "doctr", "mmocr", "kraken"]
        self.weights = self.profile.get("fusion", {}).get("weights", {})
        self.preprocessing_config = {**DEFAULT_PREPROCESSING, **self.profile.get("preprocessing", {})}
        self.layout_config = {**DEFAULT_LAYOUT, **self.profile.get("layout", {})}
        self.rotation_search = tuple(
            int(angle) % 360 for angle in self.preprocessing_config.get("rotation_search_degrees", [0, 90, 180, 270])
        )
        self.backends: list[OCRBackendBase] = [
            self.BACKEND_TYPES[name](engines_cfg, self.layout_config)
            for name in enabled_engines
            if name in self.BACKEND_TYPES
        ]
        self.fuser = TextEnsembleFuser(self.weights)

    @staticmethod
    def _load_profile(profile_path: str | None) -> dict[str, Any]:
        if not profile_path or not Path(profile_path).exists():
            return {}
        with Path(profile_path).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def extract_page_text(self, pdf_path: str, page_num: int) -> tuple[str, dict[str, Any]]:
        with fitz.open(pdf_path) as document:
            if page_num >= len(document):
                return "", {"method": "ensemble", "error": "page_out_of_range"}

            page = document[page_num]
            scale = self.render_dpi / 72.0
            pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)

        base_variants = PageImageVariants.from_pixmap(pixmap, preprocessing_config=self.preprocessing_config)
        orientation_results: list[dict[str, Any]] = []
        all_errors: dict[str, str] = {}

        for angle in self.rotation_search or (0,):
            variants = base_variants.rotated(angle)
            candidates, backend_errors = self._collect_candidates(variants)
            all_errors.update({f"{name}@{angle}": error for name, error in backend_errors.items()})
            if not candidates:
                continue

            fused = self.fuser.fuse(candidates)
            fused.meta["rotation_angle"] = angle
            score = self._orientation_score(fused, candidates)
            orientation_results.append(
                {
                    "angle": angle,
                    "score": score,
                    "fused": fused,
                    "candidates": candidates,
                    "errors": backend_errors,
                }
            )

        if not orientation_results:
            return "", {"method": "ensemble", "engines_used": [], "errors": all_errors}

        best = max(orientation_results, key=lambda item: item["score"])
        fused = best["fused"]
        fused_meta = fused.meta if isinstance(fused.meta, dict) else {}
        word_winners = fused_meta.get("word_winners", {})
        if word_winners:
            summary = ", ".join(
                f"{eng}={count}" for eng, count in sorted(word_winners.items(), key=lambda kv: -kv[1])
            )
            logger.info("page %d word-winner counts: %s", page_num, summary)
        meta = {
            "method": "ensemble",
            "engines_used": [candidate.engine for candidate in best["candidates"]],
            "winner": fused_meta.get("winner", "ensemble"),
            "word_winners": word_winners,
            "errors": all_errors,
            "rotation_angle": best["angle"],
            "orientation_scores": {str(item["angle"]): round(float(item["score"]), 4) for item in orientation_results},
        }
        return fused.text, meta

    def _collect_candidates(self, variants: PageImageVariants) -> tuple[list[OCRCandidate], dict[str, str]]:
        candidates: list[OCRCandidate] = []
        backend_errors: dict[str, str] = {}
        for backend in self.backends:
            candidate = backend.infer(variants)
            if candidate:
                candidates.append(candidate)
            elif backend._failed_reason:
                backend_errors[backend.name] = backend._failed_reason
        return candidates, backend_errors

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
