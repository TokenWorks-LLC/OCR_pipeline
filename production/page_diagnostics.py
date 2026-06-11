#!/usr/bin/env python3
"""Page diagnostics for multilingual OCR decision support.

This module provides language-agnostic per-page signals used to decide whether
text-layer extraction is trustworthy and what OCR strategy/preprocessing profile
is likely best for a page.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any

import fitz

from production.preprocessing_profiles import (
    PROFILE_CLEAN_SCAN,
    PROFILE_COMPLEX_ACADEMIC_PAGE,
    PROFILE_FADED_PAGE,
    PROFILE_NOISY_SCAN,
    PROFILE_TRANSLITERATION_DIACRITIC,
    PROFILE_UNKNOWN_SAFE_DEFAULT,
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


CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def _normalize_whitespace(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in (text or "").splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _to_bool_string(value: bool) -> str:
    return "true" if value else "false"


@dataclass
class PageDiagnostics:
    page_number: int
    input_file: str
    width: float
    height: float
    dpi: int
    render_scale: float
    is_born_digital: bool
    text_layer_char_count: int
    text_layer_word_count: int
    text_density: float
    foreground_ratio: float
    estimated_skew_degrees: float | None
    blur_score: float
    contrast_score: float
    noise_score: float
    connected_component_count: int
    estimated_column_count: int
    has_large_images: bool
    has_tables_estimate: bool
    layout_complexity_score: float
    recommended_preprocessing_profile: str
    recommended_ocr_strategy: str
    language_hint: str = "unknown"
    page_image_rendered: bool = True
    render_failure_reason: str = ""
    page_has_usable_text_layer: bool = False
    text_layer_usable: bool = False
    text_layer_accepted: bool = False
    text_layer_rejected_reason: str = ""
    text_layer_quality_score: float = 0.0
    text_layer_suspicious_patterns: list[str] = field(default_factory=list)
    text_layer_suspicious_reasons: list[str] = field(default_factory=list)
    is_mostly_blank: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.estimated_skew_degrees is None:
            data["estimated_skew_degrees"] = None
        else:
            data["estimated_skew_degrees"] = round(float(self.estimated_skew_degrees), 4)

        for field_name in (
            "text_density",
            "foreground_ratio",
            "blur_score",
            "contrast_score",
            "noise_score",
            "layout_complexity_score",
            "text_layer_quality_score",
            "width",
            "height",
            "render_scale",
        ):
            data[field_name] = round(float(data[field_name]), 6)
        return data

    def to_pipeline_metadata(self) -> dict[str, Any]:
        data = self.to_dict()
        data["text_layer_usable"] = _to_bool_string(self.text_layer_usable)
        data["text_layer_accepted"] = _to_bool_string(self.text_layer_accepted)
        data["page_has_usable_text_layer"] = _to_bool_string(self.page_has_usable_text_layer)
        data["page_image_rendered"] = _to_bool_string(self.page_image_rendered)
        data["is_born_digital"] = _to_bool_string(self.is_born_digital)
        data["is_mostly_blank"] = _to_bool_string(self.is_mostly_blank)
        data["has_large_images"] = _to_bool_string(self.has_large_images)
        data["has_tables_estimate"] = _to_bool_string(self.has_tables_estimate)
        data["text_layer_suspicious_reasons"] = "|".join(self.text_layer_suspicious_reasons)
        data["text_layer_suspicious_patterns"] = "|".join(self.text_layer_suspicious_patterns)
        return data


class PageDiagnosticsAnalyzer:
    """Compute per-page diagnostics and extraction strategy recommendations."""

    def __init__(self, dpi: int = 300):
        self.dpi = int(dpi) if dpi else 300
        self.render_scale = float(self.dpi) / 72.0

    @classmethod
    def from_profile(cls, profile_path: str | None, default_dpi: int = 300) -> "PageDiagnosticsAnalyzer":
        dpi = int(default_dpi)
        if profile_path and Path(profile_path).exists():
            try:
                with Path(profile_path).open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                rendering = payload.get("rendering", {}) if isinstance(payload, dict) else {}
                dpi = int(rendering.get("dpi", dpi))
            except Exception:
                dpi = int(default_dpi)
        return cls(dpi=dpi)

    def inspect_page(
        self,
        pdf_path: str,
        page_num: int,
        language_hint: str = "unknown",
    ) -> tuple[PageDiagnostics, str]:
        with fitz.open(pdf_path) as document:
            if page_num >= len(document):
                return self._empty_diagnostics(pdf_path, page_num, language_hint), ""

            page = document[page_num]
            width = float(page.rect.width)
            height = float(page.rect.height)

            text_layer_raw = page.get_text("text") or ""
            text_layer_text = _normalize_whitespace(text_layer_raw)
            text_char_count = len(text_layer_text)
            text_word_count = len(WORD_RE.findall(text_layer_text))
            area_mp = max((width * height) / 1_000_000.0, 1e-6)
            text_density = float(text_char_count) / area_mp

            text_layer_usable, suspicious_reasons, text_layer_quality_score = self._assess_text_layer_quality(
                text_layer_text,
                text_char_count,
                text_word_count,
            )

            has_large_images = self._has_large_images(page)

            page_image_rendered = True
            render_failure_reason = ""
            image_stats: dict[str, Any]
            try:
                pixmap = page.get_pixmap(matrix=fitz.Matrix(self.render_scale, self.render_scale), alpha=False)
                image_stats = self._image_stats_from_pixmap(pixmap)
            except Exception as render_exc:
                page_image_rendered = False
                render_failure_reason = f"{type(render_exc).__name__}:{render_exc}"
                image_stats = {
                    "foreground_ratio": 0.0,
                    "estimated_skew_degrees": None,
                    "blur_score": 0.0,
                    "contrast_score": 0.0,
                    "noise_score": 0.0,
                    "connected_component_count": 0,
                    "estimated_column_count": 1,
                    "has_tables_estimate": False,
                    "layout_complexity_score": 0.0,
                }

        if page_image_rendered and image_stats["foreground_ratio"] >= 0.04 and text_char_count < 12:
            suspicious_reasons.append("text_density_mismatch")
        if not page_image_rendered:
            suspicious_reasons.append("render_failure")

        # Deduplicate while preserving order for reproducible diagnostics.
        suspicious_reasons = list(dict.fromkeys(suspicious_reasons))
        text_layer_quality_score = self._score_text_layer_quality(suspicious_reasons)
        hard_reject_reasons = {"broken_unicode", "repeated_junk", "mostly_symbols", "render_failure"}
        text_layer_usable = bool(
            text_char_count >= 16
            and text_layer_quality_score >= 0.62
            and not any(reason in hard_reject_reasons for reason in suspicious_reasons)
        )

        text_layer_accepted = bool(text_layer_usable)
        text_layer_rejected_reason = "" if text_layer_accepted else (suspicious_reasons[0] if suspicious_reasons else "insufficient_text")
        page_has_usable_text_layer = bool(text_layer_usable and text_char_count > 0)

        is_mostly_blank = bool(
            image_stats["foreground_ratio"] < 0.01
            or (text_char_count < 8 and image_stats["connected_component_count"] < 8)
        )

        # Language/domain hints are attached only as metadata and do not alter
        # diagnostics feature extraction.
        language_hint_value = (language_hint or "unknown").strip() or "unknown"

        is_born_digital = bool(text_layer_usable and text_char_count >= 24 and not has_large_images)

        recommended_profile = self._recommended_preprocessing_profile(
            skew=image_stats["estimated_skew_degrees"],
            contrast=image_stats["contrast_score"],
            noise=image_stats["noise_score"],
            columns=image_stats["estimated_column_count"],
            has_tables=image_stats["has_tables_estimate"],
            layout_complexity=image_stats["layout_complexity_score"],
            language_hint=language_hint_value,
        )
        recommended_strategy = self._recommended_ocr_strategy(
            text_layer_usable=text_layer_usable,
            text_char_count=text_char_count,
            is_mostly_blank=is_mostly_blank,
            columns=image_stats["estimated_column_count"],
            has_tables=image_stats["has_tables_estimate"],
            layout_complexity=image_stats["layout_complexity_score"],
        )

        diagnostics = PageDiagnostics(
            page_number=page_num + 1,
            input_file=Path(pdf_path).name,
            width=width,
            height=height,
            dpi=self.dpi,
            render_scale=self.render_scale,
            is_born_digital=is_born_digital,
            text_layer_char_count=text_char_count,
            text_layer_word_count=text_word_count,
            text_density=text_density,
            foreground_ratio=image_stats["foreground_ratio"],
            estimated_skew_degrees=image_stats["estimated_skew_degrees"],
            blur_score=image_stats["blur_score"],
            contrast_score=image_stats["contrast_score"],
            noise_score=image_stats["noise_score"],
            connected_component_count=image_stats["connected_component_count"],
            estimated_column_count=image_stats["estimated_column_count"],
            has_large_images=has_large_images,
            has_tables_estimate=image_stats["has_tables_estimate"],
            layout_complexity_score=image_stats["layout_complexity_score"],
            recommended_preprocessing_profile=recommended_profile,
            recommended_ocr_strategy=recommended_strategy,
            language_hint=language_hint_value,
            page_image_rendered=page_image_rendered,
            render_failure_reason=render_failure_reason,
            page_has_usable_text_layer=page_has_usable_text_layer,
            text_layer_usable=text_layer_usable,
            text_layer_accepted=text_layer_accepted,
            text_layer_rejected_reason=text_layer_rejected_reason,
            text_layer_quality_score=text_layer_quality_score,
            text_layer_suspicious_patterns=suspicious_reasons,
            text_layer_suspicious_reasons=suspicious_reasons,
            is_mostly_blank=is_mostly_blank,
        )
        return diagnostics, text_layer_text

    def _empty_diagnostics(self, pdf_path: str, page_num: int, language_hint: str) -> PageDiagnostics:
        return PageDiagnostics(
            page_number=page_num + 1,
            input_file=Path(pdf_path).name,
            width=0.0,
            height=0.0,
            dpi=self.dpi,
            render_scale=self.render_scale,
            is_born_digital=False,
            text_layer_char_count=0,
            text_layer_word_count=0,
            text_density=0.0,
            foreground_ratio=0.0,
            estimated_skew_degrees=None,
            blur_score=0.0,
            contrast_score=0.0,
            noise_score=0.0,
            connected_component_count=0,
            estimated_column_count=1,
            has_large_images=False,
            has_tables_estimate=False,
            layout_complexity_score=0.0,
            recommended_preprocessing_profile=PROFILE_UNKNOWN_SAFE_DEFAULT,
            recommended_ocr_strategy="ocr_standard",
            language_hint=(language_hint or "unknown").strip() or "unknown",
            page_image_rendered=False,
            render_failure_reason="page_out_of_range",
            page_has_usable_text_layer=False,
            text_layer_usable=False,
            text_layer_accepted=False,
            text_layer_rejected_reason="page_out_of_range",
            text_layer_quality_score=0.0,
            text_layer_suspicious_patterns=["page_out_of_range"],
            text_layer_suspicious_reasons=["page_out_of_range"],
            is_mostly_blank=True,
        )

    def _has_large_images(self, page: fitz.Page) -> bool:
        page_area = max(float(page.rect.width * page.rect.height), 1.0)
        try:
            images = page.get_images(full=True)
        except Exception:
            return False

        for image_info in images:
            xref = int(image_info[0])
            try:
                rects = page.get_image_rects(xref)
            except Exception:
                continue

            for rect in rects:
                ratio = (float(rect.width) * float(rect.height)) / page_area
                if ratio >= 0.35:
                    return True
        return False

    def _assess_text_layer_quality(
        self,
        text: str,
        char_count: int,
        word_count: int,
    ) -> tuple[bool, list[str], float]:
        reasons: list[str] = []
        if char_count < 24:
            reasons.append("too_few_characters")

        if self._looks_repeated_junk(text):
            reasons.append("repeated_junk")

        if self._looks_broken_unicode(text):
            reasons.append("broken_unicode")

        if self._mostly_symbols(text):
            reasons.append("mostly_symbols")

        compact = "".join(ch for ch in text if not ch.isspace())
        if len(compact) >= 24:
            unique_ratio = len(set(compact)) / max(len(compact), 1)
            if unique_ratio < 0.08:
                reasons.append("low_character_diversity")

        whitespace_count = sum(1 for ch in text if ch.isspace())
        if char_count >= 40 and whitespace_count < 2 and word_count <= 2:
            reasons.append("missing_whitespace")

        quality_score = self._score_text_layer_quality(reasons)
        usable = char_count >= 16 and quality_score >= 0.62 and not any(
            reason in {"broken_unicode", "repeated_junk", "mostly_symbols"} for reason in reasons
        )
        return usable, reasons, quality_score

    @staticmethod
    def _score_text_layer_quality(reasons: list[str]) -> float:
        penalties = {
            "too_few_characters": 0.18,
            "repeated_junk": 0.30,
            "broken_unicode": 0.35,
            "mostly_symbols": 0.28,
            "missing_whitespace": 0.12,
            "low_character_diversity": 0.14,
            "text_density_mismatch": 0.10,
            "render_failure": 0.55,
        }
        score = 1.0
        for reason in reasons:
            score -= penalties.get(reason, 0.08)
        return max(0.0, min(score, 1.0))

    @staticmethod
    def _mostly_symbols(text: str) -> bool:
        stripped = "".join(ch for ch in text if not ch.isspace())
        if len(stripped) < 12:
            return False
        alnum_ratio = sum(1 for ch in stripped if ch.isalnum()) / max(len(stripped), 1)
        return alnum_ratio < 0.25

    @staticmethod
    def _looks_broken_unicode(text: str) -> bool:
        if not text:
            return False
        if "\ufffd" in text:
            return True
        control_count = len(CONTROL_CHAR_RE.findall(text))
        return (control_count / max(len(text), 1)) > 0.01

    @staticmethod
    def _looks_repeated_junk(text: str) -> bool:
        compact = "".join(ch for ch in text if not ch.isspace())
        if len(compact) < 20:
            return False

        counts: dict[str, int] = {}
        for char in compact:
            counts[char] = counts.get(char, 0) + 1
        max_ratio = max(counts.values()) / max(len(compact), 1)
        if max_ratio >= 0.35:
            return True

        tokens = [token for token in re.split(r"\s+", text) if token]
        if len(tokens) < 6:
            return False
        most_common = max(tokens.count(token) for token in set(tokens))
        return (most_common / len(tokens)) >= 0.55

    def _image_stats_from_pixmap(self, pixmap: fitz.Pixmap) -> dict[str, Any]:
        if not NUMPY_AVAILABLE:
            return {
                "foreground_ratio": 0.0,
                "estimated_skew_degrees": None,
                "blur_score": 0.0,
                "contrast_score": 0.0,
                "noise_score": 0.0,
                "connected_component_count": 0,
                "estimated_column_count": 1,
                "has_tables_estimate": False,
                "layout_complexity_score": 0.0,
            }

        array = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
        if array.ndim == 3 and array.shape[2] >= 3:
            if CV2_AVAILABLE:
                gray = cv2.cvtColor(array[:, :, :3], cv2.COLOR_RGB2GRAY)
            else:
                gray = np.mean(array[:, :, :3], axis=2).astype(np.uint8)
        else:
            gray = array.astype(np.uint8)

        contrast_score = float(np.std(gray) / 255.0)

        if CV2_AVAILABLE:
            blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            median = cv2.medianBlur(gray, 3)
            noise_score = float(np.mean(np.abs(gray.astype(np.float32) - median.astype(np.float32))) / 255.0)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresholded = binary
        else:
            diff = np.diff(gray.astype(np.float32), axis=0)
            blur_score = float(np.var(diff))
            noise_score = float(np.std(gray.astype(np.float32) - np.mean(gray)) / 255.0)
            threshold = int(np.mean(gray))
            thresholded = np.where(gray > threshold, 255, 0).astype(np.uint8)

        foreground = thresholded < 200
        foreground_ratio = float(np.mean(foreground))

        connected_components = 0
        if CV2_AVAILABLE:
            components = cv2.connectedComponents((foreground.astype(np.uint8) * 255), connectivity=8)
            connected_components = max(int(components[0]) - 1, 0)

        skew = self._estimate_skew_from_foreground(foreground)
        column_count = self._estimate_columns(foreground)
        has_tables = self._estimate_tables(foreground)
        layout_complexity = self._layout_complexity(
            column_count=column_count,
            connected_components=connected_components,
            noise_score=noise_score,
            has_tables=has_tables,
            foreground_ratio=foreground_ratio,
        )

        return {
            "foreground_ratio": foreground_ratio,
            "estimated_skew_degrees": skew,
            "blur_score": blur_score,
            "contrast_score": contrast_score,
            "noise_score": noise_score,
            "connected_component_count": connected_components,
            "estimated_column_count": column_count,
            "has_tables_estimate": has_tables,
            "layout_complexity_score": layout_complexity,
        }

    def _estimate_skew_from_foreground(self, foreground: Any) -> float | None:
        if not (CV2_AVAILABLE and NUMPY_AVAILABLE):
            return None

        points = np.column_stack(np.where(foreground))
        if points.shape[0] < 200:
            return None

        rect = cv2.minAreaRect(points.astype(np.float32))
        angle = float(rect[-1])
        if angle < -45.0:
            skew = -(90.0 + angle)
        else:
            skew = -angle
        if not np.isfinite(skew):
            return None
        if abs(skew) > 45.0:
            return None
        return float(skew)

    @staticmethod
    def _estimate_columns(foreground: Any) -> int:
        if not NUMPY_AVAILABLE or foreground.size == 0:
            return 1

        vertical_density = foreground.mean(axis=0)
        width = vertical_density.shape[0]
        if width < 64:
            return 1

        kernel = max(5, width // 120)
        kernel_vec = np.ones(kernel, dtype=np.float32) / float(kernel)
        smooth = np.convolve(vertical_density, kernel_vec, mode="same")
        valley_threshold = min(0.08, float(np.percentile(smooth, 25)) + 0.01)
        valleys = smooth <= valley_threshold

        run_lengths: list[int] = []
        run = 0
        for value in valleys:
            if value:
                run += 1
            elif run:
                run_lengths.append(run)
                run = 0
        if run:
            run_lengths.append(run)

        significant_gaps = [length for length in run_lengths if length >= max(8, width // 28)]
        if not significant_gaps:
            return 1

        # One major inter-column gap implies two columns. More major gaps imply
        # additional columns, capped for stability.
        return min(1 + len(significant_gaps), 4)

    def _estimate_tables(self, foreground: Any) -> bool:
        if not (CV2_AVAILABLE and NUMPY_AVAILABLE):
            return False

        binary = (foreground.astype(np.uint8) * 255)
        width = binary.shape[1]
        height = binary.shape[0]
        if width < 120 or height < 120:
            return False

        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, width // 18), 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, height // 18)))

        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

        h_ratio = float(np.mean(h_lines > 0))
        v_ratio = float(np.mean(v_lines > 0))
        return (h_ratio > 0.015 and v_ratio > 0.01) or (h_ratio + v_ratio) > 0.04

    @staticmethod
    def _layout_complexity(
        column_count: int,
        connected_components: int,
        noise_score: float,
        has_tables: bool,
        foreground_ratio: float,
    ) -> float:
        column_component = min(max(column_count - 1, 0) / 3.0, 1.0)
        component_density = min(connected_components / 500.0, 1.0)
        noise_component = min(max(noise_score, 0.0) * 4.0, 1.0)
        table_component = 1.0 if has_tables else 0.0
        fg_component = min(max(foreground_ratio, 0.0) * 2.0, 1.0)

        score = (
            (0.30 * column_component)
            + (0.25 * table_component)
            + (0.20 * component_density)
            + (0.15 * noise_component)
            + (0.10 * fg_component)
        )
        return float(max(0.0, min(score, 1.0)))

    @staticmethod
    def _recommended_preprocessing_profile(
        skew: float | None,
        contrast: float,
        noise: float,
        columns: int,
        has_tables: bool,
        layout_complexity: float,
        language_hint: str,
    ) -> str:
        hint = (language_hint or "unknown").strip().lower()
        if any(token in hint for token in ("akkadian", "transliteration", "diacritic")):
            return PROFILE_TRANSLITERATION_DIACRITIC

        if columns > 1 or has_tables or layout_complexity >= 0.55:
            return PROFILE_COMPLEX_ACADEMIC_PAGE
        if skew is not None and abs(skew) >= 2.0:
            return PROFILE_NOISY_SCAN
        if contrast < 0.10:
            return PROFILE_FADED_PAGE
        if noise > 0.12:
            return PROFILE_NOISY_SCAN
        if contrast >= 0.16 and noise <= 0.08 and (skew is None or abs(skew) < 1.0):
            return PROFILE_CLEAN_SCAN
        return PROFILE_UNKNOWN_SAFE_DEFAULT

    @staticmethod
    def _recommended_ocr_strategy(
        text_layer_usable: bool,
        text_char_count: int,
        is_mostly_blank: bool,
        columns: int,
        has_tables: bool,
        layout_complexity: float,
    ) -> str:
        if text_layer_usable and text_char_count >= 24:
            return "text_layer"
        if is_mostly_blank:
            return "skip_or_light_ocr"
        if columns > 1 or has_tables or layout_complexity >= 0.55:
            return "ocr_ensemble_layout"
        return "ocr_standard"
