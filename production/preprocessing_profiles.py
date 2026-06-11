#!/usr/bin/env python3
"""Adaptive preprocessing profiles for multilingual OCR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


PROFILE_CLEAN_SCAN = "clean_scan"
PROFILE_NOISY_SCAN = "noisy_scan"
PROFILE_FADED_PAGE = "faded_page"
PROFILE_COMPLEX_ACADEMIC_PAGE = "complex_academic_page"
PROFILE_TRANSLITERATION_DIACRITIC = "transliteration_or_diacritic_heavy"
PROFILE_UNKNOWN_SAFE_DEFAULT = "unknown_safe_default"
PROFILE_AUTO = "auto"


@dataclass(frozen=True)
class PreprocessingProfile:
    name: str
    render_dpi: int
    preprocessing_overrides: dict[str, Any]
    description: str


PREPROCESSING_PROFILES: dict[str, PreprocessingProfile] = {
    PROFILE_CLEAN_SCAN: PreprocessingProfile(
        name=PROFILE_CLEAN_SCAN,
        render_dpi=300,
        preprocessing_overrides={
            "enable_denoise": False,
            "enable_adaptive_threshold": False,
            "enable_morphology": False,
            "contrast_factor": 1.15,
            "sharpen_radius": 1.0,
            "sharpen_percent": 120,
            "deskew": False,
            "enable_background_normalization": False,
            "avoid_aggressive_binarization": True,
            "preserve_diacritics": False,
            "profile_primary_variant": "contrast",
            "profile_variant_order": ["contrast", "sharpen", "original", "autocontrast"],
        },
        description="Light-touch profile for clean scans and born-digital pages.",
    ),
    PROFILE_NOISY_SCAN: PreprocessingProfile(
        name=PROFILE_NOISY_SCAN,
        render_dpi=320,
        preprocessing_overrides={
            "enable_denoise": True,
            "denoise_strength": 16,
            "enable_adaptive_threshold": True,
            "adaptive_block_size": 35,
            "adaptive_C": 9,
            "enable_morphology": True,
            "morphology_kernel_size": 2,
            "deskew": True,
            "deskew_min_abs_degrees": 0.2,
            "deskew_max_abs_degrees": 45.0,
            "speckle_removal": True,
            "contrast_factor": 1.8,
            "profile_primary_variant": "adaptive",
            "profile_variant_order": ["adaptive", "morphology", "contrast", "sharpen", "original"],
        },
        description="Noise-robust profile with denoising, adaptive thresholding, deskew, and speckle cleanup.",
    ),
    PROFILE_FADED_PAGE: PreprocessingProfile(
        name=PROFILE_FADED_PAGE,
        render_dpi=360,
        preprocessing_overrides={
            "enable_background_normalization": True,
            "enable_denoise": True,
            "denoise_strength": 10,
            "enable_adaptive_threshold": True,
            "adaptive_block_size": 41,
            "adaptive_C": 8,
            "enable_morphology": False,
            "deskew": True,
            "contrast_factor": 2.1,
            "avoid_aggressive_binarization": False,
            "profile_primary_variant": "adaptive",
            "profile_variant_order": ["adaptive", "contrast", "autocontrast", "sharpen", "original"],
        },
        description="Contrast and background normalization profile for faded or uneven pages.",
    ),
    PROFILE_COMPLEX_ACADEMIC_PAGE: PreprocessingProfile(
        name=PROFILE_COMPLEX_ACADEMIC_PAGE,
        render_dpi=320,
        preprocessing_overrides={
            "enable_denoise": False,
            "enable_adaptive_threshold": False,
            "enable_morphology": False,
            "deskew": True,
            "deskew_min_abs_degrees": 0.2,
            "deskew_max_abs_degrees": 20.0,
            "contrast_factor": 1.25,
            "preserve_layout": True,
            "avoid_aggressive_binarization": True,
            "profile_primary_variant": "autocontrast",
            "profile_variant_order": ["autocontrast", "contrast", "sharpen", "grayscale", "original"],
        },
        description="Layout-preserving profile for multi-column academic pages with footnotes/captions.",
    ),
    PROFILE_TRANSLITERATION_DIACRITIC: PreprocessingProfile(
        name=PROFILE_TRANSLITERATION_DIACRITIC,
        render_dpi=420,
        preprocessing_overrides={
            "enable_denoise": False,
            "enable_adaptive_threshold": False,
            "enable_morphology": False,
            "deskew": True,
            "deskew_min_abs_degrees": 0.2,
            "deskew_max_abs_degrees": 20.0,
            "preserve_diacritics": True,
            "avoid_aggressive_binarization": True,
            "contrast_factor": 1.35,
            "sharpen_radius": 1.0,
            "sharpen_percent": 130,
            "profile_primary_variant": "sharpen",
            "profile_variant_order": ["sharpen", "contrast", "grayscale", "original"],
        },
        description="High-DPI conservative profile preserving diacritics and tiny transliteration marks.",
    ),
    PROFILE_UNKNOWN_SAFE_DEFAULT: PreprocessingProfile(
        name=PROFILE_UNKNOWN_SAFE_DEFAULT,
        render_dpi=300,
        preprocessing_overrides={
            "enable_denoise": True,
            "denoise_strength": 8,
            "enable_adaptive_threshold": True,
            "adaptive_block_size": 35,
            "adaptive_C": 11,
            "enable_morphology": False,
            "deskew": True,
            "deskew_min_abs_degrees": 0.2,
            "deskew_max_abs_degrees": 45.0,
            "contrast_factor": 1.45,
            "avoid_aggressive_binarization": True,
            "preserve_diacritics": False,
            "profile_primary_variant": "contrast",
            "profile_variant_order": ["contrast", "adaptive", "autocontrast", "sharpen", "original"],
        },
        description="Conservative multilingual-safe default preserving maximal signal.",
    ),
}


def available_preprocessing_profiles() -> tuple[str, ...]:
    return tuple(PREPROCESSING_PROFILES.keys())


def get_preprocessing_profile(name: str | None) -> PreprocessingProfile:
    normalized = (name or "").strip().lower()
    if normalized in PREPROCESSING_PROFILES:
        return PREPROCESSING_PROFILES[normalized]
    return PREPROCESSING_PROFILES[PROFILE_UNKNOWN_SAFE_DEFAULT]


def resolve_preprocessing_profile(
    diagnostics: dict[str, Any] | None = None,
    language_hint: str = "unknown",
    requested_profile: str = PROFILE_AUTO,
) -> str:
    """Select profile from diagnostics + optional requested override."""
    requested = (requested_profile or PROFILE_AUTO).strip().lower()
    if requested and requested != PROFILE_AUTO:
        if requested in PREPROCESSING_PROFILES:
            return requested
        return PROFILE_UNKNOWN_SAFE_DEFAULT

    data = diagnostics or {}
    recommended = str(data.get("recommended_preprocessing_profile", "")).strip().lower()
    if recommended in PREPROCESSING_PROFILES:
        return recommended

    hint = (language_hint or str(data.get("language_hint", "unknown"))).strip().lower()
    if any(token in hint for token in ("akkadian", "transliteration", "diacritic")):
        return PROFILE_TRANSLITERATION_DIACRITIC

    skew = _safe_float(data.get("estimated_skew_degrees"))
    contrast = _safe_float(data.get("contrast_score"))
    noise = _safe_float(data.get("noise_score"))
    columns = _safe_int(data.get("estimated_column_count"))
    has_tables = _safe_bool(data.get("has_tables_estimate"))
    layout_complexity = _safe_float(data.get("layout_complexity_score"))

    if columns > 1 or has_tables or layout_complexity >= 0.55:
        return PROFILE_COMPLEX_ACADEMIC_PAGE
    if abs(skew) >= 2.0 or noise >= 0.12:
        return PROFILE_NOISY_SCAN
    if contrast <= 0.10:
        return PROFILE_FADED_PAGE
    if contrast >= 0.16 and noise <= 0.08 and abs(skew) < 1.0:
        return PROFILE_CLEAN_SCAN
    return PROFILE_UNKNOWN_SAFE_DEFAULT


def merge_profile_with_base(
    base_preprocessing_config: dict[str, Any],
    profile_name: str,
) -> tuple[dict[str, Any], int]:
    profile = get_preprocessing_profile(profile_name)
    merged = {**base_preprocessing_config, **profile.preprocessing_overrides}
    return merged, int(profile.render_dpi)


def _safe_float(value: Any) -> float:
    try:
        if value in (None, ""):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        if value in (None, ""):
            return 0
        return int(float(value))
    except Exception:
        return 0


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "t"}
