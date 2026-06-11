from __future__ import annotations

import json
from pathlib import Path

import fitz
from PIL import Image

from production.ensemble_ocr import DEFAULT_PREPROCESSING, FortifiedOCREnsemble, OCRCandidate, PageImageVariants
from production.layout_analysis import LayoutPageResult, LayoutRegion
from production.page_diagnostics import PageDiagnosticsAnalyzer
from production.preprocessing_profiles import (
    PROFILE_AUTO,
    PROFILE_TRANSLITERATION_DIACRITIC,
    PROFILE_UNKNOWN_SAFE_DEFAULT,
    available_preprocessing_profiles,
    merge_profile_with_base,
    resolve_preprocessing_profile,
)


def _write_pdf_with_text(pdf_path: Path, text: str, width: int = 300, height: int = 200) -> None:
    doc = fitz.open()
    page = doc.new_page(width=width, height=height)
    page.insert_text((20, 40), text, fontsize=12)
    doc.save(str(pdf_path))
    doc.close()


def _write_blank_pdf(pdf_path: Path, width: int = 300, height: int = 200) -> None:
    doc = fitz.open()
    doc.new_page(width=width, height=height)
    doc.save(str(pdf_path))
    doc.close()


def _write_lightweight_profile(profile_path: Path) -> None:
    payload = {
        "rendering": {"dpi": 300},
        "engines": {"enabled": []},
    }
    profile_path.write_text(json.dumps(payload), encoding="utf-8")


def test_profile_selector_respects_language_hint_for_transliteration():
    selected = resolve_preprocessing_profile(
        diagnostics={
            "estimated_skew_degrees": 0.0,
            "contrast_score": 0.2,
            "noise_score": 0.01,
            "estimated_column_count": 1,
        },
        language_hint="Akkadian transliteration",
        requested_profile=PROFILE_AUTO,
    )
    assert selected == PROFILE_TRANSLITERATION_DIACRITIC


def test_page_variants_build_for_all_named_profiles():
    img = Image.new("RGB", (200, 120), color=(245, 245, 245))
    for profile in available_preprocessing_profiles():
        config, _ = merge_profile_with_base(DEFAULT_PREPROCESSING, profile)
        variants = PageImageVariants(img, preprocessing_config=config)
        names = set(variants.variant_names())
        assert "original" in names
        assert "contrast" in names
        assert "adaptive" in names
        assert "morphology" in names


def test_ensemble_writes_profile_debug_artifacts(tmp_path: Path):
    pdf_path = tmp_path / "fixture.pdf"
    _write_pdf_with_text(pdf_path, "debug artifact fixture")
    profile_path = tmp_path / "light_profile.json"
    _write_lightweight_profile(profile_path)

    class StubBackend:
        name = "stub"
        _failed_reason = None

        def infer(self, variants):
            return OCRCandidate(
                "stub",
                "Recovered text",
                confidence=0.93,
                variant="contrast",
                meta={"rotation_angle": variants.rotation_angle, "line_count": 1},
            )

    ensemble = FortifiedOCREnsemble(profile_path=str(profile_path))
    ensemble.backends = [StubBackend()]
    ensemble.rotation_search = (0,)

    debug_dir = tmp_path / "debug"
    text, meta = ensemble.extract_page_text(
        str(pdf_path),
        0,
        preprocessing_profile=PROFILE_UNKNOWN_SAFE_DEFAULT,
        diagnostics={"recommended_preprocessing_profile": PROFILE_UNKNOWN_SAFE_DEFAULT},
        debug_artifacts_dir=str(debug_dir),
        debug_artifact_prefix="fixture_p0001",
    )

    assert text == "Recovered text"
    assert meta.get("preprocessing_profile") == PROFILE_UNKNOWN_SAFE_DEFAULT
    produced = {path.name for path in debug_dir.iterdir()}
    assert any(name.endswith("_original.png") for name in produced)
    assert any(name.endswith("_preprocessed.png") for name in produced)
    assert any(name.endswith("_ocr.txt") for name in produced)
    assert any(name.endswith("_metadata.json") for name in produced)


def test_no_crash_on_blank_page_with_profile(tmp_path: Path):
    pdf_path = tmp_path / "blank.pdf"
    _write_blank_pdf(pdf_path)
    profile_path = tmp_path / "light_profile.json"
    _write_lightweight_profile(profile_path)

    class EmptyBackend:
        name = "stub"
        _failed_reason = None

        def infer(self, variants):
            return None

    ensemble = FortifiedOCREnsemble(profile_path=str(profile_path))
    ensemble.backends = [EmptyBackend()]
    ensemble.rotation_search = (0,)

    text, meta = ensemble.extract_page_text(
        str(pdf_path),
        0,
        preprocessing_profile=PROFILE_UNKNOWN_SAFE_DEFAULT,
        diagnostics={"recommended_preprocessing_profile": PROFILE_UNKNOWN_SAFE_DEFAULT},
    )

    assert text == ""
    assert meta.get("failure_reason") in {"no_candidate_text", "empty_fused_text"}


def test_no_crash_on_low_resolution_page(tmp_path: Path):
    pdf_path = tmp_path / "low_res.pdf"
    _write_pdf_with_text(pdf_path, "tiny", width=40, height=40)
    profile_path = tmp_path / "light_profile.json"
    _write_lightweight_profile(profile_path)

    class StubBackend:
        name = "stub"
        _failed_reason = None

        def infer(self, variants):
            return OCRCandidate(
                "stub",
                "tiny",
                confidence=0.5,
                variant="original",
                meta={"rotation_angle": variants.rotation_angle, "line_count": 1},
            )

    ensemble = FortifiedOCREnsemble(profile_path=str(profile_path))
    ensemble.backends = [StubBackend()]
    ensemble.rotation_search = (0,)

    text, meta = ensemble.extract_page_text(str(pdf_path), 0, preprocessing_profile=PROFILE_UNKNOWN_SAFE_DEFAULT)
    assert text == "tiny"
    assert meta.get("preprocessing_profile") == PROFILE_UNKNOWN_SAFE_DEFAULT


def test_no_crash_on_multilingual_unicode_signals(tmp_path: Path):
    pdf_path = tmp_path / "multi.pdf"
    _write_pdf_with_text(pdf_path, "šarrum über العربية 中文")

    analyzer = PageDiagnosticsAnalyzer(dpi=300)
    diagnostics, _ = analyzer.inspect_page(str(pdf_path), 0, language_hint="unknown")

    assert diagnostics.recommended_preprocessing_profile in set(available_preprocessing_profiles())


def test_region_level_ocr_returns_structured_layout(tmp_path: Path):
    pdf_path = tmp_path / "region_layout.pdf"
    _write_pdf_with_text(pdf_path, "layout region test", width=420, height=300)
    profile_path = tmp_path / "light_profile.json"
    _write_lightweight_profile(profile_path)

    class StubBackend:
        name = "stub"
        _failed_reason = None

        def infer(self, variants):
            return OCRCandidate(
                "stub",
                "Region OCR text",
                confidence=0.88,
                variant="original",
                meta={"rotation_angle": variants.rotation_angle, "line_count": 1},
            )

    class StubLayoutDetector:
        def analyze_page(self, page, page_number, language_hint="unknown"):
            regions = [
                LayoutRegion(region_id=f"p{page_number}_r1", type="paragraph", bbox=(20, 20, 180, 120), reading_order=1),
                LayoutRegion(region_id=f"p{page_number}_r2", type="paragraph", bbox=(220, 20, 390, 140), reading_order=2),
            ]
            return LayoutPageResult(
                page=page_number,
                page_size=(float(page.rect.width), float(page.rect.height)),
                regions=regions,
                column_count=2,
                column_mode="two_column",
                has_footnotes=False,
                has_table_interruptions=False,
                reading_order_confidence=0.91,
                ordering_source="inferred",
                text_direction="ltr",
            )

        def is_complex_layout(self, result):
            return True

        def regions_for_ocr(self, result):
            return sorted(result.regions, key=lambda item: item.reading_order)

        def save_debug_artifacts(self, pixmap, result, output_dir, prefix, region_outputs=None):
            return None

    ensemble = FortifiedOCREnsemble(profile_path=str(profile_path))
    ensemble.backends = [StubBackend()]
    ensemble.layout_detector = StubLayoutDetector()
    ensemble.rotation_search = (0,)
    ensemble.preprocessing_config["rotation_search_degrees"] = [0]

    text, meta = ensemble.extract_page_text(
        str(pdf_path),
        0,
        preprocessing_profile=PROFILE_UNKNOWN_SAFE_DEFAULT,
        diagnostics={"recommended_preprocessing_profile": "complex_academic_page"},
    )

    assert text.strip() != ""
    assert meta.get("region_ocr_used") is True
    structured = meta.get("structured_layout")
    assert isinstance(structured, dict)
    assert structured.get("regions")
    assert any("engine" in region for region in structured.get("regions", []))
