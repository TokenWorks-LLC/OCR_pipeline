from __future__ import annotations

import importlib.util
from pathlib import Path

import fitz

from production.ensemble_ocr import CuReDBackend, FortifiedOCREnsemble, OCRCandidate, OCRLine, PageLayoutAnalyzer, TextEnsembleFuser


ROOT = Path(__file__).resolve().parents[1]
PROTECTED_RUNNER = ROOT / ".merge_protect" / "tools" / "run_page_text.py"


def _load_protected_runner_module():
    spec = importlib.util.spec_from_file_location("_protected_run_page_text_test", PROTECTED_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load protected runner from {PROTECTED_RUNNER}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_text_ensemble_fuser_prefers_diacritic_preserving_consensus():
    fuser = TextEnsembleFuser({"paddle": 1.0, "doctr": 1.0, "mmocr": 1.0})
    fused = fuser.fuse(
        [
            OCRCandidate("paddle", "sarrum und uber", confidence=0.80),
            OCRCandidate("doctr", "šarrum und über", confidence=0.78),
            OCRCandidate("mmocr", "šarrum und über", confidence=0.77),
        ]
    )

    assert fused.engine == "ensemble"
    assert fused.text == "šarrum und über"


def test_text_ensemble_fuser_prefers_consensus_over_outlier():
    fuser = TextEnsembleFuser({"paddle": 1.0, "doctr": 1.0, "mmocr": 1.0})
    fused = fuser.fuse(
        [
            OCRCandidate("paddle", "The evidence is clear.", confidence=0.81),
            OCRCandidate("doctr", "The evidence is clear.", confidence=0.79),
            OCRCandidate("mmocr", "The evideuce 1s clear.", confidence=0.65),
        ]
    )

    assert fused.text == "The evidence is clear."


def test_layout_analyzer_reorders_two_columns_left_to_right():
    analyzer = PageLayoutAnalyzer()
    ordered = analyzer.reorder(
        [
            OCRLine("Right top", bbox=(310, 40, 470, 60)),
            OCRLine("Left top", bbox=(40, 45, 200, 65)),
            OCRLine("Right bottom", bbox=(312, 90, 475, 110)),
            OCRLine("Left bottom", bbox=(42, 92, 205, 112)),
        ],
        page_size=(520, 720),
    )

    assert [line.text for line in ordered] == ["Left top", "Left bottom", "Right top", "Right bottom"]


def test_fortified_ensemble_selects_best_rotation(tmp_path):
    pdf_path = tmp_path / "rotated.pdf"
    document = fitz.open()
    page = document.new_page(width=300, height=200)
    page.insert_text((72, 72), "rotation fixture", fontsize=12)
    document.save(str(pdf_path))
    document.close()

    class RotationAwareBackend:
        name = "stub"
        _failed_reason = None

        def infer(self, variants):
            if variants.rotation_angle == 90:
                return OCRCandidate(
                    "stub",
                    "Recovered upright text",
                    confidence=0.95,
                    variant="adaptive",
                    meta={"rotation_angle": variants.rotation_angle, "line_count": 1},
                )
            if variants.rotation_angle == 0:
                return OCRCandidate(
                    "stub",
                    "x",
                    confidence=0.05,
                    variant="original",
                    meta={"rotation_angle": variants.rotation_angle, "line_count": 1},
                )
            return None

    ensemble = FortifiedOCREnsemble()
    ensemble.backends = [RotationAwareBackend()]
    ensemble.rotation_search = (0, 90, 180, 270)

    text, meta = ensemble.extract_page_text(str(pdf_path), 0)

    assert text == "Recovered upright text"
    assert meta["rotation_angle"] == 90
    assert meta["orientation_scores"]["90"] > meta["orientation_scores"]["0"]


def test_pdf_text_extractor_uses_ensemble_when_text_layer_fails(monkeypatch):
    module = _load_protected_runner_module()
    extractor = module.PDFTextExtractor(
        prefer_text_layer=True,
        ocr_fallback="ensemble",
        profile_path=str(ROOT / "profiles" / "akkadian_strict.json"),
    )

    class StubEnsemble:
        def extract_page_text(self, pdf_path: str, page_num: int):
            assert pdf_path == "dummy.pdf"
            assert page_num == 0
            return "šar-ru-um", {"engines_used": ["paddle", "doctr"]}

    extractor.ensemble = StubEnsemble()
    monkeypatch.setattr(extractor, "_extract_text_layer", lambda pdf_path, page_num: ("", False))

    text, used_text_layer, meta = extractor.extract_page_text("dummy.pdf", 0)

    assert text == "šar-ru-um"
    assert used_text_layer is False
    assert meta["method"] == "ocr_ensemble"


def test_pdf_text_extractor_accepts_text_layer_when_strategy_allows(monkeypatch):
    module = _load_protected_runner_module()
    extractor = module.PDFTextExtractor(
        prefer_text_layer=True,
        ocr_fallback=None,
        profile_path=str(ROOT / "profiles" / "akkadian_strict.json"),
    )

    class StubDiagnostics:
        def to_pipeline_metadata(self):
            return {
                "text_layer_char_count": "120",
                "text_layer_word_count": "20",
                "text_layer_usable": "true",
                "text_layer_suspicious_reasons": "",
                "estimated_column_count": "1",
                "layout_complexity_score": "0.20",
                "contrast_score": "0.20",
                "noise_score": "0.04",
                "recommended_preprocessing_profile": "clean_scan",
                "language_hint": "unknown",
            }

    monkeypatch.setattr(
        extractor.diagnostics,
        "inspect_page",
        lambda pdf_path, page_num, language_hint="unknown": (StubDiagnostics(), "Text-layer content"),
    )

    text, used_text_layer, meta = extractor.extract_page_text("dummy.pdf", 0)

    assert text == "Text-layer content"
    assert used_text_layer is True
    assert meta["text_layer_acceptance"] == "accepted"
    assert meta["method"] == "text_layer"


def test_pdf_text_extractor_rejects_text_layer_and_uses_ocr_fallback(monkeypatch):
    module = _load_protected_runner_module()
    extractor = module.PDFTextExtractor(
        prefer_text_layer=True,
        ocr_fallback="ensemble",
        profile_path=str(ROOT / "profiles" / "akkadian_strict.json"),
    )

    class StubDiagnostics:
        def to_pipeline_metadata(self):
            return {
                "text_layer_char_count": "180",
                "text_layer_word_count": "30",
                "text_layer_usable": "true",
                "text_layer_suspicious_reasons": "broken_unicode",
                "estimated_column_count": "1",
                "layout_complexity_score": "0.20",
                "contrast_score": "0.20",
                "noise_score": "0.04",
                "recommended_preprocessing_profile": "clean_scan",
                "language_hint": "unknown",
            }

    class StubEnsemble:
        def extract_page_text(self, pdf_path: str, page_num: int):
            return "OCR fallback text", {"engines_used": ["paddle"], "final_output_source": "paddle"}

        def get_engine_performance_summary(self):
            return {}

    extractor.ensemble = StubEnsemble()
    monkeypatch.setattr(
        extractor.diagnostics,
        "inspect_page",
        lambda pdf_path, page_num, language_hint="unknown": (StubDiagnostics(), "Suspicious layer text"),
    )

    text, used_text_layer, meta = extractor.extract_page_text("dummy.pdf", 0)

    assert text == "OCR fallback text"
    assert used_text_layer is False
    assert str(meta["text_layer_acceptance"]).startswith("rejected")
    assert meta["method"] == "ocr_ensemble"


def test_pdf_text_extractor_force_ocr_overrides_text_layer(monkeypatch):
    module = _load_protected_runner_module()
    extractor = module.PDFTextExtractor(
        prefer_text_layer=True,
        ocr_fallback="ensemble",
        force_ocr=True,
        profile_path=str(ROOT / "profiles" / "akkadian_strict.json"),
    )

    class StubEnsemble:
        def extract_page_text(self, pdf_path: str, page_num: int):
            assert pdf_path == "dummy.pdf"
            assert page_num == 0
            return "ocr text", {"engines_used": ["paddle", "doctr"]}

    extractor.ensemble = StubEnsemble()
    monkeypatch.setattr(extractor, "_extract_text_layer", lambda pdf_path, page_num: ("layer text fallback", True))

    text, used_text_layer, meta = extractor.extract_page_text("dummy.pdf", 0)

    assert text == "ocr text"
    assert used_text_layer is False
    assert meta["method"] == "ocr_ensemble"


def test_pdf_text_extractor_force_ocr_falls_back_to_text_layer_when_ocr_fails(monkeypatch):
    module = _load_protected_runner_module()
    extractor = module.PDFTextExtractor(
        prefer_text_layer=True,
        ocr_fallback="ensemble",
        force_ocr=True,
        profile_path=str(ROOT / "profiles" / "akkadian_strict.json"),
    )

    class StubEnsemble:
        def extract_page_text(self, pdf_path: str, page_num: int):
            assert pdf_path == "dummy.pdf"
            assert page_num == 0
            return "", {"engines_used": []}

    extractor.ensemble = StubEnsemble()
    monkeypatch.setattr(extractor, "_extract_text_layer", lambda pdf_path, page_num: ("layer text fallback", True))

    text, used_text_layer, meta = extractor.extract_page_text("dummy.pdf", 0)

    assert text == "layer text fallback"
    assert used_text_layer is True
    assert meta["method"] == "text_layer"


def test_cured_backend_registered_in_ensemble():
    """CuReD must be a recognized backend so the ensemble can route to it."""
    assert "cured" in FortifiedOCREnsemble.BACKEND_TYPES
    assert FortifiedOCREnsemble.BACKEND_TYPES["cured"] is CuReDBackend


def test_cured_fusion_weight_higher_than_default():
    """akkadian_strict.json gives CuReD a 1.8x weight — verify profile loads it."""
    import json

    profile_path = ROOT / "profiles" / "akkadian_strict.json"
    with open(profile_path) as fh:
        profile = json.load(fh)

    weights = profile["fusion"]["weights"]
    assert "cured" in weights
    assert weights["cured"] > weights["paddle"]


def test_cured_backend_infer_produces_candidate(monkeypatch):
    """CuReD inference with mocked Kraken returns a valid OCRCandidate."""

    class FakeRecord:
        def __init__(self, prediction, confidence, bbox):
            self.prediction = prediction
            self.confidence = confidence
            self.bbox = bbox

    fake_predictions = [
        FakeRecord("KÙ.BABBAR 5 GÍN", 0.91, (10, 10, 200, 30)),
        FakeRecord("KISIB ša-lim-a-šur", 0.87, (10, 35, 200, 55)),
    ]

    # Stub out kraken imports so the test works without kraken installed
    import types
    kraken_mod = types.ModuleType("kraken")
    binarization_mod = types.ModuleType("kraken.binarization")
    pageseg_mod = types.ModuleType("kraken.pageseg")
    rpred_mod = types.ModuleType("kraken.rpred")
    kraken_lib_mod = types.ModuleType("kraken.lib")
    kraken_models_mod = types.ModuleType("kraken.lib.models")

    binarization_mod.nlbin = lambda img: img
    pageseg_mod.segment = lambda img: {}
    rpred_mod.rpred = lambda engine, binary, segments: fake_predictions
    kraken_models_mod.load_any = lambda path: "fake_engine"

    kraken_mod.binarization = binarization_mod
    kraken_mod.pageseg = pageseg_mod
    kraken_mod.rpred = rpred_mod
    kraken_mod.lib = kraken_lib_mod
    kraken_lib_mod.models = kraken_models_mod

    monkeypatch.setitem(__import__("sys").modules, "kraken", kraken_mod)
    monkeypatch.setitem(__import__("sys").modules, "kraken.binarization", binarization_mod)
    monkeypatch.setitem(__import__("sys").modules, "kraken.pageseg", pageseg_mod)
    monkeypatch.setitem(__import__("sys").modules, "kraken.rpred", rpred_mod)
    monkeypatch.setitem(__import__("sys").modules, "kraken.lib", kraken_lib_mod)
    monkeypatch.setitem(__import__("sys").modules, "kraken.lib.models", kraken_models_mod)

    backend = CuReDBackend(config={"cured_model_path": "/dev/null"})
    # Bypass model loading since we mocked kraken
    backend._engine = "fake_engine"

    from production.ensemble_ocr import PageImageVariants
    from PIL import Image

    img = Image.new("L", (400, 100), color=200)
    variants = PageImageVariants(img)

    candidate = backend._infer_variant(variants, "original")

    assert candidate is not None
    assert candidate.engine == "cured"
    assert "KÙ.BABBAR" in candidate.text
    assert "ša-lim-a-šur" in candidate.text
    assert len(candidate.lines) == 2
    assert candidate.confidence > 0.0


def test_cured_backend_returns_none_on_empty_predictions(monkeypatch):
    """CuReD returns None when Kraken produces no predictions."""

    import types
    kraken_mod = types.ModuleType("kraken")
    binarization_mod = types.ModuleType("kraken.binarization")
    pageseg_mod = types.ModuleType("kraken.pageseg")
    rpred_mod = types.ModuleType("kraken.rpred")

    binarization_mod.nlbin = lambda img: img
    pageseg_mod.segment = lambda img: {}
    rpred_mod.rpred = lambda engine, binary, segments: []

    monkeypatch.setitem(__import__("sys").modules, "kraken", kraken_mod)
    monkeypatch.setitem(__import__("sys").modules, "kraken.binarization", binarization_mod)
    monkeypatch.setitem(__import__("sys").modules, "kraken.pageseg", pageseg_mod)
    monkeypatch.setitem(__import__("sys").modules, "kraken.rpred", rpred_mod)

    backend = CuReDBackend(config={})
    backend._engine = "fake_engine"

    from production.ensemble_ocr import PageImageVariants
    from PIL import Image

    img = Image.new("L", (400, 100), color=200)
    variants = PageImageVariants(img)

    candidate = backend._infer_variant(variants, "original")
    assert candidate is None
