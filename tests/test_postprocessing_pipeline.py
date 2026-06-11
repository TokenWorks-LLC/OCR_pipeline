from __future__ import annotations

from production.postprocessing import GuardedModelCorrector, InMemoryLexicon, PostprocessingPipeline


def test_general_cleanup_preserves_structure_and_removes_obvious_garbage() -> None:
    pipeline = PostprocessingPipeline()
    result = pipeline.process(
        "  Foo\u00a0\u00a0bar   \n\n\n~~~~~~ //////\n(42) [abc] !?  ",
        language_hint="english",
    )

    assert "Foo bar" in result.cleaned_text
    assert "(42)" in result.cleaned_text
    assert "[abc]" in result.cleaned_text
    assert "!?" in result.cleaned_text
    assert "~~~~~~" not in result.cleaned_text
    assert "//////" not in result.cleaned_text


def test_unicode_normalization_composes_characters() -> None:
    pipeline = PostprocessingPipeline()
    result = pipeline.process("Cafe\u0301", language_hint="french")

    assert "\u00e9" in result.cleaned_text


def test_language_specific_adapters_are_selected_by_hint() -> None:
    pipeline = PostprocessingPipeline()

    english = pipeline.process("This is evidence.", language_hint="english")
    german = pipeline.process("Das ist ein uber Text.", language_hint="german")
    french = pipeline.process("Ceci est un texte francais.", language_hint="french")

    assert english.adapter_used == "english"
    assert german.adapter_used == "german"
    assert french.adapter_used == "french"


def test_transliteration_hint_routes_to_akkadian_adapter() -> None:
    pipeline = PostprocessingPipeline()
    text = "\u0161arrum a-na qibi-ma [x x]"

    result = pipeline.process(
        text,
        language_hint="Akkadian transliteration",
        script_hint="latin_diacritic",
    )

    assert result.adapter_used == "akkadian_transliteration"
    assert result.protected_character_changes == 0


def test_diacritics_are_preserved_for_akkadian_adapter() -> None:
    pipeline = PostprocessingPipeline()
    text = "\u0161 \u1e63 \u1e6d \u1e2b \u012b \u016b"

    result = pipeline.process(
        text,
        language_hint="akkadian",
        script_hint="transliteration",
    )

    assert result.corrected_text == result.cleaned_text
    assert result.protected_character_changes == 0
    assert result.quality_metrics.get("diacritic_preservation") == 1.0


def test_lexicon_rule_correction_is_auditable() -> None:
    pipeline = PostprocessingPipeline()
    result = pipeline.process("The evideuce is clear.", language_hint="english")

    assert "evidence" in result.corrected_text.lower()
    assert result.corrections_applied
    assert any(item.get("reason") in {"lexicon_suggestion", "confusion_map"} for item in result.corrections_applied)


def test_overcorrection_prevention_keeps_scholarly_tokens() -> None:
    pipeline = PostprocessingPipeline()
    text = "a-na qibi-ma [x x]"

    result = pipeline.process(
        text,
        language_hint="akkadian",
        script_hint="transliteration",
    )

    assert result.corrected_text == result.cleaned_text
    assert result.corrections_applied == []


def test_unknown_language_falls_back_to_default_latin() -> None:
    pipeline = PostprocessingPipeline()
    result = pipeline.process("Neutral sample text", language_hint="unknown", script_hint="unknown")

    assert result.adapter_used == "default_latin"


def test_lexicon_interface_supports_domain_lookup_and_suggestions() -> None:
    lexicon = InMemoryLexicon()
    lexicon.load_word_list(["analysis", "analyst"], domain="english", frequencies={"analysis": 12, "analyst": 2})

    assert lexicon.lookup_exact("analysis", domain="english")
    assert lexicon.lookup_normalized("Analysis", domain="english")

    suggestions = lexicon.suggest("analysis", domain="english", max_distance=1, max_results=3)
    assert suggestions
    assert suggestions[0].token == "analysis"


def test_optional_model_correction_is_guarded_by_structure() -> None:
    def bad_proposer(text: str, context: dict[str, str]) -> str:
        return text + "\nnew line"

    pipeline = PostprocessingPipeline(
        model_corrector=GuardedModelCorrector(
            proposer=bad_proposer,
            edit_budget_ratio=0.10,
            preserve_line_structure=True,
        ),
        enable_model_correction=True,
    )

    result = pipeline.process("This is text.", language_hint="english")

    assert result.model_reason == "model_correction_line_structure_violation"
    assert result.corrected_text == result.cleaned_text
