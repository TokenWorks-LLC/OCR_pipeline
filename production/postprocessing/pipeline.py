from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .adapters import CorrectionAudit, LanguageAdapter, adapter_registry, select_adapter_name, serialize_corrections
from .cleanup import CleanupResult, general_cleanup
from .lexicon import InMemoryLexicon
from .model_correction import GuardedModelCorrector, ModelCorrectionOutcome


def _default_word_lists() -> dict[str, list[str]]:
    return {
        "general": [
            "the",
            "and",
            "for",
            "with",
            "text",
            "page",
            "document",
            "line",
            "notes",
            "chapter",
            "section",
            "table",
            "figure",
            "evidence",
            "review",
            "quality",
            "output",
        ],
        "english": [
            "the",
            "this",
            "that",
            "evidence",
            "clear",
            "analysis",
            "document",
            "language",
            "history",
            "scholarly",
        ],
        "german": [
            "uber",
            "uberlieferung",
            "gross",
            "strasse",
            "wissenschaft",
            "beleg",
            "text",
            "seite",
        ],
        "french": [
            "etude",
            "analyse",
            "texte",
            "histoire",
            "francais",
            "langue",
            "document",
        ],
        "transliteration": [
            "a-na",
            "i-na",
            "sa-ra-am",
            "dumu",
            "beli",
            "qibi",
            "sarrum",
        ],
        "akkadian": [
            "a-na",
            "i-na",
            "sa-ra-am",
            "dumu",
            "beli",
            "qibi",
            "sarrum",
            "\u0161arrum",
            "\u1e63arrum",
        ],
    }


def build_default_lexicon(extra_lexicon_paths: dict[str, str] | None = None) -> InMemoryLexicon:
    lexicon = InMemoryLexicon()
    for domain, words in _default_word_lists().items():
        lexicon.load_word_list(words=words, domain=domain)

    for domain, path in (extra_lexicon_paths or {}).items():
        lexicon.load_from_file(path, domain=domain, has_frequency=False)

    return lexicon


@dataclass(frozen=True)
class PostprocessingResult:
    raw_text: str
    cleaned_text: str
    corrected_text: str
    adapter_used: str
    corrections_applied: list[dict[str, Any]]
    correction_confidence: float
    lexicon_coverage: float
    unknown_token_rate: float
    protected_character_changes: int
    needs_human_review: bool
    quality_score: float
    correction_diff: str
    quality_metrics: dict[str, Any]
    model_reason: str

    def to_metadata(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "cleaned_text": self.cleaned_text,
            "corrected_text": self.corrected_text,
            "adapter_used": self.adapter_used,
            "corrections_applied": self.corrections_applied,
            "correction_confidence": round(float(self.correction_confidence), 6),
            "lexicon_coverage": round(float(self.lexicon_coverage), 6),
            "unknown_token_rate": round(float(self.unknown_token_rate), 6),
            "protected_character_changes": int(self.protected_character_changes),
            "needs_human_review": bool(self.needs_human_review),
            "postprocess_quality_score": round(float(self.quality_score), 6),
            "correction_diff": self.correction_diff,
            "postprocess_quality_metrics": self.quality_metrics,
            "model_correction_reason": self.model_reason,
        }


class PostprocessingPipeline:
    """Modular multilingual postprocessing with adapter-specific behavior."""

    def __init__(
        self,
        *,
        adapters: dict[str, LanguageAdapter] | None = None,
        lexicon: InMemoryLexicon | None = None,
        enable_rule_corrections: bool = True,
        enable_model_correction: bool = False,
        model_corrector: GuardedModelCorrector | None = None,
        rule_edit_budget_ratio: float = 0.05,
    ) -> None:
        self.adapters = dict(adapters or adapter_registry())
        self.lexicon = lexicon or build_default_lexicon()
        self.enable_rule_corrections = bool(enable_rule_corrections)
        self.enable_model_correction = bool(enable_model_correction)
        self.rule_edit_budget_ratio = max(0.0, float(rule_edit_budget_ratio or 0.0))
        self.model_corrector = model_corrector or GuardedModelCorrector(
            proposer=None,
            edit_budget_ratio=0.05,
            preserve_line_structure=True,
        )

    @staticmethod
    def spacy_support_status() -> dict[str, Any]:
        try:
            import spacy

            return {
                "available": True,
                "version": str(getattr(spacy, "__version__", "unknown")),
                "recommended": [
                    "Use spaCy tokenizer only when language-specific tokenization quality materially improves over adapter regex tokenization.",
                    "Use EntityRuler/Matcher for deterministic postprocessing patterns.",
                    "Avoid training spaCy models unless enough labeled OCR correction data exists.",
                ],
            }
        except Exception as exc:
            return {
                "available": False,
                "reason": str(exc),
                "recommended": [
                    "Keep lightweight adapter tokenizers as default.",
                    "Add optional spaCy integration later for languages with clear tokenization gains.",
                ],
            }

    def process(
        self,
        raw_text: str,
        *,
        language_hint: str = "unknown",
        script_hint: str = "unknown",
        adapter_hint: str | None = None,
        page_number: int | None = None,
        region_id: str = "",
    ) -> PostprocessingResult:
        cleanup = general_cleanup(raw_text)
        cleaned_text = cleanup.cleaned_text

        adapter_name = select_adapter_name(
            text=cleaned_text,
            language_hint=language_hint,
            script_hint=script_hint,
            adapter_hint=adapter_hint,
        )
        adapter = self.adapters.get(adapter_name, self.adapters["default_latin"])

        corrected_text = cleaned_text
        correction_audits: list[CorrectionAudit] = []

        if cleaned_text and self.enable_rule_corrections:
            corrected_text, correction_audits, _ = adapter.apply_rule_corrections(
                cleaned_text,
                self.lexicon,
                edit_budget_ratio=self.rule_edit_budget_ratio,
            )

        quality_metrics = adapter.compute_quality_metrics(
            cleaned_text=cleaned_text,
            corrected_text=corrected_text,
            lexicon=self.lexicon,
            corrections=correction_audits,
        )

        model_outcome = ModelCorrectionOutcome(
            corrected_text=corrected_text,
            applied=False,
            confidence=0.0,
            reason="model_correction_disabled",
            edit_ratio=0.0,
            diff="",
        )

        if cleaned_text and self.enable_model_correction:
            quality_before = float(quality_metrics.get("quality_score", 0.0) or 0.0)

            def _quality_eval(candidate: str) -> float:
                candidate_metrics = adapter.compute_quality_metrics(
                    cleaned_text=cleaned_text,
                    corrected_text=candidate,
                    lexicon=self.lexicon,
                    corrections=correction_audits,
                )
                return float(candidate_metrics.get("quality_score", 0.0) or 0.0)

            model_outcome = self.model_corrector.apply(
                original_text=corrected_text,
                adapter=adapter,
                quality_before=quality_before,
                quality_evaluator=_quality_eval,
                context={
                    "language_hint": language_hint,
                    "script_hint": script_hint,
                    "adapter": adapter.name,
                    "page_number": page_number,
                    "region_id": region_id,
                },
            )

            if model_outcome.applied:
                corrected_text = model_outcome.corrected_text
                correction_audits.append(
                    CorrectionAudit(
                        token_before="<model>",
                        token_after="<model>",
                        reason="model_correction_applied",
                        source="model",
                        confidence=float(model_outcome.confidence),
                        start=0,
                        end=max(0, len(corrected_text)),
                        suspicious=False,
                    )
                )
                quality_metrics = adapter.compute_quality_metrics(
                    cleaned_text=cleaned_text,
                    corrected_text=corrected_text,
                    lexicon=self.lexicon,
                    corrections=correction_audits,
                )

        correction_confidence = (
            sum(float(item.confidence) for item in correction_audits) / len(correction_audits)
            if correction_audits
            else 1.0
        )

        unknown_rate = float(quality_metrics.get("unknown_token_rate", 0.0) or 0.0)
        suspicious_rate = float(quality_metrics.get("suspicious_correction_rate", 0.0) or 0.0)
        protected_changes = int(quality_metrics.get("protected_character_changes", 0) or 0)
        quality_score = float(quality_metrics.get("quality_score", 0.0) or 0.0)

        needs_human_review = bool(
            protected_changes > 0
            or unknown_rate > float(adapter.unknown_review_threshold)
            or suspicious_rate > float(adapter.suspicious_review_threshold)
            or quality_score < 0.45
        )

        if self.enable_model_correction and not model_outcome.applied and model_outcome.reason not in {
            "model_correction_disabled",
        }:
            needs_human_review = True

        correction_diff = model_outcome.diff
        if not correction_diff and correction_audits:
            correction_diff = json.dumps(serialize_corrections(correction_audits), ensure_ascii=True, sort_keys=True)

        quality_metrics = {
            **quality_metrics,
            "cleanup_removed_token_count": cleanup.removed_token_count,
            "adapter": adapter.name,
            "language_hint": str(language_hint or "unknown"),
            "script_hint": str(script_hint or "unknown"),
            "page_number": page_number if page_number is not None else "",
            "region_id": region_id,
        }

        return PostprocessingResult(
            raw_text=str(raw_text or ""),
            cleaned_text=cleaned_text,
            corrected_text=corrected_text,
            adapter_used=adapter.name,
            corrections_applied=serialize_corrections(correction_audits),
            correction_confidence=round(float(correction_confidence), 6),
            lexicon_coverage=float(quality_metrics.get("lexicon_coverage", 0.0) or 0.0),
            unknown_token_rate=unknown_rate,
            protected_character_changes=protected_changes,
            needs_human_review=needs_human_review,
            quality_score=quality_score,
            correction_diff=correction_diff,
            quality_metrics=quality_metrics,
            model_reason=str(model_outcome.reason),
        )

    @classmethod
    def from_optional_lexicon_paths(
        cls,
        *,
        lexicon_paths: dict[str, str] | None = None,
        enable_rule_corrections: bool = True,
        enable_model_correction: bool = False,
    ) -> "PostprocessingPipeline":
        return cls(
            lexicon=build_default_lexicon(extra_lexicon_paths=lexicon_paths or {}),
            enable_rule_corrections=enable_rule_corrections,
            enable_model_correction=enable_model_correction,
        )
