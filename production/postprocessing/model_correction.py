from __future__ import annotations

from dataclasses import dataclass
from difflib import unified_diff
from typing import Any, Callable

from .adapters import LanguageAdapter


@dataclass(frozen=True)
class ModelCorrectionOutcome:
    corrected_text: str
    applied: bool
    confidence: float
    reason: str
    edit_ratio: float
    diff: str


class GuardedModelCorrector:
    """Optional model correction wrapper with strict safety guardrails."""

    def __init__(
        self,
        proposer: Callable[[str, dict[str, Any]], str] | None = None,
        edit_budget_ratio: float = 0.05,
        preserve_line_structure: bool = True,
    ) -> None:
        self.proposer = proposer
        self.edit_budget_ratio = max(0.0, float(edit_budget_ratio or 0.0))
        self.preserve_line_structure = bool(preserve_line_structure)

    @staticmethod
    def _edit_distance(left: str, right: str) -> int:
        if left == right:
            return 0

        prev = list(range(len(right) + 1))
        for i, left_char in enumerate(left, start=1):
            current = [i]
            for j, right_char in enumerate(right, start=1):
                substitution = 0 if left_char == right_char else 1
                current.append(
                    min(
                        prev[j] + 1,
                        current[j - 1] + 1,
                        prev[j - 1] + substitution,
                    )
                )
            prev = current
        return prev[-1]

    @staticmethod
    def _diff_text(before: str, after: str) -> str:
        diff_lines = list(
            unified_diff(
                str(before or "").splitlines(),
                str(after or "").splitlines(),
                fromfile="original",
                tofile="corrected",
                lineterm="",
            )
        )
        return "\n".join(diff_lines)

    def apply(
        self,
        original_text: str,
        adapter: LanguageAdapter,
        quality_before: float,
        quality_evaluator: Callable[[str], float],
        context: dict[str, Any] | None = None,
    ) -> ModelCorrectionOutcome:
        if self.proposer is None:
            return ModelCorrectionOutcome(
                corrected_text=str(original_text or ""),
                applied=False,
                confidence=0.0,
                reason="model_correction_disabled",
                edit_ratio=0.0,
                diff="",
            )

        source = str(original_text or "")
        context_payload = dict(context or {})

        try:
            proposal = self.proposer(source, context_payload)
        except Exception as exc:
            return ModelCorrectionOutcome(
                corrected_text=source,
                applied=False,
                confidence=0.0,
                reason=f"model_correction_exception:{type(exc).__name__}",
                edit_ratio=0.0,
                diff="",
            )

        proposed_text = str(proposal or "")
        if not proposed_text.strip():
            return ModelCorrectionOutcome(
                corrected_text=source,
                applied=False,
                confidence=0.0,
                reason="model_correction_empty",
                edit_ratio=0.0,
                diff="",
            )

        if self.preserve_line_structure and len(source.splitlines()) != len(proposed_text.splitlines()):
            return ModelCorrectionOutcome(
                corrected_text=source,
                applied=False,
                confidence=0.0,
                reason="model_correction_line_structure_violation",
                edit_ratio=0.0,
                diff="",
            )

        protected_changes = adapter.count_protected_character_changes(source, proposed_text)
        if protected_changes > 0:
            return ModelCorrectionOutcome(
                corrected_text=source,
                applied=False,
                confidence=0.0,
                reason="model_correction_protected_character_violation",
                edit_ratio=0.0,
                diff="",
            )

        edit_distance = self._edit_distance(source, proposed_text)
        denominator = max(len(source), 1)
        edit_ratio = edit_distance / float(denominator)

        if edit_ratio > self.edit_budget_ratio:
            return ModelCorrectionOutcome(
                corrected_text=source,
                applied=False,
                confidence=0.0,
                reason="model_correction_edit_budget_exceeded",
                edit_ratio=round(float(edit_ratio), 6),
                diff="",
            )

        quality_after = float(quality_evaluator(proposed_text) or 0.0)
        if quality_after < float(quality_before):
            return ModelCorrectionOutcome(
                corrected_text=source,
                applied=False,
                confidence=0.0,
                reason="model_correction_quality_degraded",
                edit_ratio=round(float(edit_ratio), 6),
                diff="",
            )

        confidence = max(0.0, min(1.0, quality_after - max(0.0, edit_ratio * 0.5)))
        return ModelCorrectionOutcome(
            corrected_text=proposed_text,
            applied=True,
            confidence=round(float(confidence), 6),
            reason="model_correction_applied",
            edit_ratio=round(float(edit_ratio), 6),
            diff=self._diff_text(source, proposed_text),
        )
