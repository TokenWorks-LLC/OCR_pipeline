from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import re
from typing import Any


ADAPTER_POLICY_VERSION = "phase20_reviewer_correction_adapter_v1"
DEFAULT_TARGET_CATEGORY = "typewritten_historical_directories"
DEFAULT_ADAPTER_NAME = "typewritten_historical_directories_linebreak_adapter_v1"


@dataclass(frozen=True)
class AdapterEdit:
    rule: str
    before: str
    after: str
    line_index: int
    confidence: float


@dataclass(frozen=True)
class CategoryAdapterResult:
    selected_adapter: str
    target_category: str
    policy_version: str
    route_reason: str
    adapter_applied: bool
    raw_ocr_text: str
    cleaned_text: str
    corrected_text: str
    edit_trace: list[dict[str, Any]]
    review_required: bool
    audit: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_adapter": self.selected_adapter,
            "target_category": self.target_category,
            "policy_version": self.policy_version,
            "route_reason": self.route_reason,
            "adapter_applied": bool(self.adapter_applied),
            "raw_ocr_text": self.raw_ocr_text,
            "cleaned_text": self.cleaned_text,
            "corrected_text": self.corrected_text,
            "edit_trace": list(self.edit_trace),
            "review_required": bool(self.review_required),
            "audit": dict(self.audit),
        }


def _normalize_newlines(text: str) -> str:
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n")


def _normalize_spacing(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in str(text or "").split("\n")]
    compact = "\n".join(lines)
    compact = re.sub(r"\n{3,}", "\n\n", compact)
    return compact.strip()


def _starts_with_lower_or_digit(text: str) -> bool:
    value = str(text or "").lstrip()
    if not value:
        return False
    first = value[0]
    return first.islower() or first.isdigit()


def _should_soft_join(current_line: str, next_line: str) -> bool:
    left = str(current_line or "").rstrip()
    right = str(next_line or "").lstrip()
    if not left or not right:
        return False

    if left.endswith(('.', '!', '?', ':', ';', ')', ']', '}', '"')):
        return False

    if len(left.split()) < 2:
        return False

    return _starts_with_lower_or_digit(right)


def _apply_linebreak_adapter(raw_text: str, max_edits: int) -> tuple[str, list[AdapterEdit], dict[str, Any]]:
    source = _normalize_newlines(raw_text)
    lines = source.split("\n")

    if len(lines) < 4:
        return source, [], {"line_count": len(lines), "hyphenated_joins": 0, "soft_wrap_joins": 0, "pattern_signal": 0}

    edits: list[AdapterEdit] = []
    output_lines: list[str] = []
    hyphenated_joins = 0
    soft_wrap_joins = 0

    buffer = lines[0].rstrip()
    buffer_line_index = 0

    for idx, line in enumerate(lines[1:], start=1):
        current = line.rstrip()

        if not buffer.strip():
            output_lines.append("")
            buffer = current
            buffer_line_index = idx
            continue

        if not current.strip():
            output_lines.append(buffer)
            output_lines.append("")
            buffer = ""
            buffer_line_index = idx
            continue

        next_l = current.lstrip()
        if buffer.endswith("-") and re.match(r"^[A-Za-z0-9]", next_l):
            before = f"{buffer}\n{current}"
            joined = buffer[:-1] + next_l
            if len(edits) < max_edits:
                edits.append(
                    AdapterEdit(
                        rule="join_hyphenated_line_break",
                        before=before,
                        after=joined,
                        line_index=buffer_line_index,
                        confidence=0.96,
                    )
                )
            buffer = joined
            hyphenated_joins += 1
            continue

        if _should_soft_join(buffer, current):
            before = f"{buffer}\n{current}"
            joined = f"{buffer} {next_l}"
            if len(edits) < max_edits:
                edits.append(
                    AdapterEdit(
                        rule="join_soft_wrapped_line",
                        before=before,
                        after=joined,
                        line_index=buffer_line_index,
                        confidence=0.82,
                    )
                )
            buffer = joined
            soft_wrap_joins += 1
            continue

        output_lines.append(buffer)
        buffer = current
        buffer_line_index = idx

    output_lines.append(buffer)
    candidate = _normalize_spacing("\n".join(output_lines))
    signal = hyphenated_joins + soft_wrap_joins
    return candidate, edits, {
        "line_count": len(lines),
        "hyphenated_joins": hyphenated_joins,
        "soft_wrap_joins": soft_wrap_joins,
        "pattern_signal": signal,
    }


def apply_reviewer_category_adapter(
    *,
    page_id: str,
    document_type: str,
    raw_ocr_text: str,
    target_category: str = DEFAULT_TARGET_CATEGORY,
    policy_enabled: bool = True,
    max_edits: int = 120,
    max_length_delta_ratio: float = 0.12,
) -> CategoryAdapterResult:
    doc_type = str(document_type or "").strip().lower() or "unknown"
    target = str(target_category or DEFAULT_TARGET_CATEGORY).strip().lower() or DEFAULT_TARGET_CATEGORY
    raw_text = _normalize_newlines(raw_ocr_text)
    cleaned_text = _normalize_spacing(raw_text)

    route_reason = "non_target_category"
    corrected_text = cleaned_text
    edits: list[AdapterEdit] = []
    heuristics: dict[str, Any] = {
        "line_count": len(cleaned_text.split("\n")) if cleaned_text else 0,
        "hyphenated_joins": 0,
        "soft_wrap_joins": 0,
        "pattern_signal": 0,
    }
    adapter_applied = False

    if not policy_enabled:
        route_reason = "adapter_disabled_by_policy"
    elif doc_type != target:
        route_reason = "non_target_category"
    elif not cleaned_text:
        route_reason = "empty_text"
    else:
        candidate_text, edits, heuristics = _apply_linebreak_adapter(cleaned_text, max_edits=max_edits)
        signal = int(heuristics.get("pattern_signal", 0) or 0)
        if signal < 2:
            route_reason = "insufficient_pattern_signal"
        elif candidate_text == cleaned_text:
            route_reason = "no_change_after_adapter"
        else:
            baseline_len = max(len(cleaned_text), 1)
            delta_ratio = abs(len(candidate_text) - len(cleaned_text)) / float(baseline_len)
            if delta_ratio > float(max_length_delta_ratio):
                route_reason = "edit_budget_exceeded"
            else:
                corrected_text = candidate_text
                adapter_applied = True
                route_reason = "adapter_applied"

    trace_payload = [asdict(item) for item in edits] if adapter_applied else []
    rule_hits = Counter(item.rule for item in edits) if adapter_applied else Counter()

    audit = {
        "page_id": str(page_id or ""),
        "document_type": doc_type,
        "target_category": target,
        "selected_adapter": DEFAULT_ADAPTER_NAME,
        "policy_version": ADAPTER_POLICY_VERSION,
        "route_reason": route_reason,
        "adapter_applied": bool(adapter_applied),
        "edit_count": len(trace_payload),
        "rule_hits": {k: int(v) for k, v in sorted(rule_hits.items())},
        "heuristics": heuristics,
        "raw_ocr_preserved": raw_text == _normalize_newlines(raw_ocr_text),
        "review_required": bool(adapter_applied),
    }

    return CategoryAdapterResult(
        selected_adapter=DEFAULT_ADAPTER_NAME,
        target_category=target,
        policy_version=ADAPTER_POLICY_VERSION,
        route_reason=route_reason,
        adapter_applied=adapter_applied,
        raw_ocr_text=raw_text,
        cleaned_text=cleaned_text,
        corrected_text=corrected_text,
        edit_trace=trace_payload,
        review_required=bool(adapter_applied),
        audit=audit,
    )
