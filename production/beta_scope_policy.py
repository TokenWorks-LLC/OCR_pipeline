from __future__ import annotations

from dataclasses import dataclass
from typing import Any

POLICY_VERSION = "phase9_private_beta_scope_v1"

STATUS_ALLOW_REVIEW_ONLY = "allow_private_beta_with_mandatory_review"
STATUS_INTERNAL_ONLY = "internal_only_with_review"
STATUS_EXCLUDED = "exclude_from_private_beta"

DOCUMENT_TYPE_POLICY: dict[str, str] = {
    "receipts_commercial_docs": STATUS_EXCLUDED,
    "typewritten_historical_directories": STATUS_ALLOW_REVIEW_ONLY,
    "historical_book": STATUS_INTERNAL_ONLY,
    "historical_printed_books": STATUS_INTERNAL_ONLY,
    "scanned_forms": STATUS_INTERNAL_ONLY,
}


@dataclass(frozen=True)
class BetaScopeDecision:
    beta_scope_status: str
    beta_scope_reason: str
    review_required: bool
    export_allowed: bool
    export_block_reason: str
    category_policy_version: str = POLICY_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "beta_scope_status": self.beta_scope_status,
            "beta_scope_reason": self.beta_scope_reason,
            "review_required": bool(self.review_required),
            "export_allowed": bool(self.export_allowed),
            "export_block_reason": self.export_block_reason,
            "category_policy_version": self.category_policy_version,
        }


def _script_tokens(script_type: str) -> set[str]:
    raw = str(script_type or "")
    parts = [p.strip().lower() for p in raw.replace(";", ",").split(",")]
    return {p for p in parts if p}


def _script_excluded(script_type: str) -> bool:
    tokens = _script_tokens(script_type)
    return "latn" in tokens and "hang" in tokens


def resolve_beta_scope_status(document_type: str, script_type: str) -> tuple[str, str]:
    doc_type = str(document_type or "").strip().lower() or "unknown"

    if _script_excluded(script_type):
        return STATUS_EXCLUDED, "script_policy_exclusion_latn_hang"

    status = DOCUMENT_TYPE_POLICY.get(doc_type, STATUS_INTERNAL_ONLY)
    if status == STATUS_ALLOW_REVIEW_ONLY:
        return status, "document_type_allowed_with_mandatory_review"
    if status == STATUS_EXCLUDED:
        return status, "document_type_excluded_from_private_beta"
    return status, "document_type_internal_only_policy"


def evaluate_beta_scope(
    *,
    document_type: str,
    script_type: str,
    review_state: str,
    target: str = "private_beta",
) -> BetaScopeDecision:
    status, reason = resolve_beta_scope_status(document_type, script_type)
    target_norm = str(target or "private_beta").strip().lower()
    review_state_norm = str(review_state or "").strip().lower()

    review_required = True
    review_complete = review_state_norm in {"accepted", "corrected", "exported"}

    export_allowed = False
    block_reason = ""

    if target_norm != "private_beta":
        export_allowed = review_complete
        if not export_allowed:
            block_reason = "review_not_completed"
    else:
        if status == STATUS_EXCLUDED:
            export_allowed = False
            block_reason = "excluded_category_or_script"
        elif status == STATUS_INTERNAL_ONLY:
            export_allowed = False
            block_reason = "internal_only_category"
        elif not review_complete:
            export_allowed = False
            block_reason = "review_not_completed"
        else:
            export_allowed = True
            block_reason = ""

    return BetaScopeDecision(
        beta_scope_status=status,
        beta_scope_reason=reason,
        review_required=review_required,
        export_allowed=export_allowed,
        export_block_reason=block_reason,
    )


def apply_beta_scope_to_row(
    row: dict[str, Any],
    *,
    review_state: str,
    target: str = "private_beta",
) -> dict[str, Any]:
    decision = evaluate_beta_scope(
        document_type=str(row.get("document_type", "") or ""),
        script_type=str(row.get("script_type", "") or ""),
        review_state=review_state,
        target=target,
    )
    enriched = dict(row)
    enriched.update(decision.to_dict())
    return enriched
