from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re

from production.beta_scope_policy import STATUS_EXCLUDED, evaluate_beta_scope, resolve_beta_scope_status


ROUTE_VERSION = "phase17_scanned_forms_route_v1"
SCANNED_FORMS_ROUTE = "scanned_forms_structured_cleanup_v1"
DEFAULT_ROUTE = "default_pipeline_no_scanned_forms_cleanup"


@dataclass(frozen=True)
class ScannedFormsRouteResult:
    selected_route: str
    route_reason: str
    route_confidence: float
    document_type: str
    raw_ocr_text: str
    structured_cleanup_text: str
    final_display_text: str
    cleanup_applied: bool
    cleanup_reasons: list[str]
    raw_ocr_preserved: bool
    review_required: bool
    export_allowed: bool
    export_block_reason: str
    route_policy_version: str
    route_audit: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_route": self.selected_route,
            "route_reason": self.route_reason,
            "route_confidence": float(self.route_confidence),
            "document_type": self.document_type,
            "raw_ocr_text": self.raw_ocr_text,
            "structured_cleanup_text": self.structured_cleanup_text,
            "final_display_text": self.final_display_text,
            "cleanup_applied": bool(self.cleanup_applied),
            "cleanup_reasons": list(self.cleanup_reasons),
            "raw_ocr_preserved": bool(self.raw_ocr_preserved),
            "review_required": bool(self.review_required),
            "export_allowed": bool(self.export_allowed),
            "export_block_reason": self.export_block_reason,
            "route_policy_version": self.route_policy_version,
            "route_audit": dict(self.route_audit),
        }


def _structured_form_cleanup(raw_text: str) -> tuple[str, list[str]]:
    text = str(raw_text or "")
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    reasons: list[str] = []

    updated = re.sub(r"\bFAX\s*NO\b", "FAX NO", cleaned, flags=re.IGNORECASE)
    if updated != cleaned:
        reasons.append("normalize_fax_header")
    cleaned = updated

    updated = re.sub(r"\s*([:\-])\s*", r"\1 ", cleaned)
    if updated != cleaned:
        reasons.append("normalize_field_spacing")
    cleaned = updated

    updated = re.sub(r"\b(\d)\s*\)\s*(\d)\b", r"\1\2", cleaned)
    if updated != cleaned:
        reasons.append("collapse_split_number_tokens")
    cleaned = updated

    updated = re.sub(r"[ \t]+", " ", cleaned)
    updated = re.sub(r"\n{3,}", "\n\n", updated)
    if updated != cleaned:
        reasons.append("normalize_whitespace")
    cleaned = updated

    cleaned = cleaned.strip()
    if not reasons:
        reasons.append("no_high_confidence_structured_cleanup_pattern")
    return cleaned, reasons


def apply_scanned_forms_route(
    *,
    page_id: str,
    document_type: str,
    script_type: str,
    raw_ocr_text: str,
    review_state: str = "needs_review",
    target: str = "private_beta",
    route_policy_enabled: bool = True,
    fail_cleanup: bool = False,
) -> ScannedFormsRouteResult:
    doc_type = str(document_type or "").strip().lower() or "unknown"
    script = str(script_type or "").strip()
    raw_text = str(raw_ocr_text or "")

    scope_status, scope_reason = resolve_beta_scope_status(doc_type, script)
    route_selected = False
    cleanup_failed = False

    selected_route = DEFAULT_ROUTE
    route_reason = "non_scanned_forms_document_type"
    route_confidence = 0.98
    structured_text = raw_text
    final_text = raw_text
    cleanup_applied = False
    cleanup_reasons: list[str] = []
    cleanup_error = ""

    if doc_type == "receipts_commercial_docs":
        route_reason = "excluded_receipts_document_type"
    elif doc_type != "scanned_forms":
        route_reason = "non_scanned_forms_document_type"
    elif scope_status == STATUS_EXCLUDED:
        route_reason = "script_excluded_by_policy"
        route_confidence = 1.0
    elif not route_policy_enabled:
        route_reason = "route_disabled_by_policy"
        route_confidence = 0.85
    else:
        selected_route = SCANNED_FORMS_ROUTE
        route_selected = True
        route_reason = "scanned_forms_route_selected"
        try:
            if fail_cleanup:
                raise RuntimeError("forced_cleanup_failure")

            structured_text, cleanup_reasons = _structured_form_cleanup(raw_text)
            cleanup_applied = structured_text != raw_text
            final_text = structured_text if cleanup_applied else raw_text
            route_confidence = 0.92 if cleanup_applied else 0.76
        except Exception as exc:
            cleanup_failed = True
            cleanup_error = str(exc)
            cleanup_reasons = ["cleanup_exception_fallback_to_raw"]
            structured_text = raw_text
            final_text = raw_text
            cleanup_applied = False
            route_reason = "cleanup_failed_fallback_to_raw"
            route_confidence = 0.30

    scope_decision = evaluate_beta_scope(
        document_type=doc_type,
        script_type=script,
        review_state=review_state,
        target=target,
    )

    review_required = bool(scope_decision.review_required)
    export_allowed = bool(scope_decision.export_allowed)
    export_block_reason = str(scope_decision.export_block_reason or "")

    if cleanup_failed:
        export_allowed = False
        review_required = True
        if not export_block_reason:
            export_block_reason = "route_cleanup_failed_requires_manual_review"

    route_audit = {
        "route_version": ROUTE_VERSION,
        "page_id": str(page_id or ""),
        "selected_route": selected_route,
        "route_reason": route_reason,
        "route_confidence": float(route_confidence),
        "route_selected": bool(route_selected),
        "cleanup_applied": bool(cleanup_applied),
        "cleanup_reasons": list(cleanup_reasons),
        "cleanup_failed": bool(cleanup_failed),
        "cleanup_error": cleanup_error,
        "document_type": doc_type,
        "script_type": script,
        "policy_scope_status": scope_status,
        "policy_scope_reason": scope_reason,
        "review_required": bool(review_required),
        "export_allowed": bool(export_allowed),
        "export_block_reason": export_block_reason,
        "raw_ocr_preserved": raw_text == str(raw_ocr_text or ""),
    }

    return ScannedFormsRouteResult(
        selected_route=selected_route,
        route_reason=route_reason,
        route_confidence=route_confidence,
        document_type=doc_type,
        raw_ocr_text=raw_text,
        structured_cleanup_text=structured_text,
        final_display_text=final_text,
        cleanup_applied=cleanup_applied,
        cleanup_reasons=cleanup_reasons,
        raw_ocr_preserved=(raw_text == str(raw_ocr_text or "")),
        review_required=review_required,
        export_allowed=export_allowed,
        export_block_reason=export_block_reason,
        route_policy_version=ROUTE_VERSION,
        route_audit=route_audit,
    )
