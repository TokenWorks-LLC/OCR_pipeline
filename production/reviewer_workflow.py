from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import json
import uuid

REVIEW_STATE_NEEDS_REVIEW = "needs_review"
REVIEW_STATE_IN_REVIEW = "in_review"
REVIEW_STATE_ACCEPTED = "accepted"
REVIEW_STATE_CORRECTED = "corrected"
REVIEW_STATE_REJECTED = "rejected"
REVIEW_STATE_BLOCKED = "blocked"
REVIEW_STATE_EXPORTED = "exported"

REVIEW_STATES = {
    REVIEW_STATE_NEEDS_REVIEW,
    REVIEW_STATE_IN_REVIEW,
    REVIEW_STATE_ACCEPTED,
    REVIEW_STATE_CORRECTED,
    REVIEW_STATE_REJECTED,
    REVIEW_STATE_BLOCKED,
    REVIEW_STATE_EXPORTED,
}

TERMINAL_REVIEW_STATES = {
    REVIEW_STATE_ACCEPTED,
    REVIEW_STATE_CORRECTED,
    REVIEW_STATE_REJECTED,
    REVIEW_STATE_BLOCKED,
    REVIEW_STATE_EXPORTED,
}


class ReviewWorkflowError(RuntimeError):
    """Raised when review workflow state transitions are invalid."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _as_bool(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _ensure_non_empty(value: str, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ReviewWorkflowError(f"{field_name} is required")
    return text


@dataclass
class ReviewItem:
    page_id: str
    document_id: str
    raw_ocr_text: str
    status: str
    quality_class: str
    needs_human_review: bool
    review_state: str = REVIEW_STATE_NEEDS_REVIEW
    reviewer_id: str = ""
    triage_label: str = "normal"
    corrected_ocr_text: str = ""
    quality_warning_overridden: bool = False
    quality_override_reason: str = ""
    beta_scope_status: str = "internal_only_with_review"
    beta_scope_reason: str = "policy_not_evaluated"
    final_reviewed_status: str = ""
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def current_text(self) -> str:
        return self.corrected_ocr_text if self.corrected_ocr_text else self.raw_ocr_text

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "document_id": self.document_id,
            "raw_ocr_text": self.raw_ocr_text,
            "corrected_ocr_text": self.corrected_ocr_text,
            "status": self.status,
            "quality_class": self.quality_class,
            "needs_human_review": bool(self.needs_human_review),
            "review_state": self.review_state,
            "reviewer_id": self.reviewer_id,
            "triage_label": self.triage_label,
            "quality_warning_overridden": bool(self.quality_warning_overridden),
            "quality_override_reason": self.quality_override_reason,
            "beta_scope_status": self.beta_scope_status,
            "beta_scope_reason": self.beta_scope_reason,
            "final_reviewed_status": self.final_reviewed_status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }


@dataclass
class AuditEvent:
    event_id: str
    timestamp: str
    page_id: str
    document_id: str
    action: str
    actor_id: str
    from_state: str
    to_state: str
    reason: str
    override_reason: str
    before_text: str
    after_text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "page_id": self.page_id,
            "document_id": self.document_id,
            "action": self.action,
            "actor_id": self.actor_id,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "reason": self.reason,
            "override_reason": self.override_reason,
            "before_text": self.before_text,
            "after_text": self.after_text,
            "metadata": dict(self.metadata),
        }


class ReviewerWorkflowStore:
    """Persistent in-memory reviewer workflow with an explicit audit trail."""

    def __init__(self) -> None:
        self._items: dict[str, ReviewItem] = {}
        self._audit_events: list[AuditEvent] = []

    @property
    def items(self) -> dict[str, ReviewItem]:
        return self._items

    @property
    def audit_events(self) -> list[AuditEvent]:
        return self._audit_events

    def create_review_queue(self, page_rows: Iterable[dict[str, Any]], actor_id: str = "system") -> list[ReviewItem]:
        created: list[ReviewItem] = []
        for row in page_rows:
            page_id = self._derive_page_id(row)
            if page_id in self._items:
                continue

            document_id = str(row.get("document_id", "")).strip() or self._derive_document_id(page_id)
            status = str(row.get("status", "success") or "success")
            quality_class = str(row.get("quality_class", "weak_ocr") or "weak_ocr")
            needs_review = _as_bool(row.get("needs_human_review", True))
            if quality_class in {"weak_ocr", "failed_ocr"}:
                needs_review = True

            item = ReviewItem(
                page_id=page_id,
                document_id=document_id,
                raw_ocr_text=str(row.get("raw_ocr_text", row.get("page_text", "")) or ""),
                status=status,
                quality_class=quality_class,
                needs_human_review=needs_review,
                triage_label=self._triage_label(
                    status=status,
                    quality_class=quality_class,
                    needs_human_review=needs_review,
                    beta_scope_status=str(row.get("beta_scope_status", "internal_only_with_review") or "internal_only_with_review"),
                ),
                beta_scope_status=str(row.get("beta_scope_status", "internal_only_with_review") or "internal_only_with_review"),
                beta_scope_reason=str(row.get("beta_scope_reason", "policy_not_evaluated") or "policy_not_evaluated"),
                metadata={
                    "pdf_name": str(row.get("pdf_name", "") or ""),
                    "page": str(row.get("page", "") or ""),
                    "dataset_id": str(row.get("dataset_id", "") or ""),
                    "document_type": str(row.get("document_type", "") or ""),
                    "script_type": str(row.get("script_type", "") or ""),
                },
            )
            self._items[item.page_id] = item
            self._record_event(
                item=item,
                action="queue_created",
                actor_id=actor_id,
                from_state="",
                to_state=item.review_state,
                reason="queue_ingestion",
                before_text="",
                after_text=item.raw_ocr_text,
            )
            created.append(item)
        return created

    def assign_reviewer(self, page_id: str, reviewer_id: str, actor_id: str = "system") -> ReviewItem:
        item = self._get_item(page_id)
        from_state = item.review_state
        old_reviewer = item.reviewer_id
        item.reviewer_id = str(reviewer_id or "").strip()
        item.updated_at = _utc_now_iso()
        action = "reviewer_assigned" if item.reviewer_id else "reviewer_unassigned"
        reason = f"from:{old_reviewer or 'none'}"
        self._record_event(
            item=item,
            action=action,
            actor_id=actor_id,
            from_state=from_state,
            to_state=item.review_state,
            reason=reason,
            before_text=item.current_text(),
            after_text=item.current_text(),
        )
        return item

    def start_review(self, page_id: str, actor_id: str) -> ReviewItem:
        item = self._get_item(page_id)
        if item.review_state in {REVIEW_STATE_REJECTED, REVIEW_STATE_BLOCKED, REVIEW_STATE_EXPORTED}:
            raise ReviewWorkflowError(f"cannot start review from state {item.review_state}")

        from_state = item.review_state
        if item.review_state == REVIEW_STATE_NEEDS_REVIEW:
            item.review_state = REVIEW_STATE_IN_REVIEW
        if not item.reviewer_id:
            item.reviewer_id = _ensure_non_empty(actor_id, "actor_id")
        item.updated_at = _utc_now_iso()
        self._record_event(
            item=item,
            action="review_started",
            actor_id=actor_id,
            from_state=from_state,
            to_state=item.review_state,
            reason="triage_to_active_review",
            before_text=item.current_text(),
            after_text=item.current_text(),
        )
        return item

    def accept_ocr(self, page_id: str, actor_id: str, reason: str = "") -> ReviewItem:
        item = self._get_item(page_id)
        self._require_reviewable_state(item)
        from_state = item.review_state
        item.review_state = REVIEW_STATE_ACCEPTED
        item.final_reviewed_status = REVIEW_STATE_ACCEPTED
        item.updated_at = _utc_now_iso()
        self._record_event(
            item=item,
            action="accepted",
            actor_id=actor_id,
            from_state=from_state,
            to_state=item.review_state,
            reason=str(reason or "accepted_as_is"),
            before_text=item.current_text(),
            after_text=item.current_text(),
        )
        return item

    def correct_ocr(self, page_id: str, corrected_text: str, actor_id: str, reason: str = "") -> ReviewItem:
        item = self._get_item(page_id)
        self._require_reviewable_state(item)
        before_text = item.current_text()
        corrected = _ensure_non_empty(corrected_text, "corrected_text")
        from_state = item.review_state
        item.corrected_ocr_text = corrected
        item.review_state = REVIEW_STATE_CORRECTED
        item.final_reviewed_status = REVIEW_STATE_CORRECTED
        item.updated_at = _utc_now_iso()
        self._record_event(
            item=item,
            action="corrected",
            actor_id=actor_id,
            from_state=from_state,
            to_state=item.review_state,
            reason=str(reason or "manual_text_correction"),
            before_text=before_text,
            after_text=corrected,
        )
        return item

    def reject_ocr(self, page_id: str, actor_id: str, reason: str) -> ReviewItem:
        item = self._get_item(page_id)
        self._require_reviewable_state(item)
        from_state = item.review_state
        item.review_state = REVIEW_STATE_REJECTED
        item.final_reviewed_status = REVIEW_STATE_REJECTED
        item.updated_at = _utc_now_iso()
        self._record_event(
            item=item,
            action="rejected",
            actor_id=actor_id,
            from_state=from_state,
            to_state=item.review_state,
            reason=_ensure_non_empty(reason, "reason"),
            before_text=item.current_text(),
            after_text=item.current_text(),
        )
        return item

    def mark_page_unusable(self, page_id: str, actor_id: str, reason: str) -> ReviewItem:
        item = self._get_item(page_id)
        self._require_reviewable_state(item)
        from_state = item.review_state
        item.review_state = REVIEW_STATE_BLOCKED
        item.final_reviewed_status = REVIEW_STATE_BLOCKED
        item.updated_at = _utc_now_iso()
        self._record_event(
            item=item,
            action="blocked",
            actor_id=actor_id,
            from_state=from_state,
            to_state=item.review_state,
            reason=_ensure_non_empty(reason, "reason"),
            before_text=item.current_text(),
            after_text=item.current_text(),
        )
        return item

    def override_quality_warning(self, page_id: str, actor_id: str, reason: str) -> ReviewItem:
        item = self._get_item(page_id)
        if item.review_state in {REVIEW_STATE_REJECTED, REVIEW_STATE_BLOCKED, REVIEW_STATE_EXPORTED}:
            raise ReviewWorkflowError(f"cannot override quality warning from state {item.review_state}")

        override_reason = _ensure_non_empty(reason, "override_reason")
        from_state = item.review_state
        item.quality_warning_overridden = True
        item.quality_override_reason = override_reason
        item.updated_at = _utc_now_iso()
        self._record_event(
            item=item,
            action="quality_warning_overridden",
            actor_id=actor_id,
            from_state=from_state,
            to_state=item.review_state,
            reason="quality_warning_override",
            override_reason=override_reason,
            before_text=item.current_text(),
            after_text=item.current_text(),
        )
        return item

    def can_export(self, page_id: str, target: str = "private_beta") -> tuple[bool, str]:
        item = self._get_item(page_id)

        if item.review_state == REVIEW_STATE_EXPORTED:
            return True, "already_exported"

        if item.review_state not in {REVIEW_STATE_ACCEPTED, REVIEW_STATE_CORRECTED}:
            return False, "review_not_completed"

        if item.quality_warning_overridden and not item.quality_override_reason:
            return False, "override_reason_missing"

        if str(target or "private_beta").strip().lower() == "private_beta":
            if item.beta_scope_status == "exclude_from_private_beta":
                return False, "category_excluded_from_private_beta"
            if item.beta_scope_status == "internal_only_with_review":
                return False, "category_internal_only"
            if item.beta_scope_status != "allow_private_beta_with_mandatory_review":
                return False, "category_not_allowed_for_private_beta"

        return True, ""

    def mark_exported(self, page_id: str, actor_id: str, target: str = "private_beta") -> ReviewItem:
        item = self._get_item(page_id)
        allowed, reason = self.can_export(page_id, target=target)
        if not allowed:
            raise ReviewWorkflowError(f"export blocked: {reason}")

        from_state = item.review_state
        if item.final_reviewed_status not in {REVIEW_STATE_ACCEPTED, REVIEW_STATE_CORRECTED}:
            item.final_reviewed_status = from_state
        item.review_state = REVIEW_STATE_EXPORTED
        item.updated_at = _utc_now_iso()
        self._record_event(
            item=item,
            action="exported",
            actor_id=actor_id,
            from_state=from_state,
            to_state=item.review_state,
            reason=f"target={target}",
            before_text=item.current_text(),
            after_text=item.current_text(),
        )
        return item

    def get_unassigned_queue(self) -> list[ReviewItem]:
        return [
            item
            for item in self._items.values()
            if item.review_state in {REVIEW_STATE_NEEDS_REVIEW, REVIEW_STATE_IN_REVIEW} and not item.reviewer_id
        ]

    def get_items_by_state(self, review_state: str) -> list[ReviewItem]:
        if review_state not in REVIEW_STATES:
            raise ReviewWorkflowError(f"unknown review state: {review_state}")
        return [item for item in self._items.values() if item.review_state == review_state]

    def items_as_rows(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self._items.values()]

    def audit_as_rows(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self._audit_events]

    def to_snapshot(self) -> dict[str, Any]:
        return {
            "version": 1,
            "items": self.items_as_rows(),
            "audit_events": self.audit_as_rows(),
        }

    def save_snapshot(self, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_snapshot()
        path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load_snapshot(cls, input_path: str | Path) -> "ReviewerWorkflowStore":
        path = Path(input_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ReviewWorkflowError("invalid snapshot format: top-level object expected")

        store = cls()
        items = data.get("items", [])
        events = data.get("audit_events", [])
        if not isinstance(items, list) or not isinstance(events, list):
            raise ReviewWorkflowError("invalid snapshot format: items/audit_events must be arrays")

        for raw in items:
            if not isinstance(raw, dict):
                raise ReviewWorkflowError("invalid snapshot item")
            item = ReviewItem(
                page_id=str(raw.get("page_id", "") or ""),
                document_id=str(raw.get("document_id", "") or ""),
                raw_ocr_text=str(raw.get("raw_ocr_text", "") or ""),
                corrected_ocr_text=str(raw.get("corrected_ocr_text", "") or ""),
                status=str(raw.get("status", "") or ""),
                quality_class=str(raw.get("quality_class", "") or ""),
                needs_human_review=_as_bool(raw.get("needs_human_review", False)),
                review_state=str(raw.get("review_state", REVIEW_STATE_NEEDS_REVIEW) or REVIEW_STATE_NEEDS_REVIEW),
                reviewer_id=str(raw.get("reviewer_id", "") or ""),
                triage_label=str(raw.get("triage_label", "normal") or "normal"),
                quality_warning_overridden=_as_bool(raw.get("quality_warning_overridden", False)),
                quality_override_reason=str(raw.get("quality_override_reason", "") or ""),
                beta_scope_status=str(raw.get("beta_scope_status", "internal_only_with_review") or "internal_only_with_review"),
                beta_scope_reason=str(raw.get("beta_scope_reason", "policy_not_evaluated") or "policy_not_evaluated"),
                final_reviewed_status=str(raw.get("final_reviewed_status", "") or ""),
                created_at=str(raw.get("created_at", _utc_now_iso()) or _utc_now_iso()),
                updated_at=str(raw.get("updated_at", _utc_now_iso()) or _utc_now_iso()),
                metadata=dict(raw.get("metadata") or {}),
            )
            if not item.page_id:
                raise ReviewWorkflowError("invalid snapshot item: page_id is required")
            if item.review_state not in REVIEW_STATES:
                raise ReviewWorkflowError(f"invalid snapshot item state: {item.review_state}")
            store._items[item.page_id] = item

        for raw in events:
            if not isinstance(raw, dict):
                raise ReviewWorkflowError("invalid snapshot event")
            event = AuditEvent(
                event_id=str(raw.get("event_id", "") or ""),
                timestamp=str(raw.get("timestamp", _utc_now_iso()) or _utc_now_iso()),
                page_id=str(raw.get("page_id", "") or ""),
                document_id=str(raw.get("document_id", "") or ""),
                action=str(raw.get("action", "") or ""),
                actor_id=str(raw.get("actor_id", "") or ""),
                from_state=str(raw.get("from_state", "") or ""),
                to_state=str(raw.get("to_state", "") or ""),
                reason=str(raw.get("reason", "") or ""),
                override_reason=str(raw.get("override_reason", "") or ""),
                before_text=str(raw.get("before_text", "") or ""),
                after_text=str(raw.get("after_text", "") or ""),
                metadata=dict(raw.get("metadata") or {}),
            )
            if not event.event_id:
                raise ReviewWorkflowError("invalid snapshot event: event_id is required")
            store._audit_events.append(event)

        return store

    def write_audit_jsonl(self, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for event in self._audit_events:
                fh.write(json.dumps(event.to_dict(), ensure_ascii=True, sort_keys=True) + "\n")

    def _record_event(
        self,
        *,
        item: ReviewItem,
        action: str,
        actor_id: str,
        from_state: str,
        to_state: str,
        reason: str,
        before_text: str,
        after_text: str,
        override_reason: str = "",
    ) -> None:
        event = AuditEvent(
            event_id=f"evt_{uuid.uuid4().hex}",
            timestamp=_utc_now_iso(),
            page_id=item.page_id,
            document_id=item.document_id,
            action=str(action or "").strip(),
            actor_id=_ensure_non_empty(actor_id, "actor_id"),
            from_state=from_state,
            to_state=to_state,
            reason=str(reason or ""),
            override_reason=str(override_reason or ""),
            before_text=str(before_text or ""),
            after_text=str(after_text or ""),
            metadata={
                "reviewer_id": item.reviewer_id,
                "quality_class": item.quality_class,
                "needs_human_review": item.needs_human_review,
                "beta_scope_status": item.beta_scope_status,
                "final_reviewed_status": item.final_reviewed_status,
            },
        )
        self._audit_events.append(event)

    def _require_reviewable_state(self, item: ReviewItem) -> None:
        if item.review_state not in {REVIEW_STATE_NEEDS_REVIEW, REVIEW_STATE_IN_REVIEW}:
            raise ReviewWorkflowError(f"action not allowed from state {item.review_state}")

    def _get_item(self, page_id: str) -> ReviewItem:
        key = _ensure_non_empty(page_id, "page_id")
        if key not in self._items:
            raise ReviewWorkflowError(f"unknown page_id: {key}")
        return self._items[key]

    @staticmethod
    def _derive_page_id(row: dict[str, Any]) -> str:
        candidate = str(row.get("page_id", "") or "").strip()
        if candidate:
            return candidate

        candidate = str(row.get("benchmark_page_id", "") or "").strip()
        if candidate:
            return candidate

        pdf_name = str(row.get("pdf_name", "") or "").strip()
        page = str(row.get("page", "") or "").strip()
        if pdf_name and page:
            return f"{pdf_name}_page_{page}"

        raise ReviewWorkflowError("unable to derive page_id for review item")

    @staticmethod
    def _derive_document_id(page_id: str) -> str:
        if "_page_" in page_id:
            return page_id.split("_page_", 1)[0]
        return page_id

    @staticmethod
    def _triage_label(
        *,
        status: str,
        quality_class: str,
        needs_human_review: bool,
        beta_scope_status: str,
    ) -> str:
        status_norm = str(status or "").strip().lower()
        quality_norm = str(quality_class or "").strip().lower()
        scope_norm = str(beta_scope_status or "").strip().lower()

        if scope_norm in {"exclude_from_private_beta", "internal_only_with_review"}:
            return "policy_restricted"
        if status_norm not in {"success", "partial_success"}:
            return "ocr_failure"
        if quality_norm in {"weak_ocr", "failed_ocr"}:
            return "quality_risk"
        if needs_human_review:
            return "manual_review_required"
        return "normal"
