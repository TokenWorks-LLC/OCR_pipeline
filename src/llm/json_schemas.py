#!/usr/bin/env python3
"""
JSON schemas for LLM OCR correction with pydantic validation.

Defines strict request/response models with validation.

Author: Senior ML Engineer
Date: 2025-10-07
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator


class EditOperation(BaseModel):
    """Single edit operation applied by LLM."""
    pos: int = Field(..., description="Character position of edit")
    from_text: str = Field(..., alias="from", description="Original text")
    to_text: str = Field(..., alias="to", description="Corrected text")
    edit_type: Literal["subst", "insert", "delete"] = Field(..., alias="type")


class CorrectionResponse(BaseModel):
    """Expected JSON response from LLM."""
    span_id: str = Field(..., description="Identifier for this span")
    mode: Literal["fix_typos_only", "protect_transliteration"] = Field(..., description="Correction mode applied")
    lang_detected: Literal["en", "de", "fr", "it", "tr", "unknown"] = Field(..., description="Detected language")
    corrected_text: str = Field(..., description="Corrected text output")
    applied_edits: List[EditOperation] = Field(default_factory=list, description="List of edits applied")
    edit_ratio: float = Field(..., ge=0.0, le=1.0, description="Fraction of characters changed")
    diacritic_restored: bool = Field(default=False, description="Whether diacritics were restored")
    confidence: float = Field(..., ge=0.0, le=1.0, description="LLM confidence in correction")
    notes: str = Field(default="", description="Optional notes or warnings")
    
    @field_validator('edit_ratio')
    @classmethod
    def validate_edit_ratio(cls, v):
        """Ensure edit ratio is reasonable."""
        if v < 0.0 or v > 1.0:
            raise ValueError("edit_ratio must be between 0.0 and 1.0")
        return v
    
    class Config:
        populate_by_name = True  # Allow alias names


class ContextInfo(BaseModel):
    """Context information for LLM."""
    prev_line: Optional[str] = Field(default=None, description="Previous line for context")
    next_line: Optional[str] = Field(default=None, description="Next line for context")


class CorrectionFlags(BaseModel):
    """Flags controlling correction behavior."""
    is_transliteration_suspected: bool = Field(default=False, description="Is this suspected Akkadian transliteration?")
    preserve_brackets: bool = Field(default=True, description="Preserve brackets [](){}")
    preserve_footnote_markers: bool = Field(default=True, description="Preserve footnote markers and superscripts")
    preserve_hyphenation: bool = Field(default=True, description="Preserve hyphens and dashes")
    max_relative_change_chars: float = Field(default=0.12, description="Maximum allowed edit ratio")


class CorrectionRequest(BaseModel):
    """Request payload for LLM correction."""
    schema_version: str = Field(default="1.0", description="Schema version")
    span_id: str = Field(..., description="Unique span identifier")
    language_hint: str = Field(..., description="Language hint (en/de/fr/it/tr)")
    original_text: str = Field(..., description="Original OCR text")
    context: ContextInfo = Field(default_factory=ContextInfo, description="Context lines")
    flags: CorrectionFlags = Field(default_factory=CorrectionFlags, description="Correction flags")
    
    def to_prompt_json(self) -> str:
        """Convert to JSON string for prompt."""
        return self.model_dump_json(indent=2, by_alias=True)


# Response schema as JSON for LLM
RESPONSE_SCHEMA_JSON = """{
  "span_id": "string (required)",
  "mode": "fix_typos_only | protect_transliteration (required)",
  "lang_detected": "en | de | fr | it | tr | unknown (required)",
  "corrected_text": "string (required)",
  "applied_edits": [
    {
      "pos": "int",
      "from": "string",
      "to": "string",
      "type": "subst | insert | delete"
    }
  ],
  "edit_ratio": "float 0.0-1.0 (required)",
  "diacritic_restored": "boolean",
  "confidence": "float 0.0-1.0 (required)",
  "notes": "string (optional)"
}"""


if __name__ == '__main__':
    # Test schema validation
    import json
    
    # Valid response
    test_response = {
        "span_id": "test_001",
        "mode": "fix_typos_only",
        "lang_detected": "en",
        "corrected_text": "Hello world",
        "applied_edits": [
            {"pos": 0, "from": "Helo", "to": "Hello", "type": "subst"}
        ],
        "edit_ratio": 0.09,
        "diacritic_restored": False,
        "confidence": 0.95,
        "notes": "Fixed typo"
    }
    
    try:
        validated = CorrectionResponse(**test_response)
        print(f"✅ Schema validation passed: {validated.span_id}")
        print(f"   Mode: {validated.mode}")
        print(f"   Edit ratio: {validated.edit_ratio}")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
