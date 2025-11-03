"""
LLM Prompt Templates for OCR Post-Correction

Constructs prompts for local LLM (Ollama) that enforce JSON-mode output
with strict constraints for typo/diacritic restoration.
"""

import json
from typing import Dict, Any, Optional
from .json_schemas import CorrectionRequest, CorrectionResponse


def get_system_message(mode: str = "fix_typos_only") -> str:
    """
    Get system message that defines LLM role and constraints.
    
    Args:
        mode: Either 'fix_typos_only' or 'protect_transliteration'
    
    Returns:
        System message string
    """
    base_message = """You are an OCR typo & diacritic restoration tool for academic texts.

**YOUR ROLE:**
- Fix ONLY obvious OCR typos and restore missing diacritics
- Keep exact word boundaries, line breaks, and token order
- Output valid JSON matching the provided schema

**STRICT RULES:**
1. NO translation or rewriting
2. NO adding/removing words
3. NO changing meaning or structure
4. Fix ONLY: obvious typos (e.g., "Hethiter" → "Hethiter"), missing diacritics (e.g., "Ubersetzung" → "Übersetzung")
5. Preserve ALL brackets: [], (), {}
6. Keep exact spacing and line breaks
7. Output ONLY valid JSON (no markdown, no explanations)
"""
    
    if mode == "protect_transliteration":
        base_message += """
**TRANSLITERATION MODE:**
- This text contains Akkadian/cuneiform transliteration (š, ṣ, ṭ, ḫ, ā, ē, ī, ū)
- Make MINIMAL changes (≤3% of characters)
- DO NOT "fix" transliteration characters to ASCII equivalents
- Preserve hyphens, colons, and determinatives (ᵈ, ᵐ)
- When in doubt, leave the text unchanged
"""
    else:
        base_message += """
**MODERN TEXT MODE:**
- Fix common OCR errors in modern European languages
- Restore diacritics: ä, ö, ü, ß, é, è, ê, à, ç, etc.
- Max changes: ≤12% of characters
"""
    
    return base_message.strip()


def build_correction_prompt(
    request: CorrectionRequest,
    include_response_schema: bool = True
) -> str:
    """
    Build user prompt from CorrectionRequest.
    
    Args:
        request: Pydantic CorrectionRequest model
        include_response_schema: Whether to include response schema in prompt
    
    Returns:
        User prompt string with JSON structure
    """
    # Convert request to dict for JSON serialization
    request_dict = request.model_dump(exclude_none=True)
    
    prompt_parts = [
        "**INPUT TEXT:**",
        json.dumps(request_dict, indent=2, ensure_ascii=False),
        "",
        "**TASK:**",
        "1. Read the 'original_text'",
        "2. Fix ONLY obvious OCR typos and restore missing diacritics",
        "3. Check language_hint and context for validation",
        "4. If is_transliteration_suspected=true, make MINIMAL changes (≤3%)",
        "5. Otherwise, limit changes to ≤12% of characters",
        "",
    ]
    
    if include_response_schema:
        # Add response schema template
        schema_example = {
            "span_id": request.span_id,
            "mode": request.flags.mode if hasattr(request.flags, 'mode') else "fix_typos_only",
            "lang_detected": request.language_hint or "unknown",
            "corrected_text": "<your corrected text here>",
            "applied_edits": [
                {
                    "pos": 0,
                    "from": "exarnple",
                    "to": "example",
                    "type": "subst"
                }
            ],
            "edit_ratio": 0.05,
            "diacritic_restored": True,
            "confidence": 0.95,
            "notes": "Fixed typo: exarnple -> example"
        }
        
        prompt_parts.extend([
            "**OUTPUT FORMAT (JSON only, no markdown):**",
            json.dumps(schema_example, indent=2, ensure_ascii=False),
            "",
            "**IMPORTANT:**",
            "- Output ONLY the JSON object above",
            "- Do NOT wrap in markdown code blocks",
            "- 'corrected_text' must match original line breaks",
            "- 'applied_edits' must list every character-level change",
            "- 'edit_ratio' = (chars changed) / (total chars)",
            "- 'confidence' = how certain you are (0.0-1.0)",
            ""
        ])
    
    return "\n".join(prompt_parts)


def build_full_prompt(
    request: CorrectionRequest,
    mode: Optional[str] = None
) -> Dict[str, str]:
    """
    Build complete prompt with system and user messages.
    
    Args:
        request: CorrectionRequest model
        mode: Override mode (defaults to request.flags.mode if available)
    
    Returns:
        Dict with 'system' and 'user' keys for LLM chat interface
    """
    # Determine mode
    if mode is None:
        mode = getattr(request.flags, 'mode', 'fix_typos_only')
    
    system_msg = get_system_message(mode=mode)
    user_msg = build_correction_prompt(request, include_response_schema=True)
    
    return {
        "system": system_msg,
        "user": user_msg
    }


def validate_response_schema(response_json: Dict[str, Any]) -> CorrectionResponse:
    """
    Validate LLM response against pydantic schema.
    
    Args:
        response_json: Raw JSON dict from LLM
    
    Returns:
        Validated CorrectionResponse model
    
    Raises:
        ValidationError: If response doesn't match schema
    """
    return CorrectionResponse(**response_json)


# Example usage for testing
if __name__ == "__main__":
    from src.llm.json_schemas import CorrectionFlags, ContextInfo
    
    # Test modern text mode
    print("=" * 80)
    print("MODERN TEXT MODE (German with missing diacritics)")
    print("=" * 80)
    
    request = CorrectionRequest(
        schema_version="1.0",
        span_id="test_line_1",
        language_hint="de",
        original_text="Die Ubersetzung der hethitischen Texte ist schwierig.",
        context=ContextInfo(
            prev_line="Kapitel 3: Hethitologie",
            next_line="Viele Forscher arbeiten daran."
        ),
        flags=CorrectionFlags(
            is_transliteration_suspected=False,
            max_relative_change_chars=0.12,
            mode="fix_typos_only"
        )
    )
    
    prompt = build_full_prompt(request)
    print("\n**SYSTEM:**")
    print(prompt["system"])
    print("\n**USER:**")
    print(prompt["user"])
    
    # Test transliteration mode
    print("\n" + "=" * 80)
    print("TRANSLITERATION MODE (Akkadian)")
    print("=" * 80)
    
    request_akk = CorrectionRequest(
        schema_version="1.0",
        span_id="test_line_2",
        language_hint="unknown",
        original_text="ša-ar-ru šarrum LUGAL ᵈUTU",
        context=ContextInfo(
            prev_line="Transliteration:",
            next_line="Translation: The king"
        ),
        flags=CorrectionFlags(
            is_transliteration_suspected=True,
            max_relative_change_chars=0.03,
            mode="protect_transliteration"
        )
    )
    
    prompt_akk = build_full_prompt(request_akk)
    print("\n**SYSTEM:**")
    print(prompt_akk["system"])
    print("\n**USER:**")
    print(prompt_akk["user"])
    
    print("\n" + "=" * 80)
    print("Schema validation test")
    print("=" * 80)
    
    # Test valid response
    valid_response = {
        "span_id": "test_line_1",
        "mode": "fix_typos_only",
        "lang_detected": "de",
        "corrected_text": "Die Übersetzung der hethitischen Texte ist schwierig.",
        "applied_edits": [
            {"pos": 4, "from": "U", "to": "Ü", "type": "subst"}
        ],
        "edit_ratio": 0.02,
        "diacritic_restored": True,
        "confidence": 0.98,
        "notes": "Restored German umlaut: Ubersetzung -> Übersetzung"
    }
    
    try:
        validated = validate_response_schema(valid_response)
        print("✅ Valid response validated successfully")
        print(f"   Corrected: {validated.corrected_text}")
        print(f"   Edits: {len(validated.applied_edits)}")
        print(f"   Edit ratio: {validated.edit_ratio:.2%}")
        print(f"   Confidence: {validated.confidence:.2f}")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
