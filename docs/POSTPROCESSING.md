# Postprocessing Framework

This document describes the modular multilingual postprocessing layer for OCR output.

## Goals

- Keep raw OCR text intact and auditable.
- Apply language-agnostic cleanup first.
- Select language/script adapters without hardcoding one language globally.
- Support optional scholarly and Akkadian transliteration behavior as adapters.
- Keep all corrections traceable and reversible.

## Processing Stages

The postprocessing pipeline in `production/postprocessing/pipeline.py` applies:

1. `raw OCR text`
2. `general cleanup` (`production/postprocessing/cleanup.py`)
3. `Unicode normalization` (NFC)
4. `language/script hint handling + adapter selection`
5. `adapter-specific lexicon/rule correction`
6. `optional guarded model correction`
7. `final quality scoring + review flags`

## Core Modules

- `production/postprocessing/cleanup.py`

  - Unicode normalization
  - whitespace normalization
  - obvious OCR garbage filtering
  - paragraph boundary preservation

- `production/postprocessing/lexicon.py`

  - in-memory lexicon with domains
  - exact + normalized lookup
  - edit-distance suggestions
  - frequency-aware ranking
  - domain-specific vocab support

- `production/postprocessing/adapters.py`

  - plugin-style language/script adapters:
    - `default_latin`
    - `english`
    - `german`
    - `french`
    - `arabic_or_rtl_placeholder`
    - `scholarly_transliteration`
    - `akkadian_transliteration`
  - adapter responsibilities:
    - tokenization
    - protected token handling
    - correction rules
    - quality metrics

- `production/postprocessing/model_correction.py`
  - optional guarded model correction
  - strict edit budget
  - line-structure preservation
  - protected-character checks
  - quality-nondegradation gate
  - diff output

## Akkadian Transliteration Adapter

Akkadian logic is adapter-local (`akkadian_transliteration`) and not global.

Protected behavior includes:

- diacritics: `\u0161 \u1e63 \u1e6d \u1e2b \u012b \u016b`
- transliteration token patterns with hyphens
- bracketed/damage-style segments
- sign-number compatible tokens

Tracked metrics include:

- diacritic preservation
- transliteration token validity
- lexicon coverage
- unknown token rate
- suspicious correction rate

## Output Fields

Per page output metadata now records:

- `raw_text`
- `cleaned_text`
- `corrected_text`
- `adapter_used`
- `corrections_applied`
- `correction_confidence`
- `lexicon_coverage`
- `unknown_token_rate`
- `protected_character_changes`
- `needs_human_review`

Plus:

- `correction_diff`
- `postprocess_quality_score`
- `postprocess_quality_metrics`
- `model_correction_reason`

Region records in `layout_regions.jsonl` include nested `postprocessing` metadata for each region with text.

## spaCy Guidance

The framework exposes a lightweight compatibility check via `PostprocessingPipeline.spacy_support_status()`.

Current recommendation:

- Prefer adapter tokenization and rule-based processing by default.
- Use spaCy tokenizer/EntityRuler only when language-specific gains are clear.
- Do not train spaCy models without enough labeled OCR correction data.

## Optional Model Correction Policy

Model correction is optional and guarded:

- disabled by default
- no hallucination-prone free rewrite stage
- strict edit budget
- reject edits that break line/page structure
- reject edits that modify protected scholarly characters
- reject edits when quality score worsens
- always keep raw OCR text and correction diff available

## How To Add A New Language Adapter

1. Add a new adapter class in `production/postprocessing/adapters.py`.
2. Define:
   - `name`
   - `lexicon_domain`
   - token pattern
   - protected chars/patterns
   - correction behavior
   - quality thresholds/metrics
3. Register the adapter in `adapter_registry()`.
4. Add selection logic in `select_adapter_name()`.
5. Add/update domain lexicon entries.
6. Add tests in `tests/test_postprocessing_pipeline.py`.
