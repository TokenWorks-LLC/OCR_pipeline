# LLM Correction System

This document describes the LLM-in-the-loop correction system with strict fix-typos-only protocol, caching, and language routing.

## Overview

The LLM correction system automatically identifies low-confidence OCR spans and routes them to a language model for typo correction while preserving the original document structure and scholarly content.

## Key Features

- **Strict Fix-Typos-Only Protocol**: LLM can only fix obvious typos, never add/remove/reorder content
- **Span-Level Routing**: Only low-confidence text spans are sent to LLM
- **Persistent Caching**: Results cached with SHA256 keys for performance
- **Language-Aware**: Different confidence thresholds per language
- **Akkadian Detection**: Preserves transliteration and cuneiform content
- **Kill Switch**: Runtime enable/disable via `LLM_ENABLED` environment variable

## System Prompt

The LLM receives this exact system prompt to ensure consistent behavior:

```
You are a text normalization tool. Only correct obvious OCR typos. Do not add, remove, reorder, merge, or split lines. Do not change layout, punctuation style, numbers, footnote markers, or scholarly brackets. Never invent text. If uncertain, keep the original. Output valid JSON only, following the schema.
```

## JSON API Schema

The LLM receives and must respond with valid JSON following this schema:

### Input Schema

```json
{
  "schema_version": "v1",
  "language": "<ISO-LANGUAGE-CODE>",
  "mode": "fix_typos_only",
  "span_id": "<PAGE_ID>:<BLOCK_ID>:<LINE_ID>",
  "original_text": "<ORIGINAL_TEXT>",
  "context": {
    "prev_line": "<OPTIONAL>",
    "next_line": "<OPTIONAL>"
  },
  "constraints": {
    "preserve_brackets": true,
    "preserve_footnote_markers": true,
    "max_relative_change_chars": 0.15
  },
  "return": {
    "corrected_text": "string",
    "applied_edits": [
      {"pos": "int", "from": "string", "to": "string"}
    ],
    "notes": "string"
  }
}
```

### Output Schema

```json
{
  "corrected_text": "The corrected text with typos fixed",
  "applied_edits": [
    {"pos": 12, "from": "teh", "to": "the"},
    {"pos": 25, "from": "recieve", "to": "receive"}
  ],
  "notes": "Fixed 2 common OCR typos"
}
```

## Routing Logic

### Confidence Thresholds

Spans are sent to LLM only if their confidence is below language-specific thresholds:

| Language | Threshold | Rationale |
|----------|-----------|-----------|
| English (en) | 0.86 | Baseline for Latin script |
| German (de) | 0.85 | Umlauts and ß can reduce confidence |
| French (fr) | 0.84 | Accented characters |
| Italian (it) | 0.84 | Accented characters |
| Turkish (tr) | 0.83 | Many diacritical marks |

### Filtering Rules

Spans are **excluded** from LLM correction if:

1. **Text too short**: Less than 8 characters
2. **Akkadian detected**: Contains transliteration patterns (`š ṣ ṭ ḫ ā ē ī ū`) or cuneiform Unicode
3. **Mostly digits**: >70% numeric content (tables, page numbers)
4. **High confidence**: Above language threshold
5. **LLM disabled**: `LLM_ENABLED=false` environment variable

### Akkadian Detection

The system preserves scholarly transliteration by detecting:

- **Diacritical patterns**: `š ṣ ṭ ḫ ā ē ī ū â ê î û`
- **Cuneiform Unicode**: Block U+12000–U+123FF
- **Diacritic density**: >3% of characters have Unicode code points >127

When Akkadian is detected, the span is **not** sent to LLM, preserving the original transliteration.

## Caching System

### Cache Key Generation

Cache keys are SHA256 hashes of:
```python
{
  'model_id': 'mistral:latest',
  'prompt_version': 'fix-typos-only-v1',
  'language': 'en',
  'text': 'normalized_text',
  'constraints': [('preserve_brackets', True), ...]
}
```

### Cache Storage

- **Format**: SQLite database at `data/.cache/llm/llm_cache.db`
- **Schema**: `(cache_key, corrected_text, applied_edits, latency_ms, model_id, prompt_version, created_at, notes)`
- **Performance**: Indexed by `prompt_version` for fast lookups

### Cache Management

```bash
# Purge old prompt versions
python -c "from src.llm_cache import LLMCache; LLMCache().purge_by_version('old-version')"

# Purge entries older than 30 days
python -c "from src.llm_cache import LLMCache; LLMCache().purge_old_entries(30)"

# Get cache statistics
python -c "from src.llm_cache import LLMCache; print(LLMCache().get_stats())"
```

## Configuration

### Environment Variables

```bash
# Enable/disable LLM correction
export LLM_ENABLED=true

# Provider configuration
export LLM_PROVIDER=ollama
export LLM_MODEL=mistral:latest
export LLM_BASE_URL=http://localhost:11434
```

### Provider Support

#### Ollama
```python
provider = "ollama"
base_url = "http://localhost:11434"
model = "mistral:latest"  # or llama2, codellama, etc.
```

#### llama.cpp Server
```python
provider = "llamacpp" 
base_url = "http://localhost:8080"
model = "llama-2-7b-chat"
```

#### Disable LLM
```python
provider = "none"
# or set LLM_ENABLED=false
```

## Performance Tuning

### Concurrent Processing

- **Max Workers**: 3 concurrent correction threads (configurable)
- **Timeout**: 30-45 seconds per request
- **Batching**: Spans processed in parallel with thread pool

### Memory Management

- **Cache Cleanup**: Automatic cleanup of old entries
- **Model Unloading**: LLM instances can be released when inactive
- **GPU Memory**: Shared GPU memory allocation with OCR engine

## Quality Controls

### Character Change Limits

Corrections are rejected if they exceed the character change limit:
- **Max relative change**: 15% of original text length
- **Example**: 100-character text can change by at most 15 characters

### Edit Validation

Applied edits are validated for:
- **Position bounds**: Edit positions within text bounds
- **Preservation**: Brackets, footnote markers, and scholarly notation unchanged
- **Consistency**: Corrected text matches applied edits

## Metrics and Telemetry

### Correction Statistics

```json
{
  "spans_processed": 1234,
  "spans_sent_to_llm": 89,
  "cache_hits": 45,
  "akkadian_detected": 12,
  "low_confidence_filtered": 89,
  "total_latency_ms": 15678,
  "average_latency_ms": 176.2,
  "cache_hit_rate": 0.506
}
```

### Per-Page Telemetry

- `ms_detect`: Detection time
- `ms_recognize`: Recognition time  
- `ms_preprocess`: Preprocessing time
- `ms_llm`: LLM correction time
- `ms_total`: Total page processing time
- `llm_spans`: Number of spans sent to LLM
- `cache_hits`: Cache hits for this page

## Troubleshooting

### Common Issues

#### LLM Not Responding
```bash
# Check LLM server status
curl http://localhost:11434/api/tags  # Ollama
curl http://localhost:8080/health     # llama.cpp

# Check logs
tail -f reports/<RUN_ID>/logs/pipeline.jsonl
```

#### Low Cache Hit Rate
- Check if prompt version changed
- Verify text normalization consistency
- Monitor cache size and purge old entries

#### Poor Correction Quality
- Verify system prompt in LLM logs
- Check character change limits
- Review applied edits in detailed results
- Consider adjusting confidence thresholds

### Performance Optimization

#### Speed Up Corrections
1. Increase `max_concurrent_corrections`
2. Reduce `timeout` for faster failures
3. Raise confidence thresholds to send fewer spans
4. Use faster LLM model

#### Improve Quality
1. Lower confidence thresholds
2. Increase `timeout` for complex corrections
3. Use larger/better LLM model
4. Add more context lines

## Integration Examples

### With Pipeline Profiles

```python
# Quality profile: LLM enabled with low thresholds
profile = profile_manager.get_profile('quality')
profile.llm.enable_llm_correction = True
profile.llm.confidence_thresholds['en'] = 0.86

# Fast profile: LLM disabled
profile = profile_manager.get_profile('fast') 
profile.llm.enable_llm_correction = False
```

### Custom Correction Pipeline

```python
from src.llm_correction_safe import FixTyposLLMCorrector
from src.multilang_ocr import OCRSpan

# Initialize corrector
corrector = FixTyposLLMCorrector(
    provider="ollama",
    model="mistral:latest",
    max_workers=3
)

# Create OCR span
span = OCRSpan(
    text="Teh quik brown fox",
    confidence=0.7,
    span_id="page1:block2:line3",
    bbox=(100, 200, 300, 25),
    language="en"
)

# Correct span
result = corrector.correct_span(span)
print(f"Original: {result.original_text}")
print(f"Corrected: {result.corrected_text}")
print(f"Edits: {result.applied_edits}")
```