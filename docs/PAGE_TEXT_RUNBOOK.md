# Page Text Extraction Runbook

Canonical operations guide for page-level extraction workflows.

This document consolidates the previously duplicated guidance from:
- `.merge_protect/README.md`
- `.merge_protect/OCR _PIPELINE_RUNBOOK.md`

## Goal

Produce clean per-page text output with Akkadian detection:

- Output columns: `pdf_name,page,page_text,has_akkadian`
- One row per page
- UTF-8 BOM encoding for spreadsheet compatibility

## Primary Command

```bash
python tools/run_page_text.py \
  --inputs "path/to/pdfs" \
  --output-root "reports/output" \
  --prefer-text-layer \
  --status-bar
```

## Recommended Modes

### Fast throughput (no LLM)

```bash
python tools/run_page_text.py \
  --manifest manifests/secondary_sources_full_SORTED.txt \
  --output-root reports/page_text_fast \
  --prefer-text-layer \
  --llm-off \
  --status-bar
```

### Quality mode (LLM correction on)

```bash
python tools/run_page_text.py \
  --manifest manifests/secondary_sources_full_SORTED.txt \
  --output-root reports/page_text_quality \
  --prefer-text-layer \
  --llm-on \
  --status-bar
```

## Inputs

- Use either `--manifest` **or** `--inputs`
- Prefer manifests for large runs (faster startup, deterministic scope)
- Enable `--ocr-fallback paddle` only when scanned pages need OCR

## Akkadian Safety Rules

- Preserve transliteration spans and diacritics (`š ṣ ṭ ḫ ā ē ī ū`)
- Guard Akkadian tokens during correction
- Reject over-aggressive rewrites using edit-ratio thresholds

## Output Artifacts

1. `client_page_text.csv` (main deliverable)
2. `progress.csv` (timing + processing-path telemetry)

## Operational Checklist

1. Validate configuration profile (`profiles/akkadian_strict.json`)
2. Run a small manifest smoke test
3. Run production manifest in fast or quality mode
4. Spot-check `has_akkadian=true` rows
5. Archive output folder under `reports/`

## Troubleshooting

- No text extracted: add `--ocr-fallback paddle`
- Akkadian misses: tune thresholds in profile
- LLM violations: lower temperature or disable LLM for batch pass
- Memory pressure: prefer `--llm-off` and split manifests
