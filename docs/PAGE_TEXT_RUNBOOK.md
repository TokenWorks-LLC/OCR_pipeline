# Page Text Extraction Runbook

Canonical operations guide for page-level extraction workflows.

This document consolidates the previously duplicated guidance from:

- `.merge_protect/README.md`
- `.merge_protect/OCR _PIPELINE_RUNBOOK.md`

## Goal

Produce clean per-page text output with Akkadian detection and fortified OCR fallback:

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

### Fast throughput

```bash
python tools/run_page_text.py \
  --manifest path/to/manifest.tsv \
  --output-root reports/page_text_fast \
  --prefer-text-layer \
  --status-bar
```

### Quality mode

```bash
python tools/run_page_text.py \
  --manifest path/to/manifest.tsv \
  --output-root reports/page_text_quality \
  --prefer-text-layer \
  --ocr-fallback ensemble \
  --status-bar
```

## Inputs

- Use either `--manifest` **or** `--inputs`
- Prefer manifests for large runs (faster startup, deterministic scope)
- Enable `--ocr-fallback ensemble` when scanned or degraded pages need OCR

## Preprocessing And Ensemble

- The active OCR fallback now builds a broader page-prep stack: grayscale, denoise, autocontrast, sharpened, binary, adaptive-thresholded, and morphology-cleaned variants
- Quality mode performs orientation search across `0`, `90`, `180`, and `270` degrees and keeps the strongest consensus result
- OCR lines with bounding boxes are re-ordered with column-aware reading order reconstruction before fusion, which improves two-column academic layouts
- Available OCR backends are combined with consensus fusion rather than trusting a single model blindly
- Fusion is script-aware and tries to preserve diacritics for Akkadian, German, Arabic, and other non-ASCII text
- If multiple engines disagree heavily, the ensemble falls back to the strongest whole-text consensus candidate

## Test Sets

- Synthetic regression coverage for ensemble behavior lives in `tests/test_ensemble_support.py`
- The current test set includes diacritic preservation, noisy outlier rejection, rotation recovery, multi-column ordering, and protected-runner ensemble fallback
- End-to-end CLI safety tests remain in `tests/test_pipeline_e2e.py`

## Akkadian Detection Notes

- Detection uses profile rules (default: `profiles/akkadian_strict.json`)
- Preserve transliteration spans and diacritics (`š ṣ ṭ ḫ ā ē ī ū`) in source documents
- Validate detection quality with targeted spot checks on `has_akkadian=true` rows

## Output Artifacts

1. `client_page_text.csv` (main deliverable)
2. `progress.csv` (timing + processing-path telemetry)

## Operational Checklist

1. Validate configuration profile (`profiles/akkadian_strict.json`)
2. Run a small manifest smoke test
3. Run production manifest in fast or quality mode
4. Spot-check `has_akkadian=true` rows and multilingual diacritics
5. Archive output folder under `reports/`

## Troubleshooting

- No text extracted: add `--ocr-fallback ensemble`
- Sideways or upside-down scans: keep ensemble enabled so rotation search can correct the page
- Two-column reading order looks wrong: prefer ensemble mode so bbox-aware reflow is applied
- Akkadian misses: tune thresholds in profile
- Memory pressure: prefer `--ocr-fallback none` and split manifests
