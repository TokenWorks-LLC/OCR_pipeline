# Gold-Set OCR Evaluation Harness

This document explains how to build gold samples and run multilingual OCR evaluation.

## Goals

The evaluator is multilingual-first and script-aware.

- Works for English, German, French, Arabic, transliterated Akkadian, and future languages/scripts.
- Keeps core metrics language-agnostic.
- Runs Akkadian/transliteration metrics only when enabled or when page metadata indicates transliteration.
- Reports failures directly (empty outputs, timeouts, and regressions are never hidden).

## 1) Gold-Set Structure

Gold rows are page-level records.

Minimum fields:

- pdf_name
- page (1-based)
- ground_truth_text
- language_primary
- script_type
- document_type
- layout_type

Recommended metadata fields:

- language_primary
- languages_present
- script_type
- document_type
- layout_type
- has_tables
- has_footnotes
- has_columns
- has_diacritics
- has_transliteration
- scan_quality
- expected_difficulty
- expected_special_handling

Use these assets as starting points:

- schema: data/gold_data/gold_pages_schema.json
- template: data/gold_data/gold_pages_template.csv

## 2) Running Evaluation

Example command:

python tools/evaluate_gold.py \
 --ocr-csv reports/eval_gold/client_page_text.csv \
 --gold-csv data/gold_data/gold_pages.csv \
 --progress-csv reports/eval_gold/progress.csv \
 --output-dir reports/eval_gold_v2

Useful options:

- --baseline-dir reports/eval_gold_previous
- --unicode-form NFC
- --normalized-unicode-form NFKC
- --whitespace-mode collapse
- --strip-punctuation
- --casefold
- --enable-akkadian-metrics
- --disable-metadata-akkadian
- --run-id <id>
- --config-file <path>
- --engine-versions-json <json-or-file>
- --model-versions-json <json-or-file>
- --gold-set-version <label>
- --experiment-history-jsonl <path>
- --disable-experiment-tracking

## 3) Output Artifacts

The evaluator writes:

- evaluation_summary.csv
- per_page_metrics.csv
- per_engine_metrics.csv
- per_language_metrics.csv
- per_layout_metrics.csv
- confusion_matrix.json
- run_metadata.json
- metrics_summary.md
- metrics_summary.csv
- metrics_by_language.csv
- metrics_by_layout.csv
- metrics_by_engine.csv
- regression_report.md
- failing_pages.csv
- experiment_run_metadata.json
- experiment_tracking.json
- experiment_history.jsonl

## 4) Core Metrics

General OCR metrics include:

- CER
- WER
- normalized CER
- normalized WER
- empty output rate
- timeout rate
- runtime per page (mean/p50/p90/p95)
- runtime per engine (mean/p50/p90/p95, best effort)
- confidence stats when available

## 5) Multilingual/Script-Aware Metrics

Evaluator computes:

- Unicode normalization checks
- diacritic preservation rate
- punctuation preservation rate
- whitespace preservation rate
- right-to-left character coverage
- script-specific character coverage

## 6) Optional Akkadian/Transliteration Metrics

When enabled, evaluator computes:

- š / ṣ / ṭ / ḫ preservation
- macron preservation (for example ī, ū)
- subscript/superscript preservation
- hyphenated sign preservation
- bracket/damage marker preservation
- unknown transliteration token rate

## 7) Regression Tracking

When a baseline is provided, run_metadata.json includes:

- improved metrics
- worsened metrics
- unchanged metrics
- statistically suspicious changes
- most improved pages
- most regressed pages

## 8) Notes

- Runtime per engine is exact when engine runtime telemetry exists; otherwise the evaluator reports best-effort split estimates.
- Keep gold metadata consistent and explicit. Better metadata means better slicing in per_language and per_layout reports.
