# Experiment Tracking Dashboard

This document explains how to track OCR progress over time with static, machine-readable artifacts.

## What is generated

When you run the evaluator, it now also writes experiment tracking outputs to the same output directory:

- `metrics_summary.md`
- `metrics_summary.csv`
- `metrics_by_language.csv`
- `metrics_by_layout.csv`
- `metrics_by_engine.csv`
- `regression_report.md`
- `failing_pages.csv`

Additional machine-readable outputs:

- `experiment_run_metadata.json`
- `experiment_tracking.json`
- `experiment_history.jsonl`
- `metrics_by_script.csv`
- `metrics_by_document_type.csv`
- `metrics_by_scan_quality.csv`
- `metrics_by_difficulty.csv`
- `metrics_by_languages_present.csv`

Optional domain-specific tracking included in `metrics_summary.csv` and `experiment_tracking.json` when data is present:

- diacritic preservation mean
- transliteration token validity mean
- lexicon coverage mean
- protected character changes (mean/total)

## Run command

```bash
python tools/evaluate_gold.py \
  --ocr-csv reports/eval_gold/client_page_text.csv \
  --gold-csv data/gold_data/gold_pages.csv \
  --progress-csv reports/eval_gold/progress.csv \
  --output-dir reports/eval_gold_tracking \
  --baseline-dir reports/eval_gold_previous
```

Optional metadata controls:

```bash
--run-id run_2026_05_19
--config-file config_eval_advanced_v3.json
--engine-versions-json '{"paddleocr":"2.7.0","kraken":"5.3.0"}'
--model-versions-json models/version_manifest.json
--gold-set-version gold-v3
--experiment-history-jsonl reports/experiment_tracking/history.jsonl
```

## Baseline-only comparison tool

You can also compare two existing runs directly:

```bash
python tools/compare_experiment_runs.py \
  --current-dir reports/eval_gold_tracking \
  --baseline-dir reports/eval_gold_previous \
  --output-dir reports/eval_gold_tracking
```

## Recommended workflow

1. Run OCR pipeline and produce `client_page_text.csv` + `progress.csv`.
2. Run `tools/evaluate_gold.py` with the same gold set and output directory.
3. Inspect `metrics_summary.md` for top-level trends.
4. Inspect `regression_report.md` for improved/regressed pages, languages, layouts, and engines.
5. Use `failing_pages.csv` to prioritize fixes.
6. Use `experiment_history.jsonl` to track progress over time.

## Notes

- All dashboard outputs are static CSV/JSON/Markdown files.
- No external tracking service is required.
- If baseline inputs are missing, the regression report is still generated with a clear no-baseline note.
- Overall regression status uses weighted relative metric deltas so large runtime swings do not hide OCR-quality regressions.
