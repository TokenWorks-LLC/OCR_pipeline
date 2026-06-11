# Quality Scoring And Launch Gates

This document describes page/document/run quality scoring and launch-gate behavior used by the page-text pipeline.

## Scope

Quality scoring is designed to be multilingual-safe and to avoid false-positive success states. It classifies quality at three levels:

- page: score and quality class per processed page
- document: aggregate of all pages in one PDF
- run: aggregate across all documents in the run

## Classes

`production/quality_scoring.py` classifies scores using three configurable thresholds:

- `production_quality` (default >= 0.85)
- `usable_with_review` (default >= 0.70)
- `weak_ocr` (default >= 0.50)
- below `weak_ocr` is `failed_ocr`

## Signals

Page scoring combines OCR confidence and multiple safety signals, including:

- text length and character diversity
- language/script compatibility
- adapter-aware lexicon metrics (only when adapter metrics are present)
- engine agreement and layout confidence
- runtime stability (timeouts/failures lower quality)
- preprocessing consistency
- postprocessing correction risk and unknown-token signal

## Launch Gate Modes

Gate mode controls whether quality failures block run success (`tools/run_page_text.py --launch-gate-mode ...`).

- `internal`: non-blocking defaults, telemetry-oriented
- `beta`: review-oriented; gate reasons are collected but quality checks do not fail the run
- `production`: quality thresholds are enforced and can fail the run
- `strict`: production enforcement plus strict-readiness requirement

Run-level gate checks include:

- empty page rate
- timeout rate
- failed page rate
- minimum average quality
- maximum review-needed rate

## Artifacts

The pipeline writes quality artifacts to the output root by default:

- `document_quality.jsonl`: per-document quality summaries
- `run_quality.json`: run summary, launch-gate result, and applied thresholds

Per-page records in CSV/JSON outputs include:

- `page_quality_score`
- `document_quality_score`
- `quality_class`
- `quality_reasons`
- `failed_gate`
- `gate_reason`

Layout region records include:

- `region_quality_score`
- `quality_class`
- `needs_review`
- `quality_reasons`

## Configuration

You can override defaults with CLI flags or a JSON config file:

- class thresholds: `--quality-threshold-production`, `--quality-threshold-usable`, `--quality-threshold-weak`
- gate overrides: `--gate-max-empty-rate`, `--gate-max-timeout-rate`, `--gate-max-failed-rate`, `--gate-min-avg-quality`, `--gate-max-review-rate`
- config path: `--quality-config`

CLI overrides take precedence over config values.
