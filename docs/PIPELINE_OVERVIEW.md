# Pipeline Overview

This document defines the current OCR pipeline in practical terms: what goes in, what happens to each page, and what comes out.

## Purpose

The pipeline extracts usable text from PDF documents while keeping the result auditable and measurable.

It is designed to:

- prefer native PDF text when that text is trustworthy
- fall back to OCR for scanned, degraded, or suspicious pages
- adapt OCR behavior to page layout and script characteristics
- preserve raw output alongside cleaned output
- emit diagnostics and quality signals that support evaluation and review

## Inputs

The pipeline accepts:

- an input directory of PDFs
- a single PDF file
- a manifest of specific `pdf<TAB>page` rows

The main entrypoints are:

- `python run_pipeline.py`
- `python tools/run_page_text.py`

## End-To-End Flow

For each page, the current pipeline follows this shape:

1. Resolve the page source from the input file or manifest.
2. Inspect the page with diagnostics:
   - text-layer usability
   - scan quality
   - skew, blur, noise, and contrast
   - layout complexity, columns, tables, and footnotes
3. Choose a routing strategy:
   - accept text layer
   - run fast OCR
   - run layout-aware OCR
   - run more conservative or multi-engine fallback
4. Apply OCR when required:
   - preprocessing profile selection
   - engine selection and fallback plan
   - optional ensemble behavior for hard pages
5. Postprocess text:
   - cleanup
   - adapter-based correction
   - protected-character handling
   - review-oriented metadata
6. Score quality and write artifacts.

## Pipeline Layers

### 1. Entry And Orchestration

- `run_pipeline.py`
  Legacy-compatible wrapper and two-pass orchestrator.
- `tools/run_page_text.py`
  Stable runner for page-text extraction.

`run_pipeline.py` is still the safest command to document for general use because it preserves existing CLI compatibility.

### 2. Diagnostics

`production/page_diagnostics.py` measures whether a page should trust the text layer, whether OCR is needed, and what kind of OCR path is likely to work best.

Important outputs include:

- text-layer acceptance signals
- layout complexity
- estimated columns and table interruptions
- preprocessing recommendations
- routing telemetry

### 3. OCR Routing

`production/ocr_strategy.py` chooses the OCR strategy from diagnostics, engine readiness, script hints, and policy thresholds.

The routing logic can favor:

- text-layer extraction
- a simpler single-engine path
- layout-first routing for complex pages
- conservative routing for diacritic-heavy or transliteration-heavy text

### 4. OCR Execution

The page-text runner can use:

- no OCR, when the text layer is accepted
- targeted OCR fallback
- adaptive multi-engine behavior on harder pages

The compatibility entrypoint can also run a two-pass flow, where selected weak or failed first-pass pages are rerun under a second-pass budget.

### 5. Postprocessing

`production/postprocessing/` turns raw OCR into a safer final text field without losing provenance.

Key goals:

- keep `raw_text`
- produce cleaned and corrected text separately
- support language or script adapters
- protect diacritics and transliteration-sensitive tokens

### 6. Quality And Launch Gates

`production/quality_scoring.py` assigns quality at page, document, and run level.

This layer supports:

- quality classes such as `production_quality`, `usable_with_review`, `weak_ocr`, and `failed_ocr`
- review-needed signals
- run-level gate failures for high empty-rate, failure-rate, timeout-rate, or low average quality

### 7. Structured Output Conversion

`production/document_model.py` converts page-level OCR output into a canonical structured document model with blocks, lines, tables, figures, and provenance.

That structured model is the bridge from page text extraction to downstream document understanding workflows.

## Main Output Contract

The primary deliverable is:

- `client_page_text.csv`

Common per-run artifacts:

- `client_page_text.json`
- `progress.csv`
- `page_diagnostics.jsonl`
- `layout_regions.jsonl`
- `document_quality.jsonl`
- `run_quality.json`

Additional artifacts may appear when ensemble analysis, second-pass fallback, or evaluation workflows are enabled.

## What This Pipeline Is Not

The current repository is not just a plain OCR script and not just a model benchmark harness.

It is a layered pipeline that combines:

- extraction
- routing
- diagnostics
- postprocessing
- quality scoring
- evaluation support
- structured output conversion

## Read Next

- [Run And Test](RUN_AND_TEST.md)
- [Page Text Runbook](PAGE_TEXT_RUNBOOK.md)
- [Routes And Backends](ROUTES_AND_BACKENDS.md)
- [Page Diagnostics](PAGE_DIAGNOSTICS.md)
- [Postprocessing](POSTPROCESSING.md)
- [Quality Scoring](QUALITY_SCORING.md)
- [Document Model Schema](document_model_schema.md)
