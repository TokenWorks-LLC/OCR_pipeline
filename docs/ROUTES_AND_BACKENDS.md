# Routes And Backends

This document describes the current routing model and backend posture of the OCR pipeline.

## Routing Model

The pipeline does not treat every page the same.

It first decides whether the page should:

- accept the PDF text layer
- run OCR on the full page
- run a layout-aware OCR path
- use a more conservative fallback path for difficult pages

That decision is driven by diagnostics, engine readiness, script hints, and quality thresholds.

## Routing Sources

The main routing logic lives in:

- `production/page_diagnostics.py`
- `production/ocr_strategy.py`
- `production/scanned_forms_route.py`

## Current Route Classes

### Text-layer route

Used when the PDF text layer is present, usable, and not suspicious.

### OCR route

Used when the page is scanned, degraded, or when the text layer fails quality checks.

### Layout-first route

Used for pages that appear structurally complex, such as pages with multiple columns, tables, or footnote-heavy layouts.

### Conservative diacritic-sensitive route

Used for pages where transliteration, diacritics, or script-sensitive handling is more important than pure speed.

### Scanned-forms cleanup route

`production/scanned_forms_route.py` applies a narrow structured-cleanup path for scanned forms while preserving raw OCR text and route audit metadata.

## Backend Posture

The pipeline can reference multiple OCR engines, but routing does not assume they are all always healthy or installed.

Backends currently represented in the runtime and checks include:

- `paddleocr`
- `doctr`
- `mmocr`
- `kraken`

The pipeline may skip a backend when:

- it is disabled by config
- a dependency is missing
- startup failed
- the backend is marked unhealthy
- previous runtime telemetry shows repeated failures or timeouts

## Fallback Modes Exposed By CLI

The page-text runner currently exposes:

- `--ocr-fallback none`
- `--ocr-fallback paddle`
- `--ocr-fallback ensemble`

The compatibility runner also supports two-pass reruns where weak first-pass pages can be retried under explicit page and runtime budgets.

## Design Constraints

Current routing behavior is built around these constraints:

- prefer usable text layers before OCR
- do not assume a single OCR engine is best for all pages
- keep route decisions explainable through diagnostics and telemetry
- preserve raw OCR output even when cleanup or correction is applied
- keep quality and review signals attached to the final result

## Related Docs

- [Pipeline Overview](PIPELINE_OVERVIEW.md)
- [Page Diagnostics](PAGE_DIAGNOSTICS.md)
- [OCR Backend Notes](ocr_backends_notes.md)
- [Quality Scoring](QUALITY_SCORING.md)
