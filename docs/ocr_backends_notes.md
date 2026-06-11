# OCR Backend Notes

This note describes how backend availability fits into the current pipeline.

## Runtime Position

The OCR system is centered on:

- `run_pipeline.py`
- `tools/run_page_text.py`
- the maintained implementation behind `.merge_protect/tools/run_page_text.py`

The pipeline can work with multiple OCR backends, but backend selection is conditional and diagnostics-driven rather than hardwired.

## Backends Referenced By Runtime Checks

- `paddleocr`
- `doctr`
- `mmocr`
- `kraken`

Other backends may appear in adapters or experiments, but the list above is the practical set to treat as the active OCR backend surface.

## Validation Commands

Portable validation:

```bash
python test_pipeline.py --allow-missing-engines
```

Strict validation:

```bash
python test_pipeline.py
```

Optional dependency and adapter checks:

```bash
python tools/backend_optional_dependency_check.py
```

## CLI Fallback Modes

`tools/run_page_text.py` exposes:

- `--ocr-fallback none`
- `--ocr-fallback paddle`
- `--ocr-fallback ensemble`

Use `--prefer-text-layer` when you want the pipeline to favor embedded PDF text before OCR.

Use `--force-ocr` when you need OCR output even for PDFs that already contain selectable text.

## Operational Guidance

- treat backend availability as environment-dependent
- validate engines before large runs
- use portable checks for general development
- use strict checks before backend-sensitive regression runs
- use evaluation reports, not anecdotal spot checks, to compare backend behavior
