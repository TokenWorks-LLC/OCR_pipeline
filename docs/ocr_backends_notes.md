# OCR Backend Notes

## Current State

This repository supports multiple OCR backends through the current page-text pipeline path:

- `run_pipeline.py` (compatibility wrapper)
- `tools/run_page_text.py` (stable page-text runner)
- `.merge_protect/tools/run_page_text.py` (protected implementation)

Backends referenced by runtime checks:

- `paddleocr`
- `doctr`
- `mmocr`
- `kraken`

The active fallback path can now run a fortified ensemble across the available backends for harder pages.

## Validation Commands

```bash
python test_pipeline.py --allow-missing-engines
python test_pipeline.py
```

- Portable mode (`--allow-missing-engines`) checks wiring without requiring every backend.
- Strict mode requires all backends to be importable.

## Backend Selection

`tools/run_page_text.py` currently exposes OCR fallback as:

- `--ocr-fallback paddle`
- `--ocr-fallback ensemble`
- `--ocr-fallback none`

Use `--prefer-text-layer` to prioritize PDF text extraction and minimize OCR use where possible.

Use `--force-ocr` with an OCR fallback when benchmarking OCR engines on PDFs that already have a text layer.

Ensemble mode:

- preprocesses the rendered page into multiple image variants
- runs best-effort inference across configured OCR backends
- fuses outputs with diacritic-aware consensus scoring
- prefers rich-script preservation for Akkadian, German, Arabic, and similar text

## Notes on Historical Docs

Older references to `src/` engine modules and `tools/run_enhanced_eval.py` do not match the current active runtime path and should be treated as historical.
