# Run And Test

This guide covers the current ways to validate, run, and test the OCR pipeline.

## Primary Commands

Use these from the repository root:

```bash
python run_pipeline.py --help
python run_pipeline.py --validate-only -c config.json
python tools/run_page_text.py --help
python -m pytest tests -q
python test_pipeline.py --allow-missing-engines
```

`run_pipeline.py` is the stable compatibility entrypoint.

`tools/run_page_text.py` is the direct page-text runner used by that entrypoint.

## Standard Runs

Run a directory of PDFs:

```bash
python run_pipeline.py --input-dir data/input --output-dir reports/output
```

Run a single PDF:

```bash
python run_pipeline.py --input-file data/input/sample.pdf --output-dir reports/output_single
```

Run the page-text runner directly:

```bash
python tools/run_page_text.py \
  --inputs data/input \
  --output-root reports/output \
  --prefer-text-layer \
  --ocr-fallback ensemble
```

## OCR Control

Use OCR even when a text layer exists:

```bash
python run_pipeline.py \
  --input-dir data/input \
  --output-dir reports/force_ocr \
  --force-ocr \
  --engine ensemble
```

Run a two-pass workflow for weak first-pass pages:

```bash
python run_pipeline.py \
  --input-dir data/input \
  --output-dir reports/two_pass \
  --two-pass-mode \
  --rerun-failed-pages \
  --fallback-on-empty
```

Useful two-pass controls:

- `--fallback-on-low-quality`
- `--max-rerun-pages`
- `--max-rerun-page-ratio`
- `--max-second-pass-ms-per-page`
- `--max-total-second-pass-ms`
- `--second-pass-engine-mode`
- `--enable-backend-warm-reuse`

## Output Artifacts

Core outputs:

- `client_page_text.csv`
- `client_page_text.json`
- `progress.csv`

Diagnostics and layout outputs:

- `page_diagnostics.jsonl`
- `layout_regions.jsonl`

Quality outputs:

- `document_quality.jsonl`
- `run_quality.json`

Additional OCR analysis artifacts may be written when fallback and ensemble paths are active.

## Test Commands

Main regression suite:

```bash
python -m pytest tests -q
```

Portable smoke checks:

```bash
python test_pipeline.py --allow-missing-engines
```

Strict engine smoke checks:

```bash
python test_pipeline.py
```

Useful targeted suites:

```bash
python -m pytest tests/test_pipeline_e2e.py -q
python -m pytest tests/test_ensemble_support.py -q
python -m pytest tests/test_ocr_strategy_selector.py -q
```

## Backend Readiness

Optional backend checks:

```bash
python tools/backend_optional_dependency_check.py
```

If you want pytest to require specific OCR engines:

```bash
REQUIRED_OCR_ENGINES=paddleocr,doctr,mmocr,kraken python -m pytest tests/test_engine_imports.py -q
```

## Docker

Use `README_docker.md` for Docker and compose-specific setup and commands.

## Related Docs

- [Pipeline Overview](PIPELINE_OVERVIEW.md)
- [Page Text Runbook](PAGE_TEXT_RUNBOOK.md)
- [Routes And Backends](ROUTES_AND_BACKENDS.md)
- [Gold-Set Evaluation](GOLDSET_EVALUATION.md)
