# Quickstart

This quickstart is the shortest path to validating and running the current OCR pipeline.

## Validate The Environment

```bash
python run_pipeline.py --help
python run_pipeline.py --validate-only -c config.json
python test_pipeline.py --allow-missing-engines
```

## Run The Default Pipeline

```bash
python run_pipeline.py --input-dir data/input --output-dir reports/output
```

This command keeps the stable compatibility entrypoint while delegating to the maintained page-text pipeline.

## Run The Page-Text Entry Point Directly

```bash
python tools/run_page_text.py \
  --inputs data/input \
  --output-root reports/output \
  --prefer-text-layer \
  --ocr-fallback ensemble
```

## Common Variants

Force OCR even when a PDF already has a text layer:

```bash
python run_pipeline.py \
  --input-dir data/input \
  --output-dir reports/force_ocr \
  --force-ocr \
  --engine ensemble
```

Run the two-pass workflow:

```bash
python run_pipeline.py \
  --input-dir data/input \
  --output-dir reports/two_pass \
  --two-pass-mode \
  --rerun-failed-pages \
  --fallback-on-empty
```

## Expected Outputs

The main deliverables are:

- `client_page_text.csv`
- `client_page_text.json`
- `progress.csv`

Typical diagnostics and quality artifacts include:

- `page_diagnostics.jsonl`
- `layout_regions.jsonl`
- `document_quality.jsonl`
- `run_quality.json`

## Run Tests

```bash
python -m pytest tests -q
python test_pipeline.py --allow-missing-engines
```

## Read Next

- [Pipeline Overview](docs/PIPELINE_OVERVIEW.md)
- [Run And Test](docs/RUN_AND_TEST.md)
- [Page Text Runbook](docs/PAGE_TEXT_RUNBOOK.md)
