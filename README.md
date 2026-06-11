# OCR Pipeline

This repository contains a document OCR pipeline for extracting page-level text from PDFs, routing difficult pages through OCR, and producing auditable outputs for downstream review, evaluation, and structured document modeling.

## What The Pipeline Does

The current pipeline is built around `run_pipeline.py` and `tools/run_page_text.py`.

At a high level it:

1. Reads PDFs from an input directory, single file, or manifest.
2. Tries to use the PDF text layer when that text is usable.
3. Runs page diagnostics to measure scan quality, layout complexity, and routing signals.
4. Applies OCR when needed, including adaptive routing and optional multi-engine fallback.
5. Postprocesses extracted text while preserving raw OCR output for auditability.
6. Writes page text, progress telemetry, diagnostics, and quality artifacts to the output directory.

## Core Runtime Shape

- `run_pipeline.py`
  Compatibility entrypoint. Supports legacy flags and can run a two-pass workflow.
- `tools/run_page_text.py`
  Stable page-text runner used by the compatibility entrypoint.
- `production/page_diagnostics.py`
  Produces per-page diagnostics before acceptance or OCR routing.
- `production/ocr_strategy.py`
  Selects text-layer, fast OCR, layout-first OCR, or conservative fallback behavior.
- `production/postprocessing/`
  Cleanup, adapter-based correction, and review-oriented postprocessing.
- `production/quality_scoring.py`
  Scores page, document, and run quality and applies launch-gate logic.
- `production/document_model.py`
  Converts pipeline output into a canonical structured document model.

## Primary Outputs

Every run centers on `client_page_text.csv`, with one row per page.

Common artifacts include:

- `client_page_text.csv`
- `client_page_text.json`
- `progress.csv`
- `page_diagnostics.jsonl`
- `layout_regions.jsonl`
- `document_quality.jsonl`
- `run_quality.json`

Depending on OCR mode, the pipeline can also emit ensemble and disagreement analysis artifacts.

## Quick Start

Validate the environment and current entrypoints:

```bash
python run_pipeline.py --help
python run_pipeline.py --validate-only -c config.json
```

Run the compatibility entrypoint on a directory of PDFs:

```bash
python run_pipeline.py --input-dir data/input --output-dir reports/output
```

Run the page-text pipeline directly:

```bash
python tools/run_page_text.py \
  --inputs data/input \
  --output-root reports/output \
  --prefer-text-layer \
  --ocr-fallback ensemble
```

Run the test suite:

```bash
python -m pytest tests -q
python test_pipeline.py --allow-missing-engines
```

## Documentation

The current canonical docs are:

- [Pipeline Overview](docs/PIPELINE_OVERVIEW.md)
- [Docs Index](docs/README.md)
- [Quickstart](QUICKSTART.md)
- [Run And Test](docs/RUN_AND_TEST.md)
- [Routes And Backends](docs/ROUTES_AND_BACKENDS.md)
- [Page Text Runbook](docs/PAGE_TEXT_RUNBOOK.md)
- [Page Diagnostics](docs/PAGE_DIAGNOSTICS.md)
- [Postprocessing](docs/POSTPROCESSING.md)
- [Quality Scoring](docs/QUALITY_SCORING.md)
- [Document Model Schema](docs/document_model_schema.md)
- [Gold-Set Evaluation](docs/GOLDSET_EVALUATION.md)
- [Experiment Tracking](docs/EXPERIMENT_TRACKING.md)

Only the docs listed above should be treated as the active description of the pipeline.

## Repository Layout

- `production/`
  Runtime modules for routing, diagnostics, postprocessing, quality scoring, and adapters.
- `tools/`
  CLI tools for extraction, evaluation, comparison, and experiments.
- `tests/`
  Regression, unit, and end-to-end coverage.
- `profiles/`
  Detection and preprocessing profile assets.
- `config/`
  Runtime schemas and policy/configuration files.
- `data/`
  Input and evaluation-support data.
- `docs/`
  Canonical technical documentation.

## Notes

- `run_pipeline.py` remains the stable top-level command even though the maintained page-text implementation lives behind `tools/run_page_text.py`.
- The repository does not currently provide a single pinned root install workflow such as `requirements.txt` or `pyproject.toml`.
- `README_docker.md` remains the Docker-specific reference for container usage.

## License

Apache-2.0. See `doc/LICENSE`.
