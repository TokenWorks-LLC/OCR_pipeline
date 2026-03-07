# OCR Pipeline

Production OCR pipeline with page-level text extraction and Akkadian detection.

This repository currently exposes two stable CLIs:

- `run_pipeline.py`: compatibility entrypoint
- `tools/run_page_text.py`: page-text pipeline entrypoint

`run_pipeline.py` maps to the page-text runner and keeps legacy flag compatibility where possible.

## Quick Start

### Local (validated)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip pytest

# Verify environment and entrypoints
python run_pipeline.py --help
python run_pipeline.py --validate-only -c config.json
python test_pipeline.py --allow-missing-engines
```

### Run on input folder

```bash
python run_pipeline.py --input-dir data/input --output-dir reports/output
```

### Run page-text directly

```bash
python tools/run_page_text.py \
  --inputs data/input \
  --output-root reports/output \
  --prefer-text-layer \
  --ocr-fallback paddle
```

Output CSV:

- `reports/output/client_page_text.csv`
- Columns: `pdf_name,page,page_text,has_akkadian`

## Canonical Commands

Use these as the source of truth from repo root:

- `python run_pipeline.py --help`
- `python tools/run_page_text.py --help`
- `python -m pytest tests -q`
- `python test_pipeline.py --allow-missing-engines` (optional smoke convenience)
- `python test_pipeline.py` (optional strict smoke)

## CI

GitHub Actions workflow `test_suite` runs on every `push` and `pull_request` and enforces:

- `ruff` lint checks on active Python paths
- `prettier --check` for Markdown/JSON/YAML files in active docs and workflow paths
- `python -m pytest tests -q`

Use branch protection required check name: `test_suite`.

## Local Hooks

GitHub Actions does not run on `git commit`; it runs on `push` and `pull_request`.

For local commit-time enforcement, this repo includes `.pre-commit-config.yaml`.
Install the hook once per clone:

```bash
python -m pip install pre-commit
pre-commit install
```

That local `pre-commit` hook runs:

- `ruff` checks on active Python paths
- `prettier --check` on active Markdown/JSON/YAML files

Detailed operations are in `docs/RUN_AND_TEST.md` and `docs/PAGE_TEXT_RUNBOOK.md`.

## Docker

Use `README_docker.md` for container commands. Docker commands are documented there but were not executed in the current environment because Docker is not installed in this dev container.

## Project Layout

- `run_pipeline.py`: compatibility wrapper
- `tools/run_page_text.py`: page-text extractor
- `test_pipeline.py`: smoke checks
- `tests/test_pipeline_e2e.py`: end-to-end regression tests for CLI workflows
- `profiles/`: detection profiles
- `config*.json`: run/evaluation configs
- `docs/`: operational documentation

## Notes

- There is no root `requirements.txt` or `setup.py` workflow in this repo.
- Legacy top-level scripts `quick_start.py` and `gold_evaluation.py` were removed as redundant/stale.
- Legacy archive/evaluation/cache artifacts were removed as part of repository cleanup.

## License

Apache-2.0. See `doc/LICENSE`.
