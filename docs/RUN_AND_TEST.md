# Run And Test Guide

This repository currently uses the page-text pipeline implementation in `.merge_protect/tools/run_page_text.py`.

For compatibility and easier onboarding, use these stable entrypoints from repo root:

- `run_pipeline.py`: legacy-compatible runner (maps to `tools/run_page_text.py`)
- `test_pipeline.py`: optional smoke-check convenience script for CLI runnability and OCR engine availability
- `tools/run_page_text.py`: stable wrapper to the protected implementation

## 1) Local (venv) Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip pytest
```

Validate entrypoints:

```bash
python run_pipeline.py --help
python run_pipeline.py --validate-only -c config.json
python test_pipeline.py --allow-missing-engines
```

Strict engine check (fails if any engine is missing):

```bash
python test_pipeline.py
```

Run with a legacy config:

```bash
python run_pipeline.py -c config_eval_advanced_v3.json --dry-run
python run_pipeline.py -c config_eval_advanced_v3.json
```

## 2) Docker / Compose (Cross-Platform)

Use compose service `ocr` on amd64 and `ocr-arm64` on Apple Silicon.

```bash
# amd64 (Linux/Windows/Intel Mac)
export COMPOSE_SERVICE=ocr

# arm64 (Apple Silicon)
export COMPOSE_SERVICE=ocr-arm64

# smoke checks in container
docker compose run --rm $COMPOSE_SERVICE python test_pipeline.py --allow-missing-engines

# strict engine checks in container
docker compose run --rm $COMPOSE_SERVICE python test_pipeline.py
```

Run OCR on a single PDF:

```bash
docker compose run --rm $COMPOSE_SERVICE python run_pipeline.py --input-file data/input/sample.pdf
```

## 3) Pytest Suite

Run compatibility and end-to-end regression tests:

```bash
python -m pytest tests -q
```

`test_pipeline.py` is not required for the pytest suite; keep it for quick environment smoke checks.

Key coverage in `tests/test_pipeline_e2e.py`:

- `tools/run_page_text.py` inputs mode end-to-end
- `tools/run_page_text.py` manifest mode parsing (header/comments)
- `run_pipeline.py` compatibility mapping for `--input-file`
- failure path for missing input file

Enforce engine imports in pytest (optional):

```bash
REQUIRED_OCR_ENGINES=paddleocr,doctr,mmocr,kraken python -m pytest tests/test_engine_imports.py -q
```

## 4) Notes

- `test_pipeline.py --allow-missing-engines` is intended for portability checks.
- `test_pipeline.py` is intended for environment readiness checks.
- If strict checks fail locally, prefer Docker/devcontainer for consistent dependencies across machines.

## 5) GitHub Actions (`test_suite`)

CI runs on every `push` and `pull_request` via `.github/workflows/test_suite.yml`.

Checks included in the required `test_suite` status:

- `ruff check run_pipeline.py test_pipeline.py tools tests`
- `prettier --check` across active Markdown/JSON/YAML docs/workflow files
- `python -m pytest tests -q`
