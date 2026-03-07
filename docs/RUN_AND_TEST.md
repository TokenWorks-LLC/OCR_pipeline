# Run And Test Guide

This repository currently uses the page-text pipeline implementation in `.merge_protect/tools/run_page_text.py`.

For compatibility and easier onboarding, use these stable entrypoints from repo root:

- `run_pipeline.py`: legacy-compatible runner (maps to `tools/run_page_text.py`)
- `test_pipeline.py`: smoke checks for CLI runnability and OCR engine availability
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
make test-smoke COMPOSE_SERVICE=$COMPOSE_SERVICE

# strict engine checks in container
make test-engines COMPOSE_SERVICE=$COMPOSE_SERVICE
```

Run OCR on a single PDF:

```bash
make ocr COMPOSE_SERVICE=$COMPOSE_SERVICE FILE="data/input/sample.pdf"
```

## 3) Pytest Suite

Run compatibility tests:

```bash
python -m pytest tests -q
```

Enforce engine imports in pytest (optional):

```bash
REQUIRED_OCR_ENGINES=paddleocr,doctr,mmocr,kraken python -m pytest tests/test_engine_imports.py -q
```

## 4) Notes

- `test_pipeline.py --allow-missing-engines` is intended for portability checks.
- `test_pipeline.py` is intended for environment readiness checks.
- If strict checks fail locally, prefer Docker/devcontainer for consistent dependencies across machines.
