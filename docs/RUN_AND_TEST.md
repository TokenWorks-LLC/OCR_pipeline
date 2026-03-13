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

Force OCR even when a PDF has a text layer (useful for layered-text benchmarking):

```bash
python run_pipeline.py \
	--input-dir data/input_pdfs \
	--output-dir reports/force_ocr_eval \
	--engine ensemble \
	--force-ocr
```

Equivalent direct command:

```bash
python tools/run_page_text.py \
	--inputs data/input_pdfs \
	--output-root reports/force_ocr_eval \
	--prefer-text-layer \
	--ocr-fallback ensemble \
	--force-ocr
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
- `run_pipeline.py` dry-run command mapping for supported flags
- `run_pipeline.py` default dry-run mapping to ensemble fallback from config
- `run_pipeline.py` mapping of `--force-ocr`
- failure path for missing input file
- `run_pipeline.py --validate-only` failure path for missing input directories
- validation error when `--force-ocr` is used without an OCR fallback
- `tools/build_manifest.py` range expansion and strict missing-PDF behavior

Additional ensemble coverage in `tests/test_ensemble_support.py`:

- diacritic-aware consensus fusion
- consensus preference over noisy outliers
- ensemble fallback plumbing inside `PDFTextExtractor`

Enforce engine imports in pytest (optional):

```bash
REQUIRED_OCR_ENGINES=paddleocr,doctr,mmocr,kraken python -m pytest tests/test_engine_imports.py -q
```

## 4) Notes

- `test_pipeline.py --allow-missing-engines` is intended for portability checks.
- `test_pipeline.py` is intended for environment readiness checks.
- `--force-ocr` requires an OCR fallback (`paddle` or `ensemble`).
- If strict checks fail locally, prefer Docker/devcontainer for consistent dependencies across machines.

## 5) GitHub Actions (`test_suite`)

CI runs on every `push` and `pull_request` via `.github/workflows/test_suite.yml`.

Checks included in the required `test_suite` status:

- `ruff check run_pipeline.py test_pipeline.py production tools tests`
- `prettier --check` across active Markdown/JSON/YAML docs/workflow files
- `python -m pytest tests -q`
- `python test_pipeline.py --allow-missing-engines`

## 5a) Dev Container Notes

The dev container creates and activates `.venv` automatically via `.devcontainer/scripts/post-create.sh`.

The all-engines devcontainer is pinned to Python 3.10 on purpose. Python 3.11 pushes Kraken onto newer dependency lines that conflict with the older MMOCR/MMDetection stack required by the active ensemble runtime.

Strict devcontainer behavior on `amd64` now validates real backend readiness, not just top-level imports:

- MMOCR must provide `mmcv._ext` native ops, not just `mmcv-lite`
- docTR dependencies such as `h5py` are installed explicitly
- a default Kraken recognition model is downloaded automatically into `.venv/share/kraken/en_best.mlmodel`
- `KRAKEN_MODEL_PATH` can still be overridden, but in strict mode it must point to an existing recognition model file

If you intentionally want a portable setup without all optional engines configured, set `OCR_STRICT_ALL_ENGINES=0` before container creation.

## 6) Local Commit Hook

To run lint/format checks on every local commit, install the repo hook once:

```bash
python -m pip install pre-commit
pre-commit install
```

After that, `git commit` will run the configured `ruff` and `prettier --check` hooks before creating the commit.
