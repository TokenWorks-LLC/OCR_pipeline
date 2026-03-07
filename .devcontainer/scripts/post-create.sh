#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/workspaces/OCR_pipeline"
cd "$ROOT_DIR"

echo "[devcontainer] Workspace: $ROOT_DIR"
echo "[devcontainer] Architecture: $(uname -m)"
python -V

if [ ! -d .venv ]; then
	echo "[devcontainer] Creating virtual environment (.venv)"
	python -m venv .venv
fi

source .venv/bin/activate
python -m pip install -U pip setuptools wheel

echo "[devcontainer] Installing OCR dependencies and engines"
bash .devcontainer/scripts/install_python_deps.sh

echo "[devcontainer] Warming engine/model imports"
python - <<'PY'
import importlib
import os

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

modules = ["paddleocr", "doctr", "mmocr", "kraken"]
for name in modules:
		importlib.import_module(name)
		print(f"[devcontainer] import ok: {name}")

from paddleocr import PaddleOCR
ocr = PaddleOCR(lang="en")
print(f"[devcontainer] PaddleOCR warm init ok: {type(ocr).__name__}")
PY

ARCH="$(dpkg --print-architecture)"
STRICT_MODE="${OCR_STRICT_ALL_ENGINES:-auto}"
if [ "$STRICT_MODE" = "auto" ]; then
	if [ "$ARCH" = "amd64" ]; then
		STRICT_MODE="1"
	else
		STRICT_MODE="0"
	fi
fi

if [ "$STRICT_MODE" = "1" ]; then
	echo "[devcontainer] Running strict smoke checks"
	python test_pipeline.py --required-engines paddleocr doctr mmocr kraken
else
	echo "[devcontainer] Running portable smoke checks"
	python test_pipeline.py --allow-missing-engines
fi

SAMPLE_PDF="$(find data -maxdepth 3 -type f -iname '*.pdf' | head -n 1 || true)"
if [ -n "$SAMPLE_PDF" ]; then
	echo "[devcontainer] Running end-to-end smoke on: $SAMPLE_PDF"
	rm -rf reports/devcontainer_e2e
	python run_pipeline.py --input-file "$SAMPLE_PDF" --output-dir reports/devcontainer_e2e --llm-off
	echo "[devcontainer] E2E output: reports/devcontainer_e2e/client_page_text.csv"
else
	echo "[devcontainer] No sample PDF found under data/. Skipping E2E run."
fi

echo "[devcontainer] Setup complete"
