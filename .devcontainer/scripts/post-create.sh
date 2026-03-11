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

# Ensure new interactive bash terminals in this workspace auto-activate the project venv.
if ! grep -q "OCR_PIPELINE_AUTO_VENV" "$HOME/.bashrc"; then
	cat >> "$HOME/.bashrc" <<'EOF'
# OCR_PIPELINE_AUTO_VENV
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f /workspaces/OCR_pipeline/.venv/bin/activate ] && [[ "$PWD" == /workspaces/OCR_pipeline* ]]; then
	. /workspaces/OCR_pipeline/.venv/bin/activate
fi
export KRAKEN_MODEL_ID="${KRAKEN_MODEL_ID:-10.5281/zenodo.2577813}"
export KRAKEN_MODEL_PATH="${KRAKEN_MODEL_PATH:-/workspaces/OCR_pipeline/.venv/share/kraken/en_best.mlmodel}"
EOF
fi

source .venv/bin/activate
export KRAKEN_MODEL_ID="${KRAKEN_MODEL_ID:-10.5281/zenodo.2577813}"
export KRAKEN_MODEL_PATH="${KRAKEN_MODEL_PATH:-/workspaces/OCR_pipeline/.venv/share/kraken/en_best.mlmodel}"
python -m pip install -U pip "setuptools<81" wheel

echo "[devcontainer] Installing OCR dependencies and engines"
bash .devcontainer/scripts/install_python_deps.sh

ARCH="$(dpkg --print-architecture)"
STRICT_MODE="${OCR_STRICT_ALL_ENGINES:-auto}"
if [ "$STRICT_MODE" = "auto" ]; then
	if [ "$ARCH" = "amd64" ]; then
		STRICT_MODE="1"
	else
		STRICT_MODE="0"
	fi
fi

export OCR_STRICT_ALL_ENGINES_EFFECTIVE="$STRICT_MODE"

echo "[devcontainer] Ensuring Kraken recognition model is available"
python - <<'PY'
import os
from pathlib import Path

from htrmopo import get_model

model_id = os.environ.get("KRAKEN_MODEL_ID", "10.5281/zenodo.2577813").strip()
model_path = Path(os.environ.get("KRAKEN_MODEL_PATH", "/workspaces/OCR_pipeline/.venv/share/kraken/en_best.mlmodel")).resolve()

if model_path.exists():
	print(f"[devcontainer] Kraken model present: {model_path}")
else:
	model_path.parent.mkdir(parents=True, exist_ok=True)
	print(f"[devcontainer] Downloading Kraken model {model_id} to {model_path.parent}")
	get_model(model_id, path=model_path.parent)
	if not model_path.exists():
		raise SystemExit(
			f"[devcontainer] ERROR: Expected Kraken model at {model_path} after downloading {model_id}"
		)
	print(f"[devcontainer] Kraken model downloaded: {model_path}")
PY

echo "[devcontainer] Validating OCR backend readiness"
python - <<'PY'
import importlib
import os
from pathlib import Path
import sys

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

modules = ["paddleocr", "doctr", "mmocr", "kraken"]
for name in modules:
	importlib.import_module(name)
	print(f"[devcontainer] import ok: {name}")

import mmcv._ext  # noqa: F401
print("[devcontainer] mmcv native ops ok: mmcv._ext")

from mmocr.apis import MMOCRInferencer  # noqa: F401
print("[devcontainer] MMOCR API import ok")

from paddleocr import PaddleOCR
ocr = PaddleOCR(lang="en")
print(f"[devcontainer] PaddleOCR warm init ok: {type(ocr).__name__}")

strict_mode = os.environ.get("OCR_STRICT_ALL_ENGINES_EFFECTIVE", "0") == "1"
kraken_model_path = os.environ.get("KRAKEN_MODEL_PATH", "").strip()
if strict_mode:
	if not kraken_model_path:
		print(
			"[devcontainer] ERROR: KRAKEN_MODEL_PATH is not configured. "
			"Set it before container creation or run with OCR_STRICT_ALL_ENGINES=0 for portable mode.",
			file=sys.stderr,
		)
		sys.exit(1)

	model_path = Path(kraken_model_path)
	if not model_path.exists():
		print(
			f"[devcontainer] ERROR: Kraken model not found at {model_path}",
			file=sys.stderr,
		)
		sys.exit(1)

	print(f"[devcontainer] Kraken model configured: {model_path}")
else:
	if kraken_model_path:
		print(f"[devcontainer] Kraken model configured: {kraken_model_path}")
	else:
		print("[devcontainer] Kraken model not configured; portable mode will allow this")
PY

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
	python run_pipeline.py --input-file "$SAMPLE_PDF" --output-dir reports/devcontainer_e2e
	echo "[devcontainer] E2E output: reports/devcontainer_e2e/client_page_text.csv"
else
	echo "[devcontainer] No sample PDF found under data/. Skipping E2E run."
fi

echo "[devcontainer] Setup complete"
