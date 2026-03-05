#!/usr/bin/env bash
set -euo pipefail

echo "[devcontainer] Architecture: $(uname -m)"
python -V

python - <<'PY'
import platform
import sys

import cv2
import fitz
import numpy
import paddle
import paddleocr
import pdf2image
from PIL import Image

print(f"[devcontainer] Python: {sys.version.split()[0]}")
print(f"[devcontainer] Platform: {platform.machine()}")
print(f"[devcontainer] OpenCV: {cv2.__version__}")
print(f"[devcontainer] PyMuPDF: {fitz.__doc__.split()[1] if fitz.__doc__ else 'ok'}")
print(f"[devcontainer] NumPy: {numpy.__version__}")
print(f"[devcontainer] Paddle: {paddle.__version__}")
print(f"[devcontainer] PaddleOCR: {getattr(paddleocr, '__version__', 'installed')}")
print(f"[devcontainer] pdf2image: {getattr(pdf2image, '__version__', 'installed')}")
print(f"[devcontainer] Pillow: {Image.__version__}")
print("[devcontainer] Core OCR imports validated")
PY
