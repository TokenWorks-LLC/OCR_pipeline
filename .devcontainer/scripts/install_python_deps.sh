#!/usr/bin/env bash
set -euo pipefail

python -m pip install -U pip setuptools wheel

# Core OCR/runtime packages used by the active pipeline and wrappers.
python -m pip install --only-binary=:all: \
  "PyMuPDF==1.24.10" \
  "opencv-python-headless==4.10.0.84" \
  "pdf2image==1.17.0" \
  "numpy>=1.24.0,<2.1" \
  "Pillow>=10.0.0" \
  "lxml==6.0.2" \
  "PyYAML>=6.0" \
  "requests>=2.32,<3" \
  "chardet>=5.2.0,<6" \
  "charset_normalizer>=3.3,<4"

ARCH="$(dpkg --print-architecture)"
if [ "$ARCH" = "arm64" ]; then
  python -m pip install --only-binary=:all: \
    "paddlepaddle>=3.0.0,<4.0.0" \
    -f https://www.paddlepaddle.org.cn/whl/linux/aarch64/
else
  python -m pip install --only-binary=:all: \
    "paddlepaddle>=3.0.0,<4.0.0"
fi

python -m pip install --only-binary=:all: "paddleocr>=3.2.0"

# Torch family: pin to versions known to work with kraken and this project stack.
python -m pip install --only-binary=:all: \
  "torch==2.9.0" \
  "torchvision==0.24.0"

# MMOCR stack with versions that satisfy import-time compatibility checks.
python -m pip install --only-binary=:all: \
  "mmengine>=0.10,<1.0" \
  "mmcv-lite>=2.0.0rc4,<2.1.0" \
  "mmdet>=3.0.0rc5,<3.2.0" \
  "mmocr>=1.0.0,<1.1.0"

# langdetect does not ship universal wheels in all environments; allow source build here.
PIP_ONLY_BINARY= python -m pip install --no-binary langdetect "langdetect==1.0.9"

# Install docTR with explicit dependencies to avoid resolver deadlocks under binary-first policies.
python -m pip install --only-binary=:all: \
  "onnx>=1.12,<3" \
  "defusedxml>=0.7.1" \
  "anyascii>=0.3.2" \
  "validators>=0.18.0"
python -m pip install --no-deps "python-doctr==1.0.1"

# Kraken is installed after torch to avoid repeated resolver churn.
python -m pip install --only-binary=:all: kraken

