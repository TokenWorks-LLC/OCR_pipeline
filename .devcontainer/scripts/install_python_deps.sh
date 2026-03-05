#!/usr/bin/env bash
set -euo pipefail

python -m pip install -U pip setuptools wheel

python -m pip install --only-binary=:all: \
  "PyMuPDF==1.24.10" \
  "opencv-python-headless==4.10.0.84" \
  "pdf2image==1.17.0" \
  "numpy>=1.24.0" \
  "Pillow>=10.0.0"

ARCH="$(dpkg --print-architecture)"
if [ "$ARCH" = "arm64" ]; then
  python -m pip install --only-binary=:all: \
    "paddlepaddle>=3.0.0,<4.0.0" \
    -f https://www.paddlepaddle.org.cn/whl/linux/aarch64/
else
  python -m pip install --only-binary=:all: \
    "paddlepaddle>=3.0.0,<4.0.0"
fi

python -m pip install --only-binary=:all: --no-deps "paddleocr>=3.2.0"

python -m pip install --only-binary=:all: \
  "shapely==2.1.2" \
  "scikit-image==0.25.2" \
  "imgaug==0.4.0" \
  "pyclipper==1.3.0.post6" \
  "lmdb==1.7.3" \
  "rapidfuzz==3.14.1" \
  "lxml==6.0.2" \
  "premailer==3.10.0" \
  "openpyxl==3.1.5" \
  "tqdm==4.67.1" \
  "pytesseract>=0.3.10" \
  "PyYAML>=6.0" \
  "typing-extensions>=4.12" \
  "paddlex>=3.2.0"

python -m pip install --only-binary=:all: --upgrade --force-reinstall \
  "requests>=2.32.3,<3" \
  "urllib3>=2.2.0,<3" \
  "charset_normalizer>=3.3.0,<4" \
  "chardet>=5.2.0,<6"
