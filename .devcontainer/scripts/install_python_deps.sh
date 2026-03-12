#!/usr/bin/env bash
set -euo pipefail

python -m pip install -U pip "setuptools<81" wheel

# Core OCR/runtime packages used by the active pipeline and wrappers.
python -m pip install --only-binary=:all: \
  "PyMuPDF==1.26.7" \
  "opencv-python-headless==4.10.0.84" \
  "pdf2image==1.17.0" \
  "numpy==1.23.5" \
  "protobuf==3.20.3" \
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

# Use the pre-PaddleX line to keep numpy compatible with Kraken.
python -m pip install --only-binary=:all: --no-deps "paddleocr==2.7.3"

# CPU-only torch stack for the devcontainer. This keeps Kraken and MMOCR on a
# compatible shared version line and avoids GPU-specific wheel churn.
python -m pip uninstall -y torch torchvision torchaudio triton || true
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  "torch==2.0.1" \
  "torchvision==0.15.2"

# MMOCR stack: install full mmcv ops support, not mmcv-lite.
python -m pip uninstall -y mmcv mmcv-lite mmdet mmocr || true
python -m pip install --only-binary=:all: openmim
python -m pip install --only-binary=:all: \
  "mmengine==0.10.7" \
  "mmdet==3.1.0"
python -m pip install --only-binary=:all: \
  "mmcv==2.0.0" \
  -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.0/index.html
python -m pip install --only-binary=:all: "mmocr==1.0.1"

# langdetect does not ship universal wheels in all environments; allow source build here.
PIP_ONLY_BINARY= python -m pip install --no-binary langdetect "langdetect==1.0.9"

# Install docTR with explicit dependencies to avoid resolver deadlocks under binary-first policies.
python -m pip install --only-binary=:all: \
  "h5py>=3.10,<4" \
  "onnx==1.16.2" \
  "defusedxml>=0.7.1" \
  "anyascii>=0.3.2" \
  "huggingface-hub>=0.20,<2" \
  "pypdfium2>=4,<5" \
  "scipy==1.10.1" \
  "shapely==1.8.5.post1" \
  "tqdm>=4.66,<5" \
  "validators>=0.18.0"
python -m pip install --no-deps "python-doctr==1.0.1"

# Kraken is installed after torch to avoid repeated resolver churn.
python -m pip install --only-binary=:all: "htrmopo==0.5"
python -m pip install --only-binary=:all: "kraken==4.3.13"

python -m pip check

