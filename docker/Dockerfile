# syntax=docker/dockerfile:1.6
FROM --platform=linux/arm64/v8 python:3.10-slim

# Prefer binary wheels; avoid source builds in CI
ENV PIP_ONLY_BINARY=:all: \
    PIP_PREFER_BINARY=1 \
    PIP_NO_BUILD_ISOLATION=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Additional Python/pip behavior settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_PROGRESS_BAR=off \
    PIP_VERBOSE=1

# Runtime libs; poppler-utils enables pdf2image (no compile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 ffmpeg poppler-utils \
  && rm -rf /var/lib/apt/lists/*

# Keep pip toolchain modern
RUN python -m pip install -U pip wheel setuptools

# Force PyMuPDF to install from wheel (prevents vendored MuPDF/Tesseract build)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --only-binary=:all: "PyMuPDF==1.24.10"

# 2) Core OCR stack (all wheels for arm64 as pinned)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --only-binary=:all: \
      "paddlepaddle==2.6.1" -f https://www.paddlepaddle.org.cn/whl/linux/aarch64/

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --only-binary=:all: "paddleocr==2.7.0.3"

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --only-binary=:all: \
      "opencv-python-headless==4.10.0.84" "numpy==1.26.*"

# 3) PDF helpers + IO
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --only-binary=:all: \
      "pdf2image==1.17.0" \
      "Pillow>=10.0.0" \
      "requests>=2.31.0" \
      "pytesseract>=0.3.10"

# (Optional) In case GUI OpenCV wheels were pulled transitively, remove them to keep image lean/headless
RUN pip uninstall -y opencv-python opencv-contrib-python || true

# Copy application code
COPY . /app
WORKDIR /app

# Install remaining project dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Health check to verify OCR components
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import paddleocr, cv2, pymupdf; print('OCR components OK')" || exit 1

# Default command
CMD ["python", "run_pipeline.py", "--help"]

# Notes:
# - If a future OpenCV pin lacks an arm64 wheel, pick another pinned version with wheels
#   or install distro package: `apt-get update && apt-get install -y --no-install-recommends python3-opencv`
#   and REMOVE any pip opencv-* installs.