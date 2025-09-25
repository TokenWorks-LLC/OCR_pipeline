# syntax=docker/dockerfile:1.6
FROM --platform=linux/arm64/v8 python:3.10-slim

# Keep Python output clean and installs predictable
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_PROGRESS_BAR=off \
    PIP_ONLY_BINARY=:all: \
    PIP_VERBOSE=1

# Runtime libs for OpenCV, Pillow, pdf2image, etc.
# Note: poppler-utils supports pdf2image without compiling anything.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 ffmpeg poppler-utils \
  && rm -rf /var/lib/apt/lists/*

# Keep pip toolchain modern
RUN python -m pip install -U pip wheel setuptools

# IMPORTANT: Pin PyMuPDF to a prebuilt manylinux2014 aarch64 wheel to avoid SWIG
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "pymupdf==1.24.10"

# OpenCV + NumPy: versions known to publish linux/aarch64 wheels
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "opencv-python-headless==4.10.0.84" "numpy==1.26.*"

# PaddlePaddle aarch64 wheels come from Paddle's index; PaddleOCR is pure Python
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "paddlepaddle==2.6.1" -f https://www.paddlepaddle.org.cn/whl/linux/aarch64/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "paddleocr==2.7.0.3" "Pillow>=10.0.0" "requests>=2.31.0" "pdf2image"

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

# Notes / Fallbacks:
# - If a future OpenCV pin lacks an arm64 wheel, either:
#   A) switch to another pinned version with wheels, or
#   B) apt-get install the distro package: `apt-get update && apt-get install -y --no-install-recommends python3-opencv`
#      and REMOVE any pip `opencv-python(-headless)` installs.