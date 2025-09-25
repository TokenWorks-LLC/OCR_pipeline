# syntax=docker/dockerfile:1.6
# Optimized Dockerfile for Apple Silicon (arm64) - Fast, deterministic builds
# This Dockerfile ensures pip only pulls binary wheels (no source compilation)
FROM --platform=linux/arm64/v8 python:3.10-slim

# Python optimization environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_PROGRESS_BAR=off \
    DEBIAN_FRONTEND=noninteractive

# Runtime dependencies for OpenCV (binary) and general OCR needs
# These are minimal runtime deps for headless OpenCV wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    ffmpeg \
    poppler-utils \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Keep pip modern - this ensures compatibility with latest wheel formats
RUN python -m pip install -U pip wheel setuptools

# CRITICAL: Fail fast if a wheel is missing (no source builds allowed)
# This prevents the long compilation times that plague arm64 builds
ENV PIP_ONLY_BINARY=:all:
# Show which packages resolve to wheels vs builds (useful in CI logs)
ENV PIP_VERBOSE=1

# Install OpenCV + Numpy (versions that publish linux/aarch64 wheels)
# OpenCV 4.10.0.84 has reliable arm64 wheels, numpy 1.26.* series is stable
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    "opencv-python-headless==4.10.0.84" \
    "numpy==1.26.*"

# PaddlePaddle (linux/aarch64 wheels via official index)
# Using official PaddlePaddle wheel repository for ARM64
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "paddlepaddle==2.6.1" \
    -f https://www.paddlepaddle.org.cn/whl/linux/aarch64/

# PaddleOCR (pure Python, but depends on Paddle being installed first)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "paddleocr==2.7.0.3"

# Additional OCR dependencies with wheel-available versions
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    "pdf2image>=3.1.0" \
    "Pillow>=10.0.0" \
    "requests>=2.31.0" \
    "pytesseract>=0.3.10"

# PyMuPDF - Install with fallback strategy
# First try binary-only, then allow source build as last resort
RUN --mount=type=cache,target=/root/.cache/pip \
    (pip install --only-binary=all "PyMuPDF>=1.24.0" || \
     echo "Warning: PyMuPDF wheel not found, trying source build..." && \
     pip install "PyMuPDF>=1.24.0")

# Set working directory
WORKDIR /app

# Copy source code
COPY . /app

# Install any additional requirements from requirements.txt if they exist
# This allows for project-specific dependencies while maintaining wheel-only policy
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ -f requirements.txt ]; then \
        pip install -r requirements.txt; \
    fi

# Health check to verify OCR components are working
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import cv2, paddleocr; print('OCR components loaded successfully')" || exit 1

# Default command - can be overridden
CMD ["python", "-m", "src.cli", "--help"]