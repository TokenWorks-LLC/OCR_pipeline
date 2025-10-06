# Multi-architecture Docker image for OCR Pipeline with GPU support
# Uses Python base with CUDA toolkit installation (following docs reliability approach)
# For Apple Silicon (ARM64), use Dockerfile.arm64 instead
FROM python:3.11-slim

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies including basic CUDA runtime support
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    poppler-utils \
    wget \
    gnupg2 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Environment variables for Python optimization and GPU
ENV PIP_ONLY_BINARY=:all: \
    PIP_PREFER_BINARY=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0

# Upgrade pip and install build tools
RUN python -m pip install -U pip setuptools wheel

# Install core dependencies with pinned versions (following docs approach)
RUN python -m pip install --only-binary=:all: \
    "PyMuPDF==1.24.10" \
    "opencv-python-headless==4.10.0.84" \
    "pdf2image==1.17.0" \
    "numpy>=1.24.0" \
    "requests>=2.31.0" \
    "Pillow>=10.0.0"

# Install PaddlePaddle GPU version (latest compatible with NVIDIA runtime)
# Using the Docker GPU runtime, this will work with your RTX 4070
RUN python -m pip install paddlepaddle-gpu -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# Install latest compatible PaddleOCR version
RUN python -m pip install --no-deps paddleocr

# Manually install required runtime deps PaddleOCR expects (binary wheels only)
RUN python -m pip install --only-binary=:all: \
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
    "typing-extensions>=4.12"

# Set working directory
WORKDIR /app

# Copy application code
COPY . /app

# Install additional requirements if present
RUN if [ -f requirements.txt ]; then python -m pip install --only-binary=:all: -r requirements.txt; fi

# Validation step: ensure all critical imports work and check GPU support
RUN python -c "import fitz, cv2, paddleocr, PIL, pdf2image; print('✅ PyMuPDF, OpenCV, PaddleOCR, Pillow, pdf2image OK')" \
    && python -c "import paddle; print('PaddlePaddle version:', paddle.__version__); print('CUDA compiled:', paddle.device.is_compiled_with_cuda()); print('GPU count:', paddle.device.cuda.device_count() if paddle.device.is_compiled_with_cuda() else 0)"

# Default command: run pipeline with help
CMD ["python", "run_pipeline.py", "--help"]