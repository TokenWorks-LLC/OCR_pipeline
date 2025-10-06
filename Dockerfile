# Multi-architecture Docker image for OCR Pipeline with GPU support
# Uses CUDA 12.1 runtime for NVIDIA RTX 4070 compatibility
# For Apple Silicon (ARM64), use Dockerfile.arm64 instead
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
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

# Make Python 3.11 the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Environment variables for Python optimization and GPU
ENV PIP_ONLY_BINARY=:all: \
    PIP_PREFER_BINARY=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0

# Upgrade pip and install build tools
RUN python -m pip install -U pip setuptools wheel

# Install core dependencies with pinned versions
RUN python -m pip install --only-binary=:all: \
    "PyMuPDF==1.24.10" \
    "opencv-python-headless==4.10.0.84" \
    "pdf2image==1.17.0" \
    "numpy>=1.24.0" \
    "requests>=2.31.0" \
    "Pillow>=10.0.0"

# Clean install of PaddlePaddle GPU for CUDA 12.1
RUN python -m pip uninstall -y paddlepaddle paddlepaddle-gpu || true \
    && python -m pip install --no-cache-dir paddlepaddle-gpu==2.6.1

# Install PaddleOCR 2.7.x stable
RUN python -m pip install paddleocr==2.7.0.3

# Manually install required runtime deps PaddleOCR expects (binary wheels only)
RUN python -m pip install --only-binary=:all: \
    "shapely==2.0.6" \
    "scikit-image==0.24.0" \
    "imgaug==0.4.0" \
    "pyclipper==1.3.0.post5" \
    "lmdb==1.4.1" \
    "rapidfuzz==3.9.7" \
    "lxml==5.3.0" \
    "premailer==3.10.0" \
    "openpyxl==3.1.5" \
    "tqdm==4.66.5" \
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
    && python -c "import paddle; print('PaddlePaddle version:', paddle.__version__); print('CUDA compiled:', paddle.is_compiled_with_cuda()); print('GPU count:', paddle.device.cuda.device_count() if paddle.is_compiled_with_cuda() else 0)"

# Default command: run pipeline with help
CMD ["python", "run_pipeline.py", "--help"]