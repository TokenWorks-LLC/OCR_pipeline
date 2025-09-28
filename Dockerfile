# Multi-architecture Docker image for OCR Pipeline
# Optimized for Intel/AMD64 platforms with binary-only wheel installs
# For Apple Silicon (ARM64), use Dockerfile.arm64 instead
FROM python:3.11-slim

# Environment variables for Python optimization and binary-only installs
ENV PIP_ONLY_BINARY=:all: \
    PIP_PREFER_BINARY=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies (no build tools to prevent source compilation)
# - libgl1: OpenGL library required by opencv-python-headless
# - libglib2.0-0: GLib library for low-level system operations  
# - poppler-utils: PDF utilities (pdfinfo, pdftoppm) for pdf2image
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip and install build tools
RUN pip install -U pip setuptools wheel

# Install Python dependencies with pinned versions for reliability
# Using --only-binary=:all: to ensure no source compilation
RUN pip install --only-binary=:all: \
    "PyMuPDF==1.24.10" \
    "opencv-python-headless==4.10.0.84" \
    "pdf2image==1.17.0" \
    "numpy>=1.24.0" \
    "requests>=2.31.0" \
    "Pillow>=10.0.0"

# Install PaddlePaddle (upgrade to 3.x for PaddleOCR 3.2.0 + PaddleX 3.2.1 compatibility)
RUN pip install --only-binary=:all: "paddlepaddle>=3.0.0,<4.0.0"

# Install PaddleOCR without deps (newer version compatible with PaddlePaddle 3.x)
RUN pip install --only-binary=:all: --no-deps "paddleocr>=3.2.0"

# Manually install required runtime deps PaddleOCR expects (binary wheels only)
RUN pip install --only-binary=:all: \
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

# Set working directory
WORKDIR /app

# Copy application code
COPY . /app

# Install additional requirements if present
RUN if [ -f requirements.txt ]; then pip install --only-binary=:all: -r requirements.txt; fi

# Validation step: ensure all critical imports work
RUN python -c "import fitz, cv2, paddleocr, PIL, pdf2image; print('✅ PyMuPDF, OpenCV, PaddleOCR, Pillow, pdf2image OK')"

# Default command: run pipeline with help
CMD ["python", "run_pipeline.py", "--help"]