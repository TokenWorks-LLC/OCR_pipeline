# Multi-architecture Docker image for OCR Pipeline
# Supports: linux/amd64 (Intel), linux/arm64 (Apple Silicon, ARM servers)
FROM python:3.11-slim

# Environment variables for Python optimization
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for OCR and image processing
# - libgl1: OpenGL library required by opencv-python-headless
# - libglib2.0-0: GLib library for low-level system operations
# - poppler-utils: PDF utilities (pdfinfo, pdftoppm) for pdf2image
# - build-essential: C compiler and build tools for PyMuPDF compilation
# - cmake: Build system generator for PyMuPDF
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip and install build tools
RUN pip install -U pip setuptools wheel

# Install Python dependencies with pinned versions for stability
# Note: Using opencv-python-headless instead of opencv-python because:
#   1. Headless version has no GUI dependencies (smaller image)
#   2. Prevents X11/display issues in containerized environments
#   3. All CV operations work the same, just no imshow/GUI functions
RUN pip install \
    paddlepaddle==2.6.1 \
    paddleocr==2.7.0 \
    opencv-python-headless>=4.8.0 \
    pdf2image \
    numpy>=1.24.0 \
    requests>=2.31.0 \
    Pillow>=10.0.0

# Install PyMuPDF separately with specific handling for ARM64
# This avoids compilation issues on Apple Silicon by using pre-built wheels when available
RUN pip install --only-binary=all pymupdf>=1.24 || \
    pip install pymupdf>=1.24

# Clean up build dependencies to reduce image size (optional)
# Uncomment the next line if you want a smaller final image
# RUN apt-get remove -y build-essential cmake && apt-get autoremove -y

# Set working directory
WORKDIR /app

# Copy application code
COPY . /app

# Important: PaddleOCR downloads models on first run (~100MB)
# Models are cached in ~/.paddleocr/ and persist between container runs
# if you mount a volume or use the same container instance

# Entrypoint: run_pipeline.py accepts PDF path and optional flags
# Usage: docker run [image] "path/to/file.pdf" --dpi 300
ENTRYPOINT ["python", "run_pipeline.py"]