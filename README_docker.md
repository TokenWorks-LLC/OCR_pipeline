# Docker Setup for OCR Pipeline

This Docker setup provides a unified Linux environment for the OCR pipeline that works consistently across **macOS** (Intel + Apple Silicon), **Windows** (Docker Desktop), and **Linux**. 

## Why Docker?

- **Solves macOS segfaults**: Unified Linux userspace with prebuilt wheels eliminates platform-specific issues
- **Cross-platform consistency**: Same environment and dependencies everywhere
- **Simplified dependencies**: No need to install system libraries (OpenGL, Poppler) on host
- **Isolated environment**: No conflicts with other Python projects

## Quick Start

### 1. Build the Image

```bash
# Option 1: Using Docker directly
docker build -t tokenworks-ocr:latest .

# Option 2: Using Docker Compose
docker compose build

# Option 3: For Apple Silicon if main build fails
docker build -f Dockerfile.arm64 -t tokenworks-ocr:latest .
```

**Note**: First build downloads the base image and Python packages. PaddleOCR will download model weights (~100MB) on first run.

**Apple Silicon Users**: If the main Dockerfile fails with PyMuPDF build errors, use `Dockerfile.arm64` which has enhanced ARM64 compatibility.

### 2. Run OCR on a PDF

#### macOS/Linux (Bash/Zsh)
```bash
# Using Docker Compose (recommended)
docker compose run --rm ocr "data/input/Innaya_v2 copy.pdf"
docker compose run --rm ocr "data/input/your-file.pdf"

# Using Docker directly
docker run --rm -v "$PWD":/app tokenworks-ocr:latest "data/input/your-file.pdf"

# With additional flags
docker compose run --rm ocr "data/input/sample.pdf" --dpi 300
```

#### Windows PowerShell
```powershell
# Using Docker Compose (recommended)
docker compose run --rm ocr "data/input/Innaya_v2 copy.pdf"
docker compose run --rm ocr "data/input/your-file.pdf"

# Using Docker directly
docker run --rm -v "${PWD}:/app" tokenworks-ocr:latest "data/input/your-file.pdf"
```

#### Windows CMD
```cmd
# Using Docker Compose (recommended)
docker compose run --rm ocr "data/input/Innaya_v2 copy.pdf"
docker compose run --rm ocr "data/input\your-file.pdf"

# Using Docker directly
docker run --rm -v %cd%:/app tokenworks-ocr:latest "data/input/your-file.pdf"
```

### 3. Makefile Shortcuts (macOS/Linux)

```bash
# Build image
make build

# Run with default file
make ocr

# Run with custom file
make ocr FILE="data/input/sample.pdf"

# Open interactive shell
make sh
```

## Platform-Specific Notes

### Apple Silicon (M1/M2/M3 Macs)
The Docker setup automatically uses ARM64 images. If you encounter issues, force the platform:

```bash
# Build for ARM64 specifically
docker build --platform=linux/arm64 -t tokenworks-ocr:latest .

# Or uncomment this line in docker-compose.yml:
# platform: linux/arm64
```

### Intel Macs/Windows/Linux
Usually works automatically. If needed, force AMD64:

```bash
# Build for AMD64 specifically
docker build --platform=linux/amd64 -t tokenworks-ocr:latest .

# Or uncomment this line in docker-compose.yml:
# platform: linux/amd64
```

### Linux File Permissions
If output files are created as root, run with your user ID:

```bash
# Set permissions to match your user
docker run --rm -u $(id -u):$(id -g) -v "$PWD":/app tokenworks-ocr:latest "data/input/file.pdf"

# Or in docker-compose.yml, uncomment and set:
# user: "${UID}:${GID}"
# Then export UID=$(id -u) GID=$(id -g) before running compose
```

## Advanced Usage

### Interactive Shell
```bash
# Get a bash shell inside the container
docker run --rm -it -v "$PWD":/app tokenworks-ocr:latest bash

# Or using Make
make sh
```

### Sanity Check
Verify the environment is working:
```bash
docker run --rm -v "$PWD":/app tokenworks-ocr:latest \
  -c "import paddle, cv2, platform; print('PaddlePaddle:', paddle.__version__, 'OpenCV: OK', 'Platform:', platform.machine())"
```

### Custom Commands
Run any Python script in the container:
```bash
docker run --rm -v "$PWD":/app tokenworks-ocr:latest python your_script.py
```

## Performance Notes

- **First run**: Downloads Docker image + PaddleOCR models (~300MB total)
- **Subsequent runs**: Fast startup, models are cached
- **Model caching**: PaddleOCR caches models in `~/.paddleocr/` inside container
- **CPU-only**: This setup runs on CPU everywhere (no GPU dependencies)

## Troubleshooting

### Docker Build Issues

#### PyMuPDF Build Failure (Apple Silicon)
If you see `make: not found` or `Failed building wheel for pymupdf` during Docker build:

**Problem**: PyMuPDF tries to compile from source but build tools are missing.

**Solution**: The updated Dockerfile includes build dependencies. If still failing:
```bash
# Option 1: Use the fixed Dockerfile (recommended)
docker build --platform=linux/arm64 -t tokenworks-ocr:latest .

# Option 2: Force AMD64 if ARM64 continues to fail
docker build --platform=linux/amd64 -t tokenworks-ocr:latest .
```

**Alternative workaround** if build still fails:
```bash
# Build without PyMuPDF initially, then install separately
# Edit Dockerfile temporarily to comment out pymupdf line, then:
docker build -t tokenworks-ocr:temp .
docker run --rm -it tokenworks-ocr:temp bash
# Inside container: pip install --only-binary=all pymupdf
```

### PDF2Image Issues
If you see Poppler-related errors, ensure you're running inside the Docker container (not host Python):
```bash
# Wrong: This uses your host Python
python run_pipeline.py "file.pdf"

# Correct: This uses Docker container with Poppler installed
docker compose run --rm ocr "file.pdf"
```

### Segfaults on macOS
The Docker setup eliminates these by using Linux userspace with properly compiled wheels.

### File Path Issues
- Always use quotes around file paths with spaces: `"Innaya_v2 copy.pdf"`
- On Windows, both forward slashes and backslashes work: `data/input/file.pdf` or `data\input\file.pdf`
- Relative paths are resolved from the repo root (mounted at `/app` in container)

### Model Download Issues
If PaddleOCR model downloads fail:
```bash
# Run with verbose output to see download progress
docker compose run --rm ocr "file.pdf" --verbose

# Or check network connectivity from container
docker run --rm -it tokenworks-ocr:latest bash
# Then: ping github.com
```

### Memory Issues
For large PDFs, you might need to increase Docker memory limits:
- **Docker Desktop**: Settings → Resources → Memory (increase to 4GB+)
- **Linux**: No limits by default, but monitor with `docker stats`

## File Structure

The Docker setup includes:
- `Dockerfile`: Multi-arch Linux environment with all dependencies
- `docker-compose.yml`: Easy service orchestration
- `.dockerignore`: Excludes unnecessary files from build context
- `Makefile`: Convenience commands for macOS/Linux
- This `README_docker.md`: Complete usage guide

All OCR output is written to the mounted repository directory, so results persist on your host machine.