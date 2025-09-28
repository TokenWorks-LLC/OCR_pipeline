# Docker Setup for OCR Pipeline

This Docker setup provides a unified Linux environment for the OCR pipeline that works consistently across **Windows**, **Intel Macs**, and **Apple Silicon Macs**. 

## Why Docker?

- **Solves platform issues**: Unified Linux environment with prebuilt wheels eliminates platform-specific compilation failures
- **Cross-platform consistency**: Same environment and dependencies everywhere
- **Simplified dependencies**: No need to install system libraries (OpenGL, Poppler) on host
- **Isolated environment**: No conflicts with other Python projects

## Platform-Specific Build Instructions

**⚠️ IMPORTANT**: Choose the correct build path for your system. Do NOT mix Intel/AMD64 and Apple Silicon instructions.

### Intel/AMD64 Platforms (Windows, Intel Macs, Linux servers)

#### Build and Run
```bash
# Build the image
docker build -t tokenworks-ocr:latest .

# Run OCR on a PDF
docker run --rm -v "$PWD":/app -w /app tokenworks-ocr:latest python run_pipeline.py "data/input/your-file.pdf"

# Using Docker Compose (recommended)
docker compose run --rm ocr "data/input/your-file.pdf"
```

**⚠️ Warning**: Do NOT use `Dockerfile.arm64` or ARM64 platform flags on Intel/AMD64 systems.

### Apple Silicon (M1/M2/M3 Macs)

#### Build and Run
```bash
# Build the ARM64-optimized image
docker build -f Dockerfile.arm64 -t tokenworks-ocr:latest .

# Run OCR on a PDF
docker run --rm -v "$PWD":/app -w /app tokenworks-ocr:latest python run_pipeline.py "data/input/your-file.pdf"

# Using Docker Compose (use the ARM64 service)
docker compose run --rm ocr-arm64 "data/input/your-file.pdf"
```

**⚠️ Warning**: Do NOT use the root `Dockerfile` or AMD64 platform flags on Apple Silicon systems.

## Quick Start Examples

### 1. Process a Single PDF

#### macOS/Linux (Bash/Zsh)
```bash
# Intel/AMD64 systems
docker compose run --rm ocr "data/input/sample.pdf"

# Apple Silicon systems  
docker compose run --rm ocr-arm64 "data/input/sample.pdf"
```

#### Windows PowerShell
```powershell
# Intel/AMD64 systems (most Windows machines)
docker compose run --rm ocr "data/input/sample.pdf"

# Using Docker directly
docker run --rm -v "${PWD}:/app" -w /app tokenworks-ocr:latest python run_pipeline.py "data/input/sample.pdf"
```

#### Windows CMD
```cmd
# Using Docker Compose (recommended)
docker compose run --rm ocr "data/input/sample.pdf"

# Using Docker directly
docker run --rm -v "%cd%:/app" -w /app tokenworks-ocr:latest python run_pipeline.py "data/input/sample.pdf"
```

### 2. Interactive Development

#### Open Shell for Development
```bash
# Intel/AMD64 systems
docker run --rm -it -v "$PWD":/app -w /app tokenworks-ocr:latest bash

# Apple Silicon systems
docker run --rm -it -v "$PWD":/app -w /app tokenworks-ocr:latest bash
```

### 3. Makefile Shortcuts (macOS/Linux)

```bash
# Build for your platform
make build          # Intel/AMD64
make build-arm64     # Apple Silicon

# Run with default file
make ocr

# Run with custom file  
make ocr FILE="data/input/sample.pdf"

# Open interactive shell
make sh
```

## Validation and Troubleshooting

### Import Test
After building, verify all components work:

```bash
# Test critical imports
docker run --rm tokenworks-ocr:latest python -c "import fitz, cv2, paddleocr, PIL, pdf2image; print('✅ PyMuPDF, OpenCV, PaddleOCR, Pillow, pdf2image OK')"
```

### Common Issues

#### Issue 1: PyMuPDF Build Errors
**Symptom**: `Failed to build installable wheels for some pyproject.toml based projects → PyMuPDF`  
**Cause**: Using wrong Dockerfile for your platform  
**Solution**: 
- Intel/AMD64: Use `docker build -t tokenworks-ocr:latest .`  
- Apple Silicon: Use `docker build -f Dockerfile.arm64 -t tokenworks-ocr:latest .`

#### Issue 2: Dependency Resolution Note
**Symptom**: `No matching distribution found for PyMuPDF<1.21.0` or `paddleocr has requirement PyMuPDF<1.21.0`  
**Cause**: PaddleOCR 2.7.0.3 declares PyMuPDF<1.21 but we use PyMuPDF==1.24.10 for Python 3.11  
**Fix Implemented**: We install `paddleocr` with `--no-deps` and explicitly install its runtime dependencies. Our pipeline uses pdf2image/fitz directly, so PaddleOCR's legacy PyMuPDF constraint doesn't apply. All installations use binary-only wheels (`--only-binary=:all:`).

#### Issue 3: Platform Mismatch
**Symptom**: Long build times or compilation errors  
**Cause**: Mixed platform instructions  
**Solution**: Follow platform-specific instructions exactly, don't mix them

#### Issue 4: Import Errors at Runtime
**Symptom**: `ModuleNotFoundError` for opencv, paddleocr, or fitz  
**Solution**: Rebuild image ensuring validation step passes

## File Structure

Your project directory should be mounted to `/app` in the container:

```
/app/                          # Container working directory
├── run_pipeline.py           # Main entry point
├── src/                      # Source code  
├── data/
│   ├── input/               # Input files (mount your files here)
│   └── output/              # Output results
├── config.json              # Configuration
└── requirements.txt         # Python dependencies
```

## Performance Notes

- **First build**: Downloads base image and wheels (~5-10 minutes)
- **Subsequent builds**: Use cached layers (~1-2 minutes)
- **First run**: PaddleOCR downloads models (~100MB, one-time)
- **Processing speed**: ~10-30 seconds per page depending on DPI and LLM settings

## Binary-Only Installation

Both Dockerfiles use `PIP_ONLY_BINARY=:all:` to prevent source compilation:
- ✅ **PyMuPDF 1.24.10**: Uses prebuilt wheels
- ✅ **OpenCV 4.10.0.84**: Headless version with reliable wheels
- ✅ **pdf2image 1.17.0**: Latest stable version
- ✅ **PaddlePaddle**: Uses official ARM64 index for Apple Silicon

This ensures fast, reliable builds without compilation failures.

## Advanced Usage

### Custom Configuration
```bash
# Mount custom config
docker run --rm \
  -v "$PWD":/app \
  -v "$PWD/custom-config.json:/app/config.json" \
  -w /app \
  tokenworks-ocr:latest python run_pipeline.py
```

### Batch Processing
```bash
# Process multiple files
for file in data/input/*.pdf; do
  docker compose run --rm ocr "$file"
done
```

### Persistent Model Cache
```bash
# Create volume for PaddleOCR models
docker volume create paddleocr-models

# Use persistent cache
docker run --rm \
  -v "$PWD":/app \
  -v paddleocr-models:/root/.paddleocr \
  -w /app \
  tokenworks-ocr:latest python run_pipeline.py "data/input/sample.pdf"
```

---

**Next Steps**: See the main [README.md](README.md) for configuration options and usage examples.