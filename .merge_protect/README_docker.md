# Docker Setup for OCR Pipeline with GPU Support

This Docker setup provides a unified Linux environment with NVIDIA GPU support for the OCR pipeline that works consistently across **Windows**, **Linux**, and **macOS** (Apple Silicon with CPU fallback). 

## Prerequisites

### NVIDIA GPU Support (Windows/Linux)

For GPU acceleration, you need:

1. **NVIDIA Driver**: Latest driver for your RTX 4070 or other CUDA-compatible GPU
2. **NVIDIA Container Toolkit**: Required for Docker GPU access

#### Windows Installation

1. **Install NVIDIA Driver**: Download from [NVIDIA Driver Downloads](https://www.nvidia.com/drivers)
2. **Install Docker Desktop**: Enable WSL2 backend
3. **Install NVIDIA Container Toolkit**:
   ```powershell
   # Open PowerShell as Administrator
   curl.exe -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o nvidia-container-toolkit-keyring.gpg
   # Follow NVIDIA's Docker installation guide for Windows
   ```

#### Linux Installation

```bash
# Add NVIDIA Container Toolkit repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/$(. /etc/os-release; echo $ID$VERSION_ID) /" | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install and configure
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Apple Silicon (M1/M2/M3 Macs)

GPU acceleration is not available. The pipeline will automatically use CPU mode with optimized parameters.

## Why Docker?

- **Solves platform issues**: Unified Linux environment with prebuilt wheels eliminates platform-specific compilation failures
- **GPU Support**: CUDA 12.1 runtime with PaddlePaddle GPU for RTX 4070 compatibility
- **Cross-platform consistency**: Same environment and dependencies everywhere
- **Simplified dependencies**: No need to install system libraries (OpenGL, Poppler, CUDA) on host
- **Isolated environment**: No conflicts with other Python projects

## Quick GPU Test

Verify your GPU setup works:

```bash
# Test GPU detection and PaddleOCR initialization
make gpu-smoke

# Or run directly:
docker compose run --rm ocr python -c "
import paddle
print('🔍 Paddle version:', paddle.__version__)
print('🔍 CUDA compiled:', paddle.is_compiled_with_cuda())
print('🔍 GPU count:', paddle.device.cuda.device_count() if paddle.is_compiled_with_cuda() else 0)
import paddleocr
ocr = paddleocr.PaddleOCR(lang='latin', use_textline_orientation=True)
print('✅ PaddleOCR init OK')
"
```

Expected output for successful GPU setup:
```
🔍 Paddle version: 2.6.1
🔍 CUDA compiled: True
🔍 GPU count: 1
✅ PaddleOCR init OK
```

## Platform-Specific Build Instructions

**⚠️ IMPORTANT**: Choose the correct build path for your system. Do NOT mix Intel/AMD64 and Apple Silicon instructions.

### Intel/AMD64 Platforms with GPU (Windows, Linux servers)

#### Build and Run
```bash
# Build the GPU-enabled image
make build-gpu

# Or build directly:
docker build -t tokenworks-ocr:latest .

# Test GPU functionality
make gpu-smoke

# Run OCR on a PDF with GPU acceleration
docker compose run --rm ocr "data/input/your-file.pdf"
```

### Intel/AMD64 Platforms without GPU

```bash
# Same commands, but PaddleOCR will automatically fall back to CPU
docker compose run --rm ocr "data/input/your-file.pdf"
```

### Apple Silicon (M1/M2/M3 Macs)

#### Build and Run
```bash
# Build the ARM64-optimized image (CPU only)
docker build -f Dockerfile.arm64 -t tokenworks-ocr:latest .

# Run OCR on a PDF
docker compose run --rm ocr-arm64 "data/input/your-file.pdf"
```

**⚠️ Warning**: Do NOT use the root `Dockerfile` or AMD64 platform flags on Apple Silicon systems.

## Pipeline Profiles

The pipeline supports different performance profiles:

### Fast Profile
- **Use case**: High-speed processing, moderate accuracy
- **Features**: Minimal preprocessing, no LLM correction, higher thresholds
- **GPU memory**: 2GB
- **DPI**: 200

```bash
make eval-fast
# Or: docker compose run --rm ocr python tools/run_gold_eval.py --profile fast
```

### Quality Profile  
- **Use case**: Maximum accuracy, full features
- **Features**: Full preprocessing, LLM correction, lower thresholds
- **GPU memory**: 6GB
- **DPI**: 300

```bash
make eval-quality  
# Or: docker compose run --rm ocr python tools/run_gold_eval.py --profile quality
```

### Balanced Profile
- **Use case**: Good balance of speed and accuracy
- **Features**: Selective preprocessing, optional LLM correction
- **GPU memory**: 4GB
- **DPI**: 300

## Evaluation and Analysis

### Gold Standard Evaluation

Run evaluation against gold standard pages:

```bash
# Fast profile evaluation
make eval-fast

# Quality profile evaluation  
make eval-quality

# Custom evaluation
docker compose run --rm ocr python tools/run_gold_eval.py --profile quality --check-targets
```

### HTML Overlays for QA

Generate interactive HTML overlays with bounding boxes and confidence scores:

```bash
make overlays

# Or generate for specific files:
docker compose run --rm ocr python -c "
from src.html_overlays import generate_overlays
generate_overlays('data/input_pdfs', 'reports/overlays', max_files=5)
"
```

Open the generated HTML files in your browser to visualize OCR results.

## Troubleshooting

### GPU Issues

#### "CUDA compiled: False"
- Verify NVIDIA drivers are installed and up to date
- Check that nvidia-smi works on the host
- Restart Docker service after installing NVIDIA Container Toolkit

#### "GPU count: 0"  
- Check GPU is not being used by other processes
- Verify docker compose includes GPU configuration
- Try: `docker run --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi`

#### PaddleOCR GPU errors
- Try CUDA 11.8 variant: change Dockerfile base image to `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`
- Reduce GPU memory allocation in profiles
- Check for conflicting CUDA installations

### CUDA Version Compatibility Matrix

| GPU Architecture | Recommended CUDA | Docker Base Image | PaddlePaddle Version |
|------------------|------------------|-------------------|---------------------|
| RTX 4070 (Ada)   | 12.1.1          | nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 | paddlepaddle-gpu==2.6.1 |
| RTX 30xx (Ampere)| 11.8.0          | nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 | paddlepaddle-gpu==2.5.2 |
| RTX 20xx (Turing)| 11.8.0          | nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 | paddlepaddle-gpu==2.5.2 |

### Memory Issues

```bash
# Monitor GPU memory usage
nvidia-smi

# Reduce GPU memory allocation in pipeline profiles
# Edit src/pipeline_profiles.py and reduce gpu_mem values
```

### Performance Issues

```bash
# Check if GPU is being used effectively
docker compose run --rm ocr python -c "
import paddle
device = paddle.device.get_device()
print(f'Current device: {device}')
"

# Monitor processing times in telemetry
tail -f reports/<RUN_ID>/logs/pipeline.jsonl
```

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
**Cause**: PaddleOCR 2.x declared restrictive PyMuPDF<1.21 but we need PyMuPDF==1.24.10 for Python 3.11  
**Fix Implemented**: Upgraded to PaddleOCR 3.2.0 ecosystem which removes legacy constraints. All packages installed with `--only-binary=:all:` for reliability.

#### Issue 3: PaddleX set_optimization_level Error (RESOLVED)
**Symptom**: `AttributeError: 'AnalysisConfig' object has no attribute 'set_optimization_level'`  
**Root Cause**: PaddleOCR 3.x auto-installs PaddleX 3.2.1 which requires PaddlePaddle 3.x API methods  
**Fix Implemented**: Upgraded to PaddlePaddle 3.2.0 + PaddleOCR 3.2.0 ecosystem for full API compatibility

#### Issue 4: Platform Mismatch
**Symptom**: Long build times or compilation errors  
**Cause**: Mixed platform instructions  
**Solution**: Follow platform-specific instructions exactly, don't mix them

#### Issue 5: Import Errors at Runtime
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
- ✅ **PyMuPDF 1.24.10**: Uses prebuilt wheels for reliable PDF processing
- ✅ **OpenCV 4.10.0.84**: Headless version with reliable computer vision wheels
- ✅ **pdf2image 1.17.0**: Latest stable PDF-to-image conversion
- ✅ **PaddlePaddle 3.2.0**: Modern deep learning framework with full API compatibility
- ✅ **PaddleOCR 3.2.0**: Latest OCR models with PaddleX 3.2.1 ecosystem support
- ✅ **PaddleX Dependencies**: PyYAML>=6.0, typing-extensions>=4.12 for model management

This ensures fast, reliable builds without compilation failures and resolves all API compatibility issues.

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
# Create volumes for model cache (PaddleOCR 3.2.0 uses PaddleX for models)
docker volume create paddleocr-models
docker volume create paddlex-models

# Use persistent cache for both PaddleOCR and PaddleX models
docker run --rm \
  -v "$PWD":/app \
  -v paddleocr-models:/root/.paddleocr \
  -v paddlex-models:/root/.paddlex \
  -w /app \
  tokenworks-ocr:latest python run_pipeline.py "data/input/sample.pdf"
```

---

**Next Steps**: See the main [README.md](README.md) for configuration options and usage examples.