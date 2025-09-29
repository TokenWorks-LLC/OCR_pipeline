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

## Platform-Specific Deployment Guides

### Windows Deployment (Comprehensive)

#### Prerequisites
1. **Install Docker Desktop**: Download from docker.com
   - Enable WSL2 backend during installation
   - Allocate 8GB+ memory: Docker Desktop → Settings → Resources → Memory
2. **Choose your terminal**: PowerShell (recommended) or CMD

#### Windows PowerShell (Recommended)
```powershell
# Navigate to project directory
cd C:\path\to\OCR_pipeline

# Build image (Intel/AMD64 - most Windows machines)
docker build -t ocr-pipeline .

# Process files with PowerShell path syntax
docker run --rm -v "${PWD}:/workspace" -w /workspace ocr-pipeline python run_pipeline.py

# Interactive development
docker run --rm -it -v "${PWD}:/workspace" -w /workspace ocr-pipeline bash

# Using docker-compose (easier management)
docker compose run --rm ocr "data/input/document.pdf"
```

#### Windows CMD (Alternative)
```cmd
REM Navigate to project directory  
cd C:\path\to\OCR_pipeline

REM Build image
docker build -t ocr-pipeline .

REM Process files with CMD path syntax
docker run --rm -v "%cd%:/workspace" -w /workspace ocr-pipeline python run_pipeline.py

REM Interactive shell
docker run --rm -it -v "%cd%:/workspace" -w /workspace ocr-pipeline bash
```

#### Windows Subsystem for Linux (WSL) - Best Performance
```bash
# Install Docker Desktop with WSL2 backend
# Open WSL terminal (Ubuntu, etc.)

# Navigate using WSL paths
cd /mnt/c/Users/YourName/path/to/OCR_pipeline

# Use Linux commands (best performance on Windows)
docker build -t ocr-pipeline .
docker run --rm -v "$PWD:/workspace" -w /workspace ocr-pipeline python run_pipeline.py
```

#### Windows-Specific GUI Tool
For non-technical users, use the included Windows batch script:
```cmd
REM Double-click or run from terminal:
run_ocr.bat

REM Provides menu-driven interface:
REM 1. Setup OCR Pipeline
REM 2. Process files 
REM 3. Configuration help
```

### Linux Deployment (Distribution-Specific)

#### Ubuntu/Debian Systems
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker

# Install system dependencies for native Python (if not using Docker)
sudo apt-get update && sudo apt-get install -y \
  libgl1 libglib2.0-0 libsm6 libxext6 \
  libfontconfig1 libxrender1 poppler-utils \
  python3-pip python3-venv

# Build and run
docker build -t ocr-pipeline .
docker run --rm -v "$PWD:/workspace" -w /workspace ocr-pipeline python run_pipeline.py
```

#### CentOS/RHEL/Rocky Linux
```bash
# Install Docker
sudo yum install -y yum-utils
sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo yum install -y docker-ce docker-ce-cli containerd.io
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

# Install system dependencies
sudo yum install -y mesa-libGL glib2-devel libSM libXext \
  libfontconfig libXrender poppler-utils python3-pip

# Build and run (same as other platforms)
docker build -t ocr-pipeline .
docker run --rm -v "$PWD:/workspace" -w /workspace ocr-pipeline python run_pipeline.py
```

#### Arch Linux
```bash
# Install Docker
sudo pacman -S docker docker-compose
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
newgrp docker

# System dependencies
sudo pacman -S glib2 libgl libsm libxext fontconfig libxrender poppler python-pip

# Build and run
docker build -t ocr-pipeline .
docker run --rm -v "$PWD:/workspace" -w /workspace ocr-pipeline python run_pipeline.py
```

### macOS Deployment (Intel vs Apple Silicon)

#### Intel Macs
```bash
# Install Docker Desktop from docker.com
# Use standard Dockerfile:
docker build -t ocr-pipeline .
docker run --rm -v "$PWD:/workspace" -w /workspace ocr-pipeline python run_pipeline.py

# For native Python installation:
brew install python@3.10 poppler
pip3 install -r requirements.txt
python3 run_pipeline.py
```

#### Apple Silicon Macs (M1/M2/M3) - Optimized
```bash
# Install Docker Desktop with Apple Silicon support
# ALWAYS use ARM64 Dockerfile for best performance:
docker build -f Dockerfile.arm64 -t ocr-pipeline .
docker run --rm -v "$PWD:/workspace" -w /workspace ocr-pipeline python run_pipeline.py

# Enable Rosetta 2 for x86 compatibility if needed:
# Docker Desktop → Settings → General → "Use Rosetta for x86/amd64 emulation"

# Native installation with Homebrew (alternative):
brew install python@3.10 poppler tesseract
pip3 install --only-binary=all -r requirements.txt
python3 run_pipeline.py
```

## Advanced Deployment Scenarios

### Corporate/Enterprise Environments

#### Behind Corporate Firewall
```bash
# Configure Docker to use corporate proxy
# Create ~/.docker/config.json:
{
  "proxies": {
    "default": {
      "httpProxy": "http://proxy.company.com:8080",
      "httpsProxy": "http://proxy.company.com:8080"
    }
  }
}

# Build with proxy settings
docker build --build-arg HTTP_PROXY=http://proxy.company.com:8080 \
             --build-arg HTTPS_PROXY=http://proxy.company.com:8080 \
             -t ocr-pipeline .

# Alternative: Use internal package mirrors
# Edit Dockerfile to use corporate PyPI mirror:
# RUN pip install -i https://pypi.company.com/simple/ -r requirements.txt
```

#### Air-Gapped/Offline Systems
```bash
# Option 1: Export Docker image from online system
docker save ocr-pipeline:latest | gzip > ocr-pipeline.tar.gz
# Transfer file to offline system, then:
gunzip -c ocr-pipeline.tar.gz | docker load

# Option 2: Create custom requirements bundle
pip download -r requirements.txt -d packages/
# Transfer packages/ directory, then install offline:
pip install --no-index --find-links packages/ -r requirements.txt
```

#### High-Security Environments
```bash
# Run with restricted capabilities
docker run --rm --read-only \
  --security-opt=no-new-privileges \
  --user 1000:1000 \
  -v "$PWD/data:/app/data" \
  ocr-pipeline python run_pipeline.py

# Scan image for vulnerabilities (if required)
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image ocr-pipeline:latest
```

### Cloud Deployment

#### AWS ECS/Fargate
```yaml
# task-definition.json
{
  "family": "ocr-pipeline",
  "cpu": "2048",
  "memory": "4096",
  "networkMode": "awsvpc",
  "containerDefinitions": [{
    "name": "ocr-worker",
    "image": "your-registry/ocr-pipeline:latest",
    "memory": 4096,
    "mountPoints": [{
      "sourceVolume": "input-data",
      "containerPath": "/app/data/input"
    }]
  }]
}
```

#### Google Cloud Run
```bash
# Build and push to Google Container Registry
docker build -t gcr.io/your-project/ocr-pipeline .
docker push gcr.io/your-project/ocr-pipeline

# Deploy to Cloud Run
gcloud run deploy ocr-service \
  --image gcr.io/your-project/ocr-pipeline \
  --memory 4Gi \
  --cpu 2 \
  --timeout 1800
```

#### Azure Container Instances
```bash
# Create resource group and deploy
az group create --name ocr-rg --location eastus
az container create \
  --resource-group ocr-rg \
  --name ocr-pipeline \
  --image your-registry/ocr-pipeline:latest \
  --cpu 2 \
  --memory 4 \
  --restart-policy Never
```

### Production Deployment Best Practices

#### Docker Compose Production
```yaml
version: '3.8'
services:
  ocr-processor:
    build: 
      context: .
      dockerfile: Dockerfile.arm64  # or Dockerfile for Intel
    volumes:
      - ./data/input:/app/data/input:ro
      - ./data/output:/app/data/output
      - ./logs:/app/logs
    environment:
      - LOG_LEVEL=INFO
      - MAX_WORKERS=4
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "from src.healthcheck import run_health_check; run_health_check()"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'
```

#### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ocr-pipeline
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ocr-pipeline
  template:
    metadata:
      labels:
        app: ocr-pipeline
    spec:
      containers:
      - name: ocr-worker
        image: ocr-pipeline:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        volumeMounts:
        - name: input-storage
          mountPath: /app/data/input
        - name: output-storage
          mountPath: /app/data/output
```

## Deployment Validation & Testing

### Quick Validation Suite

#### 1. System Compatibility Check
```bash
# Check Docker installation
docker --version
docker compose --version

# Check system architecture (important for choosing Dockerfile)
uname -m  
# x86_64 = Intel/AMD64 (use Dockerfile)
# arm64/aarch64 = Apple Silicon (use Dockerfile.arm64)

# Check available resources
docker system df  # Docker disk usage
docker system info | grep -E "(CPUs|Total Memory)"  # Available resources
```

#### 2. Build Validation
```bash
# Build with detailed logging
docker build --progress=plain -f Dockerfile.arm64 -t ocr-pipeline-test . 2>&1 | tee build.log

# Check for success indicators in build.log:
grep -E "Successfully built|Successfully tagged" build.log

# Verify no compilation occurred (should see only wheel installations):
grep -i "building wheel" build.log  # Should return EMPTY (no results)
grep -E "Using cached|Downloading.*whl" build.log  # Should show wheel installations
```

#### 3. Container Functionality Test
```bash
# Test Python environment
docker run --rm ocr-pipeline-test python --version

# Test critical imports
docker run --rm ocr-pipeline-test python -c "
import sys, platform
print(f'✅ Python {sys.version.split()[0]} on {platform.machine()}')

import cv2, fitz, paddleocr, PIL, pdf2image, numpy
print(f'✅ OpenCV: {cv2.__version__}')
print(f'✅ PyMuPDF: {fitz.VersionBind}')
print(f'✅ NumPy: {numpy.__version__}')
print('✅ All OCR dependencies loaded successfully!')
"

# Test main application
docker run --rm ocr-pipeline-test python run_pipeline.py --help
```

#### 4. End-to-End Processing Test
```bash
# Create test file
echo "This is a test document for OCR validation." > test_input.txt

# Convert to PDF for testing (if you have pandoc installed)
# Or use any existing PDF file
cp /path/to/any/document.pdf data/input/test.pdf

# Process with dry-run first
docker run --rm -v "$PWD:/workspace" -w /workspace ocr-pipeline-test \
  python run_pipeline.py --dry-run

# Process actual file
docker run --rm -v "$PWD:/workspace" -w /workspace ocr-pipeline-test \
  python run_pipeline.py

# Verify output was created
ls -la data/output/  # Should contain processed results
```

### Performance Benchmarks

#### Expected Build Times
| Platform | First Build | Cached Rebuild | With --no-cache |
|----------|-------------|---------------|-----------------|
| Apple Silicon (M1/M2/M3) | 2-5 min | 30-60 sec | 3-6 min |
| Intel Mac | 5-10 min | 1-2 min | 6-12 min |
| Linux (good connection) | 3-8 min | 45-90 sec | 4-10 min |
| Windows (WSL2) | 4-10 min | 1-3 min | 5-12 min |

#### Expected Processing Speeds
| Document Type | Pages | Processing Time | Notes |
|---------------|-------|----------------|--------|
| Simple PDF | 1-5 pages | 10-30 seconds | Text-based PDFs |
| Scanned PDF | 1-5 pages | 30-120 seconds | Image-based, depends on DPI |
| High-res images | 1-5 images | 60-300 seconds | 300+ DPI images |
| Batch processing | 10-50 files | 5-30 minutes | Parallel processing enabled |

### Troubleshooting Validation

#### Build Issues Diagnosis
```bash
# Check specific error patterns
grep -i -E "(error|failed|fatal)" build.log | head -10

# Common ARM64 issues check
grep -i -E "(swig|cmake|gcc|compilation)" build.log
# Should return EMPTY if using correct Dockerfile.arm64

# Check wheel vs source installation
grep -E "(Building wheel|Preparing metadata)" build.log
# Should be minimal/empty for optimized builds

# Memory issues during build
docker system info | grep -i memory
# Should show 4GB+ available for reliable builds
```

#### Runtime Issues Diagnosis
```bash
# Test container startup
docker run --rm -it ocr-pipeline-test bash
# Should open shell promptly without errors

# Check import issues
docker run --rm ocr-pipeline-test python -c "
try:
    import cv2
    print('✅ OpenCV OK')
except Exception as e:
    print(f'❌ OpenCV Error: {e}')

try:
    import fitz
    print('✅ PyMuPDF OK')
except Exception as e:
    print(f'❌ PyMuPDF Error: {e}')

try:
    from paddleocr import PaddleOCR
    print('✅ PaddleOCR OK')
except Exception as e:
    print(f'❌ PaddleOCR Error: {e}')
"

# Test file processing capabilities
docker run --rm -v "$PWD:/workspace" -w /workspace ocr-pipeline-test \
  python -c "
from src.healthcheck import run_health_check
run_health_check()
"
```

### Deployment Health Monitoring

#### Continuous Health Checks
```bash
# Add to monitoring scripts
#!/bin/bash
# ocr_health_check.sh

echo "$(date): Starting OCR Pipeline health check"

# Check container health
if ! docker run --rm ocr-pipeline python -c "
import cv2, fitz, paddleocr
print('Health check passed')
" 2>/dev/null; then
    echo "❌ Container health check failed"
    exit 1
fi

# Check processing capability
if ! docker run --rm -v "$PWD:/workspace" -w /workspace ocr-pipeline \
    python run_pipeline.py --validate-only 2>/dev/null; then
    echo "❌ Configuration validation failed"
    exit 1
fi

echo "✅ OCR Pipeline health check passed"
```

#### Performance Monitoring
```bash
# Monitor container resource usage
docker stats ocr-pipeline --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"

# Monitor processing queue (if using batch processing)
watch -n 5 'ls -la data/input/ | wc -l; ls -la data/output/ | wc -l'
```

### Cross-Platform Compatibility Matrix

| Feature | Windows | macOS Intel | macOS ARM64 | Linux |
|---------|---------|-------------|-------------|--------|
| Docker build | ✅ | ✅ | ✅ (optimized) | ✅ |
| Native Python | ⚠️ (complex) | ✅ | ⚠️ (use Docker) | ✅ |
| GPU acceleration | ✅ (WSL2) | ❌ | ❌ | ✅ |
| Batch processing | ✅ | ✅ | ✅ | ✅ |
| LLM integration | ✅ | ✅ | ✅ | ✅ |

**Legend:** ✅ Fully supported, ⚠️ Supported with limitations, ❌ Not available

---