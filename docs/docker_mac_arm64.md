# Docker on macOS Apple Silicon (ARM64) - Complete Guide

## Overview

This guide explains how to build the OCR pipeline Docker image quickly and deterministically on Apple Silicon Macs (M1, M2, M3, etc.). The key insight is to **never compile from source** - only use pre-built binary wheels.

## The Problem

By default, pip will compile packages from source when binary wheels aren't available for your platform. On ARM64 systems, packages like OpenCV and PaddlePaddle often trigger compilation, which can take 20-60 minutes and may fail due to missing build dependencies.

## The Solution

Our optimized `Dockerfile.arm64` uses these strategies:

### 1. Force Binary-Only Installation
```dockerfile
ENV PIP_ONLY_BINARY=:all:
```
This environment variable tells pip to **fail immediately** if a binary wheel isn't available, rather than attempting source compilation.

### 2. Platform-Specific Base Image
```dockerfile
FROM --platform=linux/arm64/v8 python:3.10-slim
```
Explicitly targets ARM64 architecture to ensure we get the right wheels.

### 3. Pinned Wheel-Available Versions
- **OpenCV**: `4.10.0.84` (has reliable linux/aarch64 wheels)
- **PaddlePaddle**: `2.6.1` (from official ARM64 index)
- **Numpy**: `1.26.*` (excellent ARM64 support)

### 4. Official ARM64 Repositories
```dockerfile
RUN pip install "paddlepaddle==2.6.1" \
    -f https://www.paddlepaddle.org.cn/whl/linux/aarch64/
```
Uses PaddlePaddle's official ARM64 wheel repository.

### 5. Build Caching
```dockerfile
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "opencv-python-headless==4.10.0.84"
```
Caches downloaded wheels across builds for faster rebuilds.

## Build Commands

### Standard Build
```bash
DOCKER_BUILDKIT=1 docker build \
  --progress=plain \
  --platform=linux/arm64/v8 \
  -t ocrx:dev -f Dockerfile.arm64 .
```

### With Cache Debugging
```bash
DOCKER_BUILDKIT=1 docker build \
  --progress=plain \
  --no-cache \
  --platform=linux/arm64/v8 \
  -t ocrx:dev -f Dockerfile.arm64 .
```

### Build with Compose
```bash
DOCKER_BUILDKIT=1 docker compose -f docker-compose.yml build \
  --build-arg BUILDKIT_INLINE_CACHE=1
```

## Verification Steps

### 1. Check Build Logs
Look for these **good** patterns in build output:
```
Using cached opencv_python_headless-4.10.0.84-...whl
Downloading paddlepaddle-2.6.1-...aarch64.whl
```

Avoid these **bad** patterns:
```
Building wheel for opencv-python-headless
Running setup.py bdist_wheel
Building wheel for package (setup.py)
```

### 2. Verify Installed Packages
```bash
docker run --rm ocrx:dev pip list | grep -E "(opencv|paddle|numpy)"

# Expected output:
# numpy                    1.26.4
# opencv-python-headless   4.10.0.84
# paddleocr                2.7.0.3
# paddlepaddle             2.6.1
```

### 3. Runtime Test
```bash
docker run --rm ocrx:dev python -c "
import cv2
import paddleocr
import numpy as np
print('OpenCV version:', cv2.__version__)
print('NumPy version:', np.__version__)
print('All imports successful!')
"
```

## Troubleshooting

### Scenario 1: Wheel Not Found Error
```
ERROR: Could not find a version that satisfies the requirement opencv-python-headless==4.10.0.84
```

**Solution**: The wheel may not exist for your exact platform. Try:
1. Update to newer version with ARM64 wheels
2. Use the fallback Dockerfile pattern:
```dockerfile
RUN pip install --only-binary=all "opencv-python-headless>=4.8.0" || \
    echo "Warning: Using APT fallback" && \
    apt-get update && apt-get install -y python3-opencv
```

### Scenario 2: Build Still Takes Long Time
**Diagnosis**: Check if `PIP_ONLY_BINARY` is being respected:
```bash
docker build --progress=plain ... 2>&1 | grep -i "building wheel"
```

**Solution**: If you see "Building wheel", a package is still compiling. Either:
- Pin to a wheel-available version
- Use system package fallback
- Remove the problematic package if not essential

### Scenario 3: Runtime Errors After Build
```
ImportError: libGL.so.1: cannot open shared object file
```

**Solution**: Missing runtime dependencies. Add to Dockerfile:
```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
```

### Scenario 4: PaddlePaddle Import Issues
```
ModuleNotFoundError: No module named 'paddle'
```

**Solution**: Ensure you're using the ARM64-specific index:
```dockerfile
RUN pip install "paddlepaddle==2.6.1" \
    -f https://www.paddlepaddle.org.cn/whl/linux/aarch64/ \
    --trusted-host www.paddlepaddle.org.cn
```

## Fallback Strategies

If binary wheels fail, here are fallback approaches:

### 1. System Package Fallback
```dockerfile
# Try wheel first, fall back to apt package
RUN pip install --only-binary=all "opencv-python-headless>=4.8.0" || \
    (apt-get update && \
     apt-get install -y python3-opencv && \
     pip install opencv-python-headless --no-deps)
```

### 2. Multi-Stage Build with Compilation
```dockerfile
# Build stage with compilers
FROM python:3.10-slim as builder
RUN apt-get update && apt-get install -y build-essential cmake
COPY requirements.txt .
RUN pip wheel --no-cache-dir -w /wheels -r requirements.txt

# Runtime stage
FROM python:3.10-slim
COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links /wheels -r requirements.txt
```

### 3. Alternative Package Versions
```dockerfile
# Try multiple versions until one has wheels
RUN pip install --only-binary=all "opencv-python-headless==4.10.0.84" || \
    pip install --only-binary=all "opencv-python-headless==4.9.0.80" || \
    pip install --only-binary=all "opencv-python-headless==4.8.1.78"
```

## Performance Tips

### 1. Layer Optimization
```dockerfile
# Bad: Each install creates a layer
RUN pip install opencv-python-headless
RUN pip install paddlepaddle
RUN pip install paddleocr

# Good: Group related packages
RUN pip install \
    opencv-python-headless==4.10.0.84 \
    numpy==1.26.*
```

### 2. Cache Mounting
```dockerfile
# Always use cache mounts for pip
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install package==version
```

### 3. Build Context Optimization
Create `.dockerignore`:
```
.git
*.pyc
__pycache__
tests/
docs/
.pytest_cache
*.egg-info
```

## Expected Build Times

With the optimized Dockerfile:
- **First build**: 3-5 minutes (downloading wheels)
- **Cached rebuild**: 30-60 seconds (using cached layers)
- **Package changes**: 1-2 minutes (only affected layers)

Compare to source compilation:
- **OpenCV compilation**: 15-30 minutes
- **Full build with compilation**: 30-60 minutes

## Architecture Notes

### Why Python 3.10?
- Excellent ARM64 wheel availability
- Stable feature set
- Good performance
- Compatible with all our dependencies

### Why `linux/arm64/v8`?
- Most specific ARM64 variant
- Best wheel compatibility
- Future-proof for newer ARM chips

### Why PaddlePaddle ARM64 Index?
- Official repository has the most up-to-date ARM64 wheels
- Better than PyPI for ARM64 PaddlePaddle packages
- Includes GPU-enabled versions if needed

## Integration with CI/CD

### GitHub Actions
```yaml
jobs:
  build-arm64:
    runs-on: macos-latest  # or self-hosted ARM64
    steps:
      - uses: actions/checkout@v4
      - name: Build ARM64 image
        run: |
          DOCKER_BUILDKIT=1 docker build \
            --platform=linux/arm64/v8 \
            -t ocrx:arm64 -f Dockerfile.arm64 .
```

### Multi-Architecture Build
```bash
# Create buildx builder
docker buildx create --name multiarch --use

# Build for both platforms
docker buildx build \
  --platform=linux/amd64,linux/arm64/v8 \
  -t ocrx:latest -f Dockerfile.arm64 \
  --push .
```

## References

- [PaddlePaddle ARM64 Wheels](https://www.paddlepaddle.org.cn/whl/linux/aarch64/)
- [OpenCV ARM64 Support](https://github.com/opencv/opencv-python/releases)
- [Docker Buildx Multi-platform](https://docs.docker.com/buildx/working-with-buildx/)
- [PIP Binary-Only Flag](https://pip.pypa.io/en/stable/cli/pip_install/#cmdoption-only-binary)