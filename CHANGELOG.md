# Changelog

## [2025-09-28] Docker Platform Standardization

### Fixed Platform Confusion
- **Resolved**: Mixed Intel/AMD64 and Apple Silicon Docker instructions causing build failures
- **Standardized**: Clear platform-specific build paths with warnings against mixing platforms
- **Updated**: All documentation to use `tokenworks-ocr` image name and `/app` workdir consistently

### Enhanced Binary-Only Installation
- **Root Dockerfile**: Optimized for Intel/AMD64 with strict `PIP_ONLY_BINARY=:all:` settings
- **Dockerfile.arm64**: ARM64-specific optimizations with PaddlePaddle aarch64 index
- **PyMuPDF 1.24.10**: Pinned to version with reliable prebuilt wheels, prevents SWIG compilation
- **OpenCV 4.10.0.84**: Headless version with stable aarch64/amd64 wheels
- **pdf2image 1.17.0**: Corrected to actual latest stable version

### Documentation Updates
- **README.md**: Clear Intel/AMD64 vs Apple Silicon build instructions with warnings
- **README_docker.md**: Comprehensive platform-specific guide with troubleshooting
- **docker-compose.yml**: Separate services for amd64 and arm64 platforms
- **Removed**: Duplicate and conflicting Docker sections

### Build Reliability Improvements
- **Added**: Import validation step in both Dockerfiles using `import fitz, cv2, paddleocr`
- **Prevented**: Source compilation fallbacks with multiple pip environment flags
- **Ensured**: Fast, deterministic builds (3-5 minutes vs 30+ minutes with compilation)

### Breaking Changes
- **Image name**: Changed from various names to standardized `tokenworks-ocr:latest`
- **Workdir**: Standardized to `/app` across all examples (was `/workspace` in some)
- **Platform targeting**: Must use correct Dockerfile for your platform

### Migration Guide
- **Intel/AMD64 users**: Use `docker build -t tokenworks-ocr:latest .`
- **Apple Silicon users**: Use `docker build -f Dockerfile.arm64 -t tokenworks-ocr:latest .`
- **Update mounts**: Change `-v "$PWD":/workspace` to `-v "$PWD":/app -w /app`