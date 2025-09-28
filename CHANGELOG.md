# Changelog

## [2025-09-28] – Upgrade to PaddlePaddle 3.x + PaddleOCR 3.2.0 ecosystem

### Major Version Upgrade
- **Upgraded**: PaddlePaddle from 2.6.x to 3.2.0 (major version jump for API compatibility)
- **Upgraded**: PaddleOCR from 2.7.0.3 to 3.2.0 (latest stable with modern API)
- **Added**: Explicit PaddleX>=3.2.0, PyYAML>=6.0, typing-extensions>=4.12 dependencies
- **Fixed**: AttributeError: 'AnalysisConfig' object has no attribute 'set_optimization_level'

### API Parameter Updates
- **Updated**: `use_angle_cls` → `use_textline_orientation` (PaddleOCR 3.2.0 API)
- **Removed**: Deprecated `show_log` parameter from PaddleOCR initialization
- **Updated**: All source files (preflight.py, ocr_utils.py, setup.py) with new parameters

### Documentation Updates
- **Updated**: README.md troubleshooting with PaddlePaddle 3.x installation instructions
- **Updated**: README_docker.md with resolved set_optimization_level issue and version details
- **Updated**: requirements.txt with PaddlePaddle 3.x constraints and new dependencies

### Build Strategy
- **Maintained**: --no-deps installation strategy and binary-only wheels for reliability
- **Enhanced**: Docker builds now include PaddleX model cache volumes (/root/.paddlex)
- **Verified**: Full Docker build and pipeline initialization testing completed

## [Unreleased] – Resolve PaddleOCR ↔ PyMuPDF conflict
- Docker: Install `paddleocr==2.7.0.3` with `--no-deps`; explicitly install required runtime deps.
- Keep `PyMuPDF==1.24.10` for Python 3.11 compatibility (binary wheels).
- Update Docker docs: add note about PaddleOCR's legacy `PyMuPDF<1.21` pin and our resolution.
- Preserve ARM64 PaddlePaddle index and binary-only strategy.

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