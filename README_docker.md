# Docker Guide

This guide covers Docker usage for the current OCR pipeline entrypoints.

## Images

- `Dockerfile`: amd64-oriented build
- `Dockerfile.arm64`: Apple Silicon build

## Build

### Intel/AMD64

```bash
docker build -t tokenworks-ocr:latest .
```

### Apple Silicon (M1/M2/M3)

```bash
docker build -f Dockerfile.arm64 -t tokenworks-ocr:latest .
```

## Run

### Compatibility entrypoint

```bash
docker run --rm -v "$PWD":/workspace -w /workspace tokenworks-ocr:latest \
  python run_pipeline.py --input-dir data/input --output-dir reports/output
```

### Page-text entrypoint

```bash
docker run --rm -v "$PWD":/workspace -w /workspace tokenworks-ocr:latest \
  python tools/run_page_text.py \
    --inputs data/input \
    --output-root reports/output \
    --prefer-text-layer \
    --ocr-fallback paddle
```

## Docker Compose

Compose services are architecture-specific:

- `ocr`: amd64 path
- `ocr-arm64`: Apple Silicon path

```bash
# amd64
docker compose run --rm ocr python test_pipeline.py --allow-missing-engines

# arm64
docker compose run --rm ocr-arm64 python test_pipeline.py --allow-missing-engines
```

## What Was Verified

The current environment does not have Docker installed, so Docker commands in this file were not executed here.
The commands were cross-checked against existing repo entrypoints and compose service names.

## Troubleshooting

### Wrong image for platform

If builds are slow or fail on Apple Silicon, use `Dockerfile.arm64`.

### No output generated

Check the mounted working directory and output root:

```bash
docker run --rm -v "$PWD":/workspace -w /workspace tokenworks-ocr:latest \
  python run_pipeline.py --validate-only -c config.json
```

### Strict engine check fails

Run portable check first:

```bash
python test_pipeline.py --allow-missing-engines
```

Use strict mode only when all backends are installed:

```bash
python test_pipeline.py
```
