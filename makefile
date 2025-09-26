# Makefile for OCR Pipeline Docker operations
# Works on macOS and Linux; Windows users can run docker commands directly

.PHONY: build ocr sh clean help

# Default target
help:
	@echo "OCR Pipeline Docker Commands:"
	@echo "  make build    - Build the Docker image"
	@echo "  make ocr      - Run OCR on a PDF file (set FILE=path/to/file.pdf)"
	@echo "  make sh       - Open an interactive shell in the container"
	@echo "  make clean    - Remove the Docker image"
	@echo ""
	@echo "Examples:"
	@echo "  make ocr FILE=\"data/input/sample.pdf\""
	@echo "  make ocr FILE=\"data/input/Innaya_v2 copy.pdf\""
	@echo "  make build-arm64  - Build ARM64-optimized image (Apple Silicon)"

# Build the Docker image
build:
	docker build -t tokenworks-ocr:latest .

# Build ARM64-optimized image for Apple Silicon
build-arm64:
	docker build -f Dockerfile.arm64 -t tokenworks-ocr:latest .

# Run OCR on a file (default to sample file)
FILE ?= "data/input/Innaya_v2 copy.pdf"
ocr:
	docker compose run --rm ocr $(FILE)

# Alternative direct docker run (if you prefer not using compose)
ocr-direct:
	docker run --rm -v "$$(pwd)":/app tokenworks-ocr:latest $(FILE)

# Open an interactive shell in the container
sh:
	docker run --rm -it -v "$$(pwd)":/app tokenworks-ocr:latest bash

# Clean up the Docker image
clean:
	docker rmi tokenworks-ocr:latest || true

# Build and run in one command
run:
	$(PY) scripts/ocr_cli.py --input data/samples --output-root data/output --run-name demo --languages en --profile fast --export csv,json,overlay

test:
	$(PY) scripts/ocr_cli.py --input data/samples --output-root data/output --run-name smoke-test --skip-llm --languages en --profile fast --export csv,json

setup:
	$(PY) setup.py


