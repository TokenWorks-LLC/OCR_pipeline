# Makefile for OCR Pipeline Docker operations
# Works on macOS and Linux; Windows users can run docker commands directly

.PHONY: build ocr sh clean help build-gpu gpu-smoke gpu-smoke-doctr gpu-smoke-mmocr gpu-smoke-kraken eval-fast eval-quality eval-baseline overlays

# Default target
help:
	@echo "OCR Pipeline Docker Commands:"
	@echo "  make build      - Build the Docker image"
	@echo "  make build-gpu  - Build the GPU-enabled Docker image"
	@echo "  make gpu-smoke  - Test GPU and PaddleOCR functionality"
	@echo "  make gpu-smoke-doctr  - Test docTR engine availability"
	@echo "  make gpu-smoke-mmocr  - Test MMOCR engine availability"
	@echo "  make gpu-smoke-kraken - Test Kraken engine availability"
	@echo "  make ocr        - Run OCR on a PDF file (set FILE=path/to/file.pdf)"
	@echo "  make sh         - Open an interactive shell in the container"
	@echo "  make clean      - Remove the Docker image"
	@echo "  make eval-fast  - Run gold evaluation with fast profile"
	@echo "  make eval-quality - Run gold evaluation with quality profile"
	@echo "  make eval-baseline - Run lightweight baseline evaluation (2 PDFs max)"
	@echo "  make overlays   - Generate HTML overlays for QA"
	@echo ""
	@echo "Examples:"
	@echo "  make ocr FILE=\"data/input/sample.pdf\""
	@echo "  make build-arm64  - Build ARM64-optimized image (Apple Silicon)"

# Build the Docker image
build:
	docker build -t tokenworks-ocr:latest .

# Build the GPU-enabled Docker image
build-gpu:
	docker build -t tokenworks-ocr:latest .

# Test GPU and PaddleOCR functionality
gpu-smoke:
	docker compose run --rm ocr python -c "import paddle, os; print('Paddle:', paddle.__version__); print('CUDA compiled:', paddle.is_compiled_with_cuda()); print('GPU count:', paddle.device.cuda.device_count() if paddle.is_compiled_with_cuda() else 0); import paddleocr; from paddleocr import PaddleOCR; print('PaddleOCR import OK'); device = 'gpu' if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0 else 'cpu'; paddle.device.set_device(device); ocr = PaddleOCR(lang='latin', use_textline_orientation=True); print('PaddleOCR init OK')"

# Test docTR engine availability and version
gpu-smoke-doctr:
	docker compose run --rm ocr python -c "try: import doctr; print('✅ docTR version:', doctr.__version__); from doctr.models import ocr_predictor; model = ocr_predictor(pretrained=True); print('✅ docTR model loaded successfully'); except ImportError as e: print('❌ docTR not available:', e); except Exception as e: print('⚠️ docTR error:', e)"

# Test MMOCR engine availability and version
gpu-smoke-mmocr:
	docker compose run --rm ocr python -c "try: import mmocr; print('✅ MMOCR version:', mmocr.__version__); from mmocr.apis import MMOCRInferencer; inferencer = MMOCRInferencer(det='DBNet', rec='ABINet'); print('✅ MMOCR inferencer loaded successfully'); except ImportError as e: print('❌ MMOCR not available:', e); except Exception as e: print('⚠️ MMOCR error:', e)"

# Test Kraken engine availability and version
gpu-smoke-kraken:
	docker compose run --rm ocr python -c "try: import kraken; print('✅ Kraken version:', kraken.__version__); from kraken import blla; print('✅ Kraken baseline detection available'); except ImportError as e: print('❌ Kraken not available:', e); except Exception as e: print('⚠️ Kraken error:', e)"

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

# Run gold evaluation with fast profile
eval-fast:
	docker compose run --rm ocr python tools/run_gold_eval.py --profile fast

# Run gold evaluation with quality profile  
eval-quality:
	docker compose run --rm ocr python tools/run_gold_eval.py --profile quality

# Run lightweight baseline evaluation (2 PDFs max)
eval-baseline:
	docker compose run --rm ocr python tools/run_baseline_eval.py --gold-csv data/gold_data/gold_pages.csv --limit-pdfs 2 --profile quality --report-md --seed 17

# Generate HTML overlays for QA
overlays:
	docker compose run --rm ocr python -c "from src.html_overlays import generate_overlays; generate_overlays('data/input_pdfs', 'reports/overlays', max_files=5)"

# Build and run in one command
run: build ocr