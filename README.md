# OCR Pipeline - Production-Ready Document Processing System

A production-ready document processing system with AI-powered OCR, multi-language support, and specialized academic text extraction capabilities. Optimized for Apple Silicon (M1/M2/M3) with fast Docker builds and supports multiple OCR engines.

## Use Cases

This OCR pipeline is designed for:

- Document digitization projects with high-volume requirements
- Academic research involving historical texts and translations  
- Multi-language document processing workflows
- AI-enhanced OCR accuracy with LLM text correction
- Batch processing of PDFs, images, and mixed document types
- Multi-engine comparison and accuracy optimization
- Akkadian/Cuneiform transliteration detection and extraction

---

## Key Features

### Multi-Engine OCR Support
Choose the best OCR engine for your specific use case:

| Engine | Best For | License | Key Features |
|--------|----------|---------|--------------|
| **PaddleOCR** | General purpose (default) | Apache-2.0 | Multi-language, production-ready |
| **docTR** | Fast processing | Apache-2.0 | PyTorch-based, end-to-end |
| **MMOCR** | High accuracy | Apache-2.0 | OpenMMLab, state-of-the-art models |
| **Kraken** | Historical documents | Apache-2.0 | BiDi support, specialized for manuscripts |

### AI-Powered Text Correction
- LLM-based typo correction using Ollama
- Language-specific correction thresholds
- Akkadian span protection during correction
- Confidence-based edit budgets

### Multi-Language Support
Supports 12+ languages including:
- English, German, French, Italian, Spanish
- Turkish, Arabic, Hebrew, Persian
- Russian, Chinese, Japanese
- Automatic language detection per page/block

### Akkadian/Cuneiform Transliteration Detection
Specialized support for ancient Near Eastern texts:
- Syllabic structure detection (e.g., `a-na-ku`, `šar-ru-um`)
- Akkadian diacritics: `šṣṭḫāēīūâêîû`
- Determinatives: `ᵈ` (divine), `ᵐ` (male), `ᶠ` (female)
- Logographic markers: `DUMU`, `LUGAL`, `KÙ.BABBAR`, `URU`, `É`
- Page-level detection with confidence scores
- Translation pairing with source texts

### Advanced Features
- **ROVER Ensemble Fusion**: Character-level alignment with confidence calibration
- **Reading Order Detection**: Multi-column layout support with K-means clustering
- **Deterministic Processing**: Byte-for-byte identical output across runs
- **Intelligent Caching**: 10x speedup with multi-stage caching
- **GPU Acceleration**: Support for NVIDIA GPUs
- **Docker Support**: Portable, reproducible execution environments

---

## Quick Start

### 30-Second Demo
```bash
# 1. Get the project
git clone https://github.com/TokenWorks-LLC/OCR_pipeline
cd OCR_pipeline

# 2. For Apple Silicon Macs (M1/M2/M3) - Optimized build
docker build -f Dockerfile.arm64 -t ocr-pipeline .

# 3. For Intel/AMD systems
docker build -t ocr-pipeline .

# 4. Test it! (processes sample files)
docker run --rm -v "$PWD":/workspace -w /workspace ocr-pipeline \
  python run_pipeline.py --input-dir data/samples
```

### Choose Your Path

#### End Users (Just want to process documents)
```bash
# One-time setup
python setup.py                    # Installs everything + validates system

# Configure your preferences  
nano config.json                   # Edit: input folders, languages, AI features

# Add files and go!
cp your_files/* data/input/         # Put your PDFs/images here  
python run_pipeline.py             # Process documents
ls data/output/                     # Your results are here
```

#### Developers (Want to understand/modify the code)
```bash
# Get familiar with the codebase
python quick_start.py               # Interactive tutorials + examples

# Set up development environment  
pip install -r requirements.txt     # Install Python dependencies
python setup.py                     # Validate everything works

# Explore the architecture
ls src/                             # Core modules
cat config.json                     # Configuration options
```

#### Docker Users (Recommended for consistency)
```bash
# Apple Silicon (M1/M2/M3) - Optimized build (8-70 seconds)
docker build -f Dockerfile.arm64 -t ocr-pipeline .

# Intel/AMD64 systems
docker build -t ocr-pipeline .

# Run interactively (best for development)
docker run --rm -it -v "$PWD":/workspace -w /workspace ocr-pipeline bash

# Run directly (best for automation)  
docker run --rm -v "$PWD":/workspace -w /workspace ocr-pipeline \
  python run_pipeline.py --input-dir data/input

# Advanced: Use docker-compose for easier workflow
docker compose build && docker compose run --rm ocr python run_pipeline.py
```

---

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Architecture](#architecture)
- [Docker Deployment](#docker-deployment)
- [Performance](#performance)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Installation

### Prerequisites

**System Requirements:**
- Python 3.8+
- 8GB RAM minimum (16GB recommended)
- 10GB disk space minimum (50GB recommended)
- Optional: NVIDIA GPU with 6GB+ VRAM

**Python:** 3.8+

### Local Installation

```bash
# Clone repository
git clone https://github.com/TokenWorks-LLC/OCR_pipeline
cd OCR_pipeline

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Validate installation
python validate_requirements.py

# Run setup wizard
python setup.py
```

### Docker Installation (Recommended)

```bash
# Clone repository
git clone https://github.com/TokenWorks-LLC/OCR_pipeline
cd OCR_pipeline

# Build Docker image
docker build -t ocr-pipeline .

# For Apple Silicon (M1/M2/M3)
docker build -f Dockerfile.arm64 -t ocr-pipeline .

# Test installation
docker run --rm ocr-pipeline python -c "import paddleocr; print('Success!')"
```

---

## Configuration

The pipeline is configured via `config.json`:

```json
{
  "ocr": {
    "engine": "paddle",
    "dpi": 300,
    "enable_gpu": true,
    "lang": "en"
  },
  "llm_correction": {
    "enabled": true,
    "provider": "ollama",
    "model": "qwen2.5:7b-instruct",
    "timeout_s": 120,
    "edit_budget": 0.12
  },
  "akkadian": {
    "enabled": true,
    "min_confidence": 0.5,
    "extract_translations": true
  },
  "paths": {
    "input_dir": "data/input_pdfs",
    "output_dir": "data/output"
  }
}
```

### Key Configuration Options

| Setting | Purpose | Default |
|---------|---------|---------|
| `ocr.engine` | OCR engine choice | `paddle` |
| `ocr.dpi` | Rendering resolution | `300` |
| `llm_correction.enabled` | Enable LLM correction | `true` |
| `akkadian.enabled` | Enable Akkadian detection | `true` |
| `performance.max_workers` | Parallel workers | `4` |

---

## Usage Examples

### Basic PDF Processing

```bash
# Process a single PDF
python run_pipeline.py --input-file document.pdf

# Process entire directory
python run_pipeline.py --input-dir data/input

# Use specific OCR engine
python run_pipeline.py --engine doctr --input-dir data/input
```

### Advanced Options

```bash
# Use page-level text extraction (for academic documents)
python tools/run_page_text.py \
    --inputs "path/to/pdfs" \
    --output-root reports/page_text \
    --prefer-text-layer \
    --llm-on

# Compare multiple OCR engines
python tools/run_enhanced_eval.py --engines paddle,doctr,mmocr

# Process with Akkadian detection
python run_pipeline.py \
    --input-dir data/cuneiform_texts \
    --akkadian-mode \
    --generate-translations-pdf
```

### Docker Usage

```bash
# Process documents with Docker
docker run --rm \
    -v "$PWD/data":/workspace/data \
    ocr-pipeline \
    python run_pipeline.py --input-dir data/input

# Interactive session
docker run --rm -it \
    -v "$PWD":/workspace \
    -w /workspace \
    ocr-pipeline bash
```

---

## Architecture

### How the OCR Engine Works

This OCR pipeline processes documents through a sophisticated multi-stage workflow designed for accuracy, speed, and reproducibility. The system combines multiple OCR engines, AI-powered correction, and specialized text detection to extract text from PDFs and images.

### Core Processing Workflow

The pipeline operates in five distinct stages:

**Stage 1: PDF Rendering**
- Converts PDF pages to high-resolution images (300-400 DPI)
- Uses PyMuPDF (fitz) for fast, accurate rendering
- Results are cached to avoid redundant processing

**Stage 2: Image Preprocessing**
- Detects and corrects page orientation
- Applies deskewing to straighten tilted documents
- Enhances image quality for better OCR accuracy
- Results are cached for efficiency

**Stage 3: Text Detection**
- Identifies text regions and generates bounding boxes
- Uses detection models from chosen OCR engine (PaddleOCR, docTR, etc.)
- Handles multi-column layouts and complex page structures
- Detection results are cached

**Stage 4: Text Recognition**
- Performs OCR on detected text regions
- Supports GPU acceleration for faster processing
- Processes images in batches for efficiency
- Each engine produces text with confidence scores

**Stage 5: Post-Processing and Output**
- **LLM Correction**: Uses local AI models (via Ollama) to fix OCR errors while preserving specialized terminology
- **Akkadian Detection**: Identifies ancient Near Eastern transliterations using pattern matching
- **Reading Order**: Arranges text in natural reading sequence (top-to-bottom, left-to-right, multi-column aware)
- **Output Generation**: Produces CSV files with text, coordinates, confidence scores, and metadata

### Core Components

The pipeline is built from modular components that work together:

**Orchestrator** (`production/comprehensive_pipeline.py`)
- Coordinates the entire processing workflow
- Manages parallel processing of multiple documents
- Ensures deterministic output (same input always produces same output)
- Handles resource allocation and monitoring

**Multi-Engine OCR System**
- Supports PaddleOCR, docTR, MMOCR, and Kraken engines
- Can run multiple engines and combine results using ROVER (Recognizer Output Voting Error Reduction)
- Character-level alignment and confidence calibration
- Adaptive engine selection based on document type

**Text Processing Pipeline**
- **Reading Order Detection**: Analyzes spatial layout to determine correct text sequence
- **Language Detection**: Identifies language per page or text block
- **Akkadian Identification**: Specialized patterns for detecting transliterations

**LLM Correction System** (`src/enhanced_llm_correction.py`)
- Integrates with Ollama for local AI inference
- Applies language-specific correction thresholds
- Protects specialized terms (Akkadian spans) from modification
- Uses confidence-based edit budgets to prevent over-correction

**Intelligent Caching** (`src/cache_store.py`)
- Multi-stage caching (rendering, detection, OCR, LLM)
- Content-addressable storage using SHA-256 hashing
- Provides 10x speedup on repeated processing
- Invalidation controls for selective cache clearing

---

## Docker Deployment

### Standard Dockerfile

```bash
# Build
docker build -t ocr-pipeline .

# Run
docker run --rm -v "$PWD":/workspace -w /workspace ocr-pipeline \
    python run_pipeline.py
```

### Apple Silicon Dockerfile

```bash
# Build (8-70 seconds)
docker build -f Dockerfile.arm64 -t ocr-pipeline .

# Run
docker run --rm -v "$PWD":/workspace -w /workspace ocr-pipeline \
    python run_pipeline.py
```

### Docker Compose

```bash
# Build and run
docker compose build
docker compose run --rm ocr python run_pipeline.py

# Interactive shell
docker compose run --rm ocr bash
```

---

## Performance

### Throughput

**Single Machine:**
- CPU-only: ~5-10 pages/minute
- GPU (NVIDIA RTX 3060): ~30-50 pages/minute
- M1 Max (Apple Silicon): ~40-60 pages/minute

**With Caching:**
- 1st run: Full processing (~60 seconds/page)
- 2nd run: 90% cache hit rate (~6 seconds/page)

### Accuracy

**OCR Quality (English academic text):**
- PaddleOCR: 95-98% character accuracy
- Ensemble (3+ engines): 97-99% character accuracy
- With LLM correction: 98-99.5% character accuracy

**Akkadian Detection:**
- Precision: ~95% (few false positives)
- Recall: ~92% (some missed subtle cases)
- F1 Score: ~93.5%

---

## API Reference

### Command-Line Interface

```bash
python run_pipeline.py [OPTIONS]

Options:
  --input-file PATH         Process single file
  --input-dir PATH          Process directory
  --output-dir PATH         Output directory
  --engine ENGINE           OCR engine (paddle|doctr|mmocr|kraken)
  --akkadian-mode          Enable Akkadian detection
  --llm-correction         Enable LLM correction
  --dry-run                Show what would be processed
  --help                   Show help message
```

### Python API

```python
from src.orchestrator import Orchestrator
from src.multi_engine_orchestrator import MultiEngineOrchestrator

# Initialize orchestrator
orchestrator = Orchestrator(
    cache_dir="cache/pipeline",
    enable_cache=True
)

# Process PDF
results = orchestrator.process_pdf(
    pdf_path="document.pdf",
    output_dir="output"
)
```

---

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone and setup
git clone https://github.com/TokenWorks-LLC/OCR_pipeline
cd OCR_pipeline
pip install -r requirements.txt

# Run tests
pytest tests/

# Run linting
flake8 src/
black src/
```

---

## Troubleshooting

### Common Issues

**Problem:** "No text extracted" warnings  
**Solution:** Use `--prefer-text-layer` flag to extract embedded PDF text first

**Problem:** LLM correction not working  
**Solution:** Ensure Ollama server is running: `ollama serve`

**Problem:** Out of memory errors  
**Solution:** Reduce `max_workers` in `config.json`

**Problem:** Akkadian false positives  
**Solution:** Increase `min_confidence` to 0.6-0.7 in `config.json`

### Debug Mode

```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG
python run_pipeline.py --input-dir data/input

# Validate configuration
python run_pipeline.py --validate-only

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

---

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

### Third-Party OCR Engines

- **PaddleOCR**: Apache-2.0 License
- **docTR**: Apache-2.0 License
- **MMOCR**: Apache-2.0 License
- **Kraken**: Apache-2.0 License

See [THIRD_PARTY_OCR_LICENSES.md](THIRD_PARTY_OCR_LICENSES.md) for details.

---

## Additional Documentation

For more detailed information, see these guides:

- **[README_docker.md](README_docker.md)** - Comprehensive Docker deployment guide with advanced configurations
- **[QUICK_START.md](QUICK_START.md)** - Step-by-step tutorials and getting started guide
- **[PROD_RUN_GUIDE.md](PROD_RUN_GUIDE.md)** - Production deployment best practices
- **[PERFORMANCE_GUIDE.md](PERFORMANCE_GUIDE.md)** - Performance optimization and tuning
- **[tools/README.md](tools/README.md)** - Documentation for CLI tools and utilities
- **[CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md)** - Recent repository cleanup and improvements

---

## Acknowledgments

- PaddleOCR team for the excellent OCR engine
- Ollama team for local LLM inference
- OpenMMLab for MMOCR
- docTR team for PyTorch-based OCR
- Kraken team for historical document OCR

---

## Support

- **Issues**: [GitHub Issues](https://github.com/TokenWorks-LLC/OCR_pipeline/issues)
- **Discussions**: [GitHub Discussions](https://github.com/TokenWorks-LLC/OCR_pipeline/discussions)
- **Email**: support@tokenworks.com

---

## Roadmap

### Upcoming Features

- [ ] Table extraction and CSV export
- [ ] Handwriting recognition support
- [ ] Web interface for document upload
- [ ] REST API service
- [ ] Fine-tuned models for domain-specific OCR
- [ ] Active learning for uncertain pages

---

**Developed by TokenWorks LLC**
