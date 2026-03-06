# Running the OCR Pipeline (Local / Native)

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.9+ installed (`python3 --version`)
- ~2GB disk for PaddleOCR models (downloaded on first run)

## One-Time Setup

```bash
cd OCR_pipeline

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install PyMuPDF opencv-python-headless pdf2image numpy requests Pillow \
  reportlab psutil PyYAML tqdm paddlepaddle paddleocr
```

## Running the Pipeline

### 1. Activate the environment

```bash
cd OCR_pipeline
source .venv/bin/activate
```

### 2. Dry run (see what files would be processed)

```bash
PYTHONPATH=src:production:. PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
  python run_pipeline.py --dry-run
```

### 3. Process all PDFs in `data/input_pdfs/`

```bash
PYTHONPATH=src:production:. PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
  python run_pipeline.py
```

### 4. Process a single PDF (minimal config, no LLM)

```bash
PYTHONPATH=src:production:. PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
python -c "
from comprehensive_pipeline import PipelineConfig, ComprehensivePipeline

config = PipelineConfig(
    llm_provider='none',
    paddle_use_gpu=False,
    enable_llm_correction=False,
    enable_akkadian_extraction=False,
    generate_translations_pdf=False,
    create_html_overlay=False,
    create_visualization=False
)
pipeline = ComprehensivePipeline(config)
pipeline.process_pdf('data/input_pdfs/YOUR_FILE.pdf', 'data/output/my_run')
"
```

Replace `YOUR_FILE.pdf` with the actual filename.

## Output

Results go to `data/output/` (or whatever you specify):

- `comprehensive_results.csv` — OCR text per page with confidence scores, language, reading order stats
- `<filename>_comprehensive_report.json` — detailed per-page metrics (timing, confidence, word count, etc.)

## Configuration

Edit `config.json` to change:

| Setting | Location | Default |
|---------|----------|---------|
| Input directory | `input.input_directory` | `data/input_pdfs` |
| OCR DPI | `ocr.dpi` | `300` |
| LLM correction | `llm_correction.enabled` | `true` (needs Ollama running) |
| Akkadian detection | `akkadian.enabled` | `true` |

## Notes

- First run downloads ~2GB of PaddleOCR v5 models from HuggingFace — subsequent runs are instant.
- Processing speed: ~27 seconds per page on CPU (Apple Silicon).
- Average confidence: ~97% for standard academic text.
- Diacritics (s with cedilla, h with breve, etc.) are a known weakness of the base PaddleOCR English model.
- Docker builds work but require 8GB+ memory allocated to Docker Desktop (PaddleOCR v5 loads 5 models simultaneously).

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Make sure `PYTHONPATH=src:production:.` is set |
| Models downloading slowly | Set `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` to skip connectivity check |
| OOM in Docker | Use native Python (above) or increase Docker Desktop memory to 8GB+ |
| `'OCRSpan' object is not subscriptable` | You're on an older version — pull the latest `fix/restore-pipeline` branch |
