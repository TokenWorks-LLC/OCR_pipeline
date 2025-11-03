# OCR Pipeline - Page Text Extraction

**Version 3.0.0** | Simplified Page-Level Text Extraction with Akkadian Detection

> **🎯 NEW:** Single entrypoint for page-level text extraction. Simple 4-column CSV output.

## 🚀 Quick Start

### Installation

```bash
# Install minimal dependencies
pip install -r requirements_minimal.txt

# Optional: For OCR fallback
pip install paddlepaddle paddleocr opencv-python-headless

# Optional: For LLM typo correction
pip install ollama requests
```

### Basic Usage

Extract page-level text from PDFs and detect Akkadian content:

```bash
# Simple extraction from a directory
python run_pipeline.py page_text \
  --inputs "path/to/pdfs" \
  --output-root "reports/output_20251009" \
  --prefer-text-layer \
  --status-bar

# With OCR fallback for scanned pages
python run_pipeline.py page_text \
  --inputs "path/to/pdfs" \
  --output-root "reports/output_20251009" \
  --prefer-text-layer \
  --ocr-fallback paddle \
  --status-bar

# With LLM typo correction (preserves Akkadian)
python run_pipeline.py page_text \
  --inputs "path/to/pdfs" \
  --output-root "reports/output_20251009" \
  --prefer-text-layer \
  --llm-on \
  --status-bar
```

### Using a Manifest

Process specific pages from specific PDFs:

```bash
# Create manifest (TSV format: pdf_path<TAB>page_number)
python tools/build_manifest.py \
  --csv data/gold/gold_pages.csv \
  --pdf-root data/pdfs \
  --out data/manifest.txt \
  --expand-ranges

# Process manifest
python run_pipeline.py page_text \
  --manifest data/manifest.txt \
  --output-root "reports/output_20251009" \
  --prefer-text-layer \
  --llm-on
```

## 📊 Output

The pipeline produces two CSV files:

### 1. `client_page_text.csv` (UTF-8 BOM, Excel-friendly)

The main output with 4 columns:

| Column | Type | Description |
|--------|------|-------------|
| `pdf_name` | string | PDF filename (without path) |
| `page` | integer | Page number (1-based) |
| `page_text` | string | Extracted text (newlines as `\n`) |
| `has_akkadian` | boolean | `true` if Akkadian detected, `false` otherwise |

**Example:**
```csv
pdf_name,page,page_text,has_akkadian
AKT_4_2006.pdf,19,"1. a-na LUGAL be-lí-ia\n2. qí-bí-ma um-ma...",true
Smith_2010.pdf,5,"The evidence suggests that trade networks...",false
```

### 2. `progress.csv`

Tracks processing progress per page:

| Column | Description |
|--------|-------------|
| `pdf_name` | PDF filename |
| `page` | Page number (1-based) |
| `ms` | Processing time in milliseconds |
| `used_text_layer` | `True` if text layer used, `False` if OCR |
| `has_akkadian` | Akkadian detection result |
| `llm_applied` | `True` if LLM correction was applied |
| `timestamp` | ISO 8601 timestamp |

## 🔍 Akkadian Detection

The pipeline uses sophisticated Akkadian detection with **any-line aggregation**:

- Detects syllabic transliteration patterns (e.g., `a-na-ku`, `šar-ru-um`)
- Recognizes Akkadian diacritics (š, ṣ, ṭ, ḫ, ā, ē, ī, ū)
- Identifies determinatives and markers (DUMU, LUGAL, KÙ.BABBAR, ᵈ, ᵐ, ᶠ)
- Uses character language model (if available) for perplexity scoring
- Filters out false positives from German/Turkish/English scholarly text

**Configuration:** Edit `profiles/akkadian_strict.json` to tune detection thresholds.

## 🤖 LLM Typo Correction

When enabled with `--llm-on`, the pipeline:

1. **Protects Akkadian spans** - Wraps them in `<AKK>...</AKK>` tags
2. **Sends to LLM** - Requests typo fixes for non-Akkadian text only
3. **Validates protection** - Ensures Akkadian text is unchanged (bit-exact)
4. **Applies edit budget** - Rejects corrections exceeding 15% edit ratio
5. **Fallback on violation** - Returns original text if protection is violated

**Requirements:**
- Ollama running locally: `ollama serve`
- Model pulled: `ollama pull qwen2.5:7b-instruct`

**Configuration:**
```bash
--llm-provider ollama
--llm-model qwen2.5:7b-instruct
--llm-base-url http://localhost:11434
--llm-temperature 0.2
--llm-top-p 0.2
```

## 📖 Command Reference

### `run_pipeline.py page_text`

**Input Source** (choose one):
- `--manifest PATH` - TSV manifest (pdf_path<TAB>page_no)
- `--inputs DIR` - Directory to scan recursively for PDFs

**Output:**
- `--output-root DIR` - Output directory (required)
- `--progress-csv PATH` - Custom progress CSV path (optional)

**Text Extraction:**
- `--prefer-text-layer` - Try PDF text layer first (recommended)
- `--ocr-fallback {paddle,none}` - OCR engine if text layer fails (default: none)

**Akkadian Detection:**
- `--profile PATH` - Detection profile JSON (default: profiles/akkadian_strict.json)

**LLM Correction:**
- `--llm-on` - Enable LLM typo correction
- `--llm-off` - Disable LLM (default)
- `--llm-provider STR` - LLM provider (default: ollama)
- `--llm-model STR` - Model name (default: qwen2.5:7b-instruct)
- `--llm-base-url URL` - API base URL (default: http://localhost:11434)
- `--llm-temperature FLOAT` - Temperature (default: 0.2)
- `--llm-top-p FLOAT` - Top-p (default: 0.2)
- `--llm-timeout INT` - Timeout in seconds (default: 120)

**UI:**
- `--status-bar` - Show progress bar

## 🧪 Testing

Run unit tests to verify Akkadian protection:

```bash
python tests/test_akkadian_protection.py
```

All 20 tests should pass, verifying:
- Akkadian span detection (syllabic, markers, determinatives)
- Protection wrapping (`<AKK>` tags)
- Validation (detects altered/missing spans)
- Page-level detection with any-line aggregation

## 📁 Project Structure

```
OCR_pipeline/
├── run_pipeline.py              # Main entrypoint
├── tools/
│   ├── run_page_text.py         # Page text extraction engine
│   └── build_manifest.py        # Manifest builder
├── tests/
│   └── test_akkadian_protection.py  # Protection unit tests
├── profiles/
│   └── akkadian_strict.json     # Detection configuration
├── src/                         # Core source code (legacy)
├── archive/                     # Archived legacy code
├── requirements_minimal.txt     # Minimal dependencies
└── README.md                    # This file
```

## 🐳 Docker (Optional)

For containerized execution:

```bash
# Build for your platform
docker build -t ocr-pipeline .                    # Intel/AMD64
docker build -f Dockerfile.arm64 -t ocr-pipeline .  # Apple Silicon

# Run
docker run -v $(pwd)/data:/data -v $(pwd)/reports:/reports \
  ocr-pipeline page_text \
  --inputs /data/pdfs \
  --output-root /reports/output \
  --prefer-text-layer
```

See `README_docker.md` for detailed Docker instructions.

## 🔧 Advanced Configuration

### Akkadian Detection Tuning

Edit `profiles/akkadian_strict.json`:

```json
{
  "akkadian_detection": {
    "threshold": 0.25,
    "require_diacritic_or_marker": true,
    "min_diacritics_per_line": 1,
    "min_syllabic_tokens": 3,
    "min_syllabic_ratio": 0.25,
    "aggregation_mode": "any-line",
    "aggregation_qual_lines_min": 3,
    "aggregation_qual_ratio_min": 0.25
  }
}
```

### Memory-Safe Large-Scale Processing

For processing thousands of PDFs:

```bash
python run_pipeline.py page_text \
  --inputs "G:\Shared drives\Secondary Sources" \
  --output-root "reports/full_run_$(date +%Y%m%d)" \
  --prefer-text-layer \
  --ocr-fallback none \
  --llm-off \
  --status-bar \
  --progress-csv "reports/full_run_$(date +%Y%m%d)/progress.csv"
```

The pipeline:
- Streams pages one-by-one (no RAM build-up)
- Appends to CSV after each page
- Flushes output immediately
- Can be safely interrupted and resumed (check progress.csv)

## 📚 Documentation

- **Full Runbook:** `OCR _PIPELINE_RUNBOOK.md` - Complete technical documentation
- **Docker Guide:** `README_docker.md` - Docker setup and deployment
- **Legacy Code:** `archive/` - Previous implementations and experiments

## 🤝 Contributing

This is a streamlined production pipeline. For development:

1. Run tests: `python tests/test_akkadian_protection.py`
2. Validate code: `python -m compileall .`
3. Follow existing patterns for new features

## 📄 License

See LICENSE file for details.

## 🆘 Troubleshooting

### No text extracted
- Check if PDF has text layer: `pdfinfo yourfile.pdf | grep "Tagged"`
- Try `--ocr-fallback paddle` for scanned PDFs
- Ensure PDF is not encrypted or corrupted

### Akkadian not detected
- Verify diacritics are present: š ṣ ṭ ḫ ā ē ī ū
- Check for syllabic hyphens: `a-na-ku`, `šar-ru-um`
- Review detection config in `profiles/akkadian_strict.json`
- Check qualified lines in detailed logs

### LLM corrections rejected
- LLM may be altering Akkadian spans (protection violated)
- Edit budget exceeded (>15% changes)
- Check logs for validation errors
- Try adjusting `--llm-temperature` (lower = more conservative)

### Out of memory
- Disable OCR fallback if not needed: `--ocr-fallback none`
- Disable LLM: `--llm-off`
- Process in smaller batches using manifest
- Reduce concurrent processing (code limitation)

---

**Version 3.0.0** - Simplified page-level text extraction  
**Updated:** October 9, 2025  
**Maintainer:** OCR Pipeline Team
