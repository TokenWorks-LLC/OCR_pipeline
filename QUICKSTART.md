# Quick Start - Page Text Extraction

## One Command to Rule Them All

```bash
python tools/run_page_text.py \
  --inputs "path/to/pdfs" \
  --output-root "reports/output" \
  --prefer-text-layer \
  --ocr-fallback ensemble \
  --status-bar

# Legacy-compatible alias (same pipeline):
python run_pipeline.py --input-dir "path/to/pdfs" --output-dir "reports/output"
```

`run_pipeline.py` reads `config.json` by default, which now selects the ensemble fallback for difficult pages.

## Output

**File:** `reports/output/client_page_text.csv`

| pdf_name       | page | page_text          | has_akkadian |
| -------------- | ---- | ------------------ | ------------ |
| AKT_4_2006.pdf | 19   | "1. a-na LUGAL..." | true         |
| Smith_2010.pdf | 5    | "The evidence..."  | false        |

## Common Patterns

### From Google Drive (Streaming)

```bash
python tools/run_page_text.py \
  --inputs "G:\Shared drives\Secondary Sources" \
  --output-root "reports/drive_$(date +%Y%m%d)" \
  --prefer-text-layer \
  --status-bar
```

### With OCR Fallback (Scanned PDFs)

```bash
python tools/run_page_text.py \
  --inputs "path/to/pdfs" \
  --output-root "reports/output" \
  --prefer-text-layer \
  --ocr-fallback ensemble \
  --status-bar
```

The ensemble fallback preprocesses each page into multiple denoised and thresholded variants, searches right-angle rotations, re-orders multi-column OCR lines, and fuses the best available OCR outputs with diacritic-aware consensus.

### Force OCR On Layered PDFs (Engine Evaluation)

```bash
python tools/run_page_text.py \
  --inputs "path/to/pdfs" \
  --output-root "reports/force_ocr_eval" \
  --prefer-text-layer \
  --ocr-fallback ensemble \
  --force-ocr \
  --status-bar
```

Use this mode when PDFs already contain selectable text but you still want OCR output to measure OCR-engine performance.

### From Manifest (Specific Pages)

```bash
# Create manifest
python tools/build_manifest.py \
  --csv data/gold/gold_pages.csv \
  --pdf-root data/pdfs \
  --out manifest.txt \
  --expand-ranges

# Process
python tools/run_page_text.py \
  --manifest manifest.txt \
  --output-root "reports/output" \
  --prefer-text-layer
```

## Key Flags

| Flag                      | Description                                    |
| ------------------------- | ---------------------------------------------- |
| `--inputs DIR`            | Scan directory for PDFs                        |
| `--manifest FILE`         | Use TSV manifest (pdf<TAB>page)                |
| `--output-root DIR`       | Output directory                               |
| `--prefer-text-layer`     | Use PDF text layer (fast, recommended)         |
| `--ocr-fallback ensemble` | Use fortified OCR ensemble if text layer fails |
| `--force-ocr`             | Run OCR even when text layer is present        |
| `--status-bar`            | Show progress bar                              |
| `--profile FILE`          | Custom Akkadian detection config               |

## Testing

```bash
# Main regression suite
python -m pytest tests -q

# Portable smoke checks (optional convenience)
python test_pipeline.py --allow-missing-engines

# Strict smoke checks (optional; fails if any OCR backend is missing)
python test_pipeline.py

# Targeted E2E regression tests
python -m pytest tests/test_pipeline_e2e.py -q

# Ensemble-specific regression tests
python -m pytest tests/test_ensemble_support.py -q

# Optional strict pytest engine enforcement:
# REQUIRED_OCR_ENGINES=paddleocr,doctr,mmocr,kraken python -m pytest tests/test_engine_imports.py -q
```

## Output Files

1. **client_page_text.csv** - Main output (4 columns, UTF-8 BOM)
2. **progress.csv** - Per-page tracking (timing, methods used)

## Help

```bash
python tools/run_page_text.py --help
```

## Troubleshooting

**No text extracted?**

- Try `--ocr-fallback ensemble` for scanned PDFs

**Akkadian not detected?**

- Verify diacritics present: š ṣ ṭ ḫ ā ē ī ū
- Check `profiles/akkadian_strict.json` config

**Out of memory?**

- Use `--ocr-fallback none`
- Process in batches via manifest

---

**Full docs:** README.md  
**Technical details:** docs/PAGE_TEXT_RUNBOOK.md
