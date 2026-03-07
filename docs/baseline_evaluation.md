# Baseline OCR Evaluation

This document describes how to run the lightweight baseline evaluation for the OCR pipeline.

## Quick Start

### Windows (PowerShell/CMD)
```cmd
.\run_baseline_eval.bat
```

### Linux/macOS (with make)
```bash
# amd64
make eval-baseline COMPOSE_SERVICE=ocr

# arm64 (Apple Silicon)
make eval-baseline COMPOSE_SERVICE=ocr-arm64
```

### Manual Docker Command (All platforms)
```bash
docker compose run --rm ${COMPOSE_SERVICE:-ocr} python tools/run_baseline_eval.py --gold-csv data/gold_data/gold_pages.csv --limit-pdfs 2 --profile fast --report-md --seed 17
```

## What It Does

The baseline evaluation:

1. **Loads gold standard data** from `data/gold_data/gold_pages.csv`
2. **Selects 2 PDFs** with the most gold standard pages (deterministic with seed=17)
3. **Processes specific pages** from the selected PDFs using basic PaddleOCR
4. **Calculates CER/WER metrics** against gold standard text
5. **Generates reports** in `reports/baseline_fast_YYYYMMDD_HHMMSS/`

## Output

### Report Files
- `reports/baseline_fast_YYYYMMDD_HHMMSS/summary.md` - Human-readable markdown report
- `reports/baseline_fast_YYYYMMDD_HHMMSS/metrics/metrics.csv` - Raw metrics data

### Key Metrics
- **CER (Character Error Rate)**: Character-level accuracy
- **WER (Word Error Rate)**: Word-level accuracy  
- **Pages Processed**: Number of pages evaluated
- **PDFs Processed**: Number of PDF files processed

## Configuration Options

The script accepts several parameters:

```bash
python tools/run_baseline_eval.py --help
```

### Common Parameters
- `--limit-pdfs 2`: Maximum number of PDFs to process
- `--profile fast`: Pipeline profile (fast/balanced/quality)
- `--seed 17`: Random seed for deterministic PDF selection
- `--report-md`: Generate markdown report
- `--gold-csv path/to/gold.csv`: Path to gold standard CSV

## Expected Baseline Performance

The baseline evaluation uses basic PaddleOCR without any optimization:

- **CER**: ~95-98% (high error rate expected)
- **WER**: ~99% (very high error rate expected)
- **Processing**: 2 PDFs, typically 4-6 pages total
- **Runtime**: ~2-3 minutes (includes model downloads on first run)

## Troubleshooting

### Common Issues

1. **"No PDFs found"**: Ensure PDFs exist in `data/input_pdfs/`
2. **"Gold CSV not found"**: Check `data/gold_data/gold_pages.csv` exists
3. **Docker errors**: Ensure Docker Desktop is running
4. **GPU issues**: GPU detection is automatic, will fallback to CPU

### Debug Mode

Add verbose logging:
```bash
docker compose run --rm ocr python -u tools/run_baseline_eval.py --gold-csv data/gold_data/gold_pages.csv --limit-pdfs 1 --profile fast --report-md

# arm64 (Apple Silicon)
docker compose run --rm ocr-arm64 python -u tools/run_baseline_eval.py --gold-csv data/gold_data/gold_pages.csv --limit-pdfs 1 --profile fast --report-md
```

## Technical Details

### Pipeline Components
- **PDF Processing**: `pdf_utils.extract_images_from_pdf()`
- **OCR Engine**: PaddleOCR with Turkish language model
- **Metrics**: Character/Word edit distance calculations
- **Language Detection**: Simple heuristic based on character patterns

### File Selection Logic
1. Count gold standard rows per PDF
2. Filter to PDFs that exist in `data/input_pdfs/`
3. Sort by gold row count (descending), then filename (ascending)
4. Select top N PDFs with seed-based deterministic ordering

### Performance Profile
- **Fast**: 200 DPI, basic OCR settings
- **Balanced**: 250 DPI, enhanced settings (not implemented in baseline)
- **Quality**: 300 DPI, full processing (not implemented in baseline)

This baseline provides a quick sanity check and establishes a performance floor for comparison with enhanced pipeline features.