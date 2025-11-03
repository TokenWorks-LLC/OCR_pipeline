# A/B Evaluation Tool - Usage Guide

## Overview

The A/B evaluation tool (`tools/ab_evaluation.py`) compares OCR pipeline performance with LLM correction **OFF vs ON**. It generates comprehensive reports with gating logic to decide whether LLM correction should be enabled in production.

## Features

- ✅ **A/B Testing**: Runs pipeline twice (LLM off, then LLM on) on same documents
- ✅ **Comprehensive Metrics**: CER, WER, mojibake detection, outlier flagging
- ✅ **Excel Reports**: Multi-sheet workbook with summary, detailed results, per-page comparison
- ✅ **Gating Logic**: Auto-recommends enabling/disabling LLM based on improvement thresholds
- ✅ **Windows Long-Path Support**: Uses `shortpath.py` to handle 260+ char paths
- ✅ **Page Range Support**: Handles "6-7" format, concatenates multi-page spans
- ✅ **Encoding Safety**: Outputs both UTF-8 and UTF-8-sig (BOM) CSVs for Excel compatibility

## Usage

### Basic A/B Test

Run full A/B evaluation on all gold pages:

```powershell
.venv\Scripts\Activate.ps1
python tools/ab_evaluation.py --ab --report-md --excel-out metrics.xlsx
```

### Command Line Options

```
--gold-csv PATH          Path to gold standard CSV (default: data/gold_data/gold_pages.csv)
--input-pdfs PATH        Directory with PDF files (default: data/input_pdfs)
--output-dir PATH        Output directory (default: ab_evaluation_results)
--ab                     Run A/B test (LLM off, then LLM on)
--llm-enabled true|false Enable LLM in single-pass mode (ignored if --ab is set)
--llm-model MODEL        LLM model to use (default: qwen2.5:7b-instruct)
--report-md              Generate markdown summary report
--excel-out FILE         Output Excel file path (e.g., metrics.xlsx)
```

### Output Files

After running `--ab --report-md --excel-out metrics.xlsx`:

```
ab_evaluation_results/
├── summary_llm_ab.md              # Markdown summary with gating decision
├── metrics_llm_off.csv            # Results with LLM disabled (UTF-8)
├── metrics_llm_off_bom.csv        # Results with LLM disabled (UTF-8-sig for Excel)
├── metrics_llm_on.csv             # Results with LLM enabled (UTF-8)
├── metrics_llm_on_bom.csv         # Results with LLM enabled (UTF-8-sig for Excel)
├── metrics.xlsx                   # Excel workbook (4 sheets)
├── llm_off/                       # Detailed per-page results (LLM OFF)
│   └── *_page_*/
│       └── comprehensive_results.csv
└── llm_on/                        # Detailed per-page results (LLM ON)
    └── *_page_*/
        └── comprehensive_results.csv
```

### Excel Workbook Structure

**Sheet 1: Summary**
- Aggregate metrics comparison (Median CER, Mean CER, Median WER, Mean WER)
- Delta calculations
- LLM correction statistics

**Sheet 2: LLM_OFF**
- Detailed per-page results with LLM disabled
- Columns: pdf_name, page_span, cer, wer, ref_len, ocr_len, is_outlier, has_mojibake, etc.

**Sheet 3: LLM_ON**
- Detailed per-page results with LLM enabled
- Includes llm_corrections count per page

**Sheet 4: Comparison**
- Side-by-side per-page comparison
- Columns: cer_off, cer_on, cer_delta, wer_off, wer_on, wer_delta, llm_corrections

## Gating Logic

The tool automatically recommends enabling/disabling LLM based on:

**Enable LLM if:**
- CER improvement ≥ 3% **OR**
- WER improvement ≥ 5%

**Example outputs:**

✅ **ENABLE LLM** if:
- Median CER: 15% → 11% (4% improvement, exceeds 3% threshold)

❌ **SUPPRESS LLM** if:
- Median CER: 8% → 7.5% (0.5% improvement, below 3% threshold)
- Median WER: 12% → 11% (1% improvement, below 5% threshold)

## Interpretation Guide

### Median CER/WER
- **<5%**: Excellent OCR quality
- **5-10%**: Good OCR quality, LLM may provide marginal improvements
- **10-20%**: Moderate OCR quality, LLM should help significantly
- **>20%**: Poor OCR quality, investigate input quality or OCR settings

### Outliers
- Pages with CER > 1000% (10.0) are flagged as outliers
- Common causes:
  - Wrong page extracted
  - Encoding issues
  - Layout/column detection failures
- Outliers are excluded from aggregate statistics

### Mojibake Detection
- Flags pages with encoding corruption (Ã¼, Ã¶, Ã§, etc.)
- Should be 0 after fixes applied
- If mojibake persists, check:
  - Input PDF encoding
  - OCR engine language settings
  - CSV output encoding

### LLM Corrections
- **0 corrections**: LLM triggers not firing (check confidence thresholds)
- **Low corrections (<10%)**: LLM being conservative (expected for high-quality OCR)
- **High corrections (>50%)**: LLM very active (check for over-editing)

## Example Session

```powershell
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Run A/B evaluation on all 39 gold pages
python tools/ab_evaluation.py `
    --ab `
    --report-md `
    --excel-out metrics.xlsx `
    --output-dir final_ab_results

# Check summary
cat final_ab_results\summary_llm_ab.md

# Open Excel report
start final_ab_results\metrics.xlsx
```

## Expected Runtime

- **Single page**: ~25 seconds (300 DPI, reading order, LLM enabled)
- **39 pages (A/B)**: ~32 minutes (16 min LLM off + 16 min LLM on)

## Troubleshooting

### Long Path Errors (Windows)

If you see path errors like "path too long":
- The tool automatically uses `shortpath.py` utility
- Paths are slugged to ≤80 chars per segment with SHA1 hashing
- Windows `\\?\` prefix applied for paths >200 chars

### LLM Not Triggering

If `llm_corrections` is 0 for all pages:
- Check OCR confidence levels (may be too high)
- Verify mojibake detection is working
- Check `utils/triggers.py` thresholds (en:0.86, de:0.85, tr:0.83)
- Run with increased logging: `--verbose`

### Mojibake Still Present

If `has_mojibake` is True in LLM ON results:
- LLM may have rejected the correction (guardrail violation)
- Check edit budget settings (0.12 modern, 0.03 Akkadian)
- Try `utils/encoding.py` fallback fixes

### Excel File Won't Open

- Use `*_bom.csv` files (UTF-8-sig encoding) for Excel compatibility
- Or open CSV in Excel with explicit encoding selection

## Next Steps After A/B Test

1. **Review summary_llm_ab.md** for gating recommendation
2. **Examine outliers** in Excel Comparison sheet
3. **If LLM recommended**:
   - Update `config.json`: `"enable_new_llm_correction": true`
   - Deploy to production
4. **If LLM not recommended**:
   - Investigate trigger sensitivity
   - Check if OCR quality is already excellent
   - Consider calibrating confidence thresholds with `tools/calibrate_confidence.py`

## Advanced: Single-Pass Mode

Run evaluation with LLM enabled (no A/B comparison):

```powershell
python tools/ab_evaluation.py `
    --llm-enabled true `
    --output-dir single_pass_results
```

Run evaluation with LLM disabled:

```powershell
python tools/ab_evaluation.py `
    --llm-enabled false `
    --output-dir baseline_results
```

## Integration with CI/CD

```yaml
# Example GitHub Actions workflow
- name: Run A/B Evaluation
  run: |
    python tools/ab_evaluation.py --ab --report-md --excel-out metrics.xlsx
    
- name: Check Gating
  run: |
    python tools/check_gating.py ab_evaluation_results/summary_llm_ab.md
```

## Dependencies

- **Required**:
  - `production.comprehensive_pipeline`
  - `cer_evaluation`
  - Python 3.8+

- **Optional** (for full functionality):
  - `pandas` (for Excel generation)
  - `openpyxl` (Excel engine)
  - `utils.shortpath` (Windows long-path support)
  - `utils.encoding` (mojibake detection)
  - `utils.triggers` (LLM trigger heuristics)

Install optional dependencies:
```powershell
pip install pandas openpyxl
```
