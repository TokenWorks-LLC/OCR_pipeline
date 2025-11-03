# OCR Pipeline Tools

This directory contains utility tools for testing, calibration, and evaluation of the OCR pipeline.

## Tools

### 📊 ab_evaluation.py
**A/B Gold Standard Evaluation**

Compares pipeline performance with LLM correction OFF vs ON. Generates Excel reports with gating recommendations.

**Usage:**
```powershell
python tools/ab_evaluation.py --ab --report-md --excel-out metrics.xlsx
```

**Features:**
- Full A/B testing (LLM off/on comparison)
- Excel report with 4 sheets (Summary, LLM_OFF, LLM_ON, Comparison)
- Markdown summary with gating decision
- Median/mean CER/WER metrics
- Outlier detection and mojibake flagging

**See:** `AB_EVALUATION_GUIDE.md` for detailed documentation

---

## Planned Tools (Task 8)

### 🧪 smoke_paths.py (Not yet implemented)
**Windows Long-Path Smoke Test**

Tests path handling on Windows for 260+ character paths.

**Planned usage:**
```powershell
python tools/smoke_paths.py
```

**Will test:**
- Path shortening with SHA1 hashing
- Windows `\\?\` prefix handling
- Directory creation with long paths
- File operations with long paths

---

### ✅ check_run_invariants.py (Not yet implemented)
**Evaluation Results Validator**

Validates evaluation results for common issues.

**Planned usage:**
```powershell
python tools/check_run_invariants.py --run-dir evaluation_results
```

**Will check:**
- No mojibake in output CSVs
- CER/WER in realistic ranges (no thousand-percent explosions)
- Akkadian corruption rate = 0%
- All expected files present
- Per-page ocr_len varies realistically

---

### 📈 calibrate_confidence.py (Not yet implemented)
**LLM Confidence Threshold Calibrator**

Fits optimal confidence thresholds from gold data.

**Planned usage:**
```powershell
python tools/calibrate_confidence.py --gold data/gold_data/gold_pages.csv
```

**Will generate:**
- Calibrated thresholds per language
- Updated config.json with optimal values
- ROC curves for threshold selection

---

## Installation

Most tools require the OCR pipeline dependencies:

```powershell
# Basic dependencies
pip install -r requirements.txt

# For Excel generation (ab_evaluation.py)
pip install pandas openpyxl
```

## Quick Start

1. **Run A/B evaluation** to test LLM correction impact:
   ```powershell
   .venv\Scripts\Activate.ps1
   python tools/ab_evaluation.py --ab --report-md --excel-out metrics.xlsx
   ```

2. **Check gating recommendation** in summary:
   ```powershell
   cat ab_evaluation_results\summary_llm_ab.md
   ```

3. **Review detailed metrics** in Excel:
   ```powershell
   start ab_evaluation_results\metrics.xlsx
   ```

## Development

To add a new tool:

1. Create `tools/your_tool.py`
2. Add CLI with `argparse`
3. Follow logging conventions: `logging.basicConfig(level=logging.INFO)`
4. Add usage example to this README
5. Update `FIXES_PROGRESS.md` status

## Testing

Run tool smoke tests:
```powershell
pytest tests/test_tools.py
```

## Support

For issues or questions:
- Check individual tool documentation
- Review `AB_EVALUATION_GUIDE.md` for A/B evaluation
- See `FIXES_PROGRESS.md` for implementation status
