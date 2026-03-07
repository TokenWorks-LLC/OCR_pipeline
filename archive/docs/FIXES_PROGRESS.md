# OCR Pipeline Fixes - Progress Tracker

## Completed Tasks ✅

### Task 1: Windows Long-Path Handling
- ✅ Created `src/utils/shortpath.py`
  - `safe_path()`: Constructs paths ≤240 chars with \\?\\ prefix
  - `_slug_segment()`: Slugifies segments to ≤80 chars with SHA1 hash
  - `ensure_dir()`: Creates directories with long-path support
  - `open_safe()`: Opens files handling \\?\\ prefix
- Status: **Complete** - Ready for integration into pipeline

### Task 2: UTF-8 Mojibake Detection & Repair
- ✅ Created `src/utils/encoding.py`
  - `has_mojibake()`: Detects Ã¼, Ã¶, Ã§, ÃŸ, etc.
  - `repair_mojibake()`: Attempts cp1252→utf-8 fix with 80% threshold
  - `apply_fallback_fixes()`: Deterministic replacements for common mojibake
  - `has_expected_diacritics()`: Checks language-specific diacritics
- Status: **Complete** - Ready for integration

### Task 3: Evaluation Math Corrections
- ✅ Fixed `run_final_evaluation.py`
  - Removed text concatenation bug (was joining all CSV rows)
  - Added per-page text isolation (first row only for single pages)
  - Added multi-page span support (concatenate with newlines)
  - Added outlier detection (CER > 1000% flagged)
  - Added diagnostic columns: `ref_len`, `ocr_len`, `len_ratio`, `is_outlier`
  - Updated summary report with outlier triage section
- Status: **Complete** - Ready for re-evaluation

### Task 4: Page Range Support
- ✅ Updated `load_gold_data()` in `run_final_evaluation.py`
  - Handles "A-B" format page ranges
  - Stores `page_span`, `partial_span`, `end_page` metadata
  - Concatenates multi-page text with newlines
  - Logs multi-page spans during processing
- Status: **Complete** - Ready for re-evaluation

### Task 5: LLM Trigger Improvements
- ✅ Created `src/utils/triggers.py`
  - `should_trigger_llm()`: OR logic for 4 triggers
    1. Low confidence (calibrated thresholds)
    2. Mojibake detection
    3. Diacritic mismatch (missing expected chars for language)
    4. Character-level anomaly (z-score > 2.5)
  - `get_calibrated_threshold()`: Per-language thresholds (en:0.86, de:0.85, tr:0.83)
  - `has_diacritic_mismatch()`: Language-specific diacritic checking
  - `calculate_char_lm_anomaly()`: N-gram frequency analysis
- ✅ Updated `src/enhanced_llm_correction.py`
  - Integrated new triggers into `_should_route_to_llm()`
  - Added fallback to legacy logic if triggers unavailable
  - Improved OCR error pattern detection
- ✅ Updated `src/utils/__init__.py`
  - Exported all new utilities
- Status: **Complete** - Ready for testing

## Pending Tasks ⏳

### Task 6: Tightened LLM Guardrails  
- ✅ Verified existing guardrails in `src/llm/corrector.py`:
  - ✅ Bracket count preservation: `_validate_bracket_preservation()` - [], (), {} counts match
  - ✅ Line break preservation: `_validate_line_breaks()` - newline counts match
  - ✅ Edit budget enforcement: `_validate_edit_budget()` - 12% modern, 3% Akkadian
  - ✅ Vocabulary explosion detection: `_validate_vocabulary_explosion()` - max 15% alpha char increase
  - ✅ All applied via `_apply_guardrails()` before accepting corrections
- ✅ Updated `should_correct_line()` to use enhanced triggers
- Status: **Complete** - Guardrails already comprehensive, enhanced triggers integrated

### Task 7: A/B Evaluation with Excel Output
- ✅ Created `tools/ab_evaluation.py`:
  - `--ab` flag for A/B testing (LLM off, then LLM on)
  - Generates metrics_llm_off.csv, metrics_llm_on.csv (utf-8 + utf-8-sig)
  - Creates summary_llm_ab.md with medians, deltas, outliers
  - Generates metrics.xlsx with 4 sheets (Summary, LLM_OFF, LLM_ON, Comparison)
  - Gating logic: Enable LLM if CER improves ≥3% OR WER ≥5%
  - Integrated safe_path, mojibake detection, outlier flagging
  - Supports page ranges and multi-page concatenation
- Status: **Complete** - Ready for testing

### Task 8: Telemetry & Invariants
- [ ] Create `tools/smoke_paths.py`:
  - Test 260+ char paths on Windows
  - Verify safe_path() utility
- [ ] Create `tools/check_run_invariants.py`:
  - Validate evaluation results
  - Check for mojibake in output CSVs
  - Verify CER/WER ranges (no thousand-percent explosions)
  - Check Akkadian corruption rate
- [ ] Create `tools/calibrate_confidence.py`:
  - Fit optimal thresholds from gold data
  - Generate calibrated config
- Status: **Not started**

## Integration Points

### Files Needing Updates
- [x] `src/utils/__init__.py` - Export new utilities
- [x] `src/enhanced_llm_correction.py` - Integrate new triggers
- [ ] `production/comprehensive_pipeline.py` - Integrate safe_path, encoding repairs
- [ ] `src/aggregated_csv.py` - Add utf-8-sig output, Excel generation
- [ ] `config.json` - Add encoding, char_lm, threshold sections

### Testing Sequence
1. `python tools/smoke_paths.py` - Verify path handling
2. `python tools/calibrate_confidence.py --gold data/gold_data/gold_pages.csv` - Calibrate thresholds
3. `python tools/run_gold_eval.py --ab --llm-enabled true --report-md --excel-out metrics.xlsx` - A/B evaluation
4. `python tools/check_run_invariants.py --run-dir evaluation_results` - Validate results

## Acceptance Criteria

- [ ] All 39 pages process without path-length errors
- [ ] No mojibake in output CSVs (proper diacritics: ü, ö, ç, ğ, ı, ş)
- [ ] Realistic CER/WER values (no thousand-percent explosions)
- [ ] Per-page `ocr_len` varies realistically
- [ ] Page ranges ("6-7") concatenated correctly
- [ ] LLM triggers on mojibake/diacritic issues
- [ ] Akkadian corruption rate remains 0%
- [ ] A/B report shows LLM benefit (or auto-suppression if harmful)
- [ ] Excel output with proper formatting
- [ ] All smoke tests pass

## Next Steps

### Immediate: Run A/B Evaluation

```powershell
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Run full A/B evaluation on all 39 gold pages
python tools/ab_evaluation.py --ab --report-md --excel-out metrics.xlsx

# Check summary report
cat ab_evaluation_results\summary_llm_ab.md

# Open Excel for detailed analysis
start ab_evaluation_results\metrics.xlsx
```

### Expected Results
- ✅ All 39/39 pages should process (safe_path fixes Windows limits)
- ✅ Realistic CER/WER values (<100%, evaluation math fixed)
- ✅ LLM corrections >0 (new triggers firing on mojibake/diacritics)
- ✅ Outliers properly flagged (CER >1000%)
- ✅ Mojibake detection working in both passes

### After Evaluation

1. **Tighten guardrails** (Task 6) → ALREADY DONE!
2. **Create A/B evaluation tool** (Task 7) → COMPLETE!
3. **Create smoke tests and invariants checker** (Task 8) → PENDING
4. **Integrate into pipeline** (safe_path, encoding repairs) → READY
5. **Re-run evaluation** on all 39 pages → READY TO RUN
6. **Validate results** with invariants checker → Pending Task 8
