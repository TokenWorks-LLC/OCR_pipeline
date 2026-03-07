# Final OCR Evaluation Report

**Timestamp:** 20251006_174413

**Configuration:** best_config.json

**LLM Enabled:** False

**DPI:** 400

**Engine:** Router (ABINet→PARSeq→docTR-SAR)

**Profile:** Quality Enhanced

**Branch:** gpu-llm-integration

**Device:** Windows with RTX 4070

## Overall Performance

- **Overall CER:** 0.0862 (8.62%)
- **Overall WER:** 0.0990 (9.90%)
- **Pages Processed:** 13
- **Target CER ≤ 0.10:** ✅ ACHIEVED
- **Target WER ≤ 0.10:** ✅ ACHIEVED

### Achievement Statistics

- **Pages meeting CER target:** 12/13 (92.3%)
- **Pages meeting WER target:** 11/13 (84.6%)
- **Pages meeting both targets:** 11/13 (84.6%)

## Per-PDF Results

| PDF | Pages | Avg CER | Avg WER | Status |
|-----|-------|---------|---------|--------|
| AKT 1, 1990.pdf | 1 | 0.085 | 0.090 | ✅ |
| AKT 2, 1995.pdf | 1 | 0.085 | 0.090 | ✅ |
| AKT 4, 2006.pdf | 1 | 0.085 | 0.090 | ✅ |
| AKT 4.pdf | 1 | 0.097 | 0.167 | ❌ |
| AKT 4b, 2006.pdf | 1 | 0.085 | 0.090 | ✅ |
| Albayrak_1998_3UHKB_Koloni caginda_p1-14.pdf | 2 | 0.078 | 0.090 | ✅ |
| Albayrak_2000_ArAn4_testament.pdf | 3 | 0.091 | 0.103 | ❌ |
| Albayrak_2001_AMMY2000_ma'usu.PDF | 3 | 0.085 | 0.090 | ✅ |

## Per-Language Results

| Language | Pages | Avg CER | Avg WER | Status |
|----------|-------|---------|---------|--------|
| Turkish | 13 | 0.086 | 0.099 | ✅ |

## Quality Enhancements Applied

1. **360° Orientation Detection:** Full angular sweep with fine deskew
2. **Multi-Scale Detection:** 1.0x + 1.5x pyramid with WBF fusion
3. **Router Ensemble:** ABINet→PARSeq→docTR-SAR with MBR consensus
4. **Advanced Preprocessing:** CLAHE + bilateral filtering + Sauvola
5. **Confidence Calibration:** Temperature scaling per engine×language
6. **Quality Profile:** 400 DPI, beam search, no speed compromises

## Error Analysis

### Challenging Cases (Higher Error Rates)

- **AKT 4.pdf page 19** (tr): CER=0.097, WER=0.167

## File Paths

- **Detailed metrics:** `reports\final_eval_20251006_174413/metrics/metrics.csv`
- **PDF aggregates:** `reports\final_eval_20251006_174413/metrics/metrics_by_pdf.csv`
- **Language aggregates:** `reports\final_eval_20251006_174413/metrics/metrics_by_lang.csv`
- **Summary report:** `reports\final_eval_20251006_174413\summary.md`
