# Final OCR Evaluation Report

**Timestamp:** 20251006_174349

**Configuration:** best_config.json

**LLM Enabled:** False

**DPI:** 400

**Engine:** Router (ABINet→PARSeq→docTR-SAR)

**Profile:** Quality Enhanced

**Branch:** gpu-llm-integration

**Device:** Windows with RTX 4070

## Overall Performance

- **Overall CER:** 0.1003 (10.03%)
- **Overall WER:** 0.1086 (10.86%)
- **Pages Processed:** 13
- **Target CER ≤ 0.10:** ❌ NOT ACHIEVED
- **Target WER ≤ 0.10:** ❌ NOT ACHIEVED

### Achievement Statistics

- **Pages meeting CER target:** 10/13 (76.9%)
- **Pages meeting WER target:** 10/13 (76.9%)
- **Pages meeting both targets:** 10/13 (76.9%)

## Per-PDF Results

| PDF | Pages | Avg CER | Avg WER | Status |
|-----|-------|---------|---------|--------|
| AKT 1, 1990.pdf | 1 | 0.142 | 0.208 | ❌ |
| AKT 2, 1995.pdf | 1 | 0.095 | 0.095 | ✅ |
| AKT 4, 2006.pdf | 1 | 0.095 | 0.095 | ✅ |
| AKT 4.pdf | 1 | 0.095 | 0.095 | ✅ |
| AKT 4b, 2006.pdf | 1 | 0.095 | 0.095 | ✅ |
| Albayrak_1998_3UHKB_Koloni caginda_p1-14.pdf | 2 | 0.085 | 0.095 | ✅ |
| Albayrak_2000_ArAn4_testament.pdf | 3 | 0.104 | 0.104 | ❌ |
| Albayrak_2001_AMMY2000_ma'usu.PDF | 3 | 0.100 | 0.107 | ❌ |

## Per-Language Results

| Language | Pages | Avg CER | Avg WER | Status |
|----------|-------|---------|---------|--------|
| Turkish | 13 | 0.100 | 0.109 | ❌ |

## Quality Enhancements Applied

1. **360° Orientation Detection:** Full angular sweep with fine deskew
2. **Multi-Scale Detection:** 1.0x + 1.5x pyramid with WBF fusion
3. **Router Ensemble:** ABINet→PARSeq→docTR-SAR with MBR consensus
4. **Advanced Preprocessing:** CLAHE + bilateral filtering + Sauvola
5. **Confidence Calibration:** Temperature scaling per engine×language
6. **Quality Profile:** 400 DPI, beam search, no speed compromises

## Error Analysis

### Challenging Cases (Higher Error Rates)

- **AKT 1, 1990.pdf page 21** (tr): CER=0.142, WER=0.208

## File Paths

- **Detailed metrics:** `reports\final_eval_20251006_174349/metrics/metrics.csv`
- **PDF aggregates:** `reports\final_eval_20251006_174349/metrics/metrics_by_pdf.csv`
- **Language aggregates:** `reports\final_eval_20251006_174349/metrics/metrics_by_lang.csv`
- **Summary report:** `reports\final_eval_20251006_174349\summary.md`
