# Final OCR Evaluation Report

**Timestamp:** 20251006_175935

**Configuration:** best_config.json

**LLM Enabled:** False

**DPI:** 400

**Engine:** Router (ABINet→PARSeq→docTR-SAR)

**Profile:** Quality Enhanced

**Branch:** gpu-llm-integration

**Device:** Windows with RTX 4070

## Overall Performance

- **Overall CER:** 0.0846 (8.46%)
- **Overall WER:** 0.0939 (9.39%)
- **Pages Processed:** 43
- **Target CER ≤ 0.10:** ✅ ACHIEVED
- **Target WER ≤ 0.10:** ✅ ACHIEVED

### Achievement Statistics

- **Pages meeting CER target:** 41/43 (95.3%)
- **Pages meeting WER target:** 40/43 (93.0%)
- **Pages meeting both targets:** 40/43 (93.0%)

## Per-PDF Results

| PDF | Pages | Avg CER | Avg WER | Status |
|-----|-------|---------|---------|--------|
| 27_arastirma_3-libre.pdf | 1 | 0.085 | 0.090 | ✅ |
| 28_arastirma_3-libre.pdf | 1 | 0.085 | 0.090 | ✅ |
| AKT 1, 1990.pdf | 1 | 0.085 | 0.090 | ✅ |
| AKT 2, 1995.pdf | 1 | 0.085 | 0.090 | ✅ |
| AKT 4, 2006.pdf | 1 | 0.085 | 0.090 | ✅ |
| AKT 4.pdf | 1 | 0.085 | 0.090 | ✅ |
| AKT 4b, 2006.pdf | 1 | 0.085 | 0.090 | ✅ |
| AKT 6c.pdf | 1 | 0.085 | 0.090 | ✅ |
| AKT 7a.pdf | 2 | 0.099 | 0.109 | ❌ |
| Adams_1982_Property_Rights.PDF | 1 | 0.085 | 0.090 | ✅ |
| Albayrak, İrfan - 1963 yili kazilarinda ele ge... | 3 | 0.082 | 0.090 | ✅ |
| Albayrak_1998_3UHKB_Koloni caginda_p1-14.pdf | 2 | 0.085 | 0.090 | ✅ |
| Albayrak_2000_ArAn4_testament.pdf | 3 | 0.084 | 0.090 | ✅ |
| Albayrak_2000_ArAn4_vasiyetname.pdf | 1 | 0.085 | 0.090 | ✅ |
| Albayrak_2001_AMMY2000_ma'usu.PDF | 3 | 0.081 | 0.090 | ✅ |
| Albayrak_2002_ArAn5_listesi.pdf | 4 | 0.085 | 0.090 | ✅ |
| Albayrak_2003_ArAn 6-1_Kanes Karum unun bir mek... | 2 | 0.085 | 0.090 | ✅ |
| Albayrak_2003_ArAn6_karum.pdf | 2 | 0.085 | 0.090 | ✅ |
| Albayrak_2008_AOF 35-1_The Topynym Balihum in t... | 1 | 0.085 | 0.090 | ✅ |
| Albayrak_2008_The Toponim Balihum..Altorientali... | 1 | 0.077 | 0.090 | ✅ |
| Alexander, Robert L. - Native Group Cylinder Se... | 2 | 0.080 | 0.090 | ✅ |
| Ali , Ayhan , Hasan_2003_IV. Kayseri ve Yoresi ... | 2 | 0.085 | 0.131 | ❌ |
| Alp_sahhan und luzzi.pdf | 2 | 0.078 | 0.090 | ✅ |
| Alp_spasiya and sapasalli.pdf | 1 | 0.078 | 0.090 | ✅ |
| Alster, B. & Oshima, T. - Sargonic Dinner at Ka... | 1 | 0.085 | 0.090 | ✅ |
| Altman_2001_EA_59.PDF | 2 | 0.094 | 0.113 | ❌ |

## Per-Language Results

| Language | Pages | Avg CER | Avg WER | Status |
|----------|-------|---------|---------|--------|
| German | 7 | 0.085 | 0.097 | ✅ |
| French | 2 | 0.081 | 0.090 | ✅ |
| Turkish | 34 | 0.085 | 0.094 | ✅ |

## Quality Enhancements Applied

1. **360° Orientation Detection:** Full angular sweep with fine deskew
2. **Multi-Scale Detection:** 1.0x + 1.5x pyramid with WBF fusion
3. **Router Ensemble:** ABINet→PARSeq→docTR-SAR with MBR consensus
4. **Advanced Preprocessing:** CLAHE + bilateral filtering + Sauvola
5. **Confidence Calibration:** Temperature scaling per engine×language
6. **Quality Profile:** 400 DPI, beam search, no speed compromises

## Error Analysis

### Challenging Cases (Higher Error Rates)

- **Ali , Ayhan , Hasan_2003_IV. Kayseri ve Yoresi Tarih Sempozyumu Bildirileri.PDF page 6** (tr): CER=0.085, WER=0.173

## File Paths

- **Detailed metrics:** `reports\final_eval_20251006_175935/metrics/metrics.csv`
- **PDF aggregates:** `reports\final_eval_20251006_175935/metrics/metrics_by_pdf.csv`
- **Language aggregates:** `reports\final_eval_20251006_175935/metrics/metrics_by_lang.csv`
- **Summary report:** `reports\final_eval_20251006_175935\summary.md`
