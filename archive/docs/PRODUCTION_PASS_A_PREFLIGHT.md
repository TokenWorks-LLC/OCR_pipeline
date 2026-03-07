# Production Run Preflight Check - Pass A (v2)

**Date**: October 9, 2025 @ 14:25  
**Run Type**: Full Production - Pass A (Heuristic Pairing, LLM OFF)  
**Pipeline**: Calibrated with Block Split + Role Tag filtering

---

## Preflight Checklist ✅

### 0. Configuration Validation
- ✅ **Profile**: `profiles/akkadian_strict.json` v1.1.0
- ✅ **block_clean enabled**: 
  - `split_enabled: true`
  - `role_tagging: true`
  - `exclude_roles_in_pairing: ["reference_meta", "header_footer", "figure_caption"]`
- ✅ **Akkadian detection calibrated**: threshold=0.25, any-line aggregation
- ✅ **Split/Role modules**: `src/block_splitter.py`, `src/block_roles.py` present

### 1. Manifest
- ✅ **File**: `manifests\secondary_sources_full_20251009_0949.txt`
- ✅ **Entries**: 189,355 pages
- ✅ **Format**: TSV (pdf_path, page_no, gold_text)

### 2. Google Drive Mount
- ✅ **Drive**: G: is accessible
- ✅ **Path**: `G:\.shortcut-targets-by-id\1VhQSilnVXpZtOWwAtGF6_s1zhfBIlLWU\Secondary Sources`
- ✅ **Streaming**: Files will be accessed on-demand (no local copy)

### 3. Previous Progress
- ℹ️ **Previous run**: `reports\prod_20251009_0949` (stopped early)
- ℹ️ **Pages in old progress.csv**: 439 rows (status column empty - incomplete)
- ✅ **New run**: Will start fresh in `reports\prod_20251009_0949_v2`

### 4. Output Directory
- ✅ **Created**: `reports\prod_20251009_0949_v2`
- ✅ **Outputs planned**:
  - `progress.csv` - Resume-safe tracking
  - `client_translations.csv` - UTF-8 BOM, incremental append
  - `summary.md` - Updated every ~1,000 pages
  - `outputs/<pdf_id>/` - Per-PDF JSONs with pairs

---

## Run Configuration (as per Runbook)

### Command
```powershell
python tools/run_manifest.py `
  --manifest "manifests\secondary_sources_full_20251009_0949.txt" `
  --profile "profiles\akkadian_strict.json" `
  --engines paddle --prefer-text-layer `
  --llm-off --pairing=heuristic `
  --resume-safe --skip-completed --status-bar `
  --output-root "reports\prod_20251009_0949_v2" `
  --progress-csv "reports\prod_20251009_0949_v2\progress.csv" `
  --client-csv "reports\prod_20251009_0949_v2\client_translations.csv" `
  --fail-fast-check-every 100 --fail-fast-min-rows 10
```

### Pipeline Stages
1. **OCR**: PaddleOCR (prefer embedded text layer, fallback to OCR)
2. **Blockification**: Extract text blocks with column detection
3. **PHASE 1 - Block Splitting**: Split mixed-content blocks (NEW ✨)
4. **PHASE 2 - Role Tagging**: Assign semantic roles (NEW ✨)
5. **Akkadian Detection**: threshold=0.25, any-line aggregation
6. **PHASE 3 - Pairing Filter**: Exclude reference_meta, header_footer, figure_caption (NEW ✨)
7. **Translation Pairing**: Hungarian algorithm (heuristic mode, no LLM)
8. **Output**: JSON per page + incremental client CSV

### Features Enabled
- ✅ **Status Bar**: tqdm progress with ETA
- ✅ **Resume-Safe**: Skip already processed pages via progress.csv
- ✅ **Fail-Fast Gate**: Check every 100 pages, require ≥10 translation pairs
- ✅ **Memory-Safe**: Incremental CSV writes, no large buffers
- ✅ **Fault Tolerance**: Continue on file-not-found or encoding errors

### Expected Behavior
- **Gate #1** (Page 100): Must have ≥10 translation pairs or ABORT
- **ETA Display**: Updates with rolling average (pages/min), estimated completion time
- **Client CSV**: UTF-8 BOM, schema: `pdf_name,page_no,akkadian_text,translation,confidence,status`
- **No duplicate headers**: Header written once at start
- **No blank rows**: Only write rows with actual pairs (pages with 0 pairs = 0 CSV rows)
- **Continuous flush**: Each CSV write opens, appends, closes (auto-flush)

---

## Acceptance Criteria (Pass A)

### Must-Pass Gates
1. **Fail-Fast (Page 100)**: ≥10 translation pairs detected ✅
2. **Fail-Fast (Page 200)**: ≥20 PDFs with ≥1 pair each ✅
3. **No fatal crashes**: Run completes or stops gracefully at gate
4. **Progress tracking**: progress.csv updates continuously

### Quality Expectations (vs. old pipeline without split+role)
- **Reference exclusion**: ≈0 reference_meta blocks ("HW s. 124 a.", catalog numbers) paired as translations
- **Split quality**: Mixed author+transliteration blocks separated correctly
- **Pairing precision**: Higher quality pairs (fewer false positives)
- **Pairing recall**: May be lower (acceptable - quality over quantity in Pass A)

### Deliverables
- `client_translations.csv` - Main output for client review
- `progress.csv` - For resume after interruption
- `summary.md` - Metrics, config, sample pairs (updated every ~1,000 pages)
- `outputs/<pdf_id>/translations.csv` - Technical QA data per PDF

---

## Runbook Compliance

### Sections Followed
✅ **Gold Data** - Test procedure (canary validated Oct 9)  
✅ **Full-Scale Run Procedure** - Phase 3: Pass A (Heuristic, LLM OFF)  
✅ **Akkadian Character LM** - Calibrated threshold=0.25  
✅ **Client Output Contract** - UTF-8 BOM, 6-column schema  
✅ **Live Progress & ETA** - tqdm with rolling average  
✅ **Risks & Mitigations** - Memory-safe, fault-tolerant  

### Runbook Additions Recommended
After this run completes, suggest adding to runbook:
- **Block Clean Pipeline** section documenting split→tag→filter stages
- **Fail-Fast Gates** detailed examples and tuning guidance
- **Resume Strategy** for multi-day runs (checkpoint every N hours)

---

## Estimated Runtime

**Assumptions**:
- Average: ~5-10 seconds per page (OCR fallback slower, text layer faster)
- Total pages: 189,355
- Throughput: ~6-12 pages/min

**Estimated Duration**: 
- Best case (12 p/min): 262 hours = **11 days**
- Typical (8 p/min): 394 hours = **16 days**
- Worst case (6 p/min): 525 hours = **22 days**

**Recommendation**: Monitor first 1,000 pages to get actual throughput, then extrapolate.

---

## Post-Run Actions

### On Successful Completion
1. **Export final CSV**: Verify row count matches expected pairs
2. **Generate summary.md**: Total pairs, PDFs with pairs, p50/p90 timing
3. **Quality spot-check**: Review 20 random pairs for accuracy
4. **Plan Pass B** (optional): LLM correction on unpaired Akkadian blocks

### On Fail-Fast Abort
1. **Review detection config**: Check if threshold=0.25 too strict for corpus
2. **Inspect failed PDFs**: Are they actually Akkadian or false positives in manifest?
3. **Adjust gate thresholds**: Lower `--fail-fast-min-rows` if corpus is sparse
4. **Re-run canary**: Test 100-200 pages with adjusted config before full retry

---

**Status**: ✅ READY TO RUN  
**Start Time**: 2025-10-09 @ 14:30 (estimated)  
**Expected Completion**: 2025-10-25 @ 18:30 (16-day estimate)

---

**Runbook Version**: v2.0 (Oct 9, 2025)  
**Agent**: GitHub Copilot  
**Session**: Block Split + Role Tag Implementation + Production Resume
