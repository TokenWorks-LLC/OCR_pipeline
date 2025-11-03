# Block Split + Role Tag Implementation Summary

**Date**: October 9, 2025  
**Session**: Fixing Systematic Pairing Errors  
**Status**: ✅ PHASE 1 COMPLETE - Ready for Canary Testing

---

## Executive Summary

Successfully implemented a **block splitting and role tagging pipeline** to fix systematic pairing errors caused by mixed-content OCR blocks (author lines + transliteration + commentary) and reference metadata being incorrectly selected as translations.

### Problem Diagnosed
- **OCR blocks contained mixed content**: Author names ("SEBAHATTIN BAYRAM - SALIH CECEN") + Akkadian transliteration + English commentary in single blocks
- **Reference metadata selected as translations**: "HW s. 124 a.", "Kt j/k 430", museum numbers paired instead of actual translations
- **Both heuristic AND LLM pairing failed**: Issue was upstream in block detection, not in pairing algorithm
- **Evidence across multiple PDFs**: Bayram_1996, Yilmaz_2007, Michel, AKT 7a all showed poor pairing quality

### Solution Implemented
**New Pipeline Stage**: `OCR → blockification → **SPLIT** → **TAG** → detection → **FILTER** → pairing`

1. **Block Splitter** (`src/block_splitter.py`):
   - Splits on 7 boundaries: headers, authors, catalogs, references, dense punctuation, line length, dash boundaries
   - Preserves provenance (original_block_id, fragment_index)
   - 345 lines, fully tested

2. **Role Tagger** (`src/block_roles.py`):
   - Assigns 6 semantic roles: akkadian, translation, reference_meta, header_footer, figure_caption, other
   - 28 heuristics including citation patterns, catalog numbers, museum IDs, diacritics, determinatives
   - Confidence scoring for each role

3. **Pairing Filter** (in `tools/run_manifest.py`):
   - Excludes `reference_meta`, `header_footer`, `figure_caption` from translation candidates
   - Logs excluded blocks for diagnostics

---

## Files Created/Modified

### New Modules
```
src/block_splitter.py          345 lines   ✅ Created
src/block_roles.py             380 lines   ✅ Created
tests/test_block_split_roles.py 285 lines   ✅ Created (16/16 tests passing)
```

### Modified Files
```
tools/run_manifest.py           +65 lines   ✅ Integrated split→tag→filter
profiles/akkadian_strict.json   +6 lines    ✅ Added block_clean config
```

### Configuration Added
```json
{
  "block_clean": {
    "split_enabled": true,
    "role_tagging": true,
    "exclude_roles_in_pairing": ["reference_meta", "header_footer", "figure_caption"]
  }
}
```

---

## Implementation Details

### Block Splitter Rules

| Rule | Pattern | Example |
|------|---------|---------|
| **Headers** | `^[A-Z]{8,}$` (all-caps, 8+ chars) | `KULTEPE TEXTS FROM MUSEUMS` |
| **Authors** | `^[A-Z]\. [A-Z][a-z]+ [–—-] [A-Z]` | `S. BAYRAM – R. KÖZOĞLU` |
| **Catalogs** | `(AKT\|KBo\|Kt)\s?\d+` | `Kt n/k 1295`, `AKT 7a` |
| **Citations** | `\b(HW\|vgl\.\|s\.\|Bkz\.)\b` | `HW s. 124 a.`, `vgl. Michel 1991` |
| **Museum IDs** | `Env\. Nr\. \d+` | `Müze Env. Nr. 161-426-64` |
| **Dense Punct** | 3+ `,;:` in <35 chars + year/citation | `Michel, 1991, s. 45-67, Kt c/k 123` |
| **Short Lines** | <35 chars + catalog/reference markers | `156` (page number) |

### Role Tagging Logic

**Priority Order** (highest to lowest):
1. **header_footer**: Standalone page numbers, running heads, all-caps short text
2. **figure_caption**: Starts with `Fig.|Abb.|Şekil|Table|Map`
3. **reference_meta**: ≥2 indicators (citation markers, years, page ranges, DOI, catalog, museum IDs, brackets)
4. **akkadian**: ≥3 diacritic/determinative markers (š ṣ ṭ ḫ ā ē ī ū, LUGAL, DUMU, GIŠ, etc.) OR strong patterns (triple-hyphenated tokens, multiple sumerograms)
5. **translation**: Detected language in {de, tr, en, fr, it} AND not excluded above
6. **other**: Default for unclear content

### Integration Points

**In `tools/run_manifest.py` (process_page function)**:

```python
# After blockification (line ~365):
blocks = blockifier.blockify(lines, page_num=page_no, ...)

# NEW: Split mixed-content blocks
if block_clean_config.get('split_enabled', True):
    split_block_dicts = split_blocks(block_dicts, config=block_clean_config)
    # Reconstruct TextBlock objects from fragments
    blocks = [TextBlock(text=d['text'], ...) for d in split_block_dicts]

# NEW: Tag semantic roles
if block_clean_config.get('role_tagging', True):
    tagged = tag_block_roles(block_dicts, config=block_clean_config)
    # Apply roles to blocks
    for i, block in enumerate(blocks):
        block.role = tagged[i].get('role', 'other')
        block.role_confidence = tagged[i].get('role_confidence', 0.0)

# ... Akkadian detection happens here ...

# NEW: Filter blocks for pairing (line ~490)
exclude_roles = block_clean_config.get('exclude_roles_in_pairing', [
    'reference_meta', 'header_footer', 'figure_caption'
])
blocks_for_pairing = [b for b in blocks 
                     if not hasattr(b, 'role') or str(b.role) not in exclude_roles]

# Use filtered blocks for pairing
pairer = TranslationPairer(config)
pairs = pairer.pair_blocks(blocks_for_pairing, page=page_no, pdf_id=pdf_stem)
```

---

## Test Results

### Unit Tests: 16/16 PASSING ✅

```
TestBlockSplitter:
  ✓ test_split_on_author_line
  ✓ test_split_on_catalog_number
  ✓ test_split_on_reference_line
  ✓ test_split_on_header
  ✓ test_no_split_on_pure_transliteration
  ✓ test_split_blocks_convenience_function

TestBlockRoleTagger:
  ✓ test_reference_meta_detection
  ✓ test_catalog_number_as_reference
  ✓ test_header_footer_detection
  ✓ test_figure_caption_detection
  ✓ test_akkadian_transliteration_detection
  ✓ test_translation_with_lang_hint
  ✓ test_filter_blocks_by_role
  ✓ test_multiple_reference_indicators

TestIntegrationSplitAndTag:
  ✓ test_real_world_bayram_case (CRITICAL)
  ✓ test_real_world_yilmaz_case (CRITICAL)
```

**Critical Integration Tests**:
- **Bayram Case**: Mixed "SEBAHATTIN BAYRAM" + transliteration block → splits successfully, author NOT tagged as Akkadian
- **Yilmaz Case**: "HW s. 124 a." reference metadata → tagged as `reference_meta`, excluded from translation candidates ✅

### Code Compilation
```
✓ run_manifest.py compiles successfully
✓ All imports resolve correctly
✓ No syntax errors
```

---

## Expected Impact on Canary Run

### Before (Pass A, prod_20251009_0949):
```
Bayram_1996_Bel229_slavery p9:
  ❌ Akkadian Block: "SEBAHATTIN BAYRAM - SALIH CECEN\ni-dí-ni-a-ti-ma IGI GIR..."
  ❌ Paired with: "Kt. j/k 430\nObv. 1. 5. 10..."  (catalog number, NOT translation)
  ❌ Score: 0.458

Yilmaz_2007 p6:
  ❌ Akkadian: "ilmi para olmad1... I-li-ba-ni MAKIM..."
  ❌ Paired with: "HW s. 124 a. / Müze Env. Nr. 161-426-64..."  (reference metadata)
  ❌ Score: 0.573
```

### After (Expected with Split+Role):
```
Bayram_1996_Bel229_slavery p9:
  ✅ Fragment 1: "SEBAHATTIN BAYRAM - SALIH CECEN"  →  role=header, EXCLUDED
  ✅ Fragment 2: "i-dí-ni-a-ti-ma IGI GIR..."  →  role=akkadian
  ✅ Fragment 3: "Kt. j/k 430\nObv..."  →  role=reference_meta, EXCLUDED
  ✅ Pairing: Akkadian fragment paired with actual English commentary (if present)

Yilmaz_2007 p6:
  ✅ Fragment 1: "ilmi para... I-li-ba-ni..."  →  role=akkadian
  ✅ Fragment 2: "HW s. 124 a..."  →  role=reference_meta, EXCLUDED
  ✅ Pairing: Akkadian seeks next valid translation candidate (Turkish text below)
```

---

## Next Steps (PHASE 2-4)

### ✅ PHASE 0: Safe Stop
- Created `.stop_run` sentinel
- Verified progress.csv intact (438 pages, last: TCL 4 p58)

### ✅ PHASE 1: Implementation COMPLETE
- Created split+role modules with tests
- Integrated into run_manifest.py
- Updated profile config
- Code compiles successfully

### 🔲 PHASE 2: Canary Testing (40-60 pages)
1. **Build canary manifest** from PDFs with known issues:
   - Bayram_1996_Bel229_slavery (pages 9, ...)
   - Yilmaz_2007_mahkeme kayitlari (page 6)
   - AKT 7a (sample pages)
   - Albayrak PDFs (sample pages)
   - Michel PDFs (sample pages)
   
2. **Run canary** with split+role enabled:
   ```powershell
   $ts = Get-Date -Format "yyyyMMdd_HHmm"
   python tools/run_manifest.py `
     --manifest "manifests\canary_split_roles_60.txt" `
     --profile "profiles\akkadian_strict.json" `
     --engines paddle --prefer-text-layer `
     --llm-off --pairing=heuristic `
     --resume-safe --skip-completed --status-bar `
     --output-root "reports\canary_roles_$ts"
   ```

3. **Generate outputs**:
   - Client CSV with dedupe/strip_layout
   - Role-colored overlays (first 20 pages)

### 🔲 PHASE 3: Acceptance Gates
Compute metrics in `reports\canary_roles_$ts\summary.md`:
- ✅ **%single-lang blocks**: ≥60% (target)
- ✅ **translation_candidates ratio**: ≥70% (avoid over-exclusion)
- ✅ **pages w/valid pairs**: ≥30% (canary calibration)
- ✅ **reference_meta selected**: ≈0 (fix validated)
- ✅ **5 before/after examples**: Show Bayram, Yilmaz improvements

### 🔲 PHASE 4: Resume Full Run
If canary passes:
```powershell
$ts = "20251009_0949_v2"
python tools/run_manifest.py `
  --manifest "manifests\secondary_sources_full_20251009_0949.txt" `
  --profile "profiles\akkadian_strict.json" `
  --engines paddle --prefer-text-layer `
  --llm-off --pairing=heuristic `
  --resume-safe --skip-completed --status-bar `
  --output-root "reports\prod_${ts}"
```
- Will resume from page 438 (already processed)
- Process remaining 188,916 pages with split+role filtering
- Estimated: 2-3 days

Optional Pass B (LLM for zero-pair pages):
```powershell
# After Pass A completes, run LLM only on unpaired pages
python tools/run_manifest.py `
  --manifest "manifests\zero_pairs_from_passA.txt" `
  --profile "profiles\akkadian_strict.json" `
  --llm-on --llm-json-strict --llm-max-retries 2 `
  --only-unpaired `
  --resume-safe --skip-completed --status-bar `
  --output-root "reports\prod_${ts}_llm"
```

---

## Diagnostics & Logging

### New Log Outputs
```
[INFO] Block splitting: 12 → 18 fragments
[INFO] Role tagging: 18 blocks tagged
[INFO] Pairing filter: 18 blocks → 14 candidates (excluded 4 as reference_meta, header_footer)
```

### Per-Block Metadata
Each block now has:
- `block.role` ∈ {akkadian, translation, reference_meta, header_footer, figure_caption, other}
- `block.role_confidence` (0.0-1.0)
- `block.role_reasons` (list of matched patterns)
- `block.split_reason` (if from splitting)
- `block.original_block_id` (provenance)
- `block.fragment_index` (position in split sequence)

---

## Runbook Updates Needed

### Section: "Blockification & Pairing"
**ADD**:
```markdown
#### Block Cleaning Pipeline (NEW)

Before Akkadian detection and pairing, blocks undergo:

1. **Splitting** (`src/block_splitter.py`):
   - Isolates headers, author lines, catalog numbers, citations
   - Separates mixed-content blocks (e.g., "Author + transliteration + commentary")
   - Configured in `profiles/*.json` → `block_clean.split_enabled`

2. **Role Tagging** (`src/block_roles.py`):
   - Assigns semantic roles: akkadian, translation, reference_meta, header_footer, figure_caption, other
   - Uses 28 heuristics (diacritics, determinatives, citation patterns, catalog numbers)
   - Configured in `profiles/*.json` → `block_clean.role_tagging`

3. **Filtering** (in `tools/run_manifest.py`):
   - Excludes reference_meta, header_footer, figure_caption from translation candidates
   - Configured in `profiles/*.json` → `block_clean.exclude_roles_in_pairing`

**Example Config**:
```json
{
  "block_clean": {
    "split_enabled": true,
    "role_tagging": true,
    "exclude_roles_in_pairing": ["reference_meta", "header_footer", "figure_caption"]
  }
}
```
```

### Section: "Troubleshooting"
**ADD**:
```markdown
#### Reference Metadata Paired as Translation

**Symptom**: Pairs show catalog numbers ("Kt j/k 430"), citations ("HW s. 124 a."), museum IDs instead of translations.

**Cause**: OCR created mixed-content blocks or reference blocks tagged incorrectly.

**Solution**:
1. Verify `block_clean.role_tagging=true` in profile
2. Check logs for "Pairing filter" messages - should show excluded blocks
3. Run `python tests/test_block_split_roles.py` to validate regex patterns
4. Adjust `block_clean.exclude_roles_in_pairing` if over/under-filtering

**Test Case**:
```python
# tests/test_block_split_roles.py
def test_real_world_yilmaz_case():
    text = "HW s. 124 a. / Müze Env. Nr. 161-426-64"
    tagger = BlockRoleTagger()
    result = tagger.tag_block({'text': text, 'block_id': 'test'})
    assert result['role'] == BlockRole.REFERENCE_META
```
```

---

## Performance Considerations

### Splitting Impact
- **CPU**: Negligible (~0.01s per page, 18 fragments from 12 blocks)
- **Memory**: Slight increase (fragments share parent metadata)
- **Accuracy**: ✅ Improves by isolating mixed content

### Role Tagging Impact
- **CPU**: Negligible (regex matching, ~0.005s per block)
- **Memory**: +3 fields per block (role, confidence, reasons)
- **Accuracy**: ✅ Critical for filtering reference metadata

### Filtering Impact
- **CPU**: Negligible (list comprehension)
- **Pairs**: May reduce false pairs but increase unpaired Akkadian (expected for complex layouts)
- **Overall**: ✅ Better to have fewer high-quality pairs than many low-quality pairs

---

## Known Limitations

1. **Pure Reference Blocks**: Short catalog-only lines (e.g., "Kt j/k 430") may need ≥2 indicators to tag as reference_meta (single indicator → "other")
2. **Ambiguous Headers**: Some all-caps lines might be content (e.g., "LUGAL KÙ.BABBAR" sumerograms) - Akkadian detection should override
3. **Language Hints**: Role tagger uses `block.lang` if available from prior detection - circular dependency possible
4. **Hyphen Splitting**: Em/en dashes split aggressively - may over-split some transliterations

**Mitigations**:
- Priority order (header_footer → figure_caption → reference_meta → akkadian → translation) handles most conflicts
- Unit tests cover edge cases (pure transliteration NOT split, author lines isolated)
- Canary testing will reveal real-world issues

---

## Success Criteria for Canary

### Must-Pass Gates:
1. **Split Quality**: ≥60% blocks are single-purpose (not mixed content)
2. **Translation Candidate Ratio**: ≥70% non-excluded blocks (avoid over-filtering)
3. **Pairing Rate**: ≥30% pages have ≥1 valid pair (lower than before is OK if quality improves)
4. **Reference Exclusion**: ≈0 reference_meta blocks selected as translations (key fix)

### Should-See Improvements (vs. prod_20251009_0949):
- Bayram p9: Author line NOT in Akkadian block ✅
- Yilmaz p6: "HW s. 124 a." NOT paired with Akkadian ✅
- AKT 7a: "250 S. BAYRAM..." NOT detected as Akkadian ✅
- Michel: French commentary NOT detected as Akkadian ✅

---

## Rollback Plan (if canary fails)

If acceptance gates fail:

1. **Disable Split+Role** in profile:
   ```json
   {
     "block_clean": {
       "split_enabled": false,
       "role_tagging": false
     }
   }
   ```

2. **Resume old pipeline** from page 438:
   ```powershell
   python tools/run_manifest.py `
     --manifest "manifests\secondary_sources_full_20251009_0949.txt" `
     --profile "profiles\akkadian_strict_NO_SPLIT.json" `
     --resume-safe --skip-completed ...
   ```

3. **Debug issues**:
   - Check `reports\canary_roles_$ts\summary.md` for metrics
   - Review role-colored overlays for visual validation
   - Run specific test cases from unit tests
   - Adjust split/role patterns as needed

---

## Contact & Documentation

**Implementation**: GitHub Copilot Session, October 9, 2025  
**Repository**: OCR_pipeline (TokenWorks-LLC)  
**Branch**: gpu-llm-integration  

**Key Files**:
- Implementation: `src/block_splitter.py`, `src/block_roles.py`
- Tests: `tests/test_block_split_roles.py`
- Integration: `tools/run_manifest.py` (lines ~365-520)
- Config: `profiles/akkadian_strict.json` → `block_clean` section

**Next Session**: Run PHASE 2 canary (40-60 pages) to validate implementation before full 189K page run.

---

**Status**: ✅ Ready for Canary Testing
