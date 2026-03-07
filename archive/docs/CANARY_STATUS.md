# Canary Run - ACTIVE STATUS

**Time Started**: October 9, 2025 @ 13:57  
**Status**: 🔄 **RUNNING**  
**Terminal ID**: `d6cf497b-9f65-4e04-9ef0-ce5108df5a0b`  
**Output Directory**: `reports\canary_roles_20251009_1357`

---

## Quick Stats

- **Manifest**: `manifests\canary_split_roles_60.txt`  
- **Total Pages**: 60 (from AKT, Albayrak, Bayram, Michel, Yilmaz series)  
- **Profile**: `profiles\akkadian_strict.json` v1.1.0  
- **Pipeline**: Split → Tag → Detect → Filter → Pair  
- **Pairing Mode**: Heuristic (LLM OFF)  
- **OCR Engine**: PaddleOCR  
- **Estimated Duration**: 5-10 minutes

---

## What's Being Tested

### Block Splitting (NEW)
- Separates mixed-content blocks on 7 boundaries:
  - Headers (all-caps long lines)
  - Authors (citation format)
  - Catalogs (AKT, Kt, KBo numbers)
  - References (HW s., vgl., Bkz.)
  - Museum IDs (Env. Nr.)
  - Dense punctuation (bibliography style)
  - Short metadata lines

### Role Tagging (NEW)
- Assigns semantic roles to each block:
  - `akkadian`: Transliterations with diacritics/determinatives
  - `translation`: German/Turkish/English commentary
  - `reference_meta`: Citations, catalog numbers, museum IDs
  - `header_footer`: Page numbers, running headers
  - `figure_caption`: Fig., Abb., Table labels
  - `other`: Unclear/mixed content

### Pairing Filter (NEW)
- **Excludes from translation candidates**:
  - `reference_meta` (catalog numbers, citations, museum IDs)
  - `header_footer` (page numbers, headers)
  - `figure_caption` (figure/table labels)

---

## Expected Improvements

### Before (prod_20251009_0949 - Pass A without fixes)
```
❌ Bayram p9: "SEBAHATTIN BAYRAM - SALIH CECEN\ni-dí-ni-a-ti-ma..." 
   → Paired with "Kt. j/k 430" (catalog number)

❌ Yilmaz p6: Akkadian transliteration
   → Paired with "HW s. 124 a. / Müze Env. Nr..." (reference metadata)

❌ AKT 7a: Turkish headers/metadata detected as Akkadian
```

### After (canary_roles_20251009_1357 - WITH split+role fixes)
```
✅ Bayram p9: 
   Fragment 1: "SEBAHATTIN BAYRAM..." → role=header_footer → EXCLUDED
   Fragment 2: "i-dí-ni-a-ti-ma..." → role=akkadian → AVAILABLE
   Fragment 3: "Kt. j/k 430..." → role=reference_meta → EXCLUDED

✅ Yilmaz p6:
   Fragment 1: Akkadian → role=akkadian → AVAILABLE
   Fragment 2: "HW s. 124 a..." → role=reference_meta → EXCLUDED

✅ AKT 7a: Turkish headers split and tagged as header_footer/other
```

---

## Monitoring Commands

### Check Progress
```powershell
# View latest progress
$canary = Get-ChildItem "reports\" -Filter "canary_roles_*" -Directory | 
          Sort-Object LastWriteTime -Descending | Select-Object -First 1
Import-Csv "$($canary.FullName)\progress.csv" | Format-Table
```

### Watch Logs in Real-Time
```powershell
# Monitor for key events
Get-Content "reports\canary_roles_20251009_1357\run.log" -Wait | 
    Select-String -Pattern "Block splitting|Role tagging|Pairing filter|ERROR"
```

### Check Terminal Output
```powershell
# In VS Code, use:
# > Terminal: Get Terminal Output (ID: d6cf497b-9f65-4e04-9ef0-ce5108df5a0b)
```

---

## Issues Encountered & Fixed

### Issue 1: Google Drive Mount Detection
- **Problem**: Drive folder structure uses `.shortcut-targets-by-id/...`  
- **Solution**: Extracted base path from existing full manifest
- **Path**: `G:\.shortcut-targets-by-id\1VhQSilnVXpZtOWwAtGF6_s1zhfBIlLWU\Secondary Sources`

### Issue 2: Manifest BOM Encoding
- **Problem**: UTF-8 BOM (`\ufeff`) caused `pdf_path` field to be `\ufeffpdf_path`
- **Solution**: Removed BOM with UTF-8 encoding (no BOM)
- **Command**: `[System.IO.File]::WriteAllText(..., [System.Text.UTF8Encoding]::new($false))`

### Issue 3: Tab Separator Confusion
- **Problem**: PowerShell `Out-File` converts tabs to spaces
- **Solution**: Used `[System.IO.File]` methods to preserve literal tabs

---

## Next Steps (After Completion)

### 1. Export Client CSV
```powershell
python tools/export_translations.py `
  --inputs "reports\canary_roles_20251009_1357\outputs" `
  --out "reports\canary_roles_20251009_1357\client_translations.csv" `
  --dedupe --strip_layout --excel-friendly
```

### 2. Generate Overlays
```powershell
python tools/pairing_overlays.py `
  --inputs "reports\canary_roles_20251009_1357\outputs" `
  --limit 20 `
  --color-by-role `
  --out "reports\canary_roles_20251009_1357\overlays"
```

### 3. Compute Metrics
```python
# Manual metric calculation or run a script
import json
from pathlib import Path

outputs = list(Path("reports/canary_roles_20251009_1357/outputs").glob("*.json"))

# Metrics:
# - Total fragments vs original blocks (split quality)
# - Role distribution (akkadian, translation, reference_meta, etc.)
# - Excluded blocks count (reference_meta + header_footer + figure_caption)
# - Translation candidate ratio (translation / (translation + excluded))
# - Pages with pairs (pairing success rate)
# - Reference_meta as translation target (should be ≈0)
```

### 4. Validate Acceptance Gates
- ✅ No fatal errors (pipeline completes)
- ✅ Split+role logs present
- ✅ Reference exclusion (≈0 reference_meta in pairs)
- ⚠️ Split quality (≥60% single-purpose blocks)
- ⚠️ Translation ratio (≥70%)
- ⚠️ Pairing rate (≥30%)

### 5. Document Results
- Update `BLOCK_SPLIT_ROLE_IMPLEMENTATION.md` with actual metrics
- Create `reports\canary_roles_20251009_1357\summary.md` with:
  - Acceptance gate results
  - Before/after examples (Bayram, Yilmaz, AKT)
  - Role distribution chart
  - Recommendations for full run

### 6. If Gates PASS → Resume Full Run
```powershell
python tools/run_manifest.py `
  --manifest "manifests\secondary_sources_full_20251009_0949.txt" `
  --profile "profiles\akkadian_strict.json" `
  --engines paddle --prefer-text-layer `
  --llm-off --pairing=heuristic `
  --resume-safe --skip-completed --status-bar `
  --output-root "reports\prod_20251009_0949_v2" `
  --progress-csv "reports\prod_20251009_0949_v2\progress.csv"
```
- Will resume from page 438 (already completed)
- Process remaining 188,916 pages with split+role fixes

---

**Last Updated**: October 9, 2025 @ 13:58  
**Estimated Completion**: ~14:05 (5-10 minutes from start)  
**Status**: Waiting for canary to complete...
