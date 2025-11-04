# Canary Run Checklist - Block Split + Role Tag Validation

**Date**: October 9, 2025  
**Status**: 🔄 IN PROGRESS  
**Canary Output**: `reports\canary_roles_20251009_1354\`

---

## Canary Configuration

### Manifest
- **File**: `manifests\canary_split_roles_60.txt`
- **Pages**: 60 entries
- **PDF Series**: AKT, Albayrak, Bayram, Michel, Yilmaz (known problematic series)
- **Strategy**: Extracted from existing full manifest (`secondary_sources_full_20251009_0949.txt`)

### Profile
- **File**: `profiles\akkadian_strict.json`
- **Version**: 1.1.0
- **Block Clean Config**:
  ```json
  {
    "split_enabled": true,
    "role_tagging": true,
    "exclude_roles_in_pairing": ["reference_meta", "header_footer", "figure_caption"]
  }
  ```

### Pipeline Settings
- **OCR Engine**: PaddleOCR only (--engines paddle)
- **Pairing Mode**: Heuristic (--llm-off --pairing=heuristic)
- **Text Layer**: Preferred (--prefer-text-layer)
- **Resume Safety**: Enabled (--resume-safe --skip-completed)
- **Status Bar**: Enabled (--status-bar)

---

## Expected Outcomes

### Split Quality
- **Target**: ≥60% blocks are single-purpose (not mixed content)
- **Measurement**: Count fragments with single role vs. original block count
- **Log Pattern**: `Block splitting: 12 → 18 fragments`

### Role Distribution
- **Akkadian blocks**: Should detect transliterations with diacritics/markers
- **Translation blocks**: German/Turkish/English commentary
- **Reference_meta blocks**: Citations (HW s.), catalog numbers (Kt n/k), museum IDs
- **Header_footer blocks**: Page numbers, running headers
- **Figure_caption blocks**: "Fig.", "Abb.", "Table", etc.

### Pairing Filter Impact
- **Target**: ≥70% translation candidates ratio
- **Calculation**: `translation_candidates / (translation_candidates + excluded_blocks)`
- **Log Pattern**: `Pairing filter: 18 blocks → 14 candidates (excluded 4 as reference_meta, header_footer)`

### Quality Gates
1. **Reference Exclusion**: ≈0 reference_meta blocks paired as translations
2. **Pages with Pairs**: ≥30% (canary may be lower than production due to sampling)
3. **No Regressions**: Bayram author lines NOT in Akkadian blocks
4. **Correct Filtering**: Yilmaz "HW s. 124 a." excluded from translation candidates

---

## Known Test Cases (from Phase 1)

### Bayram_1996_Bel229_slavery.pdf
- **Page**: 9 (or similar pages in canary)
- **Before**: "SEBAHATTIN BAYRAM - SALIH CECEN" + transliteration in same block
- **Expected After**:
  - Fragment 1: "SEBAHATTIN BAYRAM..." → role=header_footer → EXCLUDED
  - Fragment 2: Akkadian transliteration → role=akkadian
  - Fragment 3: Catalog numbers → role=reference_meta → EXCLUDED

### Yilmaz_2007_mahkeme kayitlari.PDF
- **Page**: 6 (if in canary)
- **Before**: "HW s. 124 a. / Müze Env. Nr. 161-426-64" paired with Akkadian
- **Expected After**:
  - Fragment: "HW s. 124 a..." → role=reference_meta → EXCLUDED from pairing

### AKT Series PDFs
- **Pages**: Various from canary
- **Before**: Turkish metadata/headers detected as Akkadian
- **Expected After**: Headers split and tagged as header_footer/other

---

## Validation Steps (After Run Completes)

### 1. Check Progress & Completion
```powershell
# View progress
Import-Csv "reports\canary_roles_20251009_1354\progress.csv" | Format-Table

# Count successful pages
$progress = Import-Csv "reports\canary_roles_20251009_1354\progress.csv"
$success = $progress | Where-Object { $_.status -eq 'success' }
Write-Host "Successful: $($success.Count) / 60 pages"
```

### 2. Check Log for Split/Role Evidence
```powershell
# Look for block splitting logs
Select-String -Path "reports\canary_roles_20251009_1354\run.log" -Pattern "Block splitting:" | Select-Object -First 10

# Look for role tagging logs
Select-String -Path "reports\canary_roles_20251009_1354\run.log" -Pattern "Role tagging:" | Select-Object -First 10

# Look for pairing filter logs
Select-String -Path "reports\canary_roles_20251009_1354\run.log" -Pattern "Pairing filter:" | Select-Object -First 10
```

### 3. Inspect Sample Outputs
```powershell
# List generated JSON outputs
Get-ChildItem "reports\canary_roles_20251009_1354\outputs" -Filter "*.json" | Select-Object -First 5

# Check a Bayram output for split blocks
$bayramJson = Get-ChildItem "reports\canary_roles_20251009_1354\outputs" -Filter "*Bayram*.json" | Select-Object -First 1
if ($bayramJson) {
    $data = Get-Content $bayramJson.FullName | ConvertFrom-Json
    Write-Host "Bayram blocks count: $($data.blocks.Count)"
    $data.blocks | ForEach-Object {
        if ($_.role) { Write-Host "  Block role=$($_.role), text=$($_.text.Substring(0, [Math]::Min(50, $_.text.Length)))..." }
    }
}
```

### 4. Export Client CSV
```powershell
python tools/export_translations.py `
  --inputs "reports\canary_roles_20251009_1354\outputs" `
  --out "reports\canary_roles_20251009_1354\client_translations.csv" `
  --dedupe --strip_layout --excel-friendly
```

### 5. Generate Role-Colored Overlays
```powershell
# If pairing_overlays.py supports --color-by-role flag
python tools/pairing_overlays.py `
  --inputs "reports\canary_roles_20251009_1354\outputs" `
  --limit 20 `
  --color-by-role `
  --out "reports\canary_roles_20251009_1354\overlays"
```

### 6. Compute Acceptance Metrics
```powershell
# Create metrics script
python tools/compute_canary_metrics.py `
  --canary-dir "reports\canary_roles_20251009_1354" `
  --out "reports\canary_roles_20251009_1354\summary.md"
```

**OR manually calculate**:
```python
import pandas as pd
import json
from pathlib import Path

canary_dir = Path("reports/canary_roles_20251009_1354/outputs")
outputs = list(canary_dir.glob("*.json"))

total_blocks = 0
total_fragments = 0
akkadian_blocks = 0
translation_blocks = 0
reference_meta_blocks = 0
excluded_blocks = 0
pages_with_pairs = 0

for json_file in outputs:
    data = json.loads(json_file.read_text(encoding='utf-8'))
    blocks = data.get('blocks', [])
    total_fragments += len(blocks)
    
    for block in blocks:
        role = block.get('role', 'other')
        if role == 'akkadian':
            akkadian_blocks += 1
        elif role == 'translation':
            translation_blocks += 1
        elif role == 'reference_meta':
            reference_meta_blocks += 1
        
        if role in ['reference_meta', 'header_footer', 'figure_caption']:
            excluded_blocks += 1
    
    if data.get('pairs'):
        pages_with_pairs += 1

print(f"Total fragments: {total_fragments}")
print(f"Akkadian blocks: {akkadian_blocks}")
print(f"Translation blocks: {translation_blocks}")
print(f"Reference_meta blocks: {reference_meta_blocks}")
print(f"Excluded from pairing: {excluded_blocks}")
print(f"Translation candidate ratio: {translation_blocks / (translation_blocks + excluded_blocks):.2%}")
print(f"Pages with pairs: {pages_with_pairs} / {len(outputs)} ({pages_with_pairs/len(outputs):.1%})")
```

---

## Acceptance Gates

### MUST PASS ✅
1. ✅ **No fatal errors**: Pipeline completes without crashes
2. ✅ **Modules load**: block_splitter and block_roles import successfully
3. ✅ **Logs present**: Block splitting, role tagging, pairing filter messages in logs
4. ✅ **Reference exclusion**: ≈0 pairs with reference_meta as translation target

### SHOULD PASS ⚠️
5. ⚠️ **Split quality**: ≥60% single-purpose blocks
6. ⚠️ **Translation ratio**: ≥70% non-excluded blocks
7. ⚠️ **Pairing rate**: ≥30% pages with valid pairs (may vary with canary sampling)

### VISUAL VALIDATION 👁️
8. 👁️ **Bayram case**: Author line NOT in Akkadian block
9. 👁️ **Yilmaz case**: "HW s. 124 a." NOT paired with Akkadian
10. 👁️ **Overlays**: Role colors correctly assigned (if overlay tool supports)

---

## Next Steps (After Validation)

### If Gates PASS ✅
1. **Document metrics** in summary.md with before/after examples
2. **Update BLOCK_SPLIT_ROLE_IMPLEMENTATION.md** with actual results
3. **Resume full run** from page 438:
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

### If Gates FAIL ❌
1. **Analyze failure mode**: Over/under filtering? Split too aggressive?
2. **Adjust regex patterns** in block_splitter.py or block_roles.py
3. **Rerun unit tests**: `python -m pytest tests/test_block_split_roles.py -v`
4. **Create focused test**: Add failing case to unit tests
5. **Iterate**: Fix → Test → Canary → Validate
6. **Rollback option**: Disable split+role in profile if fundamentally broken

---

## Log Monitoring

Watch for these patterns in real-time:
```powershell
# Tail the log (PowerShell doesn't have native tail, use Get-Content -Wait)
Get-Content "reports\canary_roles_20251009_1354\run.log" -Wait | Select-String -Pattern "Block splitting|Role tagging|Pairing filter|ERROR|WARNING"
```

Key indicators:
- ✅ `Block splitting: X → Y fragments` (Y > X means splits happening)
- ✅ `Role tagging: Y blocks tagged` (confirms role assignment)
- ✅ `Pairing filter: Y blocks → Z candidates (excluded N as ...)` (shows filtering)
- ❌ `ERROR` lines (stop immediately and debug)
- ⚠️ `WARNING` lines (may indicate edge cases but not fatal)

---

**Status**: Canary running in background (Terminal ID: 23b31802-fb1d-43fb-8656-3b9f54485197)  
**Next Check**: Wait for completion (~5-10 minutes for 60 pages), then validate outputs
