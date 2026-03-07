# Canary Run - Bug Fixes & Restart

**Date**: October 9, 2025 @ 14:05  
**Status**: 🔄 **RUNNING (RESTARTED AFTER ALL FIXES)**  
**Terminal ID**: `d1e51df9-29a3-4140-8585-0b2923a217f3`  
**Output Directory**: `reports\canary_roles_20251009_1405`

---

## Bugs Encountered & Fixed

### Error #1: PHASE 1 Block Splitting (Line ~381)
```
2025-10-09 13:57:21,916 - ERROR - Error processing Bayram_1998_CRRAI34_Turkish.pdf page 5: 
'TextBlock' object has no attribute 'column'
```

**Location**: `tools/run_manifest.py` lines ~367-402  
**Issue**: Integration code referenced `block.column`, but TextBlock dataclass uses `column_index`

**Fix**:
```python
# ❌ BEFORE (line ~381)
block_dicts.append({
    'block_id': f"p{page_no}_c{block.column}_b{i}",
    'column': block.column,
})
...
new_block = TextBlock(
    column=split_dict.get('column', ...),
)

# ✅ AFTER
block_dicts.append({
    'block_id': f"p{page_no}_c{block.column_index}_b{i}",
    'column': block.column_index,
})
...
new_block = TextBlock(
    column_index=split_dict.get('column', original.column_index ...),
)
```

### Error #2: PHASE 2 Role Tagging (Line 425)
```
2025-10-09 14:00:18,940 - ERROR - Error processing Bayram_Cecen_1995_ArAn1_6neue.pdf page 7: 
'TextBlock' object has no attribute 'column'
```

**Location**: `tools/run_manifest.py` line 425  
**Issue**: Same issue in PHASE 2 role tagging block conversion

**Fix**:
```python
# ❌ BEFORE (line 425)
block_dicts.append({
    'block_id': getattr(block, 'block_id', 'unknown'),
    'text': block.text,
    'bbox': block.bbox,
    'column': block.column  # ❌
})

# ✅ AFTER
block_dicts.append({
    'block_id': getattr(block, 'block_id', 'unknown'),
    'text': block.text,
    'bbox': block.bbox,
    'column': block.column_index  # ✅
})
```

---

## Root Cause Analysis

### TextBlock Dataclass Structure
From `src/blockification.py` (lines 31-48):
```python
@dataclass
class TextBlock:
    """Logical block of text with language/Akkadian metadata."""
    block_id: str
    page: int
    bbox: Tuple[int, int, int, int]
    text: str
    mean_conf: float  # ✅ not 'confidence'
    lines: List[TextLine]
    
    lang: str
    is_akk: bool
    akk_conf: float
    
    column_index: int = 0  # ✅ not 'column'
    reading_order: int = 0
```

### Why This Happened
1. **Incomplete API knowledge**: Integration code assumed `column` attribute based on common naming
2. **Unit tests didn't catch it**: `tests/test_block_split_roles.py` used mock blocks without dataclass enforcement
3. **Bytecode caching**: First fix didn't take effect due to Python `.pyc` cache

### Prevention
- ✅ Added `@dataclass` validation awareness
- ✅ Cleared all `__pycache__` directories
- ✅ Compiled code before running
- ⚠️ **TODO**: Update unit tests to use actual TextBlock dataclass instances

---

## Fix Verification

### Search for Remaining Issues
```powershell
Select-String -Path "tools\run_manifest.py" -Pattern "\.column\b"
# Result: No matches ✅
```

### Compilation Check
```powershell
python -m py_compile tools/run_manifest.py
# Result: Success ✅
```

### Cache Cleared
```powershell
Get-ChildItem -Recurse -Include "__pycache__","*.pyc" | Remove-Item -Recurse -Force
# Result: All cache cleared ✅
```

---

## Canary Restart

### Command
```powershell
python tools/run_manifest.py `
  --manifest "manifests\canary_split_roles_60.txt" `
  --profile "profiles\akkadian_strict.json" `
  --engines paddle --prefer-text-layer `
  --pairing=heuristic --llm-off `
  --resume-safe --skip-completed --status-bar `
  --output-root "reports\canary_roles_20251009_1402" `
  --progress-csv "reports\canary_roles_20251009_1402\progress.csv"
```

### Resume Safety
The `--resume-safe --skip-completed` flags will skip pages already processed in previous failed run (if any progress.csv exists from the 13:57 run).

---

## Monitoring

### Check Progress
```powershell
# View latest canary
$canary = Get-ChildItem "reports\" -Filter "canary_roles_*" -Directory | 
          Sort-Object LastWriteTime -Descending | Select-Object -First 1

# Check progress.csv
if (Test-Path "$($canary.FullName)\progress.csv") {
    $progress = Import-Csv "$($canary.FullName)\progress.csv"
    Write-Host "Processed: $($progress.Count - 1) / 60 pages"
    $progress | Select-Object -Last 5 | Format-Table
}
```

### Watch for Errors
```powershell
# Monitor log file
Get-Content "reports\canary_roles_20251009_1402\run.log" -Wait | 
    Select-String -Pattern "ERROR|Block splitting|Role tagging|Pairing filter"
```

---

## Expected Timeline

- **Total Pages**: 60
- **Estimated Duration**: 5-10 minutes (with text layer), up to 15 minutes (with OCR fallbacks)
- **ETA**: ~14:10 - 14:15

---

## Next Steps (After Completion)

1. ✅ **Verify no more 'column' errors** in log
2. ✅ **Check for block splitting logs**: `Block splitting: X → Y fragments`
3. ✅ **Check for role tagging logs**: `Role tagging: Y blocks tagged`
4. ✅ **Check for pairing filter logs**: `Pairing filter: Y blocks → Z candidates (excluded N as ...)`
5. **Export client CSV**
6. **Generate overlays** (if tool supports --color-by-role)
7. **Compute acceptance metrics**
8. **Write summary.md**

---

**Last Updated**: October 9, 2025 @ 14:03  
**Status**: Canary running with fixed code ✅
