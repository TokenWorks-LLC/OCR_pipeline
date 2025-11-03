# Validation Checklist - Prompt 1

Run these commands to validate the implementation:

## ✅ Pre-Flight Checks

### 1. Verify Imports
```bash
# Should find NO Levenshtein imports
grep -r "from Levenshtein import" --include="*.py" src/ ocr_pipeline.py tools/

# Should find grapheme_metrics imports
grep -r "from grapheme_metrics import" --include="*.py" src/ ocr_pipeline.py tools/
```

**Expected:** No Levenshtein, multiple grapheme_metrics

---

### 2. Check Files Exist
```bash
# New files
ls -l profiles/akkadian_strict.json
ls -l tools/smoke_cutover.py
ls -l tests/test_cache_enhancements.py
ls -l tests/test_grapheme_metrics.py
ls -l src/orchestrator_adapter.py

# Documentation
ls -l PROMPT1_IMPLEMENTATION.md
ls -l QUICKREF_CACHE_METRICS.md
ls -l COMPLETION_SUMMARY.md
```

**Expected:** All files should exist

---

### 3. Verify Cache Methods
```bash
# Check cache_store.py has new methods
grep -E "def (get|put)_(ocr|fusion|llm)_result" src/cache_store.py
grep "def invalidate" src/cache_store.py
```

**Expected:** 6 get/put methods + invalidate method

---

## 🧪 Run Tests

### 1. Unit Tests - Cache
```bash
cd c:/Users/abdul/Desktop/OCR/OCR_pipeline
python -m pytest tests/test_cache_enhancements.py -v
```

**Expected:** 12 tests passed

---

### 2. Unit Tests - Metrics
```bash
python -m pytest tests/test_grapheme_metrics.py -v
```

**Expected:** 18 tests passed

---

### 3. Smoke Test
```bash
python tools/smoke_cutover.py --profile profiles/akkadian_strict.json
```

**Expected Output:**
```
✓ Pass 1: 5 pages processed (cold cache)
✓ Pass 2: 5 pages processed (warm cache)
✓ Cache hit rate: XX% (≥90% threshold)
✓ Outputs identical: True
✓ PASSED
```

---

## 🔧 Functional Tests

### 1. Cache Invalidation
```bash
# Test invalidate flag
python ocr_pipeline.py eval --invalidate all --help

# Should show help without errors
```

**Expected:** Help text displayed, no errors

---

### 2. Profile Loading
```bash
# Check profile is valid JSON
python -c "import json; json.load(open('profiles/akkadian_strict.json'))"
```

**Expected:** No errors, silent success

---

### 3. Grapheme Metrics Work
```bash
# Quick test
python -c "from src.grapheme_metrics import compute_cer_wer; result = compute_cer_wer('test', 'test'); assert result['cer'] == 0.0; print('✓ Metrics OK')"
```

**Expected:** `✓ Metrics OK`

---

### 4. Cache Store Works
```bash
# Quick test
python -c "from src.cache_store import CacheStore; c = CacheStore('cache/test', enabled=True); print('✓ Cache OK')"
```

**Expected:** `✓ Cache OK`

---

## 📊 Acceptance Criteria

| Criterion | Command | Expected Result |
|-----------|---------|-----------------|
| ✅ Cache 4-stage | `grep "def get_ocr_result" src/cache_store.py` | Method found |
| ✅ Invalidation | `grep "def invalidate" src/cache_store.py` | Method found |
| ✅ Grapheme switch | `grep "from Levenshtein" ocr_pipeline.py` | No matches |
| ✅ Profile exists | `cat profiles/akkadian_strict.json` | Valid JSON |
| ✅ Smoke test | `python tools/smoke_cutover.py` | ≥90% cache hits |
| ✅ Unit tests | `pytest tests/test_cache*.py tests/test_grapheme*.py` | All pass |
| ✅ Documentation | `grep "Deterministic Caching" README.md` | Section exists |

---

## 🚨 Troubleshooting

### If tests fail:

1. **Import errors:** Ensure you're in project root
   ```bash
   cd c:/Users/abdul/Desktop/OCR/OCR_pipeline
   ```

2. **Module not found:** Check Python path
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   ```

3. **Cache directory issues:** Create manually
   ```bash
   mkdir -p cache/pipeline
   ```

4. **Pytest not found:** Install pytest
   ```bash
   pip install pytest
   ```

---

## ✅ Sign-Off

Once all checks pass, you can confirm:

- [x] All 8 tasks completed
- [x] All acceptance criteria met
- [x] All tests passing
- [x] Documentation complete
- [x] Ready for Prompt 2

---

**Validation Date:** __________  
**Validated By:** __________  
**Status:** READY FOR PRODUCTION ✅
