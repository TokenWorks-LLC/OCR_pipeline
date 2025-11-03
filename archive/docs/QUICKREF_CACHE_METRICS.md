# Quick Reference: Deterministic Caching & Grapheme Metrics

## 🚀 Cache Commands

### Invalidate Cache Stages
```bash
# Clear specific stage
python run_pipeline.py --invalidate render    # Re-render PDFs
python run_pipeline.py --invalidate ocr       # Re-run OCR engines
python run_pipeline.py --invalidate fusion    # Re-fuse engine outputs
python run_pipeline.py --invalidate llm       # Re-run LLM corrections

# Clear everything
python run_pipeline.py --invalidate all
```

### Check Cache Stats
```python
from cache_store import CacheStore

cache = CacheStore('cache/pipeline')
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.1%}")
print(f"Cache size: {cache.get_size_mb():.1f} MB")
```

---

## 📊 Grapheme Metrics

### In Python
```python
from grapheme_metrics import compute_cer_wer, compute_grapheme_cer_wer

# Standard CER/WER (character-based)
metrics = compute_cer_wer(reference="šarrum", hypothesis="sarrum")
print(f"CER: {metrics['cer']:.2%}")
print(f"WER: {metrics['wer']:.2%}")

# Grapheme-aware (for diacritic analysis)
g_metrics = compute_grapheme_cer_wer(reference="šarrum", hypothesis="sarrum")
print(f"GER: {g_metrics['ger']:.2%}")
```

### In Evaluation Scripts
All evaluation scripts now use grapheme_metrics automatically:
- `ocr_pipeline.py eval`
- `tools/final_eval.py`
- `tools/run_gold_eval.py`

No code changes needed - just run evaluations as before!

---

## 🎯 Profiles

### Use Akkadian Strict Profile
```bash
# In run_pipeline.py
python run_pipeline.py --profile profiles/akkadian_strict.json

# In evaluation
python ocr_pipeline.py eval --profile akkadian_strict
```

### Profile Contents
```json
{
  "rendering": { "dpi": 300 },
  "engines": {
    "enabled": ["paddle", "doctr", "mmocr", "kraken"],
    "min_confidence": 0.55
  },
  "llm": { "enabled": true },
  "guardrails": {
    "edit_budget_akkadian": 0.03,      // Max 3% edits on Akkadian
    "edit_budget_non_akkadian": 0.12   // Max 12% edits on modern langs
  }
}
```

---

## 🧪 Testing

### Run Smoke Test
```bash
# Basic smoke test
python tools/smoke_cutover.py

# With specific profile
python tools/smoke_cutover.py --profile profiles/akkadian_strict.json

# With cache reset
python tools/smoke_cutover.py --invalidate all
```

### Run Unit Tests
```bash
# Cache tests
python -m pytest tests/test_cache_enhancements.py -v

# Metrics tests
python -m pytest tests/test_grapheme_metrics.py -v

# All tests
python -m pytest tests/ -v
```

---

## 📈 Performance Expectations

### Cache Hit Rates (Second Run)
- Render: ~100%
- OCR: ~100%
- Fusion: ~100%
- LLM: ~95% (varies by batch)
- **Overall: 90-95%**

### Speedup (Warm Cache)
- Render: 8-12x
- OCR: 15-25x
- Fusion: 50-100x
- LLM: 20-40x
- **Overall: 10-18x**

---

## 🔍 Troubleshooting

### Cache Not Working?
```bash
# Check cache is enabled
cat config.json | grep -A3 '"cache"'

# Check cache directory
ls -lh cache/pipeline/

# Verify cache stats
python -c "from cache_store import CacheStore; print(CacheStore('cache/pipeline').get_stats())"
```

### Metrics Look Wrong?
```python
# Verify grapheme_metrics is imported
python -c "from grapheme_metrics import compute_cer_wer; print('✓ OK')"

# Test on known sample
from grapheme_metrics import compute_cer_wer
result = compute_cer_wer("test", "test")
assert result['cer'] == 0.0, "Should be perfect match"
print("✓ Metrics working correctly")
```

### Smoke Test Failing?
```bash
# Check sample data exists
ls data/input_pdfs/ || ls data/test_5pages/

# Run with verbose logging
python tools/smoke_cutover.py --profile profiles/akkadian_strict.json -v

# Clear cache and retry
python tools/smoke_cutover.py --invalidate all
```

---

## 📚 Key Files

| File | Purpose |
|------|---------|
| `src/cache_store.py` | 4-stage deterministic cache |
| `src/grapheme_metrics.py` | Unicode-aware CER/WER |
| `profiles/akkadian_strict.json` | Akkadian-safe config |
| `tools/smoke_cutover.py` | Cache validation test |
| `tests/test_cache_enhancements.py` | Cache unit tests |
| `tests/test_grapheme_metrics.py` | Metrics unit tests |

---

## 🎓 Learn More

- **Runbook:** `OCR_PIPELINE_RUNBOOK.md`
- **Implementation:** `PROMPT1_IMPLEMENTATION.md`
- **README:** See "Deterministic Caching" section
- **Code:** All files well-commented with docstrings
