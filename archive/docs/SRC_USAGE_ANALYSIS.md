# OCR Pipeline Source Code Usage Analysis
**Generated:** October 7, 2025
**Purpose:** Identify which src/ modules are actively used vs. legacy/unused code

## Summary

### ✅ **Currently Used in ocr_pipeline.py (New Unified Script)**

1. **config.py** - Used for TESSERACT_CMD
2. **engines/easyocr_engine.py** - EasyOCR wrapper
3. **enhanced_llm_correction.py** - LLM post-correction with guardrails

### ⚠️ **Available in src/ but NOT used in ocr_pipeline.py**

#### **Core Pipeline Components** (Should be integrated)
- `orchestrator.py` - Full parallel OCR pipeline with caching
- `pipeline.py` - Original OCR pipeline
- `pdf_utils.py` - PDF rendering utilities
- `ocr_utils.py` - OCR helper functions
- `preprocess.py` - Image preprocessing

#### **OCR Engines** (Partially integrated)
- `engines/__init__.py` - Engine registry (has 6 engines)
- `engines/doctr_engine.py` - DocTR wrapper
- `engines/kraken_engine.py` - Kraken wrapper  
- `engines/mmocr_engine.py` - MMOCR wrapper
- `multi_engine.py` - Multi-engine orchestration
- `ensemble.py` - Ensemble voting system
- `rover_fusion.py` - ROVER fusion for ensemble

#### **LLM Correction System** (Partially integrated)
- `llm/` directory - Full LLM system with guardrails
- `llm_correction.py` - Original LLM corrector
- `llm_correction_safe.py` - Safe LLM corrector
- `llm_cache.py` - LLM caching

#### **Akkadian/Transliteration Support**
- `akkadian_extract.py` - Akkadian text detection
- `lang_and_akkadian.py` - Language + Akkadian detection
- `lang_and_extract.py` - Language detection & extraction
- `translit_norm.py` - Transliteration normalization
- `diacritic_restoration.py` - Diacritic restoration

#### **Text Processing & Analysis**
- `grapheme_metrics.py` - CER/WER calculation with grapheme awareness
- `unicode_utils.py` - Unicode normalization
- `alignment_report.py` - Text alignment for comparison
- `reading_order.py` - Reading order detection
- `page_cleanup.py` - Page text cleanup
- `confusion_prior.py` - OCR confusion patterns
- `lexicon_bias.py` - Dictionary-based correction

#### **Evaluation & Metrics**
- `cer_evaluation.py` - Character error rate evaluation
- `baseline_accuracy.py` - Baseline accuracy measurement
- `benchmark_tracker.py` - Performance benchmarking
- `summary_analysis.py` - Analysis summaries
- `simple_analysis.py` - Simple analysis tools

#### **Advanced Features**
- `advanced_preprocessing.py` - Advanced image preprocessing
- `detection.py` - Text detection
- `recognition_router.py` - Recognition routing
- `layout_classifier.py` - Page layout classification
- `gpu_determinism.py` - GPU determinism for reproducibility
- `tta_augment.py` - Test-time augmentation
- `char_lm_decoder.py` - Character language model decoder
- `python_char_lm.py` - Python char LM implementation

#### **Output & Reporting**
- `csv_writer.py` - CSV output generation
- `aggregated_csv.py` - Aggregated CSV reports
- `translations_pdf.py` - PDF with translations overlay
- `html_overlays.py` - HTML overlay generation

#### **Infrastructure**
- `cache_store.py` - Caching infrastructure
- `structured_logging.py` - Structured logging
- `telemetry.py` - Telemetry tracking
- `healthcheck.py` - Health check endpoints
- `cli.py` - CLI interface
- `preflight.py` - Pre-flight checks
- `pipeline_profiles.py` - Pipeline configuration profiles

#### **Utilities**
- `utils/` directory - Utility functions
- `page_utils.py` - Page utilities
- `filters.py` - Image filters
- `multilang_ocr.py` - Multi-language OCR
- `ocr_engine.py` - Base OCR engine interface
- `orientation.py` - Orientation detection
- `reading_order_triage.py` - Reading order triage

---

## Recommendations

### 🎯 **Immediate Actions**

1. **Use orchestrator.py instead of custom implementation**
   - Our `ocr_pipeline.py` re-implements rendering and processing
   - `orchestrator.py` has full parallel pipeline with caching already built
   - Benefits: Caching, parallelization, deterministic hashing

2. **Integrate full engine registry**
   - `engines/__init__.py` has 6 engines registered (not just 3)
   - DocTR, MMOCR, Kraken are available but unused
   - Ensemble could use all 6 engines

3. **Use proper evaluation metrics**
   - Switch from Levenshtein distance to `grapheme_metrics.py`
   - Handles combining characters properly for Akkadian
   - More accurate CER/WER for transliteration

4. **Integrate Akkadian detection**
   - `akkadian_extract.py` has is_akkadian_transliteration()
   - Would enable proper handling of transliteration vs modern text
   - Critical for applying correct guardrails (3% vs 12% edit budget)

5. **Use existing LLM system**
   - Full LLM system in `llm/` directory
   - Has all guardrails implemented and tested
   - JSON schema validation, edit budgets, bracket preservation

### 📦 **Code Consolidation**

**DELETE (Legacy/Duplicate):**
- `pipeline.py` - Superseded by orchestrator.py
- `llm_correction.py` - Use enhanced_llm_correction.py
- `llm_correction_safe.py` - Use llm/ directory version
- `simple_analysis.py` - Duplicate of functionality
- Old demo files (if any)

**KEEP & INTEGRATE:**
- `orchestrator.py` - Core parallel pipeline
- `enhanced_llm_correction.py` or `llm/` - Full LLM system
- `grapheme_metrics.py` - Proper evaluation
- `akkadian_extract.py` - Akkadian detection
- All engine wrappers

---

## Current Architecture vs. Optimized Architecture

### **Current (ocr_pipeline.py)**
```
User → ocr_pipeline.py → Custom OCR wrappers → Engines (3/6)
         ↓
      Basic ensemble (confidence-based)
         ↓
      Simple LLM integration (partial)
         ↓
      Basic Levenshtein CER
```

### **Optimized (Using existing src/)**
```
User → ocr_pipeline.py → orchestrator.py → Engine Registry (6/6)
                              ↓
                         Cache (deterministic)
                              ↓
                         Parallel processing
                              ↓
                         rover_fusion.py (ensemble)
                              ↓
                         llm/ (full guardrails)
                              ↓
                         grapheme_metrics.py (accurate CER)
                              ↓
                         akkadian_extract.py (language routing)
```

---

## Integration Priority

### **Phase 1: Core Pipeline** (Immediate)
1. Replace custom rendering with `pdf_utils.py`
2. Use `orchestrator.py` for processing
3. Integrate `cache_store.py` for caching
4. Use `grapheme_metrics.py` for evaluation

### **Phase 2: Engines** (High Priority)
1. Use `engines/__init__.py` registry
2. Add DocTR, MMOCR, Kraken to ensemble
3. Implement `rover_fusion.py` for better voting
4. Use `multi_engine.py` orchestration

### **Phase 3: LLM** (High Priority)  
1. Integrate full `llm/` directory
2. Enable `akkadian_extract.py` for language routing
3. Use proper guardrails from `llm/corrector.py`
4. Enable `llm_cache.py` for performance

### **Phase 4: Advanced Features** (Medium Priority)
1. `advanced_preprocessing.py` for quality
2. `layout_classifier.py` for document types
3. `reading_order.py` for complex layouts
4. `diacritic_restoration.py` for Akkadian

### **Phase 5: Reporting** (Low Priority)
1. `benchmark_tracker.py` for metrics
2. `html_overlays.py` for visual reports
3. `aggregated_csv.py` for batch results

---

## File Status Table

| Module | Status | Used in ocr_pipeline.py | Should Use | Priority |
|--------|--------|------------------------|------------|----------|
| orchestrator.py | ✅ Core | ❌ | ✅ | HIGH |
| pdf_utils.py | ✅ Core | ❌ | ✅ | HIGH |
| grapheme_metrics.py | ✅ Core | ❌ | ✅ | HIGH |
| engines/__init__.py | ✅ Core | ⚠️ Partial | ✅ | HIGH |
| llm/ | ✅ Core | ⚠️ Partial | ✅ | HIGH |
| akkadian_extract.py | ✅ Important | ❌ | ✅ | HIGH |
| rover_fusion.py | ✅ Important | ❌ | ✅ | MEDIUM |
| cache_store.py | ✅ Important | ❌ | ✅ | MEDIUM |
| advanced_preprocessing.py | ✅ Quality | ❌ | ⚠️ | MEDIUM |
| layout_classifier.py | ✅ Quality | ❌ | ⚠️ | MEDIUM |
| reading_order.py | ✅ Quality | ❌ | ⚠️ | LOW |
| pipeline.py | ⚠️ Legacy | ❌ | ❌ | DELETE |
| llm_correction.py | ⚠️ Duplicate | ❌ | ❌ | DELETE |
| simple_analysis.py | ⚠️ Duplicate | ❌ | ❌ | DELETE |

---

## Action Items

### ✅ **Completed**
- [x] Created unified ocr_pipeline.py
- [x] Consolidated test scripts
- [x] Basic ensemble (3 engines)
- [x] Basic LLM integration

### 🔄 **In Progress**
- [ ] Ensemble evaluation running
- [ ] LLM correction testing

### ❌ **TODO - High Priority**
- [ ] Integrate orchestrator.py (replaces custom pipeline)
- [ ] Use grapheme_metrics.py (better evaluation)
- [ ] Integrate full engine registry (6 engines)
- [ ] Integrate full llm/ system (all guardrails)
- [ ] Add akkadian_extract.py (language routing)
- [ ] Use cache_store.py (deterministic caching)
- [ ] Implement rover_fusion.py (better ensemble)

### ❌ **TODO - Medium Priority**
- [ ] Delete legacy files (pipeline.py, old LLM)
- [ ] Add advanced_preprocessing.py
- [ ] Add layout_classifier.py
- [ ] Add benchmark_tracker.py

### ❌ **TODO - Low Priority**
- [ ] Integrate reading_order.py
- [ ] Add html_overlays.py
- [ ] Add aggregated_csv.py
- [ ] Full documentation update

---

## Estimated Impact

**Current Implementation:**
- Uses ~10% of available src/ modules
- Basic functionality working
- Missing advanced features

**After Full Integration:**
- 60-70% of src/ modules actively used
- Professional-grade pipeline
- All features available
- Better accuracy (ensemble + LLM)
- Faster (caching + parallel)
- More accurate evaluation (grapheme-aware)
- Proper Akkadian support

**Performance Improvement Estimates:**
- Accuracy: 43.5% → 60-70% (with full ensemble + LLM)
- Speed: 2x faster with caching
- Akkadian: Proper handling vs. generic
- Reproducibility: Deterministic with cache_store

