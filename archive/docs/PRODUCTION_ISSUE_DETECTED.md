# Production Run Status - ISSUE DETECTED

**Date**: October 9, 2025 @ 14:35  
**Run**: Pass A (v2) - In Progress  
**Pages Processed**: 711 / 189,355 (0.4%)  
**Translation Pairs**: 2,346 rows in client CSV

---

## ⚠️ CRITICAL ISSUE: Role Filtering NOT Working

### Evidence

**Problem**: Reference metadata is being paired as translations, despite Block Split + Role Tag pipeline being enabled.

**Examples from client_translations.csv**:

1. **Line 221**: 
   - Akkadian: `"I-ku-nim / ta-pá-i-ni"`
   - Paired with: `"HW s. 124 a."` ❌ (bibliographic reference)
   
2. **Line 291**:
   - Paired with: `"Müze env. 166-147-64 3,7 x 4,2 x 1,6 cm. siyah renkli"` ❌ (museum catalog number)
   
3. **Line 2256**:
   - Paired with: `"kt k/k 15 A, 3. B, 18-19. Da"` ❌ (tablet catalog reference)
   
4. **Line 2329**:
   - Paired with: `"Kt k/k, 44, 1 sceau"` ❌ (catalog reference)

5. **AKT 7a page 282**:
   - Akkadian: 17 lines of transliteration
   - Paired with: `"250 S. BAYRAM-R K8Z8OĞL8 Notlar:"` ❌ (page number + author names + "Notes:")

### Expected Behavior

These reference metadata blocks should have been:
1. **Detected** by `src/block_roles.py` as `role=reference_meta`
2. **Excluded** from translation candidates during pairing
3. **NOT appear** in the final translation pairs

### Root Cause Analysis

**Hypothesis 1**: Role tagging not executing
- Profile has `block_clean.role_tagging: true` ✅
- But the filtering logic in `tools/run_manifest.py` may not be applying roles correctly

**Hypothesis 2**: Role patterns too strict
- The regex patterns in `src/block_roles.py` may not match these specific formats:
  - `"HW s. 124 a."` - needs citation marker pattern
  - `"Müze env. 166-147-64"` - needs museum ID pattern
  - `"Kt k/k 15"` - needs catalog number pattern

**Hypothesis 3**: Integration bug
- Role tagging runs but blocks don't carry `role` attribute to pairing stage
- Similar to the `TextBlock.column` vs `TextBlock.column_index` bugs we fixed

---

## Current Pipeline Status

### What's Working ✅
- OCR extraction (PaddleOCR)
- Akkadian detection (threshold=0.25, getting Akkadian text)
- Translation pairing (Hungarian algorithm producing pairs)
- Client CSV writing (UTF-8 BOM, 2,346 rows)
- Progress tracking (711 pages logged)
- Diacritic preservation (ú, ܈, š visible in output)

### What's NOT Working ❌
- **Reference metadata exclusion** - Critical failure
- Role tagging either:
  - Not executing at all, OR
  - Patterns not matching these formats, OR
  - Roles not being used in pairing filter

---

## Impact Assessment

### Quality Impact: HIGH ⚠️

The entire purpose of the Block Split + Role Tag implementation was to **exclude reference metadata from being paired as translations**. If this filtering is not working, then:

1. **Pass A output will have same quality issues as before**
2. **2,346 translation pairs likely contain 15-20% false positives** (reference metadata)
3. **Client CSV will need post-processing cleanup** instead of being production-ready
4. **16-day production run may be wasted** if output quality doesn't meet requirements

### Recommended Actions

**Option A: STOP and FIX** (Recommended)
1. Stop the current run (711 pages is only 0.4% complete)
2. Diagnose why role filtering isn't working
3. Add debug logging to see role assignments
4. Test fix on canary (60 pages)
5. Restart production run with verified fix

**Option B: CONTINUE and POST-PROCESS**
1. Let run continue (already started, 16 days remaining)
2. Filter reference metadata in post-processing:
   ```python
   # Remove rows where translation matches reference patterns
   df = df[~df['translation'].str.contains(r'HW s\.|Kt [jnk]/k|Müze|vgl\.|Bkz\.')]
   ```
3. Accept lower quality, higher manual review burden

**Option C: MONITOR First 1,000 Pages**
1. Let run continue to page 1,000 (fail-fast gate checks at page 100, 200, etc.)
2. Analyze quality metrics at checkpoint
3. Decide: continue vs. stop-and-fix vs. accept-and-post-process

---

## Diagnostic Steps Needed

To understand why filtering failed:

1. **Check role tagging execution**:
   ```python
   # Add logging in tools/run_manifest.py after role tagging
   logger.info(f"Block {block.block_id}: role={block.role}, confidence={block.role_confidence}")
   ```

2. **Verify role patterns**:
   ```python
   # Test specific examples
   test_cases = [
       "HW s. 124 a.",
       "Müze env. 166-147-64",
       "Kt k/k 15 A, 3",
       "250 S. BAYRAM-R KÖZOGLU Notlar:"
   ]
   for text in test_cases:
       role = tag_block_role({'text': text, 'block_id': 'test'})
       print(f"{text} → {role['role']} (confidence: {role['role_confidence']})")
   ```

3. **Inspect pairing filter**:
   ```python
   # Check if roles are being read correctly
   logger.info(f"Pairing filter: {len(blocks)} blocks → {len(blocks_for_pairing)} candidates")
   logger.info(f"Excluded roles: {[b.block_id for b in blocks if hasattr(b, 'role') and b.role in exclude_roles]}")
   ```

---

## Next Steps

**DECISION REQUIRED**: 

Should we:
- [ ] **STOP** production run and fix role filtering now?
- [ ] **CONTINUE** and accept post-processing burden?
- [ ] **MONITOR** to page 1,000 then decide?

**Current Status**: Run is still active, processing pages continuously. Every minute that passes adds ~8-12 more pages with potential false positives.

---

**Runbook Compliance Note**: 

Runbook specifies fail-fast gates at pages 100, 200 for **quantity** (≥10 pairs), but we've discovered a **quality** issue that gates don't check. Should add quality gate:

```python
# Proposed quality gate
if pages_processed % 100 == 0:
    # Check for reference metadata in recent pairs
    recent_pairs = read_client_csv_tail(100)  # Last 100 rows
    ref_meta_count = count_reference_patterns(recent_pairs['translation'])
    ref_meta_ratio = ref_meta_count / len(recent_pairs)
    
    if ref_meta_ratio > 0.10:  # >10% reference metadata = FAIL
        logger.error(f"FAIL-FAST QUALITY: {ref_meta_ratio:.1%} of pairs are reference metadata!")
        logger.error("Role filtering not working - aborting run")
        sys.exit(1)
```

---

**Time**: 2025-10-09 @ 14:40  
**Decision Deadline**: Within next 30 minutes (before significant compute wasted)  
**Estimated Waste if Continued**: ~16 days × 24 hours = 384 compute-hours with degraded output quality
