# Pipeline Profiles

This document describes the available OCR pipeline profiles and their parameters.

## Fast Profile

**Description:** Optimized for speed with reduced accuracy

### OCR Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| det_db_thresh | 0.5 | Detection threshold (lower = more text boxes) |
| det_db_box_thresh | 0.7 | Box threshold (lower = more boxes) |
| rec_score_thresh | 0.6 | Recognition threshold (lower = keep more text) |
| rec_batch_num | 12 | Batch size (higher = faster, more memory) |
| det_limit_side_len | 640 | Max image side length (higher = more detail) |
| gpu_mem | 2000 | GPU memory allocation (MB) |

### Preprocessing Features

| Feature | Enabled | Description |
|---------|---------|-------------|
| Deskew | False | Correct image rotation |
| Denoise | False | Reduce image noise |
| Contrast | True | Enhance text contrast |
| Column Detection | False | Detect multi-column layouts |
| Footnote Detection | False | Separate footnotes |
| Reading Order | False | Calculate text reading order |

### LLM Correction

**Enabled:** False

### Performance Settings

- **DPI:** 200
- **GPU Enabled:** True
- **HTML Overlays:** False
- **Telemetry:** False

---

## Balanced Profile

**Description:** Balanced speed and accuracy for general use

### OCR Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| det_db_thresh | 0.3 | Detection threshold (lower = more text boxes) |
| det_db_box_thresh | 0.6 | Box threshold (lower = more boxes) |
| rec_score_thresh | 0.5 | Recognition threshold (lower = keep more text) |
| rec_batch_num | 6 | Batch size (higher = faster, more memory) |
| det_limit_side_len | 960 | Max image side length (higher = more detail) |
| gpu_mem | 4000 | GPU memory allocation (MB) |

### Preprocessing Features

| Feature | Enabled | Description |
|---------|---------|-------------|
| Deskew | True | Correct image rotation |
| Denoise | False | Reduce image noise |
| Contrast | True | Enhance text contrast |
| Column Detection | True | Detect multi-column layouts |
| Footnote Detection | True | Separate footnotes |
| Reading Order | True | Calculate text reading order |

### LLM Correction

**Enabled:** True

**Max Concurrent:** 2

**Timeout:** 30s

**Confidence Thresholds by Language:**

- en: 0.88
- de: 0.87
- fr: 0.86
- it: 0.86
- tr: 0.85

### Performance Settings

- **DPI:** 300
- **GPU Enabled:** True
- **HTML Overlays:** True
- **Telemetry:** True

---

## Quality Profile

**Description:** Optimized for maximum accuracy with full preprocessing

### OCR Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| det_db_thresh | 0.2 | Detection threshold (lower = more text boxes) |
| det_db_box_thresh | 0.5 | Box threshold (lower = more boxes) |
| rec_score_thresh | 0.4 | Recognition threshold (lower = keep more text) |
| rec_batch_num | 3 | Batch size (higher = faster, more memory) |
| det_limit_side_len | 1280 | Max image side length (higher = more detail) |
| gpu_mem | 6000 | GPU memory allocation (MB) |

### Preprocessing Features

| Feature | Enabled | Description |
|---------|---------|-------------|
| Deskew | True | Correct image rotation |
| Denoise | True | Reduce image noise |
| Contrast | True | Enhance text contrast |
| Column Detection | True | Detect multi-column layouts |
| Footnote Detection | True | Separate footnotes |
| Reading Order | True | Calculate text reading order |

### LLM Correction

**Enabled:** True

**Max Concurrent:** 3

**Timeout:** 45s

**Confidence Thresholds by Language:**

- en: 0.86
- de: 0.85
- fr: 0.84
- it: 0.84
- tr: 0.83

### Performance Settings

- **DPI:** 300
- **GPU Enabled:** True
- **HTML Overlays:** True
- **Telemetry:** True

---

## Language-Specific Parameters

The Quality profile includes optimized parameters for different languages:

### Turkish (tr)
- Lower detection thresholds for better text capture
- More aggressive unclipping ratio: 1.8
- Recognition threshold: 0.4

### German (de)
- Moderate detection thresholds for Umlauts and ß
- Recognition threshold: 0.45

### French (fr) & Italian (it)
- Standard parameters optimized for accent handling
- Recognition threshold: 0.5

### English (en)
- Baseline parameters
- Recognition threshold: 0.5

---

## Profile Selection Guidelines

### Use **Fast** profile when:
- Processing large document batches
- Real-time/interactive processing needed
- Accuracy requirements are moderate
- System resources are limited

### Use **Balanced** profile when:
- General-purpose document processing
- Mixed document types
- Moderate accuracy requirements
- Good balance of speed and quality needed

### Use **Quality** profile when:
- High accuracy is critical
- Processing academic/scholarly documents
- Akkadian/multilingual content
- Time is less critical than accuracy
- Full analysis and correction features needed