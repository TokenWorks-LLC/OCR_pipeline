# Page Diagnostics Layer

This document summarizes the PageDiagnostics layer added to the multilingual OCR pipeline.

## What It Does

Before OCR/text-layer acceptance, each page is inspected and a structured diagnostics record is produced.

The diagnostics are language-agnostic by default and include optional language/domain hints as metadata only.

## Signals Tracked Per Page

Core geometry and source:

- `page_number`
- `input_file`
- `width`
- `height`
- `dpi`
- `render_scale`

Text-layer diagnostics:

- `is_born_digital`
- `text_layer_char_count`
- `text_layer_word_count`
- `text_density`
- `text_layer_usable`
- `text_layer_suspicious_reasons`
- `text_layer_acceptance`

Image diagnostics:

- `foreground_ratio`
- `estimated_skew_degrees`
- `blur_score`
- `contrast_score`
- `noise_score`
- `connected_component_count`
- `estimated_column_count`
- `has_large_images`
- `has_tables_estimate`
- `layout_complexity_score`
- `is_mostly_blank`

Strategy recommendation:

- `recommended_preprocessing_profile`
- `applied_preprocessing_profile`
- `recommended_ocr_strategy`
- `language_hint`

Layout and reading-order telemetry:

- `detected_page_layout_mode`
- `detected_region_count`
- `detected_has_footnotes`
- `detected_has_table_interruptions`
- `reading_order_confidence`
- `reading_order_source`
- `ordering_source`
- `region_ocr_used`
- `region_ocr_attempted`

Adaptive routing telemetry:

- `selected_strategy`
- `strategy_mode`
- `strategy_use_text_layer`
- `strategy_use_full_page_ocr`
- `strategy_use_region_ocr`
- `strategy_primary_engine`
- `strategy_fallback_engines`
- `strategy_ensemble_needed`
- `strategy_max_engines_per_page`
- `engines_attempted`
- `engines_skipped`
- `engine_skip_reasons`
- `fallback_path`
- `final_output_source`
- `runtime_per_engine_ms`
- `confidence_per_engine`

Named preprocessing profiles currently used by the pipeline:

- `clean_scan`
- `noisy_scan`
- `faded_page`
- `complex_academic_page`
- `transliteration_or_diacritic_heavy`
- `unknown_safe_default`

## Output Artifacts

The pipeline now writes diagnostics to:

- `page_diagnostics.jsonl` (one structured record per page)
- `layout_regions.jsonl` (structured regions, reading order, and plain-text reconstruction)
- `progress.csv` (diagnostics fields appended as columns)
- `debug_artifacts/page_diagnostics/*.json` (if `--debug` is enabled)

Schema reference:

- `config/page_diagnostics_schema.json`

## How This Helps Improve OCR

The diagnostics layer supports measurable, page-level decision making:

- Identifies when text-layer extraction should be accepted vs rejected.
- Flags low-quality scans (blur, low contrast, high noise) before OCR.
- Detects likely skew and layout complexity to guide preprocessing/profile choice.
- Surfaces column/table likelihood for downstream layout-aware OCR/fusion.
- Produces machine-readable signals for evaluation slicing, regression tracking, and targeted improvements.

The current phase focuses on measurement and decision support. It avoids large behavioral changes and keeps extraction flow compatible with existing pipeline outputs.
