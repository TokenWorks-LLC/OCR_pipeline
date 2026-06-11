# Gold Registry Converters

This folder contains modular converter scaffolding for normalizing heterogeneous
datasets into `data/gold_registry/gold_manifest.jsonl` records.

Converters are organized by annotation format:

- `PAGE_XML`
- `ALTO_XML`
- `COCO`
- `JSON_BOXES`
- `line_pairs`

Each converter should:

1. Preserve original source files.
2. Emit normalized manifest records without mutating upstream data.
3. Keep dataset-specific logic isolated to its converter module.

The orchestration entrypoint is `tools/gold_registry/build_gold_manifest.py`.
