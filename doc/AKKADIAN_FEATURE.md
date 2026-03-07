# Akkadian Detection Notes

Akkadian detection is part of the page-text pipeline output.

## Current CLI Path

Use one of these entrypoints:

```bash
python run_pipeline.py --input-dir data/input --output-dir reports/output
```

or

```bash
python tools/run_page_text.py \
  --inputs data/input \
  --output-root reports/output \
  --prefer-text-layer \
  --ocr-fallback paddle
```

The output CSV includes:

- `pdf_name`
- `page`
- `page_text`
- `has_akkadian`

## Profiles

Default detection profile:

- `profiles/akkadian_strict.json`

Override with:

```bash
python tools/run_page_text.py --inputs data/input --output-root reports/output --profile profiles/akkadian_strict.json
```

## Important

Legacy flags such as `--akkadian-mode` and `--generate-translations-pdf` are not part of the current CLI surface.
