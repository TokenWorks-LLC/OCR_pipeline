# Gold/Non-Gold Sorter (Minimal)

This folder is intentionally trimmed to only what is needed for the sorting model.

## Required files

- `gold_unclean_sorter.py` – train profile + classify rows as `gold` or `unclean`
- `training_data/` – your current pure-gold training corpus
- `data/gold_profile.json` – trained profile artifact (generated)
- `README.md` – usage and workflow

## Train on gold data

```bash
python abhiram_test_file/gold_unclean_sorter.py \
  --input "abhiram_test_file/training_data" \
  --reference-dir "abhiram_test_file/training_data" \
  --train-profile-out "abhiram_test_file/data/gold_profile.json" \
  --train-only \
  --min-reference-translit-density 0.08 \
  --max-reference-noise-density 0.02
```

## Classify new testing dataset

```bash
python abhiram_test_file/gold_unclean_sorter.py \
  --input "abhiram_test_file/testing_data_new" \
  --profile-in "abhiram_test_file/data/gold_profile.json" \
  --output-dir "abhiram_test_file/data/sorted_output" \
  --threshold 0.42
```

## Outputs

- `gold.csv`
- `unclean.csv`
- `review_queue.csv` (closest to threshold; inspect first)
- `summary.json`

## Notes

- No API keys are needed for this sorter.
- Uses Python standard library only.
- Input supports `.csv`, `.txt`, `.tsv`.
