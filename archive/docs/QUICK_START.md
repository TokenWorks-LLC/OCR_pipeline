# Quick Start Guide - Running the OCR Pipeline

## Prerequisites

1. Python environment activated:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. Ollama running (for LLM correction):
   ```powershell
   # In a separate terminal
   ollama serve
   ```

## Running the Pipeline

### Option 1: Run Integration Test (Recommended First Step)

Validates all components work together:

```powershell
python tools/test_pairing_integration.py
```

Expected output:
```
✅ Blockification works
✅ Pairing works
✅ CSV export works
✅ Profile config present
STATUS: READY FOR PIPELINE INTEGRATION
```

### Option 2: Run on Sample Data

Process 3 PDFs from the input directory:

```powershell
python tools/run_gold_test.py --limit 3
```

### Option 3: Run Full Pipeline

Process a specific PDF:

```powershell
python run_pipeline.py data/samples/your_file.pdf -o data/output/test_run
```

Process with specific configuration:

```powershell
python run_pipeline.py data/samples/your_file.pdf ^
    -o data/output/test_run ^
    --config profiles/akkadian_strict.json
```

## Output Files

After running, check:

```powershell
# Main results CSV
dir data\output\<pdf_name>\comprehensive_results.csv

# Translation pairs (if pairing enabled)
dir data\output\<pdf_name>\translations.csv

# Akkadian translations report (if generated)
dir data\output\<pdf_name>\*_akkadian_translations.pdf
```

## Verification Steps

### 1. Check Components

```powershell
# Test LLM guardrails (15 unit tests)
python -m pytest tests/test_router_guardrails.py -v

# Test integration
python tools/test_pairing_integration.py
```

### 2. Inspect CSV Output

```powershell
# View first few lines
Get-Content data\output\integration_test\translations.csv -Head 5

# Count rows
(Get-Content data\output\integration_test\translations.csv).Length
```

### 3. Check Profile Configuration

```powershell
# View pairing config
python -c "import json; print(json.dumps(json.load(open('profiles/akkadian_strict.json'))['pairing'], indent=2))"
```

## Pipeline Architecture

```
PDF Input
    ↓
[OCR Engines] → Paddle, docTR, MMOCR, Kraken
    ↓
[ROVER Fusion] → Ensemble voting
    ↓
[Blockification] → Group lines into blocks, detect columns
    ↓
[Language Detection] → Tag Akkadian vs modern languages
    ↓
[LLM Router] → Correct non-Akkadian low-confidence blocks
    ↓         (with guardrails: 3%/12% edit budgets)
    ↓
[Translation Pairing] → Match Akkadian ↔ Translation blocks
    ↓                  (Hungarian algorithm, 6-component scoring)
    ↓
[Output]
    ├─ comprehensive_results.csv (all OCR results)
    ├─ translations.csv (paired Akkadian ↔ translations)
    └─ *_akkadian_translations.pdf (visual report)
```

## Key Features

### Akkadian-Safe Guardrails
- Preserves diacritics: š ṣ ṭ ḫ ā ē ī ū
- Preserves determinatives: ᵈ ᵐ ᶠ
- Edit budget limits: 3% for Akkadian, 12% for modern languages
- Line count invariance
- Bracket/numeral preservation

### Translation Pairing Algorithm
- **Distance scoring** (40%): Closer pairs score higher
- **Column awareness** (20%): Same/adjacent column bonus
- **Language matching** (15%): Prefers de/tr/en/fr/it
- **Lexical markers** (10%): Detects "Übersetzung", "translation", etc.
- **Reading order** (10%): Checks vertical sequence
- **Font size** (5%): Similar size bonus

### Caching
- Content-addressed keys (SHA1)
- LLM results cached (sqlite)
- Fusion results cached
- 90%+ hit rate on reruns

## Troubleshooting

### "Module not found" errors

```powershell
# Ensure you're in the right directory
cd c:\Users\abdul\Desktop\OCR\OCR_pipeline

# Activate environment
.\.venv\Scripts\Activate.ps1
```

### "Ollama connection refused"

```powershell
# Start Ollama in another terminal
ollama serve

# Check it's running
curl http://localhost:11434
```

### "No PDF files found"

```powershell
# Check input directory
dir data\input_pdfs\*.pdf
dir data\samples\*.pdf

# Or specify full path
python run_pipeline.py "C:\full\path\to\file.pdf" -o data\output\test
```

### CSV is empty

Check the log file:
```powershell
Get-Content pipeline.log -Tail 50
```

Common causes:
- OCR found no text (increase DPI or check PDF quality)
- Language detection failed (check text is readable)
- Pairing found no matches (check distance threshold)

## Configuration

### Edit Pairing Settings

Edit `profiles/akkadian_strict.json`:

```json
{
  "pairing": {
    "max_dist_px": 800,        // Increase for distant pairs
    "weights": {
      "distance": 0.40,        // Adjust scoring weights
      "column": 0.20,
      "language": 0.15
    },
    "lexical_markers": [       // Add custom markers
      "übersetzung",
      "translation",
      "my_custom_marker"
    ]
  }
}
```

### Edit LLM Settings

Edit `profiles/akkadian_strict.json`:

```json
{
  "llm": {
    "model": "qwen2.5:7b-instruct",  // Change model
    "temperature": 0.3,               // Lower = more conservative
    "max_retries": 1
  },
  "guardrails": {
    "edit_budget_akkadian": 0.03,     // Max 3% edits for Akkadian
    "edit_budget_non_akkadian": 0.12  // Max 12% edits for modern
  }
}
```

## Next Steps

1. ✅ **Validate components** - Run `test_pairing_integration.py`
2. ✅ **Test on samples** - Run `run_gold_test.py --limit 3`
3. ⏳ **Process real data** - Run pipeline on actual PDFs
4. ⏳ **Review outputs** - Check CSV quality and pairing accuracy
5. ⏳ **Generate overlays** - Visual inspection of pairings
6. ⏳ **Measure metrics** - Acceptance gate validation

## Support

See full documentation:
- `PIPELINE_READY_STATUS.md` - Detailed status report
- `OCR_PIPELINE_RUNBOOK.md` - Complete runbook
- `IMPLEMENTATION_SUMMARY.md` - Implementation details

For issues, check:
- `pipeline.log` - Execution logs
- `tests/test_router_guardrails.py` - Unit test examples
- `tools/test_pairing_integration.py` - Integration test code
