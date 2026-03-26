# Synthetic Pages: Transliteration + Modern Language Wrapper

Generate synthetic “academic-style” pages that mix **Akkadian/cuneiform transliteration** with **modern language** (English, German, French, Italian, Turkish, Arabic, Hebrew, etc.) for OCR pipeline testing and training alignment.

## Goals

- **Pipeline testing**: Use gold pages (or clean data) and/or synthetic image PDFs that produce “bad” OCR outcomes to stress-test the pipeline.
- **Training alignment**: Produce (image/PDF, ideal text) pairs where ideal text has a defined format: line numbers, document refs, and mixed transliteration + modern commentary.

## Pipeline Quick Reference

To get the pipeline running for testing (from OCR_pipeline repo root):

```bash
# Page-level text extraction (recommended entry)
# If the script lives under .merge_protect: python .merge_protect/tools/run_page_text.py ...
python tools/run_page_text.py \
  --inputs "path/to/pdfs" \
  --output-root "reports/output" \
  --prefer-text-layer \
  --ocr-fallback paddle \
  --llm-on \
  --status-bar

# With a manifest (e.g. gold pages)
python tools/build_manifest.py --csv data/gold/gold_pages.csv --pdf-root data/pdfs --out manifest.txt --expand-ranges
python tools/run_page_text.py --manifest manifest.txt --output-root reports/output --prefer-text-layer --llm-on
```

Output: `client_page_text.csv` with columns `pdf_name`, `page`, `page_text`, `has_akkadian`. Gold pages (Adam’s work) can be used for testing; alternatively use synthetic PDFs from this folder to create “bad outcome” OCR inputs.

## Transliteration Data Sources

| Source | Row = | Document ID | Transliteration column / structure |
|--------|--------|-------------|------------------------------------|
| **Finaldf.csv** | (see file) | (see file) | (see file) |
| **Oracc** | word | `id_text` | word row → aggregate by `id_text` |
| **Archibab** (archibab_texts_compiled *.csv) | document | first column | line array e.g. `['word word', 'word word']` |
| **OA_Published.csv** | document | first column | (see file) |
| **OARE** | document | first column | `transliteration_orig` |
| **Transliteration.csv** | (see file) | (see file) | (see file) |
| **CDLI** | document | `id_text` | `transliteration` |

Place CSV files in `abhiram_test_file/data/` (or set paths in the notebook).

## Modern Language Wrapper

- **Main language** ~70%: one of English, German, French, Italian, Turkish, Arabic, Hebrew.
- **LLM**: generate short commentary, captions, or paragraph text that “wraps around” the transliteration lines (e.g. citation, line refs, translation snippets).
- **Agent/LLM options**: OpenClaw (https://openclaw.ai/), Ollama (local), or any OpenAI-compatible API.

### API key (OpenAI / OpenAI-compatible)

To use the OpenAI (or OpenAI-compatible) API from `llm_wrapper.py`:

1. **From a `.env` file (recommended)**  
   From the `abhiram_test_file` directory:
   ```bash
   cp .env.example .env
   # Edit .env and set OPENAI_API_KEY=sk-your-actual-key
   ```
   The `.env` file is gitignored; never commit your key.

2. **From the environment**  
   ```bash
   export OPENAI_API_KEY=sk-your-actual-key
   ```

Then call with `provider="openai"` (and optionally `api_base=...` for a custom endpoint). The wrapper reads `OPENAI_API_KEY` automatically.

## Output Format (Ideal for OCR Alignment)

Numbered lines with document refs and mixed transliteration + modern text, e.g.:

```
2) AfO 12, s. 347 ve AC s. 194, n. 3'te kaydedilen KTP 45: 2[...] ma-na KÙ.BABBAR
3[û-]sé-bi4-1ac-ma lu i-na 4[i-]sé-ra-tim lu i-na 5şa-fı-dim 1/3 ma-na i-bi-ti-iq (!)
3) CCT IV 4a 411u i-na şa-ru-pi-im 421u i-na sé-ra-tim 43mu-tâ e...
4) Kay.13: x+20[ ] lu nu-şa-id lu i-/na] 214é(!)-ra-tim im-ti
5) Yine AfO 12 s. 347'de kaydedilen TC II 14 metni: 213 2/3 GIN sa Şax-e-dim 225 GIN
```

## Contents

- `loaders.py` – Load transliteration from Oracc, Archibab, OARE, CDLI, and generic CSV.
- `wrap_transliteration.ipynb` – Jupyter notebook: load a dataset, wrap modern language around transliteration via LLM, emit text (and optionally image/PDF).
- `data/` – Optional: put your transliteration CSVs here (or configure paths in the notebook).

## Requirements

- Python 3.8+
- pandas
- jupyter
- For LLM: `ollama` (local) or `openai` / requests for OpenClaw or other APIs.
- Optional: reportlab or fpdf2 or imgkit for PDF/image export.
