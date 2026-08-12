# Citation / Bibliography Parsing

Structured parsing of bibliography and reference text produced by the OCR
pipeline. This turns the noisy text of a reference block into machine-readable
citation records (authors, year, title, journal/series, volume, pages, DOI),
each with a confidence score so uncertain parses can be routed to review.

- Module: `production/postprocessing/citation_parser.py`
- CLI: `tools/parse_bibliography.py`
- Tests: `tests/test_citation_parser.py`

## Why this lives in the pipeline

Reference sections are one of the highest-value, most structured targets in the
corpus: once a bibliography is parsed into fields it can be linked to external
authorities (journal abbreviations, author identities, the TokenWorks Wikibase /
FactGrid items) and used to cross-check OCR quality. It is also a natural OCR
*post-processing* step — it consumes the same page/block text the pipeline
already emits.

The parser follows the same principles as the rest of the postprocessing stage:

- **Preserve the raw text.** The source reference is never mutated; `raw_text` is
  carried through verbatim.
- **High-precision first.** DOI, URL, year, pages, and journal-abbreviation +
  volume are extracted before the fuzzier author/title attribution.
- **Auditable.** Every field carries a confidence, the unattributed remainder is
  reported as `unparsed`, and an overall confidence drives a `needs_review`
  flag — consistent with the pipeline's launch-gate / review philosophy.
- **Dependency-free and diacritic-safe.** Pure standard library (`re`,
  `unicodedata`), NFC-normalized, so Turkish and transliteration diacritics
  (İ, ğ, ş, ç, ö, ü, û, …) survive intact. Runs on the same minimal environment
  as the rest of CI.

## Supported citation styles

Grounded in the project's hand-typed gold pages, which mix two conventions:

| Style | Example | Notes |
|-------|---------|-------|
| Western academic | `Albayrak, İrfan (2008). The Toponym Balihum in the Kültepe Texts. AOF 35, 21-32.` | parenthesised year, `pp.` pages |
| Assyriological / Turkish | `Bilgiç, DTCFD VI, 5, s. 506` | journal abbreviations, Roman-numeral volumes, `s.` (sayfa) = page |

Dense, `;`-separated inline citation lists are split before parsing, e.g.
`berger, ArOr XVIII 1-2, s. 338; Hardy, AJSL 58, s. 177-216; Bilgiç, DTCFD VI, 5, s. 506`
→ three citations. Wrapped multi-line entries (hanging indents, a page range on
its own line) are merged; numbered bibliographies (`12. Smith, J. …`) are kept as
separate entries with the entry number stripped from the author field.

## Output fields

Each `ParsedCitation` (see `to_dict()`):

| Field | Meaning |
|-------|---------|
| `authors` | list of `Surname, Initials` (or bare surname) strings |
| `year` | 4-digit year, with optional disambiguation letter (`1999a`) |
| `title` | best-effort title span |
| `source` | journal / series abbreviation or name |
| `volume`, `issue` | volume (Arabic or Roman) and issue/part |
| `pages` | page or page range (`21-32`) |
| `doi`, `url` | when present |
| `unparsed` | text not attributed to any field |
| `style` | `western` \| `assyriological` \| `unknown` |
| `field_confidence` | per-field confidence (0–1) |
| `overall_confidence` | weighted blend, penalised by unattributed remainder |
| `needs_review` | `True` when confidence is low or year+authors are both missing |

## Confidence model

Each extracted field is assigned a confidence from the strength of the pattern
that produced it (a DOI regex match is near-certain; a bare trailing page range
is weaker than a labelled `pp.` range). The overall confidence is a
field-weighted average — structurally reliable fields (DOI, year, pages, volume,
authors) are weighted above the fuzzy title — then scaled down by the fraction of
the reference left unparsed, with a small bonus for breadth of coverage. A parse
is flagged `needs_review` when the overall confidence is below
`REVIEW_CONFIDENCE_THRESHOLD` (0.6) or when it recovered neither a year nor any
author. The goal is not a single correct/incorrect verdict but a triage signal:
high-confidence citations can flow downstream, low-confidence ones go to a human.

## Usage

```bash
# Free-form file (blank-line-separated bibliography blocks)
python tools/parse_bibliography.py --input refs.txt --output-dir reports/cites

# CSV or JSONL with a text column (auto-detected, or name it with --text-column)
python tools/parse_bibliography.py --input refs.csv --output-dir reports/cites

# Straight from pipeline output, only reference/bibliography blocks
python tools/parse_bibliography.py \
    --page-text-csv reports/output/client_page_text.csv \
    --reference-only \
    --output-dir reports/cites
```

Outputs `citations.jsonl` (one record per line) and `citations.csv` (flat
columns), plus a summary of count, mean confidence, and the share flagged for
review.

### Programmatic

```python
from production.postprocessing.citation_parser import (
    parse_reference,
    parse_bibliography_block,
)

one = parse_reference("Bilgiç, DTCFD VI, 5, s. 506")
many = parse_bibliography_block(block_text)  # splits then parses each entry
```

## Pipeline integration

The `production/document_model.py` block model already recognises a
`bibliography` block type, and `profiles/akkadian_strict.json` tags a
`reference_meta` role. The intended wiring is: after page text and layout roles
are assigned, feed blocks whose role/type is reference/bibliography through
`parse_bibliography_block`, and emit the structured citations as an additional
per-run artifact (`citations.jsonl` / `citations.csv`) alongside
`client_page_text.csv`. The CLI already supports this path via
`--page-text-csv --reference-only`, so it can run today as a standalone
post-processing step over any completed run.

## Scope and roadmap

This is a deliberately lightweight, transparent **first pass**, not a
replacement for a trained citation model. Mature reference parsers exist
(GROBID, AnyStyle, Neural ParsCit) and use CRF / sequence models trained on
large citation corpora. They are heavier to deploy and are tuned for Western
STM citations rather than Assyriological abbreviations and Turkish conventions.

The value of this module is that it is dependency-free, diacritic-safe,
tuned for the corpus we actually have, and auditable (per-field confidence +
review flags). It is a good baseline and a clean data-labelling front-end.

Planned next steps:

1. **Journal-abbreviation authority list.** Resolve `AOF`, `ArOr`, `AJSL`,
   `UHKB`, `OrNS`, … against a curated Assyriology abbreviation table to lift
   `source` confidence and normalise names.
2. **Evaluation set.** Hand-label a few dozen reference lines from the gold pages
   to measure per-field precision/recall and calibrate the confidence thresholds.
3. **CRF / sequence-model upgrade.** Once labelled data exists, train a token
   classifier (the rules output makes cheap weak-supervision labels) and keep the
   rules parser as a fallback and confidence cross-check.
4. **Wikibase linking.** Map high-confidence citations to Wikibase / FactGrid
   items to close the loop between OCR output and the knowledge base.
