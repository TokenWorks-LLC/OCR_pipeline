"""Unit and CLI tests for the citation / bibliography parser.

The reference strings here are drawn from (or modelled closely on) real lines in
the project's hand-typed gold pages, so the parser is exercised against the
Assyriological + Turkish citation conventions it will actually see, not just
idealised Western citations.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from production.postprocessing.citation_parser import (
    REVIEW_CONFIDENCE_THRESHOLD,
    parse_bibliography_block,
    parse_reference,
    split_references,
)


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def test_western_journal_citation_extracts_core_fields():
    citation = parse_reference(
        "Albayrak, İrfan (2008). The Toponym Balihum in the Kültepe Texts. AOF 35, 21-32."
    )
    assert citation.year == "2008"
    assert citation.pages == "21-32"
    assert citation.source == "AOF"
    assert citation.volume == "35"
    assert citation.style == "western"
    # Diacritics must be preserved end to end.
    assert "İrfan" in citation.authors[0]
    assert "Kültepe" in (citation.title or "")
    assert citation.needs_review is False


def test_doi_and_multiple_authors_are_extracted():
    citation = parse_reference(
        "Smith, J. and Doe, A. (2019). Cuneiform segmentation with neural nets. "
        "Journal of Assyriology 12, 45-67. https://doi.org/10.1234/abc.2019.5"
    )
    assert citation.year == "2019"
    assert citation.pages == "45-67"
    assert citation.doi == "10.1234/abc.2019.5"
    assert len(citation.authors) == 2
    assert any("Smith" in a for a in citation.authors)
    assert any("Doe" in a for a in citation.authors)


def test_assyriological_abbreviation_and_roman_volume():
    # Modelled on a real gold line: "Hardy, AJSL 58, s. 177-216".
    citation = parse_reference("Bilgiç, DTCFD VI, 5, s. 506")
    assert citation.source == "DTCFD"
    assert citation.volume == "VI"  # Roman-numeral volume
    assert citation.pages == "506"  # Turkish "s." (sayfa) page marker
    assert citation.style == "assyriological"
    assert any("Bilgiç" in a for a in citation.authors)


def test_turkish_page_marker_range():
    citation = parse_reference("Hardy, AJSL 58, s. 177-216")
    assert citation.source == "AJSL"
    assert citation.volume == "58"
    assert citation.pages == "177-216"


def test_dense_inline_citation_list_is_split():
    dense = "berger, ArOr XVIII 1-2, s. 338; Hardy, AJSL 58, s. 177-216; Bilgiç, DTCFD VI, 5, s. 506"
    refs = split_references(dense)
    assert len(refs) == 3
    citations = parse_bibliography_block(dense)
    assert [c.source for c in citations] == ["ArOr", "AJSL", "DTCFD"]
    assert [c.pages for c in citations] == ["338", "177-216", "506"]


def test_wrapped_multiline_block_merges_continuation_lines():
    block = (
        "Albayrak, İrfan (2008). The Toponym Balihum in the Kültepe Texts. AOF 35,\n"
        "  21-32.\n"
        "Bilgiç, E. (1992). Ebla in Cuneiform texts. ArOr 60, 512-537."
    )
    citations = parse_bibliography_block(block)
    assert len(citations) == 2
    assert citations[0].pages == "21-32"  # merged from the wrapped continuation line
    assert citations[0].year == "2008"
    assert citations[1].pages == "512-537"
    assert citations[1].year == "1992"


def test_numbered_bibliography_entries_are_not_merged():
    block = (
        "12. Smith, J. (2019). Cuneiform OCR. Journal X 4, 1-20.\n"
        "13. Doe, A. (2020). Sign detection. Journal Y 5, 21-40."
    )
    citations = parse_bibliography_block(block)
    assert len(citations) == 2
    # The leading entry number must not leak into the author field.
    assert citations[0].authors and not citations[0].authors[0].startswith("12")
    assert citations[0].year == "2019"
    assert citations[1].year == "2020"


def test_garbage_input_is_low_confidence_and_flagged():
    citation = parse_reference("xz;; //// |||| ^^^^")
    assert citation.overall_confidence < REVIEW_CONFIDENCE_THRESHOLD
    assert citation.needs_review is True
    assert citation.year is None
    assert citation.authors == ()


def test_empty_input_is_handled():
    citation = parse_reference("")
    assert citation.overall_confidence == 0.0
    assert citation.needs_review is True
    assert citation.to_dict()["authors"] == []


def test_raw_text_is_preserved_verbatim():
    raw = "  Albayrak,  İrfan   (2008).  AOF 35, 21-32.  "
    citation = parse_reference(raw)
    assert citation.raw_text == raw  # never mutated
    assert citation.year == "2008"


def test_to_dict_is_json_serializable():
    citation = parse_reference("Bilgiç, DTCFD VI, 5, s. 506")
    payload = json.dumps(citation.to_dict(), ensure_ascii=False)
    restored = json.loads(payload)
    assert restored["source"] == "DTCFD"
    assert restored["needs_review"] in (True, False)


def test_cli_parses_text_file(tmp_path: Path):
    refs = tmp_path / "refs.txt"
    refs.write_text(
        "Albayrak, İrfan (2008). The Toponym Balihum in the Kültepe Texts. AOF 35, 21-32.\n\n"
        "Bilgiç, E. (1992). Ebla in Cuneiform texts. ArOr 60, 512-537.\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    proc = subprocess.run(
        [
            PYTHON,
            "tools/parse_bibliography.py",
            "--input",
            str(refs),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    jsonl_path = output_dir / "citations.jsonl"
    csv_path = output_dir / "citations.csv"
    assert jsonl_path.exists()
    assert csv_path.exists()

    records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert {r["year"] for r in records} == {"2008", "1992"}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert all("overall_confidence" in row for row in rows)


def test_cli_parses_page_text_csv_reference_only(tmp_path: Path):
    page_csv = tmp_path / "client_page_text.csv"
    with page_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["page", "block_type", "cleaned_text"])
        writer.writeheader()
        writer.writerow({"page": "1", "block_type": "paragraph", "cleaned_text": "Body text, not a citation."})
        writer.writerow(
            {
                "page": "2",
                "block_type": "bibliography",
                "cleaned_text": "Bilgiç, E. (1992). Ebla in Cuneiform texts. ArOr 60, 512-537.",
            }
        )
    output_dir = tmp_path / "out"

    proc = subprocess.run(
        [
            PYTHON,
            "tools/parse_bibliography.py",
            "--page-text-csv",
            str(page_csv),
            "--reference-only",
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    records = [
        json.loads(line)
        for line in (output_dir / "citations.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    # Only the bibliography row should have been parsed.
    assert len(records) == 1
    assert records[0]["year"] == "1992"
    assert records[0]["source"] == "ArOr"
