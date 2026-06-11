from __future__ import annotations

from pathlib import Path

from tools.gold_registry.converters.local_gold_converter import (
    build_source_page_key,
    canonical_lookup_key,
    extract_local_gold_text,
)


def test_canonical_lookup_key_normalizes_diacritics_and_symbols() -> None:
    value = "Albayrak_İrfan_-_Kültepe_page_3.pdf"
    key = canonical_lookup_key(value)
    assert key == "albayrak_irfan_kultepe_page_3"


def test_build_source_page_key_uses_source_stem_and_page() -> None:
    source = "data/input_pdfs/Alp_spasiya and sapasalli.pdf"
    key = build_source_page_key(source, 2)
    assert key.endswith("page_2")
    assert "alp_spasiya_and_sapasalli" in key


def test_extract_plain_text_file(tmp_path: Path) -> None:
    text_path = tmp_path / "sample.txt"
    text_path.write_text("  hello   world\n", encoding="utf-8")

    text, warning = extract_local_gold_text(
        source_path=text_path,
        annotation_format="plain_text",
        page_id="sample_page_1",
        pdf_name="sample_page_1.pdf",
        page_number=1,
    )

    assert warning == ""
    assert text == "hello world"


def test_extract_from_csv_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "gold.csv"
    csv_path.write_text(
        "pdf_name,page,gold_text\n"
        "doc_a_page_1.pdf,1,alpha beta\n"
        "doc_b_page_2.pdf,1,gamma delta\n",
        encoding="utf-8",
    )

    text, warning = extract_local_gold_text(
        source_path=csv_path,
        annotation_format="LOCAL_GOLD_CSV",
        page_id="doc_b_page_2",
        pdf_name="doc_b_page_2.pdf",
        page_number=1,
    )

    assert warning == ""
    assert text == "gamma delta"


def test_extract_from_tsv_without_header(tmp_path: Path) -> None:
    tsv_path = tmp_path / "manifest_from_gold.txt"
    tsv_path.write_text(
        "data\\input_pdfs\\Secondary Sources\t4\tlegacy text for page\n"
        "data\\input_pdfs\\Other.pdf\t1\tother text\n",
        encoding="utf-8",
    )

    text, warning = extract_local_gold_text(
        source_path=tsv_path,
        annotation_format="LOCAL_GOLD_MANIFEST_TSV",
        page_id="secondary_sources_page_4",
        pdf_name="Secondary Sources",
        page_number=4,
    )

    assert warning == ""
    assert text == "legacy text for page"


def test_extract_from_json_records(tmp_path: Path) -> None:
    json_path = tmp_path / "gold.json"
    json_path.write_text(
        "["
        "{\"page_id\": \"x_page_1\", \"text\": \"first\"},"
        "{\"page_id\": \"y_page_2\", \"ground_truth_text\": \"second\"}"
        "]",
        encoding="utf-8",
    )

    text, warning = extract_local_gold_text(
        source_path=json_path,
        annotation_format="LOCAL_GOLD_JSON",
        page_id="y_page_2",
        pdf_name="y_page_2.pdf",
        page_number=1,
    )

    assert warning == ""
    assert text == "second"


def test_extract_returns_warning_when_row_not_found(tmp_path: Path) -> None:
    csv_path = tmp_path / "gold.csv"
    csv_path.write_text("pdf_name,page,gold_text\ndoc_page_1.pdf,1,hello\n", encoding="utf-8")

    text, warning = extract_local_gold_text(
        source_path=csv_path,
        annotation_format="LOCAL_GOLD_CSV",
        page_id="missing_page_9",
        pdf_name="missing_page_9.pdf",
        page_number=1,
    )

    assert text == ""
    assert warning == "local_gold_row_not_found"
