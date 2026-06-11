from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from production.layout_analysis import LayoutAnalyzer


def _new_doc_page(width: float = 600, height: float = 820) -> tuple[fitz.Document, fitz.Page]:
    doc = fitz.open()
    page = doc.new_page(width=width, height=height)
    return doc, page


def test_single_column_layout_detection(tmp_path: Path):
    doc, page = _new_doc_page()
    page.insert_text((72, 70), "Single Column Title", fontsize=20)
    page.insert_textbox(
        fitz.Rect(72, 120, 520, 360),
        "This is a normal paragraph block in a single-column page.\n" * 6,
        fontsize=11,
    )

    pdf_path = tmp_path / "single_column.pdf"
    doc.save(str(pdf_path))
    doc.close()

    analyzer = LayoutAnalyzer()
    with fitz.open(str(pdf_path)) as loaded:
        result = analyzer.analyze_page(loaded[0], page_number=1)

    assert result.column_count == 1
    assert result.column_mode == "single_column"
    assert any(region.type in {"title", "heading", "paragraph"} for region in result.regions)


def test_two_column_layout_detection(tmp_path: Path):
    doc, page = _new_doc_page()
    left_text = "Left column text line\n" * 14
    right_text = "Right column text line\n" * 14
    page.insert_textbox(fitz.Rect(48, 80, 280, 620), left_text, fontsize=10)
    page.insert_textbox(fitz.Rect(320, 80, 560, 620), right_text, fontsize=10)

    pdf_path = tmp_path / "two_column.pdf"
    doc.save(str(pdf_path))
    doc.close()

    analyzer = LayoutAnalyzer()
    with fitz.open(str(pdf_path)) as loaded:
        result = analyzer.analyze_page(loaded[0], page_number=1)

    assert result.column_count >= 2
    assert result.column_mode in {"two_column", "multi_column"}


def test_footnote_detection(tmp_path: Path):
    doc, page = _new_doc_page()
    page.insert_textbox(
        fitz.Rect(72, 90, 540, 540),
        "Main body text appears in the upper section of the page.\n" * 8,
        fontsize=11,
    )
    page.insert_textbox(
        fitz.Rect(72, 690, 540, 790),
        "1) Footnote with compact text and citation details.",
        fontsize=8,
    )

    pdf_path = tmp_path / "footnote.pdf"
    doc.save(str(pdf_path))
    doc.close()

    analyzer = LayoutAnalyzer()
    with fitz.open(str(pdf_path)) as loaded:
        result = analyzer.analyze_page(loaded[0], page_number=1)

    assert result.has_footnotes
    assert any(region.type == "footnote" for region in result.regions)


def test_table_interrupt_detection(tmp_path: Path):
    doc, page = _new_doc_page()
    page.insert_textbox(
        fitz.Rect(72, 90, 540, 220),
        "Introductory paragraph before the table appears here.\n" * 3,
        fontsize=11,
    )
    page.insert_textbox(
        fitz.Rect(72, 260, 540, 420),
        "Year   Count   Value\n2020   10   5.3\n2021   12   5.9\n2022   11   5.7",
        fontsize=10,
    )
    page.insert_textbox(
        fitz.Rect(72, 460, 540, 620),
        "Follow-up discussion after the table continues in prose.\n" * 3,
        fontsize=11,
    )

    pdf_path = tmp_path / "table_interrupt.pdf"
    doc.save(str(pdf_path))
    doc.close()

    analyzer = LayoutAnalyzer()
    with fitz.open(str(pdf_path)) as loaded:
        result = analyzer.analyze_page(loaded[0], page_number=1)

    assert any(region.type == "table" for region in result.regions)
    assert result.has_table_interruptions


def test_image_and_caption_region_detection(tmp_path: Path):
    pillow = pytest.importorskip("PIL.Image")

    img_path = tmp_path / "figure.png"
    img = pillow.new("RGB", (300, 160), color=(200, 210, 220))
    img.save(img_path)

    doc, page = _new_doc_page()
    page.insert_image(fitz.Rect(120, 180, 480, 380), filename=str(img_path))
    page.insert_textbox(
        fitz.Rect(120, 390, 500, 430),
        "Figure 1. Caption for the inserted image.",
        fontsize=10,
    )

    pdf_path = tmp_path / "image_caption.pdf"
    doc.save(str(pdf_path))
    doc.close()

    analyzer = LayoutAnalyzer()
    with fitz.open(str(pdf_path)) as loaded:
        result = analyzer.analyze_page(loaded[0], page_number=1)

    types = {region.type for region in result.regions}
    assert "image" in types
    assert "caption" in types


def test_reading_order_reconstruction_two_columns(tmp_path: Path):
    doc, page = _new_doc_page()
    page.insert_textbox(fitz.Rect(48, 90, 280, 180), "Left column intro", fontsize=10)
    page.insert_textbox(fitz.Rect(48, 220, 280, 310), "Left column continuation", fontsize=10)
    page.insert_textbox(fitz.Rect(320, 90, 560, 180), "Right column intro", fontsize=10)
    page.insert_textbox(fitz.Rect(320, 220, 560, 310), "Right column continuation", fontsize=10)

    pdf_path = tmp_path / "reading_order.pdf"
    doc.save(str(pdf_path))
    doc.close()

    analyzer = LayoutAnalyzer()
    with fitz.open(str(pdf_path)) as loaded:
        result = analyzer.analyze_page(loaded[0], page_number=1)

    ordered_text = [region.text for region in sorted(result.regions, key=lambda item: item.reading_order)]
    left_idx = next(i for i, text in enumerate(ordered_text) if "Left column intro" in text)
    right_idx = next(i for i, text in enumerate(ordered_text) if "Right column intro" in text)

    assert left_idx < right_idx
    assert result.reading_order_confidence >= 0.5
