from __future__ import annotations

from pathlib import Path

import fitz
from PIL import Image

from tools.gold_registry.source_input_resolver import resolve_ocr_input


def test_single_page_pdf_index_reset(tmp_path: Path) -> None:
    pdf_path = tmp_path / "one_page.pdf"
    with fitz.open() as doc:
        doc.new_page(width=200, height=200)
        doc.save(str(pdf_path))

    rec = {
        "page_id": "p1",
        "page_index": 57,
        "local_pdf_path": str(pdf_path.relative_to(Path.cwd())) if pdf_path.is_relative_to(Path.cwd()) else str(pdf_path),
        "source_file": "sample.xml",
    }
    # Use absolute path in this test to avoid cwd coupling.
    rec["local_pdf_path"] = str(pdf_path)

    resolved = resolve_ocr_input(rec)
    assert resolved.ocr_input_type == "pdf"
    assert resolved.resolved_page_index == 0
    assert resolved.source_resolution_method in {"single_page_pdf_index_reset", "local_pdf"}


def test_tiff_image_to_pdf(tmp_path: Path) -> None:
    img_path = tmp_path / "sample.tif"
    Image.new("RGB", (64, 32), color=(255, 255, 255)).save(img_path)

    rec = {
        "page_id": "p2",
        "page_index": 0,
        "local_pdf_path": "",
        "local_image_path": str(img_path),
        "source_file": "sample.xml",
    }

    resolved = resolve_ocr_input(rec)
    assert resolved.ocr_input_type == "image_as_pdf"
    assert resolved.resolved_page_index == 0
    assert resolved.ocr_input_path.endswith(".pdf")
    assert Path(resolved.ocr_input_path).exists()


def test_missing_source_returns_unresolved() -> None:
    rec = {
        "page_id": "p3",
        "page_index": 0,
        "local_pdf_path": "",
        "local_image_path": "",
        "source_file": "missing.json",
    }
    resolved = resolve_ocr_input(rec)
    assert resolved.ocr_input_type == "missing"
    assert resolved.render_or_conversion_status == "failed"
