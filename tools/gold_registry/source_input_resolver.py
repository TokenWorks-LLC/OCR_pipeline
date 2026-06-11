from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "reports" / "source_input_cache"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}
XML_SUFFIXES = {".xml"}
JSON_SUFFIXES = {".json"}


@dataclass
class ResolvedInput:
    ocr_input_path: str
    ocr_input_type: str
    source_resolution_method: str
    render_or_conversion_status: str
    render_or_conversion_warning: str
    image_width: int
    image_height: int
    image_file_size: int
    is_blank_or_nearly_blank: str
    requested_page_index: int
    resolved_page_index: int
    source_file_type: str
    source_file_exists: str


def _to_int(v: Any, default: int = 0) -> int:
    try:
        return int(float(str(v).strip()))
    except Exception:
        return default


def _is_blank_image(path: Path) -> str:
    if Image is None or (not path.exists()):
        return "unknown"
    try:
        with Image.open(path) as im:
            gray = im.convert("L")
            hist = gray.histogram()
            total = sum(hist) or 1
            bright = sum(hist[245:256]) / total
            dark = sum(hist[0:10]) / total
            if bright > 0.985 or dark > 0.985:
                return "true"
            return "false"
    except Exception:
        return "unknown"


def _pdf_page_count(path: Path) -> int:
    try:
        with fitz.open(str(path)) as doc:
            return len(doc)
    except Exception:
        return 0


def _render_pdf_page_to_pdf(pdf_path: Path, page_index: int, cache_key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = CACHE_DIR / f"{cache_key}_p{page_index}.pdf"
    if out_pdf.exists():
        return out_pdf

    with fitz.open(str(pdf_path)) as src:
        if page_index < 0 or page_index >= len(src):
            raise IndexError("page_out_of_range")
        page = src[page_index]
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
        tmp_png = CACHE_DIR / f"{cache_key}_p{page_index}.png"
        pix.save(str(tmp_png))

    with fitz.open() as out_doc:
        rect = fitz.Rect(0, 0, pix.width, pix.height)
        out_page = out_doc.new_page(width=rect.width, height=rect.height)
        out_page.insert_image(rect, filename=str(tmp_png))
        out_doc.save(str(out_pdf))
    return out_pdf


def _image_to_pdf(image_path: Path, cache_key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = CACHE_DIR / f"{cache_key}.pdf"
    if out_pdf.exists():
        return out_pdf

    if Image is not None:
        with Image.open(image_path) as im:
            rgb = im.convert("RGB")
            rgb.save(out_pdf, "PDF", resolution=300.0)
        return out_pdf

    # Fallback to fitz insertion path.
    with fitz.open() as out_doc:
        with fitz.open(str(image_path)) as maybe_doc:
            page = maybe_doc[0]
            pix = page.get_pixmap(alpha=False)
            rect = fitz.Rect(0, 0, pix.width, pix.height)
            out_page = out_doc.new_page(width=rect.width, height=rect.height)
            png_path = CACHE_DIR / f"{cache_key}.png"
            pix.save(str(png_path))
            out_page.insert_image(rect, filename=str(png_path))
        out_doc.save(str(out_pdf))
    return out_pdf


def resolve_ocr_input(record: dict[str, Any]) -> ResolvedInput:
    requested_page = _to_int(record.get("page_index"), 0)
    local_pdf = str(record.get("local_pdf_path", "")).strip()
    local_img = str(record.get("local_image_path", "")).strip()
    source_file = str(record.get("source_file", "")).strip()

    pdf_path = (ROOT / local_pdf) if local_pdf else None
    img_path = (ROOT / local_img) if local_img else None
    src_path = (ROOT / source_file) if source_file else None

    suffix = (src_path.suffix.lower() if src_path else "")
    source_type = "unknown"
    if suffix in XML_SUFFIXES:
        source_type = "xml"
    elif suffix in JSON_SUFFIXES:
        source_type = "json"
    elif suffix in IMAGE_SUFFIXES:
        source_type = "image"
    elif suffix == ".pdf":
        source_type = "pdf"

    cache_key = hashlib.sha1((str(record.get("page_id", "")) + local_pdf + local_img).encode("utf-8")).hexdigest()[:16]

    if pdf_path and pdf_path.exists():
        pages = _pdf_page_count(pdf_path)
        resolved_index = requested_page
        warning = ""
        method = "local_pdf"
        if pages == 1 and requested_page != 0:
            resolved_index = 0
            warning = "requested_page_out_of_single_page_pdf_reset_to_0"
            method = "single_page_pdf_index_reset"
        elif requested_page < 0 or requested_page >= max(pages, 1):
            # Convert requested page if possible; otherwise clamp to 0 for single-page fallback.
            if pages > 0:
                resolved_index = min(max(requested_page, 0), pages - 1)
                warning = "requested_page_out_of_range_clamped"
                method = "pdf_index_clamped"
        return ResolvedInput(
            ocr_input_path=str(pdf_path),
            ocr_input_type="pdf",
            source_resolution_method=method,
            render_or_conversion_status="ok",
            render_or_conversion_warning=warning,
            image_width=0,
            image_height=0,
            image_file_size=pdf_path.stat().st_size,
            is_blank_or_nearly_blank="unknown",
            requested_page_index=requested_page,
            resolved_page_index=resolved_index,
            source_file_type=source_type,
            source_file_exists="true",
        )

    if img_path and img_path.exists():
        image_pdf = _image_to_pdf(img_path, cache_key)
        width = height = 0
        if Image is not None:
            with Image.open(img_path) as im:
                width, height = im.size
        return ResolvedInput(
            ocr_input_path=str(image_pdf),
            ocr_input_type="image_as_pdf",
            source_resolution_method="local_image_to_pdf",
            render_or_conversion_status="ok",
            render_or_conversion_warning="",
            image_width=width,
            image_height=height,
            image_file_size=img_path.stat().st_size,
            is_blank_or_nearly_blank=_is_blank_image(img_path),
            requested_page_index=requested_page,
            resolved_page_index=0,
            source_file_type=source_type,
            source_file_exists="true",
        )

    return ResolvedInput(
        ocr_input_path="",
        ocr_input_type="missing",
        source_resolution_method="unresolved",
        render_or_conversion_status="failed",
        render_or_conversion_warning="missing_source",
        image_width=0,
        image_height=0,
        image_file_size=0,
        is_blank_or_nearly_blank="unknown",
        requested_page_index=requested_page,
        resolved_page_index=requested_page,
        source_file_type=source_type,
        source_file_exists="false",
    )
