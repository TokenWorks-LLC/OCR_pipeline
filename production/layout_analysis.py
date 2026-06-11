#!/usr/bin/env python3
"""Layout detection, region segmentation, and reading-order reconstruction.

This module is language-agnostic and intended for multilingual OCR pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import re
import unicodedata
from typing import Any

import fitz

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False

try:
    from PIL import Image, ImageDraw

    PIL_AVAILABLE = True
except ImportError:
    Image = None
    ImageDraw = None
    PIL_AVAILABLE = False


logger = logging.getLogger(__name__)


REGION_TYPES = {
    "title",
    "paragraph",
    "heading",
    "footnote",
    "table",
    "caption",
    "image",
    "header",
    "footer",
    "page_number",
    "marginal_note",
    "bibliography",
    "unknown",
}


DEFAULT_LAYOUT_ANALYSIS = {
    "header_band_ratio": 0.10,
    "footer_band_ratio": 0.12,
    "margin_band_ratio": 0.12,
    "column_gap_ratio": 0.13,
    "min_text_region_area_ratio": 0.0010,
    "min_image_region_area_ratio": 0.0080,
    "footnote_min_y_ratio": 0.72,
    "footnote_font_ratio": 0.88,
    "table_digit_ratio": 0.28,
    "table_min_tabbed_lines": 2,
    "max_region_count": 96,
}

_PAGE_NUMBER_RE = re.compile(r"^\(?\s*[ivxlcdm]+\s*\)?$|^\(?\s*\d+[a-z]?\s*\)?$", re.IGNORECASE)
_HEADING_RE = re.compile(r"^(chapter|section|appendix|part)\b", re.IGNORECASE)
_CAPTION_RE = re.compile(r"^(figure|fig\.|table|plate|chart|map)\b", re.IGNORECASE)
_BIB_RE = re.compile(r"^(references|bibliography|works cited|literature)\b", re.IGNORECASE)


def _normalize_text(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in (text or "").splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, (x1 - x0)) * max(0.0, (y1 - y0))


def _clip_bbox(
    bbox: tuple[float, float, float, float],
    width: float,
    height: float,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    return (
        max(0.0, min(float(x0), width)),
        max(0.0, min(float(y0), height)),
        max(0.0, min(float(x1), width)),
        max(0.0, min(float(y1), height)),
    )


def _has_rtl_chars(text: str) -> bool:
    rtl_count = 0
    visible_count = 0
    for ch in text:
        if not ch.strip():
            continue
        visible_count += 1
        if unicodedata.bidirectional(ch) in {"R", "AL", "AN"}:
            rtl_count += 1
    if visible_count == 0:
        return False
    return (rtl_count / visible_count) >= 0.25


@dataclass
class LayoutLine:
    text: str
    bbox: tuple[float, float, float, float] | None = None
    reading_order: int = 0
    confidence: float = 0.0
    ordering_source: str = "engine_provided"

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "bbox": list(self.bbox) if self.bbox else None,
            "reading_order": int(self.reading_order),
            "confidence": round(float(self.confidence), 6),
            "ordering_source": self.ordering_source,
        }


@dataclass
class LayoutRegion:
    region_id: str
    type: str
    bbox: tuple[float, float, float, float]
    reading_order: int = 0
    confidence: float = 0.0
    ordering_source: str = "inferred"
    text: str = ""
    line_reading_confidence: float = 0.0
    source_index: int = -1
    lines: list[LayoutLine] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "type": self.type,
            "bbox": [round(float(value), 3) for value in self.bbox],
            "reading_order": int(self.reading_order),
            "confidence": round(float(self.confidence), 6),
            "ordering_source": self.ordering_source,
            "line_reading_confidence": round(float(self.line_reading_confidence), 6),
            "text": self.text,
            "line_ordering": [line.to_dict() for line in self.lines],
        }


@dataclass
class LayoutPageResult:
    page: int
    page_size: tuple[float, float]
    regions: list[LayoutRegion]
    column_count: int
    column_mode: str
    has_footnotes: bool
    has_table_interruptions: bool
    reading_order_confidence: float
    ordering_source: str
    text_direction: str

    def to_structured_output(self) -> dict[str, Any]:
        return {
            "page": int(self.page),
            "column_count": int(self.column_count),
            "column_mode": self.column_mode,
            "has_footnotes": bool(self.has_footnotes),
            "has_table_interruptions": bool(self.has_table_interruptions),
            "reading_order_confidence": round(float(self.reading_order_confidence), 6),
            "ordering_source": self.ordering_source,
            "text_direction": self.text_direction,
            "regions": [region.to_dict() for region in sorted(self.regions, key=lambda r: r.reading_order)],
        }


class LayoutAnalyzer:
    """Detect regions, columns, and reading order on heterogeneous pages."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = {**DEFAULT_LAYOUT_ANALYSIS, **(config or {})}

    def analyze_page(
        self,
        page: fitz.Page,
        page_number: int,
        language_hint: str = "unknown",
    ) -> LayoutPageResult:
        width = float(page.rect.width)
        height = float(page.rect.height)
        raw_regions = self._extract_raw_regions(page, width, height)

        text_sample = "\n".join(item.get("text", "") for item in raw_regions if item.get("kind") == "text")
        direction = self._detect_direction(text_sample, language_hint)

        regions = self._build_layout_regions(raw_regions, width, height, page_number)
        if not regions:
            regions = self._image_only_regions(page, width, height, page_number)

        column_count, column_mode, column_confidence = self._detect_columns(regions, width, direction)
        has_footnotes = any(region.type == "footnote" for region in regions)
        has_table_interruptions = self._has_table_interruptions(regions)

        reading_order_confidence, ordering_source = self._assign_reading_order(
            regions,
            column_count,
            direction,
            column_confidence,
        )

        return LayoutPageResult(
            page=int(page_number),
            page_size=(width, height),
            regions=regions,
            column_count=int(column_count),
            column_mode=column_mode,
            has_footnotes=bool(has_footnotes),
            has_table_interruptions=bool(has_table_interruptions),
            reading_order_confidence=float(reading_order_confidence),
            ordering_source=ordering_source,
            text_direction=direction,
        )

    def is_complex_layout(self, result: LayoutPageResult) -> bool:
        if result.column_count > 1 or result.has_footnotes or result.has_table_interruptions:
            return True
        complex_types = {"table", "caption", "marginal_note", "image", "bibliography"}
        return any(region.type in complex_types for region in result.regions)

    def regions_for_ocr(self, result: LayoutPageResult) -> list[LayoutRegion]:
        excluded = {"header", "footer", "page_number"}
        selected = [region for region in sorted(result.regions, key=lambda item: item.reading_order) if region.type not in excluded]
        if selected:
            return selected

        fallback_regions = [region for region in result.regions if region.type != "image"]
        if fallback_regions:
            return sorted(fallback_regions, key=lambda item: item.reading_order)
        return sorted(result.regions, key=lambda item: item.reading_order)

    def save_debug_artifacts(
        self,
        pixmap: fitz.Pixmap,
        result: LayoutPageResult,
        output_dir: str,
        prefix: str,
        region_outputs: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        if not PIL_AVAILABLE:
            return

        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", prefix).strip("_") or "page"

        if pixmap.alpha:
            pixmap = fitz.Pixmap(fitz.csRGB, pixmap)
        image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)

        scale_x = float(image.width) / max(result.page_size[0], 1.0)
        scale_y = float(image.height) / max(result.page_size[1], 1.0)

        region_view = image.copy()
        order_view = image.copy()
        draw_regions = ImageDraw.Draw(region_view)
        draw_order = ImageDraw.Draw(order_view)

        centers: list[tuple[float, float]] = []
        ordered_regions = sorted(result.regions, key=lambda item: item.reading_order)
        for region in ordered_regions:
            x0, y0, x1, y1 = region.bbox
            box = (
                int(round(x0 * scale_x)),
                int(round(y0 * scale_y)),
                int(round(x1 * scale_x)),
                int(round(y1 * scale_y)),
            )
            color = self._region_color(region.type)
            label = f"{region.reading_order}:{region.type}"
            draw_regions.rectangle(box, outline=color, width=3)
            draw_regions.text((box[0] + 2, max(0, box[1] - 12)), label, fill=color)

            draw_order.rectangle(box, outline=color, width=2)
            draw_order.text((box[0] + 2, max(0, box[1] - 12)), str(region.reading_order), fill=color)
            centers.append(((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0))

            region_crop = image.crop(box)
            crop_name = f"{safe_prefix}_{region.region_id}_{region.type}.png"
            region_crop.save(target_dir / crop_name)

            if region_outputs and region.region_id in region_outputs:
                payload = region_outputs[region.region_id]
                with (target_dir / f"{safe_prefix}_{region.region_id}_ocr.json").open("w", encoding="utf-8") as handle:
                    json_payload = {
                        "region": region.to_dict(),
                        "ocr": payload,
                    }
                    json.dump(json_payload, handle, ensure_ascii=False, indent=2)

        for index in range(len(centers) - 1):
            draw_order.line([centers[index], centers[index + 1]], fill=(255, 64, 64), width=3)

        region_view.save(target_dir / f"{safe_prefix}_regions.png")
        order_view.save(target_dir / f"{safe_prefix}_reading_order.png")

    def _extract_raw_regions(self, page: fitz.Page, width: float, height: float) -> list[dict[str, Any]]:
        raw: list[dict[str, Any]] = []
        page_area = max(width * height, 1.0)

        try:
            text_dict = page.get_text("dict")
        except Exception:
            text_dict = {"blocks": []}

        for index, block in enumerate(text_dict.get("blocks", [])):
            bbox_raw = block.get("bbox")
            if not bbox_raw or len(bbox_raw) != 4:
                continue
            bbox = _clip_bbox((bbox_raw[0], bbox_raw[1], bbox_raw[2], bbox_raw[3]), width, height)
            area_ratio = _bbox_area(bbox) / page_area
            if area_ratio < float(self.config["min_text_region_area_ratio"]):
                continue

            block_type = int(block.get("type", 0))
            if block_type == 1:
                raw.append(
                    {
                        "kind": "image",
                        "bbox": bbox,
                        "text": "",
                        "line_count": 0,
                        "word_count": 0,
                        "avg_font_size": 0.0,
                        "source_index": index,
                    }
                )
                continue

            lines: list[LayoutLine] = []
            fonts: list[float] = []
            texts: list[str] = []
            for line in block.get("lines", []):
                line_bbox_raw = line.get("bbox")
                line_bbox = None
                if line_bbox_raw and len(line_bbox_raw) == 4:
                    line_bbox = _clip_bbox(
                        (line_bbox_raw[0], line_bbox_raw[1], line_bbox_raw[2], line_bbox_raw[3]),
                        width,
                        height,
                    )

                spans = line.get("spans", [])
                span_text = "".join(str(span.get("text", "")) for span in spans)
                normalized_span_text = _normalize_text(span_text)
                if normalized_span_text:
                    texts.append(normalized_span_text)
                    lines.append(LayoutLine(text=normalized_span_text, bbox=line_bbox, ordering_source="engine_provided"))

                for span in spans:
                    try:
                        fonts.append(float(span.get("size", 0.0)))
                    except Exception:
                        continue

            text = _normalize_text("\n".join(texts))
            if not text and not lines:
                continue

            words = [token for token in re.split(r"\s+", text) if token]
            raw.append(
                {
                    "kind": "text",
                    "bbox": bbox,
                    "text": text,
                    "line_count": max(len(lines), 1),
                    "word_count": len(words),
                    "avg_font_size": (sum(fonts) / len(fonts)) if fonts else 0.0,
                    "lines": lines,
                    "source_index": index,
                }
            )

        return raw[: int(self.config["max_region_count"])]

    def _build_layout_regions(
        self,
        raw_regions: list[dict[str, Any]],
        width: float,
        height: float,
        page_number: int,
    ) -> list[LayoutRegion]:
        text_font_values = [float(item.get("avg_font_size", 0.0)) for item in raw_regions if item.get("kind") == "text"]
        median_font = self._median(text_font_values) if text_font_values else 0.0

        regions: list[LayoutRegion] = []
        for idx, raw in enumerate(raw_regions, start=1):
            bbox = raw["bbox"]
            kind = raw.get("kind")
            text = str(raw.get("text", "") or "")
            region_type = "unknown"

            if kind == "image":
                region_type = "image"
            elif kind == "text":
                region_type = self._classify_text_region(
                    text=text,
                    bbox=bbox,
                    width=width,
                    height=height,
                    avg_font=float(raw.get("avg_font_size", 0.0)),
                    median_font=median_font,
                    word_count=int(raw.get("word_count", 0)),
                    line_count=int(raw.get("line_count", 0)),
                )

            if region_type not in REGION_TYPES:
                region_type = "unknown"

            confidence = self._region_confidence(region_type, text, bbox, width, height)
            region = LayoutRegion(
                region_id=f"p{int(page_number)}_r{idx}",
                type=region_type,
                bbox=bbox,
                confidence=confidence,
                text=text,
                source_index=int(raw.get("source_index", idx - 1)),
                lines=list(raw.get("lines", [])),
            )
            regions.append(region)

        return regions

    def _image_only_regions(
        self,
        page: fitz.Page,
        width: float,
        height: float,
        page_number: int,
    ) -> list[LayoutRegion]:
        if not (NUMPY_AVAILABLE and CV2_AVAILABLE):
            return [
                LayoutRegion(
                    region_id=f"p{int(page_number)}_r1",
                    type="unknown",
                    bbox=(0.0, 0.0, width, height),
                    confidence=0.25,
                    text="",
                    source_index=0,
                )
            ]

        try:
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
            array = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
            gray = cv2.cvtColor(array[:, :, :3], cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            component_count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        except Exception:
            return [
                LayoutRegion(
                    region_id=f"p{int(page_number)}_r1",
                    type="unknown",
                    bbox=(0.0, 0.0, width, height),
                    confidence=0.20,
                    text="",
                    source_index=0,
                )
            ]

        min_area = max(64, int(0.0015 * binary.shape[0] * binary.shape[1]))
        scale_x = width / max(binary.shape[1], 1)
        scale_y = height / max(binary.shape[0], 1)

        regions: list[LayoutRegion] = []
        for index in range(1, component_count):
            x, y, w, h, area = stats[index]
            if area < min_area:
                continue
            bbox = _clip_bbox((x * scale_x, y * scale_y, (x + w) * scale_x, (y + h) * scale_y), width, height)
            regions.append(
                LayoutRegion(
                    region_id=f"p{int(page_number)}_r{len(regions) + 1}",
                    type="unknown",
                    bbox=bbox,
                    confidence=0.20,
                    text="",
                    source_index=index,
                )
            )

        if not regions:
            regions = [
                LayoutRegion(
                    region_id=f"p{int(page_number)}_r1",
                    type="unknown",
                    bbox=(0.0, 0.0, width, height),
                    confidence=0.20,
                    text="",
                    source_index=0,
                )
            ]

        return regions[: int(self.config["max_region_count"])]

    def _classify_text_region(
        self,
        text: str,
        bbox: tuple[float, float, float, float],
        width: float,
        height: float,
        avg_font: float,
        median_font: float,
        word_count: int,
        line_count: int,
    ) -> str:
        x0, y0, x1, y1 = bbox
        center_x = (x0 + x1) / 2.0
        top_ratio = y0 / max(height, 1.0)
        bottom_ratio = y1 / max(height, 1.0)
        left_margin = x0 <= (float(self.config["margin_band_ratio"]) * width)
        right_margin = x1 >= ((1.0 - float(self.config["margin_band_ratio"])) * width)

        compact_text = _normalize_text(text)
        lower = compact_text.casefold()
        words = [token for token in re.split(r"\s+", compact_text) if token]
        digit_ratio = (sum(ch.isdigit() for ch in compact_text) / max(len(compact_text), 1)) if compact_text else 0.0

        if compact_text and _PAGE_NUMBER_RE.fullmatch(compact_text) and abs(center_x - (width / 2.0)) <= (0.25 * width):
            if top_ratio <= 0.18 or bottom_ratio >= 0.82:
                return "page_number"

        if top_ratio <= float(self.config["header_band_ratio"]):
            if word_count <= 14 or compact_text.isupper():
                return "header"

        if bottom_ratio >= (1.0 - float(self.config["footer_band_ratio"])):
            if _PAGE_NUMBER_RE.fullmatch(compact_text):
                return "page_number"
            if word_count <= 16:
                if re.match(r"^\s*\d+\s*[).-]", compact_text) or avg_font <= max(median_font * 0.9, 6.0):
                    return "footnote"
                return "footer"

        if _BIB_RE.match(compact_text):
            return "bibliography"
        if "doi" in lower or "isbn" in lower:
            return "bibliography"

        if _CAPTION_RE.match(compact_text):
            return "caption"

        if line_count <= 3 and (avg_font >= max(median_font * 1.30, 12.0)) and top_ratio <= 0.35 and word_count <= 20:
            return "title"

        if line_count <= 3 and (avg_font >= max(median_font * 1.12, 9.0) or _HEADING_RE.match(compact_text) or compact_text.isupper()):
            return "heading"

        if self._looks_like_table_text(compact_text, digit_ratio):
            return "table"

        if bottom_ratio >= float(self.config["footnote_min_y_ratio"]) and (
            re.match(r"^\s*(\d+|\*|†|‡)\s*[).-]?", compact_text)
            or avg_font <= max(median_font * float(self.config["footnote_font_ratio"]), 6.0)
        ):
            return "footnote"

        if (left_margin or right_margin) and (x1 - x0) <= (0.30 * width) and word_count <= 40:
            return "marginal_note"

        if not compact_text:
            return "unknown"
        return "paragraph"

    def _looks_like_table_text(self, text: str, digit_ratio: float) -> bool:
        if not text:
            return False

        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) < 2:
            return False

        tabbed_lines = 0
        for line in lines:
            if len(re.findall(r"\S+\s{2,}\S+", line)) > 0:
                tabbed_lines += 1

        if tabbed_lines >= int(self.config["table_min_tabbed_lines"]):
            return True

        separators = text.count("|") + text.count("\t")
        if digit_ratio >= float(self.config["table_digit_ratio"]) and len(lines) >= 3:
            return True
        return digit_ratio >= float(self.config["table_digit_ratio"]) and (separators > 2 or tabbed_lines > 0)

    def _detect_columns(
        self,
        regions: list[LayoutRegion],
        page_width: float,
        direction: str,
    ) -> tuple[int, str, float]:
        candidates = [
            region for region in regions
            if region.type in {"paragraph", "heading", "bibliography", "table", "caption", "footnote", "unknown"}
        ]
        if len(candidates) < 2:
            return 1, "single_column", 0.35

        centers = sorted(((region.bbox[0] + region.bbox[2]) / 2.0) for region in candidates)
        if len(centers) < 2:
            return 1, "single_column", 0.35

        gaps = [centers[idx + 1] - centers[idx] for idx in range(len(centers) - 1)]
        min_gap = max(26.0, float(self.config["column_gap_ratio"]) * page_width)
        significant_gaps = [gap for gap in gaps if gap >= min_gap]
        if not significant_gaps:
            return 1, "single_column", 0.42

        count = min(1 + len(significant_gaps), 4)
        if count <= 1:
            mode = "single_column"
        elif count == 2:
            mode = "two_column"
        else:
            mode = "multi_column"

        confidence = min(max((max(significant_gaps) / max(page_width, 1.0)) * 2.2, 0.45), 0.96)
        if direction == "rtl":
            confidence = min(confidence + 0.01, 0.97)
        return count, mode, confidence

    def _assign_reading_order(
        self,
        regions: list[LayoutRegion],
        column_count: int,
        direction: str,
        column_confidence: float,
    ) -> tuple[float, str]:
        if not regions:
            return 0.0, "inferred"

        headers = [region for region in regions if region.type in {"header", "title"}]
        footnotes = [region for region in regions if region.type == "footnote"]
        footers = [region for region in regions if region.type in {"footer", "page_number"}]
        body = [
            region
            for region in regions
            if region.type not in {"header", "title", "footnote", "footer", "page_number"}
        ]

        ordered: list[LayoutRegion] = []
        ordered.extend(self._sort_regions_flow(headers, max(1, min(column_count, 2)), direction))
        ordered.extend(self._sort_regions_flow(body, max(column_count, 1), direction))
        ordered.extend(self._sort_regions_flow(footnotes, 1, direction))
        ordered.extend(self._sort_regions_flow(footers, 1, direction))

        for index, region in enumerate(ordered, start=1):
            region.reading_order = index
            region.ordering_source = "inferred"
            self._assign_line_order(region, direction)

        if column_count <= 1:
            source_index_order = [region.source_index for region in ordered]
            if source_index_order == sorted(source_index_order):
                for region in ordered:
                    region.ordering_source = "engine_provided"

        overlap_penalty = min(self._region_overlap_ratio(ordered) * 0.5, 0.35)
        unknown_penalty = min(sum(1 for region in ordered if region.type == "unknown") / max(len(ordered), 1) * 0.3, 0.25)
        confidence = max(0.1, min(0.98, 0.62 + (column_confidence * 0.32) - overlap_penalty - unknown_penalty))
        return confidence, "inferred"

    def _sort_regions_flow(
        self,
        regions: list[LayoutRegion],
        column_count: int,
        direction: str,
    ) -> list[LayoutRegion]:
        if not regions:
            return []

        if column_count <= 1 or len(regions) <= 2:
            if direction == "rtl":
                return sorted(regions, key=lambda region: (region.bbox[1], -region.bbox[2], region.bbox[0]))
            return sorted(regions, key=lambda region: (region.bbox[1], region.bbox[0]))

        centers = sorted(((region.bbox[0] + region.bbox[2]) / 2.0) for region in regions)
        gaps = []
        for idx in range(len(centers) - 1):
            gaps.append((centers[idx + 1] - centers[idx], idx))
        gaps = sorted(gaps, key=lambda item: item[0], reverse=True)

        boundaries: list[float] = []
        for _, idx in gaps[: max(0, column_count - 1)]:
            boundaries.append((centers[idx] + centers[idx + 1]) / 2.0)
        boundaries = sorted(boundaries)

        columns: dict[int, list[LayoutRegion]] = {}
        for region in regions:
            center = (region.bbox[0] + region.bbox[2]) / 2.0
            col = 0
            for boundary in boundaries:
                if center > boundary:
                    col += 1
            columns.setdefault(col, []).append(region)

        ordered_cols = sorted(columns.keys(), reverse=(direction == "rtl"))
        ordered: list[LayoutRegion] = []
        for col in ordered_cols:
            ordered.extend(sorted(columns[col], key=lambda region: (region.bbox[1], region.bbox[0])))
        return ordered

    def _assign_line_order(self, region: LayoutRegion, direction: str) -> None:
        if not region.lines:
            region.line_reading_confidence = 0.0
            return

        if direction == "rtl":
            sorted_lines = sorted(
                region.lines,
                key=lambda line: ((line.bbox[1] if line.bbox else 0.0), -(line.bbox[2] if line.bbox else 0.0)),
            )
        else:
            sorted_lines = sorted(
                region.lines,
                key=lambda line: ((line.bbox[1] if line.bbox else 0.0), (line.bbox[0] if line.bbox else 0.0)),
            )

        same_order = [id(item) for item in sorted_lines] == [id(item) for item in region.lines]
        source = "engine_provided" if same_order else "inferred"
        confidence = 0.92 if same_order else 0.76
        for index, line in enumerate(sorted_lines, start=1):
            line.reading_order = index
            line.ordering_source = source
            line.confidence = confidence
        region.lines = sorted_lines
        region.line_reading_confidence = confidence

    def _has_table_interruptions(self, regions: list[LayoutRegion]) -> bool:
        tables = [region for region in regions if region.type == "table"]
        paragraphs = [region for region in regions if region.type in {"paragraph", "heading", "bibliography"}]
        if not tables or len(paragraphs) < 2:
            return False

        paragraph_y = sorted(((region.bbox[1] + region.bbox[3]) / 2.0) for region in paragraphs)
        min_p = paragraph_y[0]
        max_p = paragraph_y[-1]
        for table in tables:
            y = (table.bbox[1] + table.bbox[3]) / 2.0
            if min_p < y < max_p:
                return True
        return False

    @staticmethod
    def _median(values: list[float]) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        mid = len(ordered) // 2
        if len(ordered) % 2 == 1:
            return float(ordered[mid])
        return float((ordered[mid - 1] + ordered[mid]) / 2.0)

    def _region_confidence(
        self,
        region_type: str,
        text: str,
        bbox: tuple[float, float, float, float],
        width: float,
        height: float,
    ) -> float:
        area_ratio = _bbox_area(bbox) / max(width * height, 1.0)
        text_len = len(text.strip())
        base = {
            "title": 0.84,
            "paragraph": 0.80,
            "heading": 0.78,
            "footnote": 0.72,
            "table": 0.70,
            "caption": 0.74,
            "image": 0.88,
            "header": 0.76,
            "footer": 0.74,
            "page_number": 0.90,
            "marginal_note": 0.64,
            "bibliography": 0.76,
            "unknown": 0.50,
        }.get(region_type, 0.5)

        if region_type != "image":
            if text_len < 4:
                base -= 0.14
            elif text_len > 120:
                base += 0.04

        if area_ratio < 0.002:
            base -= 0.07
        elif area_ratio > 0.20:
            base += 0.03

        return max(0.15, min(base, 0.98))

    @staticmethod
    def _detect_direction(text_sample: str, language_hint: str) -> str:
        hint = (language_hint or "unknown").strip().lower()
        if any(token in hint for token in ("arabic", "hebrew", "persian", "urdu", "rtl")):
            return "rtl"
        if _has_rtl_chars(text_sample):
            return "rtl"
        return "ltr"

    @staticmethod
    def _region_overlap_ratio(regions: list[LayoutRegion]) -> float:
        if len(regions) < 2:
            return 0.0

        overlap_acc = 0.0
        comparisons = 0
        for i, first in enumerate(regions):
            ax0, ay0, ax1, ay1 = first.bbox
            a_area = max((ax1 - ax0) * (ay1 - ay0), 1.0)
            for second in regions[i + 1 :]:
                bx0, by0, bx1, by1 = second.bbox
                inter_x0 = max(ax0, bx0)
                inter_y0 = max(ay0, by0)
                inter_x1 = min(ax1, bx1)
                inter_y1 = min(ay1, by1)
                if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
                    continue
                inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
                overlap_acc += inter_area / a_area
                comparisons += 1

        if comparisons == 0:
            return 0.0
        return overlap_acc / comparisons

    @staticmethod
    def _region_color(region_type: str) -> tuple[int, int, int]:
        palette = {
            "title": (219, 68, 55),
            "paragraph": (52, 168, 83),
            "heading": (66, 133, 244),
            "footnote": (251, 188, 4),
            "table": (244, 180, 0),
            "caption": (0, 172, 193),
            "image": (121, 85, 72),
            "header": (142, 36, 170),
            "footer": (94, 53, 177),
            "page_number": (2, 136, 209),
            "marginal_note": (0, 151, 167),
            "bibliography": (67, 160, 71),
            "unknown": (120, 120, 120),
        }
        return palette.get(region_type, (120, 120, 120))
