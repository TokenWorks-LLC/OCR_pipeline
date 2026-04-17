#!/usr/bin/env python3
"""Per-word bounding-box extraction with confidence scores using PaddleOCR.

Adapted from Vedashnii's Detectron/Tesseract notebook to use PaddleOCR
(the only reliable engine in the pipeline so far). Produces:

  1. Annotated PNG per page (green boxes + confidence labels)
  2. predictions.csv  (page, x0, y0, x1, y1, predicted_text, confidence)
  3. Optional evaluation against gold data (word-level CER/WER + confidence stats)

Usage:
    # Basic extraction
    python tools/word_confidence.py --pdf "data/input_pdfs/AKT 7a.pdf"

    # With gold-data evaluation
    python tools/word_confidence.py \\
        --pdf "data/input_pdfs/AKT 7a.pdf" \\
        --gold-csv data/gold_data/gold_pages.csv \
        --output reports/word_confidence

    # Specific pages only
    python tools/word_confidence.py \\
        --pdf "data/input_pdfs/AKT 7a.pdf" \\
        --pages 1-5
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from statistics import mean, median

import fitz
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# PaddleOCR initialisation (version-aware, mirrors ensemble_ocr.py logic)
# ---------------------------------------------------------------------------

_paddle_engine = None
_is_v3 = False


def _get_paddle():
    global _paddle_engine, _is_v3
    if _paddle_engine is not None:
        return _paddle_engine, _is_v3

    from paddleocr import PaddleOCR
    try:
        import paddleocr
        _is_v3 = int(paddleocr.__version__.split(".")[0]) >= 3
    except Exception:
        _is_v3 = False

    if _is_v3:
        _paddle_engine = PaddleOCR(lang="en")
    else:
        _paddle_engine = PaddleOCR(lang="en", use_textline_orientation=True)
    return _paddle_engine, _is_v3


# ---------------------------------------------------------------------------
# Bounding-box helpers (from ensemble_ocr.py)
# ---------------------------------------------------------------------------

def _bbox_from_polygon(points):
    """Convert polygon [[x,y], ...] -> (x0, y0, x1, y1)."""
    try:
        if points is None:
            return None
        arr = np.asarray(points, dtype=float)
        if arr.size == 0:
            return None
        xs = arr[:, 0]
        ys = arr[:, 1]
        return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
    except Exception:
        return None


def _split_line_into_words(text, bbox, confidence):
    """Split a line-level OCR result into per-word entries.

    Estimates each word's bounding box proportionally by character count
    within the line bbox.  Not pixel-perfect, but good enough for
    evaluation and visualisation.
    """
    words = text.split()
    if not words or bbox is None:
        return [{"text": text, "bbox": bbox, "confidence": confidence}] if text.strip() else []

    x0, y0, x1, y1 = bbox
    total_chars = sum(len(w) for w in words)
    if total_chars == 0:
        return [{"text": text, "bbox": bbox, "confidence": confidence}]

    results = []
    cursor_x = x0
    line_width = x1 - x0
    for word in words:
        word_width = (len(word) / total_chars) * line_width
        word_bbox = (cursor_x, y0, cursor_x + word_width, y1)
        results.append({"text": word, "bbox": word_bbox, "confidence": confidence})
        cursor_x += word_width

    return results


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_words_from_page(page_img_np):
    """Run PaddleOCR on a numpy image and return per-word results.

    Returns list of dicts: {text, bbox (x0,y0,x1,y1), confidence}
    """
    engine, is_v3 = _get_paddle()

    if is_v3 and hasattr(engine, "predict"):
        return _extract_v3(engine, page_img_np)
    return _extract_v2(engine, page_img_np)


def _extract_v2(engine, image):
    result = engine.ocr(image, cls=True)
    words = []
    if result and result[0]:
        for item in result[0]:
            if item and len(item) >= 2:
                text = str(item[1][0]).strip()
                if not text:
                    continue
                try:
                    confidence = float(item[1][1])
                except Exception:
                    confidence = 0.0
                bbox = _bbox_from_polygon(item[0])
                words.extend(_split_line_into_words(text, bbox, confidence))
    return words


def _extract_v3(engine, image):
    results = engine.predict(image)
    words = []
    for result in results:
        rec_texts = getattr(result, "rec_texts", None)
        if rec_texts is None and isinstance(result, dict):
            rec_texts = result.get("rec_texts", [])
        rec_texts = rec_texts or []

        rec_scores = getattr(result, "rec_scores", None)
        if rec_scores is None and isinstance(result, dict):
            rec_scores = result.get("rec_scores", [])
        rec_scores = rec_scores or []

        dt_polys = getattr(result, "dt_polys", None)
        if dt_polys is None and isinstance(result, dict):
            dt_polys = result.get("dt_polys", [])
        dt_polys = dt_polys or []

        for i, txt in enumerate(rec_texts):
            text = str(txt).strip()
            if not text:
                continue
            confidence = 0.0
            if i < len(rec_scores):
                try:
                    confidence = float(rec_scores[i])
                except Exception:
                    pass
            bbox = _bbox_from_polygon(dt_polys[i]) if i < len(dt_polys) else None
            words.extend(_split_line_into_words(text, bbox, confidence))
    return words


# ---------------------------------------------------------------------------
# Render PDF page -> numpy at 2x resolution (same as Vedashnii's notebook)
# ---------------------------------------------------------------------------

def render_page(doc, page_num, zoom=2):
    page = doc[page_num]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = img[:, :, :3]
    return img


# ---------------------------------------------------------------------------
# Annotated image output
# ---------------------------------------------------------------------------

def draw_annotations(page_img_np, word_results, min_conf_highlight=50):
    """Draw bounding boxes + confidence on the page image.

    Green = confident (>= min_conf_highlight), Red = low confidence.
    Returns a PIL Image.
    """
    pil_img = Image.fromarray(page_img_np).convert("RGB")
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except Exception:
            font = ImageFont.load_default()

    for w in word_results:
        bbox = w["bbox"]
        if bbox is None:
            continue
        x0, y0, x1, y1 = [int(v) for v in bbox]
        conf = w["confidence"]
        conf_pct = int(conf * 100)

        color = (0, 200, 0) if conf_pct >= min_conf_highlight else (220, 40, 40)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)

        label = f"{conf_pct}%"
        draw.text((x0, max(y0 - 16, 0)), label, fill=color, font=font)

    return pil_img


# ---------------------------------------------------------------------------
# Gold-data evaluation (word-level CER/WER + confidence analysis)
# ---------------------------------------------------------------------------

def _normalize(text):
    return re.sub(r"\s+", " ", text).strip()


def _edit_distance(a, b):
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            ins = dp[j] + 1
            delete = dp[j - 1] + 1
            replace = prev + cost
            prev = dp[j]
            dp[j] = min(ins, delete, replace)
    return dp[m]


def _cer(ref, hyp):
    ref, hyp = _normalize(ref), _normalize(hyp)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)


def _wer(ref, hyp):
    ref_w, hyp_w = _normalize(ref).split(), _normalize(hyp).split()
    if not ref_w:
        return 0.0 if not hyp_w else 1.0
    return _edit_distance(ref_w, hyp_w) / len(ref_w)


def load_gold(gold_csv):
    """Load gold CSV -> {normalized_key: gold_text}."""
    gold = {}
    with open(gold_csv, encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pdf_name = row.get("PDF LINK", "").strip()
            page = row.get("PAGE", "").strip()
            text = row.get("HANDTYPED", "").strip()
            if pdf_name and text:
                stem = Path(pdf_name).stem
                key = _normalize_key(f"{stem}_page_{page}")
                gold[key] = text
    return gold


def _normalize_key(key):
    key = key.lower()
    key = re.sub(r"[^a-z0-9]+", "_", key)
    key = re.sub(r"_+", "_", key).strip("_")
    key = re.sub(r"_page_(\d+)_\1$", r"_page_\1", key)
    return key


def evaluate_page(word_results, gold_text):
    """Evaluate OCR words against gold text for a single page.

    Returns dict with CER, WER, confidence stats, and low-confidence words.
    """
    ocr_text = " ".join(w["text"] for w in word_results)
    cer_val = _cer(gold_text, ocr_text)
    wer_val = _wer(gold_text, ocr_text)

    confidences = [w["confidence"] for w in word_results if w["confidence"] > 0]
    low_conf = [w for w in word_results if w["confidence"] < 0.5]

    return {
        "cer": round(cer_val, 4),
        "wer": round(wer_val, 4),
        "ocr_text": ocr_text,
        "gold_chars": len(_normalize(gold_text)),
        "ocr_chars": len(_normalize(ocr_text)),
        "word_count": len(word_results),
        "mean_confidence": round(mean(confidences), 4) if confidences else 0.0,
        "median_confidence": round(median(confidences), 4) if confidences else 0.0,
        "low_confidence_count": len(low_conf),
        "low_confidence_words": [(w["text"], round(w["confidence"], 3)) for w in low_conf[:20]],
    }


# ---------------------------------------------------------------------------
# Page range parsing
# ---------------------------------------------------------------------------

def parse_pages(pages_str, total_pages):
    """Parse '1-5,8,10-12' into a sorted list of 0-based page indices."""
    if not pages_str:
        return list(range(total_pages))
    indices = set()
    for part in pages_str.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            lo, hi = int(lo), int(hi)
            for p in range(lo, hi + 1):
                if 1 <= p <= total_pages:
                    indices.add(p - 1)
        else:
            p = int(part)
            if 1 <= p <= total_pages:
                indices.add(p - 1)
    return sorted(indices)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Per-word bounding-box extraction and confidence scoring with PaddleOCR"
    )
    parser.add_argument("--pdf", required=True, help="Path to input PDF")
    parser.add_argument("--pages", default=None, help="Page range, e.g. '1-5,8' (1-based). Default: all pages")
    parser.add_argument("--gold-csv", default=None, help="Path to gold_pages.csv for evaluation")
    parser.add_argument("--output", default="reports/word_confidence", help="Output directory")
    parser.add_argument("--zoom", type=int, default=2, help="Render zoom factor (default: 2)")
    parser.add_argument("--no-images", action="store_true", help="Skip annotated image output")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "annotated"
    if not args.no_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    page_indices = parse_pages(args.pages, len(doc))
    print(f"Processing {pdf_path.name}: {len(page_indices)} page(s) at {args.zoom}x zoom")

    # Load gold data if provided
    gold = {}
    if args.gold_csv:
        gold = load_gold(args.gold_csv)
        print(f"Loaded {len(gold)} gold entries")

    all_words = []
    eval_results = []
    pdf_stem = pdf_path.stem

    for idx in page_indices:
        page_num = idx + 1  # 1-based for display
        print(f"  Page {page_num}/{len(doc)}...", end=" ", flush=True)

        page_img = render_page(doc, idx, zoom=args.zoom)
        words = extract_words_from_page(page_img)
        print(f"{len(words)} words found", end="")

        # Store words with page number
        for w in words:
            row = {
                "page": page_num,
                "x0": round(w["bbox"][0], 1) if w["bbox"] else "",
                "y0": round(w["bbox"][1], 1) if w["bbox"] else "",
                "x1": round(w["bbox"][2], 1) if w["bbox"] else "",
                "y1": round(w["bbox"][3], 1) if w["bbox"] else "",
                "predicted_text": w["text"],
                "confidence": round(w["confidence"], 4),
            }
            all_words.append(row)

        # Draw annotated image
        if not args.no_images and words:
            annotated = draw_annotations(page_img, words)
            img_path = images_dir / f"{pdf_stem}_page_{page_num}.png"
            annotated.save(str(img_path))

        # Evaluate against gold data if available
        gold_key = _normalize_key(f"{pdf_stem}_page_{page_num}")
        if gold_key in gold:
            ev = evaluate_page(words, gold[gold_key])
            ev["page"] = page_num
            ev["gold_key"] = gold_key
            eval_results.append(ev)
            print(f"  | CER={ev['cer']:.1%} WER={ev['wer']:.1%} conf={ev['mean_confidence']:.1%}", end="")

        print()

    doc.close()

    # Write predictions CSV
    pred_csv = output_dir / "predictions.csv"
    with open(pred_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["page", "x0", "y0", "x1", "y1", "predicted_text", "confidence"])
        writer.writeheader()
        writer.writerows(all_words)
    print(f"\nPredictions: {pred_csv} ({len(all_words)} words)")

    # Write evaluation results
    if eval_results:
        eval_csv = output_dir / "evaluation.csv"
        with open(eval_csv, "w", newline="", encoding="utf-8") as fh:
            fieldnames = ["page", "cer", "wer", "gold_chars", "ocr_chars",
                          "word_count", "mean_confidence", "median_confidence",
                          "low_confidence_count"]
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for ev in eval_results:
                writer.writerow({k: ev[k] for k in fieldnames})

        # Summary
        avg_cer = mean(ev["cer"] for ev in eval_results)
        avg_wer = mean(ev["wer"] for ev in eval_results)
        avg_conf = mean(ev["mean_confidence"] for ev in eval_results)
        total_low = sum(ev["low_confidence_count"] for ev in eval_results)

        print(f"Evaluation: {eval_csv} ({len(eval_results)} pages matched gold)")
        print(f"\n{'='*50}")
        print(f"  SUMMARY ({len(eval_results)} pages)")
        print(f"{'='*50}")
        print(f"  Avg CER:            {avg_cer:.4f} ({avg_cer*100:.1f}%)")
        print(f"  Avg WER:            {avg_wer:.4f} ({avg_wer*100:.1f}%)")
        print(f"  Avg Confidence:     {avg_conf:.4f} ({avg_conf*100:.1f}%)")
        print(f"  Low-conf words:     {total_low} (< 50% confidence)")
        print(f"{'='*50}")

        # Print worst pages
        worst = sorted(eval_results, key=lambda e: e["cer"], reverse=True)[:5]
        if worst:
            print(f"\n  Worst pages by CER:")
            for ev in worst:
                print(f"    Page {ev['page']:>3}: CER={ev['cer']:.1%}  WER={ev['wer']:.1%}  "
                      f"conf={ev['mean_confidence']:.1%}  words={ev['word_count']}")

        # Print low-confidence words across all pages
        all_low = []
        for ev in eval_results:
            for word, conf in ev["low_confidence_words"]:
                all_low.append((word, conf, ev["page"]))
        if all_low:
            all_low.sort(key=lambda x: x[1])
            print(f"\n  Lowest confidence words:")
            for word, conf, pg in all_low[:15]:
                print(f"    '{word}' (conf={conf:.1%}, page {pg})")

    if not args.no_images:
        print(f"\nAnnotated images: {images_dir}/")

    print("Done.")


if __name__ == "__main__":
    main()
