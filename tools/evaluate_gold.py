#!/usr/bin/env python3
"""Compare OCR pipeline output against gold (hand-typed) reference text.

Produces per-page CER/WER and summary statistics.

Usage:
    python tools/evaluate_gold.py \
        --ocr-csv reports/eval_gold/client_page_text.csv \
        --gold-csv data/gold_data/gold_pages.csv \
        --output reports/eval_gold/evaluation_results.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


def _normalize(text: str) -> str:
    """Collapse whitespace and strip for fair comparison."""
    return re.sub(r"\s+", " ", text).strip()


def _char_error_rate(reference: str, hypothesis: str) -> float:
    """Character Error Rate via edit distance."""
    ref = _normalize(reference)
    hyp = _normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0

    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            insert = dp[j] + 1
            delete = dp[j - 1] + 1
            replace = prev + cost
            prev = dp[j]
            dp[j] = min(insert, delete, replace)
    return dp[m] / n


def _word_error_rate(reference: str, hypothesis: str) -> float:
    """Word Error Rate via edit distance on word tokens."""
    ref_words = _normalize(reference).split()
    hyp_words = _normalize(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    n, m = len(ref_words), len(hyp_words)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            insert = dp[j] + 1
            delete = dp[j - 1] + 1
            replace = prev + cost
            prev = dp[j]
            dp[j] = min(insert, delete, replace)
    return dp[m] / n


def _normalize_key(key: str) -> str:
    """Normalize a page key for matching: lowercase, collapse non-alnum to _."""
    key = key.lower()
    key = re.sub(r"[^a-z0-9]+", "_", key)
    key = re.sub(r"_+", "_", key).strip("_")
    # Strip trailing page range duplicates like page_33_33 -> page_33
    key = re.sub(r"_page_(\d+)_\1$", r"_page_\1", key)
    return key


def _load_gold(gold_csv: str) -> dict[str, str]:
    """Load gold CSV into {normalized_key -> gold_text} mapping."""
    gold = {}
    with open(gold_csv, encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pdf_name = row.get("PDF LINK", "").strip()
            page = row.get("PAGE", "").strip()
            text = row.get("HANDTYPED", "").strip()
            if pdf_name and text:
                stem = Path(pdf_name).stem
                raw_key = f"{stem}_page_{page}"
                gold[_normalize_key(raw_key)] = text
    return gold


def _load_ocr(ocr_csv: str) -> dict[str, str]:
    """Load OCR output CSV into {normalized_key -> ocr_text} mapping."""
    ocr = {}
    with open(ocr_csv, encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pdf_name = row.get("pdf_name", "").strip()
            text = row.get("page_text", "").strip()
            if pdf_name:
                stem = Path(pdf_name).stem
                ocr[_normalize_key(stem)] = text
    return ocr


def main():
    parser = argparse.ArgumentParser(description="Evaluate OCR output against gold reference")
    parser.add_argument("--ocr-csv", required=True, help="Path to OCR client_page_text.csv")
    parser.add_argument("--gold-csv", required=True, help="Path to gold_pages.csv")
    parser.add_argument("--output", required=True, help="Path to write evaluation results CSV")
    args = parser.parse_args()

    gold = _load_gold(args.gold_csv)
    ocr = _load_ocr(args.ocr_csv)

    matched = sorted(set(gold.keys()) & set(ocr.keys()))
    if not matched:
        print("ERROR: No matching pages found between OCR output and gold data.", file=sys.stderr)
        print(f"  Gold keys (sample): {sorted(gold.keys())[:5]}", file=sys.stderr)
        print(f"  OCR keys (sample): {sorted(ocr.keys())[:5]}", file=sys.stderr)
        sys.exit(1)

    results = []
    total_cer, total_wer = 0.0, 0.0
    for key in matched:
        cer = _char_error_rate(gold[key], ocr[key])
        wer = _word_error_rate(gold[key], ocr[key])
        total_cer += cer
        total_wer += wer
        results.append({
            "page": key,
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "gold_chars": len(_normalize(gold[key])),
            "ocr_chars": len(_normalize(ocr[key])),
        })

    avg_cer = total_cer / len(matched)
    avg_wer = total_wer / len(matched)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["page", "cer", "wer", "gold_chars", "ocr_chars"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nEvaluation complete: {len(matched)} pages matched out of {len(gold)} gold / {len(ocr)} OCR")
    print(f"  Unmatched gold: {len(gold) - len(matched)}")
    print(f"  Unmatched OCR:  {len(ocr) - len(matched)}")
    print(f"\n  Average CER: {avg_cer:.4f} ({avg_cer*100:.1f}%)")
    print(f"  Average WER: {avg_wer:.4f} ({avg_wer*100:.1f}%)")
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
