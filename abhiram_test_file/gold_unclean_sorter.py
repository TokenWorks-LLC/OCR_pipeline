#!/usr/bin/env python3
"""
Sort raw dataset rows into `gold` vs `unclean` buckets.

This is a local "LLM-style" scorer that uses the reference corpus in
`testing_data/` to learn what clean transliteration/translation text looks like.
It does not require network APIs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


TEXT_COLUMN_HINTS = (
    "text",
    "transliteration",
    "translation",
    "akkadian",
    "english",
    "french",
    "line",
    "content",
    "sentence",
)


def normalize_text(value: str) -> str:
    value = value.replace("\u00a0", " ").replace("\t", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def tokenize(value: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9ḫḪšŠṣṢṭṬāēīūĀĒĪŪ'’\-\.]+", value)


def transliteration_density(value: str) -> float:
    if not value:
        return 0.0
    tokens = tokenize(value)
    if not tokens:
        return 0.0
    translit_hits = 0
    for token in tokens:
        has_marker = (
            any(ch in token for ch in "ḫḪšŠṣṢṭṬāēīūĀĒĪŪ")
            or "." in token
            or token.isupper()
            or bool(re.search(r"\d", token))
            or token in {"KÙ.BABBAR", "AN.NA", "GIN", "GÍN", "TÚG.HI.A"}
        )
        if has_marker:
            translit_hits += 1
    return translit_hits / max(len(tokens), 1)


def noise_density(value: str) -> float:
    if not value:
        return 1.0
    weird = re.findall(r"[�]|[^\w\s\.,;:!?'\-\/\[\]\(\){}ḫḪšŠṣṢṭṬāēīūĀĒĪŪ]", value)
    gap_markers = re.findall(r"\[x\+?\]|\.\.\.|<unk>|<gap>", value, flags=re.IGNORECASE)
    return min(1.0, (len(weird) + 0.5 * len(gap_markers)) / max(len(value), 1))


def short_or_blank_penalty(value: str) -> float:
    s = value.strip()
    if not s:
        return 1.0
    if len(s) < 8:
        return 0.7
    if len(s.split()) < 2:
        return 0.4
    return 0.0


def text_ngrams(value: str, n: int = 3) -> List[str]:
    v = normalize_text(value).lower()
    if len(v) < n:
        return [v] if v else []
    return [v[i : i + n] for i in range(len(v) - n + 1)]


def choose_text_columns(fieldnames: Sequence[str]) -> List[str]:
    lowered = {name.lower(): name for name in fieldnames}
    picked = []
    for hint in TEXT_COLUMN_HINTS:
        for lower_name, original in lowered.items():
            if hint in lower_name and original not in picked:
                picked.append(original)
    if picked:
        return picked
    # Fall back to all non-id style columns.
    for name in fieldnames:
        lname = name.lower()
        if lname not in {"id", "doc_id", "document_id", "no", "page", "line_no"}:
            picked.append(name)
    return picked


def extract_rows_from_csv(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return rows
        text_columns = choose_text_columns(reader.fieldnames)
        for idx, row in enumerate(reader, start=1):
            parts = [normalize_text(row.get(col, "")) for col in text_columns]
            joined = " | ".join(p for p in parts if p)
            rows.append(
                {
                    "source_file": str(path.name),
                    "row_id": idx,
                    "text": joined,
                    "raw": row,
                }
            )
    return rows


def extract_rows_from_txt(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f, start=1):
            text = normalize_text(line)
            if not text:
                continue
            rows.append({"source_file": str(path.name), "row_id": idx, "text": text, "raw": {"line": text}})
    return rows


def extract_rows_from_path(path: Path) -> List[dict]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return extract_rows_from_csv(path)
    if suffix in {".txt", ".tsv"}:
        return extract_rows_from_txt(path)
    return []


def collect_reference_texts(reference_dir: Path) -> List[str]:
    refs: List[str] = []
    for file_path in sorted(reference_dir.glob("*")):
        if file_path.suffix.lower() not in {".txt", ".csv"}:
            continue
        for row in extract_rows_from_path(file_path):
            text = normalize_text(row["text"])
            if len(text) >= 8:
                refs.append(text)
    return refs


@dataclass
class ReferenceProfile:
    token_vocab: set
    ngram_vocab: set
    avg_length: float


def build_reference_profile(refs: Sequence[str]) -> ReferenceProfile:
    token_vocab = set()
    ngram_vocab = set()
    total_length = 0
    for ref in refs:
        total_length += len(ref)
        token_vocab.update(t.lower() for t in tokenize(ref))
        ngram_vocab.update(text_ngrams(ref, n=3))
    avg_length = total_length / max(len(refs), 1)
    return ReferenceProfile(token_vocab=token_vocab, ngram_vocab=ngram_vocab, avg_length=avg_length)


@dataclass
class ScoreBreakdown:
    score: float
    profile_similarity: float
    translit_density: float
    noise_density: float
    short_penalty: float


def profile_similarity(text: str, profile: ReferenceProfile) -> float:
    tokens = [t.lower() for t in tokenize(text)]
    ngrams = text_ngrams(text, n=3)
    if not tokens and not ngrams:
        return 0.0

    token_hit = 0.0
    if tokens:
        token_hit = sum(1 for t in tokens if t in profile.token_vocab) / len(tokens)

    ngram_hit = 0.0
    if ngrams:
        ngram_hit = sum(1 for ng in ngrams if ng in profile.ngram_vocab) / len(ngrams)

    return 0.45 * token_hit + 0.55 * ngram_hit


def score_text(text: str, profile: ReferenceProfile) -> ScoreBreakdown:
    sim = profile_similarity(text, profile)
    translit = transliteration_density(text)
    noise = noise_density(text)
    short_penalty = short_or_blank_penalty(text)

    # Weighted blend: bias toward reference similarity while still capturing
    # transliteration shape and obvious OCR noise.
    score = 0.62 * sim + 0.30 * translit - 0.45 * noise - 0.35 * short_penalty
    score = max(0.0, min(1.0, score))
    return ScoreBreakdown(
        score=score,
        profile_similarity=sim,
        translit_density=translit,
        noise_density=noise,
        short_penalty=short_penalty,
    )


def classify_rows(rows: Iterable[dict], profile: ReferenceProfile, threshold: float) -> List[dict]:
    out: List[dict] = []
    for row in rows:
        details = score_text(row["text"], profile)
        label = "gold" if details.score >= threshold else "unclean"
        enriched = dict(row)
        enriched.update(
            {
                "label": label,
                "score": round(details.score, 4),
                "profile_similarity": round(details.profile_similarity, 4),
                "translit_density": round(details.translit_density, 4),
                "noise_density": round(details.noise_density, 4),
                "short_penalty": round(details.short_penalty, 4),
            }
        )
        out.append(enriched)
    return out


def write_split_csv(rows: Sequence[dict], output_dir: Path, threshold: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_file",
        "row_id",
        "text",
        "label",
        "score",
        "profile_similarity",
        "translit_density",
        "noise_density",
        "short_penalty",
    ]

    for label in ("gold", "unclean"):
        out_path = output_dir / f"{label}.csv"
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                if row["label"] != label:
                    continue
                writer.writerow({k: row.get(k, "") for k in fields})

    summary = Counter(row["label"] for row in rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "total_rows": len(rows),
                "gold_rows": summary.get("gold", 0),
                "unclean_rows": summary.get("unclean", 0),
                "threshold": round(float(threshold), 4),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def resolve_input_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    files: List[Path] = []
    for child in sorted(path.rglob("*")):
        if child.suffix.lower() in {".csv", ".txt", ".tsv"}:
            files.append(child)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sort raw text rows into gold vs unclean using testing_data references."
    )
    parser.add_argument("--input", required=True, help="Input file or directory with CSV/TXT files.")
    parser.add_argument(
        "--reference-dir",
        default=str(Path(__file__).resolve().parent / "testing_data"),
        help="Directory containing reference gold-like examples.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "data" / "sorted_output"),
        help="Where to write gold.csv / unclean.csv / summary.json",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.42,
        help="Score threshold for gold label (0..1). Higher is stricter.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    reference_dir = Path(args.reference_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    refs = collect_reference_texts(reference_dir)
    if not refs:
        raise SystemExit(f"No reference texts found in: {reference_dir}")
    profile = build_reference_profile(refs)

    input_files = resolve_input_files(input_path)
    if not input_files:
        raise SystemExit(f"No readable CSV/TXT input files found at: {input_path}")

    all_rows: List[dict] = []
    for file_path in input_files:
        all_rows.extend(extract_rows_from_path(file_path))

    if not all_rows:
        raise SystemExit("Input loaded but produced zero text rows.")

    classified = classify_rows(all_rows, profile=profile, threshold=args.threshold)
    write_split_csv(classified, output_dir=output_dir, threshold=args.threshold)

    counts = Counter(row["label"] for row in classified)
    print(f"References loaded: {len(refs)}")
    print(f"Rows processed: {len(classified)}")
    print(f"Gold: {counts.get('gold', 0)}")
    print(f"Unclean: {counts.get('unclean', 0)}")
    print(f"Wrote: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
