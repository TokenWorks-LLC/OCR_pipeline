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
import unicodedata
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
    value = unicodedata.normalize("NFC", value)
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


def line_marker_density(value: str) -> float:
    if not value:
        return 0.0
    markers = re.findall(r"\b\d+[)\].:]|\b[ivxlcdm]+\.", value, flags=re.IGNORECASE)
    words = value.split()
    return min(1.0, len(markers) / max(len(words), 1))


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
    if suffix == ".tsv":
        return extract_rows_from_csv(path)
    if suffix == ".txt":
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


def filter_reference_texts(
    refs: Sequence[str],
    min_translit_density: float,
    max_noise_density: float,
    min_chars: int,
) -> List[str]:
    kept: List[str] = []
    for text in refs:
        if len(text) < min_chars:
            continue
        if transliteration_density(text) < min_translit_density:
            continue
        if noise_density(text) > max_noise_density:
            continue
        kept.append(text)
    return kept


@dataclass
class ReferenceProfile:
    token_vocab: set
    ngram_vocab: set
    avg_length: float


def save_reference_profile(profile: ReferenceProfile, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "avg_length": profile.avg_length,
        "token_vocab": sorted(profile.token_vocab),
        "ngram_vocab": sorted(profile.ngram_vocab),
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def load_reference_profile(path: Path) -> ReferenceProfile:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return ReferenceProfile(
        token_vocab=set(payload.get("token_vocab", [])),
        ngram_vocab=set(payload.get("ngram_vocab", [])),
        avg_length=float(payload.get("avg_length", 0.0)),
    )


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
    gold_similarity: float
    non_gold_similarity: float
    contrast_similarity: float
    translit_density: float
    line_marker_density: float
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


def score_text(
    text: str,
    gold_profile: ReferenceProfile,
    non_gold_profile: ReferenceProfile | None = None,
) -> ScoreBreakdown:
    gold_sim = profile_similarity(text, gold_profile)
    non_gold_sim = profile_similarity(text, non_gold_profile) if non_gold_profile else 0.0
    contrast_similarity = max(0.0, gold_sim - (0.7 * non_gold_sim))
    translit = transliteration_density(text)
    line_markers = line_marker_density(text)
    noise = noise_density(text)
    short_penalty = short_or_blank_penalty(text)

    # Contrastive blend: prioritize closeness to gold references while penalizing
    # non-gold similarity and obvious OCR-like corruption.
    score = (
        0.52 * gold_sim
        + 0.20 * contrast_similarity
        + 0.20 * translit
        + 0.10 * line_markers
        - 0.18 * non_gold_sim
        - 0.42 * noise
        - 0.35 * short_penalty
    )
    score = max(0.0, min(1.0, score))
    return ScoreBreakdown(
        score=score,
        gold_similarity=gold_sim,
        non_gold_similarity=non_gold_sim,
        contrast_similarity=contrast_similarity,
        translit_density=translit,
        line_marker_density=line_markers,
        noise_density=noise,
        short_penalty=short_penalty,
    )


def classify_rows(
    rows: Iterable[dict],
    gold_profile: ReferenceProfile,
    threshold: float,
    non_gold_profile: ReferenceProfile | None = None,
) -> List[dict]:
    out: List[dict] = []
    for row in rows:
        details = score_text(row["text"], gold_profile=gold_profile, non_gold_profile=non_gold_profile)
        label = "gold" if details.score >= threshold else "unclean"
        enriched = dict(row)
        enriched.update(
            {
                "label": label,
                "score": round(details.score, 4),
                "gold_similarity": round(details.gold_similarity, 4),
                "non_gold_similarity": round(details.non_gold_similarity, 4),
                "contrast_similarity": round(details.contrast_similarity, 4),
                "translit_density": round(details.translit_density, 4),
                "line_marker_density": round(details.line_marker_density, 4),
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
        "gold_similarity",
        "non_gold_similarity",
        "contrast_similarity",
        "translit_density",
        "line_marker_density",
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

    uncertain = sorted(rows, key=lambda r: abs(float(r.get("score", 0.0)) - threshold))
    review_path = output_dir / "review_queue.csv"
    with review_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in uncertain[:250]:
            writer.writerow({k: row.get(k, "") for k in fields})

    summary = Counter(row["label"] for row in rows)
    score_values = [float(row.get("score", 0.0)) for row in rows]
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "total_rows": len(rows),
                "gold_rows": summary.get("gold", 0),
                "unclean_rows": summary.get("unclean", 0),
                "threshold": round(float(threshold), 4),
                "score_min": round(min(score_values), 4),
                "score_max": round(max(score_values), 4),
                "score_mean": round(sum(score_values) / max(len(score_values), 1), 4),
                "review_queue": str(review_path.name),
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
        default=str(Path(__file__).resolve().parent / "training_data"),
        help="Directory containing gold-like examples.",
    )
    parser.add_argument(
        "--profile-in",
        default="",
        help="Optional path to a pre-trained gold profile JSON.",
    )
    parser.add_argument(
        "--train-profile-out",
        default="",
        help="Optional path to save a trained gold profile JSON.",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Train and save profile, then exit without classifying rows.",
    )
    parser.add_argument(
        "--non-gold-reference-dir",
        default="",
        help="Optional directory containing non-gold examples to improve contrastive scoring.",
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
    parser.add_argument(
        "--min-reference-translit-density",
        type=float,
        default=0.05,
        help="Minimum transliteration density for reference rows kept in training profile.",
    )
    parser.add_argument(
        "--max-reference-noise-density",
        type=float,
        default=0.03,
        help="Maximum noise density for reference rows kept in training profile.",
    )
    parser.add_argument(
        "--min-reference-chars",
        type=int,
        default=8,
        help="Minimum character length for reference rows kept in training profile.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    reference_dir = Path(args.reference_dir).expanduser().resolve()
    profile_in = Path(args.profile_in).expanduser().resolve() if args.profile_in else None
    train_profile_out = (
        Path(args.train_profile_out).expanduser().resolve() if args.train_profile_out else None
    )
    non_gold_reference_dir = (
        Path(args.non_gold_reference_dir).expanduser().resolve()
        if args.non_gold_reference_dir
        else None
    )
    output_dir = Path(args.output_dir).expanduser().resolve()

    gold_refs_count = 0
    if profile_in:
        if not profile_in.exists():
            raise SystemExit(f"No profile found at: {profile_in}")
        gold_profile = load_reference_profile(profile_in)
    else:
        gold_refs = collect_reference_texts(reference_dir)
        if not gold_refs:
            raise SystemExit(f"No reference texts found in: {reference_dir}")
        filtered_gold_refs = filter_reference_texts(
            gold_refs,
            min_translit_density=args.min_reference_translit_density,
            max_noise_density=args.max_reference_noise_density,
            min_chars=args.min_reference_chars,
        )
        gold_refs_count = len(filtered_gold_refs)
        if not filtered_gold_refs:
            raise SystemExit(
                "Reference filtering removed all rows. Lower "
                "--min-reference-translit-density or increase --max-reference-noise-density."
            )
        gold_profile = build_reference_profile(filtered_gold_refs)
        if train_profile_out:
            save_reference_profile(gold_profile, train_profile_out)

    if args.train_only:
        if not train_profile_out:
            raise SystemExit("--train-only requires --train-profile-out")
        print(f"Gold references loaded: {gold_refs_count}")
        print(f"Trained profile saved: {train_profile_out}")
        return 0

    non_gold_profile = None
    if non_gold_reference_dir:
        non_gold_refs = collect_reference_texts(non_gold_reference_dir)
        if not non_gold_refs:
            raise SystemExit(f"No non-gold reference texts found in: {non_gold_reference_dir}")
        non_gold_profile = build_reference_profile(non_gold_refs)

    input_files = resolve_input_files(input_path)
    if not input_files:
        raise SystemExit(f"No readable CSV/TXT input files found at: {input_path}")

    all_rows: List[dict] = []
    for file_path in input_files:
        all_rows.extend(extract_rows_from_path(file_path))

    if not all_rows:
        raise SystemExit("Input loaded but produced zero text rows.")

    classified = classify_rows(
        all_rows,
        gold_profile=gold_profile,
        threshold=args.threshold,
        non_gold_profile=non_gold_profile,
    )
    write_split_csv(classified, output_dir=output_dir, threshold=args.threshold)

    counts = Counter(row["label"] for row in classified)
    if profile_in:
        print(f"Gold profile loaded from: {profile_in}")
    else:
        print(f"Gold references loaded: {gold_refs_count}")
    if non_gold_profile:
        print(f"Non-gold references: enabled ({non_gold_reference_dir})")
    else:
        print("Non-gold references: disabled")
    print(f"Rows processed: {len(classified)}")
    print(f"Gold: {counts.get('gold', 0)}")
    print(f"Unclean: {counts.get('unclean', 0)}")
    print(f"Wrote: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
