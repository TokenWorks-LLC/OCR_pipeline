from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
GOLD_MANIFEST = ROOT / "data" / "gold_registry" / "gold_manifest.jsonl"


def _load_manifest_rows(limit: int = 300) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with GOLD_MANIFEST.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def test_gold_manifest_exists() -> None:
    assert GOLD_MANIFEST.exists(), f"Missing canonical manifest: {GOLD_MANIFEST}"


def test_ground_truth_paths_resolve_from_canonical_manifest() -> None:
    rows = _load_manifest_rows(limit=300)
    assert rows, "Expected non-empty gold manifest sample"

    missing_paths: list[str] = []
    for row in rows:
        rel_path = str(row.get("ground_truth_text_path", "")).strip()
        if not rel_path:
            missing_paths.append("<missing ground_truth_text_path field>")
            continue

        gt_path = ROOT / rel_path
        if not gt_path.exists():
            missing_paths.append(rel_path)

    assert not missing_paths, (
        "Some canonical ground-truth paths are missing in sampled manifest rows: "
        + ", ".join(missing_paths[:10])
    )
