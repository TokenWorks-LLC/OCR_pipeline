from __future__ import annotations

import json
from typing import Dict, Iterable

from .base import ConversionContext


class LinePairsConverter:
    annotation_format = "line_pairs"

    def iter_records(self, context: ConversionContext) -> Iterable[Dict[str, object]]:
        index_path = context.source_root / "index.jsonl"
        if not index_path.exists():
            return []

        records: list[Dict[str, object]] = []
        with index_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                payload = line.strip()
                if not payload:
                    continue
                row = json.loads(payload)
                if str(row.get("annotation_format", "")).lower() != self.annotation_format:
                    continue
                records.append(row)
        return records
