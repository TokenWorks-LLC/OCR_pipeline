from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Protocol


@dataclass
class ConversionContext:
    dataset_id: str
    source_root: Path
    output_root: Path


class Converter(Protocol):
    annotation_format: str

    def iter_records(self, context: ConversionContext) -> Iterable[Dict[str, object]]:
        ...
