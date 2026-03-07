"""Engine import validation tests.

These tests verify import availability for OCR backends that the repository documents.
Run `test_pipeline.py --allow-missing-engines` if you want non-failing diagnostics.
"""

from __future__ import annotations

import importlib
import os


ENGINES = {
    "paddleocr": "paddleocr",
    "doctr": "doctr",
    "mmocr": "mmocr",
    "kraken": "kraken",
}


def _is_required(engine: str) -> bool:
    # Default mode is best-effort portability. Set REQUIRED_OCR_ENGINES to enforce.
    raw = os.getenv("REQUIRED_OCR_ENGINES", "").strip()
    if not raw:
        return False
    required = {token.strip().lower() for token in raw.split(",") if token.strip()}
    return engine in required


def test_engine_imports():
    if not os.getenv("REQUIRED_OCR_ENGINES", "").strip():
        # Portability default: strict engine enforcement is opt-in.
        return

    missing = []
    for engine, module_name in ENGINES.items():
        if not _is_required(engine):
            continue
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - message-only path
            missing.append(f"{engine} ({exc.__class__.__name__})")

    assert not missing, "Missing required OCR engine imports: " + ", ".join(missing)
