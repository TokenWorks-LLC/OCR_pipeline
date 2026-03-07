#!/usr/bin/env python3
"""Legacy comprehensive pipeline compatibility module.

This module no longer performs LLM/Ollama correction. It keeps the old
entrypoint shape and routes execution to the maintained `run_pipeline.py`
compatibility command.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PipelineConfig:
    """Minimal retained config for compatibility callers."""

    dpi: int = 300
    enable_reading_order: bool = True


class ComprehensivePipeline:
    """Compatibility facade that delegates to run_pipeline.py."""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()

    def process_pdf(
        self,
        pdf_path: str,
        output_dir: str | Path,
        start_page: int = 1,
        end_page: int | None = None,
    ) -> dict[str, Any]:
        del start_page
        del end_page

        output_dir = str(output_dir)
        cmd = [
            sys.executable,
            "run_pipeline.py",
            "--input-file",
            pdf_path,
            "--output-dir",
            output_dir,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return {
                "error": result.stderr.strip() or "run_pipeline.py failed",
                "returncode": result.returncode,
            }

        return {
            "pages_processed": 1,
            "total_processing_time": "n/a",
            "output_csv": str(Path(output_dir) / "client_page_text.csv"),
        }


def main() -> int:
    """Main entry point for legacy callers."""
    parser = argparse.ArgumentParser(description="Comprehensive OCR Pipeline (compatibility mode)")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("-s", "--start-page", type=int, default=1, help="Start page (default: 1)")
    parser.add_argument("-e", "--end-page", type=int, help="End page (default: all pages)")
    parser.add_argument("--disable-reading-order", action="store_true", help="Legacy flag")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for PDF rendering")

    args = parser.parse_args()

    config = PipelineConfig(
        enable_reading_order=not args.disable_reading_order,
        dpi=args.dpi,
    )

    output_dir = args.output
    if not output_dir:
        pdf = Path(args.pdf_path)
        output_dir = str(pdf.parent / f"{pdf.stem}_comprehensive_ocr")

    pipeline = ComprehensivePipeline(config)
    result = pipeline.process_pdf(
        pdf_path=args.pdf_path,
        output_dir=output_dir,
        start_page=args.start_page,
        end_page=args.end_page,
    )

    if "error" in result:
        print(f"Pipeline failed: {result['error']}", file=sys.stderr)
        return 1

    print("Pipeline completed successfully")
    print(f"Output: {result['output_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
