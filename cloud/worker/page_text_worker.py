"""
Page-text worker for AWS Batch.
Downloads manifest shard, runs tools/run_page_text.py, uploads results to S3.
"""
import os
import subprocess
import sys
import pathlib
import json
import shutil


def run(cmd):
    """Execute command with logging."""
    print("+", " ".join(cmd))
    sys.stdout.flush()
    return subprocess.run(cmd, check=True)


def main():
    # Required env
    S3_INPUT = os.environ["S3_INPUT"]             # s3://bucket/prefix/
    S3_OUTPUT = os.environ["S3_OUTPUT"]           # s3://bucket/prefix/
    SHARD_KEY = os.environ.get("SHARD_KEY")       # e.g., manifests/shard_00.txt
    SHARD_IDX = os.environ.get("SHARD_IDX", "0")
    IMAGE_TAG = os.environ.get("IMAGE_TAG", "local")

    work = pathlib.Path("/workspace")
    out_root = work / f"reports/page_text_{IMAGE_TAG}_shard_{SHARD_IDX}"
    out_root.mkdir(parents=True, exist_ok=True)

    # Pull shard manifest
    mdir = work / "manifests"
    mdir.mkdir(exist_ok=True)
    if SHARD_KEY:
        run(["aws", "s3", "cp", f"{S3_INPUT}{SHARD_KEY}", str(mdir / pathlib.Path(SHARD_KEY).name)])
        manifest_path = str(mdir / pathlib.Path(SHARD_KEY).name)
    else:
        # fallback: expect manifests/shard_${SHARD_IDX}.txt in input
        shard_name = f"manifests/shard_{int(SHARD_IDX):02}.txt"
        run(["aws", "s3", "cp", f"{S3_INPUT}{shard_name}", str(mdir / f"shard_{int(SHARD_IDX):02}.txt")])
        manifest_path = str(mdir / f"shard_{int(SHARD_IDX):02}.txt")

    # Run page-text extraction
    cmd = [
        "python", "tools/run_page_text.py",
        "--manifest", manifest_path,
        "--output-root", str(out_root),
        "--prefer-text-layer", "--llm-off",
        "--status-bar",
        "--progress-csv", str(out_root / "progress.csv")
    ]
    run(cmd)

    # Sync results
    run(["aws", "s3", "sync", str(out_root), f"{S3_OUTPUT}{out_root.name}/", "--only-show-errors"])
    print("DONE shard", SHARD_IDX)


if __name__ == "__main__":
    main()
