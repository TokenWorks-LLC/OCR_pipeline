"""
Shard manifest into N equal parts for parallel AWS Batch processing.

Usage:
    python tools/shard_manifest.py <manifest.txt> <num_shards>

Output:
    manifests/shard_00.txt
    manifests/shard_01.txt
    ...
    manifests/shard_NN.txt
"""
import sys
import math
import pathlib

if len(sys.argv) < 3:
    print("usage: shard_manifest.py <manifest.txt> <num_shards>")
    sys.exit(1)

src = pathlib.Path(sys.argv[1])
N = int(sys.argv[2])

lines = src.read_text(encoding="utf-8").splitlines()
hdr, rows = lines[0], [r for r in lines[1:] if r.strip()]
size = math.ceil(len(rows) / N)

outdir = pathlib.Path("manifests")
outdir.mkdir(exist_ok=True)

for i in range(N):
    part = rows[i * size:(i + 1) * size]
    (outdir / f"shard_{i:02}.txt").write_text("\n".join([hdr] + part), encoding="utf-8")

print(f"wrote {N} shards under manifests/")
