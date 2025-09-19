#!/usr/bin/env python3
"""
One-command wrapper to run the existing pipeline on one or many documents,
create a timestamped run folder, save per-doc outputs and aggregate CSV/JSON.
"""

import argparse
import os
import subprocess
import json
import csv
import datetime
import subprocess as sp

def iso_timestamp_for_dir(ts=None):
    ts = ts or datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    return ts.replace(":", "-")  # make filename safe

def get_git_sha():
    try:
        return sp.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode()
    except Exception:
        return None

def make_run_dir(output_root, run_name, ts=None):
    ts = ts or datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(output_root, f"{ts}_{run_name}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, ts

def get_input_files(path):
    if os.path.isfile(path):
        return [path]
    exts = (".pdf", ".png", ".jpg", ".jpeg", ".tiff")
    files = []
    for root, _, filenames in os.walk(path):
        for f in filenames:
            if f.lower().endswith(exts):
                files.append(os.path.join(root, f))
    return sorted(files)

def call_pipeline_for_doc(doc_path, doc_out_dir, skip_llm=False, formats=None):
    cmd = ["python", "run_pipeline.py", "--input", doc_path, "--output", doc_out_dir]
    if skip_llm:
        cmd.append("--skip-llm")
    if formats:
        cmd += ["--formats", formats]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def aggregate_results(run_dir, per_doc_results):
    csv_path = os.path.join(run_dir, "results.csv")
    json_path = os.path.join(run_dir, "results.json")
    if per_doc_results:
        keys = sorted(per_doc_results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(per_doc_results)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(per_doc_results, f, indent=2)
    print("Wrote aggregates:", csv_path, json_path)

def main(args):
    run_dir, ts = make_run_dir(args.output_root, args.run_name, args.timestamp)

    input_files = get_input_files(args.input)
    if not input_files:
        raise SystemExit("No input files found under: " + args.input)

    per_doc_results = []
    for doc in input_files:
        doc_id = os.path.splitext(os.path.basename(doc))[0]
        doc_out_dir = os.path.join(run_dir, doc_id)
        os.makedirs(doc_out_dir, exist_ok=True)

        try:
            call_pipeline_for_doc(doc, doc_out_dir, skip_llm=args.skip_llm, formats=args.formats)
        except subprocess.CalledProcessError as e:
            print(f"Pipeline failed for {doc}: {e}")
            continue

        per_doc_results.append({"doc_id": doc_id, "input_path": doc, "output_dir": doc_out_dir})

    aggregate_results(run_dir, per_doc_results)

    meta = {"timestamp": ts, "git_sha": get_git_sha()}
    with open(os.path.join(run_dir, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Run complete:", run_dir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="file or directory input")
    p.add_argument("--output-root", default="data/output", help="root folder for runs")
    p.add_argument("--run-name", default="demo", help="short name for the run")
    p.add_argument("--timestamp", default=None, help="optional ISO timestamp override")
    p.add_argument("--skip-llm", action="store_true", help="skip LLM calls (useful for tests)")
    p.add_argument("--formats", default="json,csv,overlay", help="comma-separated output formats")
    args = p.parse_args()
    main(args)
