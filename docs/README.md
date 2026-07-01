# Documentation Index

This directory now keeps only the current technical documentation for the active OCR pipeline.

If a document is not listed here, it should not be treated as part of the canonical explanation of the system.

## Start Here

- [Root Overview](../README.md)
- [Quickstart](../QUICKSTART.md)
- [Pipeline Overview](PIPELINE_OVERVIEW.md)

## Core System Docs

- [Run And Test](RUN_AND_TEST.md)
- [Page Text Runbook](PAGE_TEXT_RUNBOOK.md)
- [Routes And Backends](ROUTES_AND_BACKENDS.md)
- [Page Diagnostics](PAGE_DIAGNOSTICS.md)
- [Postprocessing](POSTPROCESSING.md)
- [Citation / Bibliography Parsing](CITATION_PARSING.md)
- [Quality Scoring](QUALITY_SCORING.md)
- [Document Model Schema](document_model_schema.md)

## Evaluation And Benchmarking

- [Gold-Set Evaluation](GOLDSET_EVALUATION.md)
- [Gold Data And Evaluation Policy](GOLD_DATA_AND_EVALUATION_POLICY.md)
- [Experiment Tracking](EXPERIMENT_TRACKING.md)
- [How To Run Benchmarks](HOW_TO_RUN_BENCHMARKS.md)

## Backend Reference

- [OCR Backend Notes](ocr_backends_notes.md)

## Documentation Rules

- `README.md` explains the repository at a high level.
- `PIPELINE_OVERVIEW.md` defines what the pipeline is and how it is layered.
- `RUN_AND_TEST.md` is the operational entrypoint for running and validating the system.
- Topic docs under `docs/` should describe current behavior only.
- Historical phase tracking, cleanup notes, and handoff documents are intentionally excluded from the active structure.
