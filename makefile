.PHONY: help setup run clean test

# Path to the venv python (relative)
PY = .venv/bin/python
REQ = requirements.txt

help:
	@echo "make setup     -> create venv and install deps"
	@echo "make run       -> run OCR pipeline (uses .venv/bin/python)"
	@echo "make clean     -> remove outputs & logs"
	@echo "make test      -> run smoke tests"

# creates a virtual env and installs dependencies
setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r $(REQ)

#calls cli script on all pdfs and saves outputs in a timestamped folder under data/output/. run name is set to demo. 
# does not skip LLM calls so it runs the full pipeline if PDFs exist
run:
	$(PY) scripts/ocr_cli.py --input data/samples --output-root data/output --run-name demo

# removes previous outputs and logs
clean:
	rm -rf data/output/*
	rm -rf logs/*

# runs a 'dry run' of the pipeline (skips LLM calls) on PDFs in data/samples/
test:
	# dry run: no LLM calls, quick smoke test
	$(PY) scripts/ocr_cli.py --input data/samples --output-root data/output --run-name smoke-test --skip-llm --formats json,csv,overlay
