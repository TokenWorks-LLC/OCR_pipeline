@echo off
REM Baseline evaluation batch script for Windows
REM Run this from the OCR_pipeline directory

echo Starting baseline OCR evaluation...
echo.

docker compose run --rm ocr python tools/run_baseline_eval.py --gold-csv data/gold_data/gold_pages.csv --limit-pdfs 2 --profile quality --report-md --seed 17

echo.
echo Baseline evaluation complete!
echo Check the reports/ directory for results.
pause