# Sample Files Directory

This directory is for sample input files to test the OCR pipeline.

## Supported File Types:
- PDF files (.pdf)
- Image files (.png, .jpg, .jpeg, .tiff, .bmp)

## Usage:
1. Place your test files in this directory
2. Update config.json to point to this directory
3. Run the pipeline: python run_pipeline.py

## Example Configuration:
```json
{
  "input": {
    "input_directory": "./data/samples",
    "process_all_files": true
  }
}
```
