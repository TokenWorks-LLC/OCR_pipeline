# Production Run - Secondary Sources Full Pipeline
# Pass A: Heuristic pairing (LLM OFF) → Pass B: LLM retry (strict JSON)

$ErrorActionPreference = "Stop"

$separator = "=" * 80
Write-Host $separator -ForegroundColor Cyan
Write-Host "PRODUCTION RUN - Secondary Sources Pipeline" -ForegroundColor Cyan
Write-Host $separator -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# STEP 1: Auto-detect Google Drive "Secondary Sources" folder
# ============================================================================
Write-Host "[1/5] Locating Google Drive 'Secondary Sources' folder..." -ForegroundColor Yellow

$drivePaths = @(
    "G:\My Drive\Shared drives\Secondary Sources",
    "G:\Shared drives\Secondary Sources",
    "C:\Users\$env:USERNAME\Google Drive\Shared drives\Secondary Sources"
)

$sourceDir = $null
foreach ($path in $drivePaths) {
    if (Test-Path $path) {
        $sourceDir = $path
        Write-Host "✓ Found: $sourceDir" -ForegroundColor Green
        break
    }
}

if (-not $sourceDir) {
    Write-Host "✗ ERROR: Could not locate 'Secondary Sources' folder in Google Drive" -ForegroundColor Red
    Write-Host "  Tried paths:" -ForegroundColor Red
    foreach ($path in $drivePaths) {
        Write-Host "    - $path" -ForegroundColor Red
    }
    exit 1
}

# ============================================================================
# STEP 2: Build full manifest (all PDFs, all pages)
# ============================================================================
Write-Host ""
Write-Host "[2/5] Building manifest from Drive PDFs..." -ForegroundColor Yellow

$ts = Get-Date -Format "yyyyMMdd_HHmm"
$manifestPath = "manifests\secondary_sources_full_$ts.txt"

New-Item -ItemType Directory -Force -Path "manifests" | Out-Null

& .\.venv\Scripts\python.exe tools/build_manifest.py `
    --inputs "$sourceDir" `
    --out "$manifestPath" `
    --scan-pages `
    --dedupe

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Manifest build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Manifest created: $manifestPath" -ForegroundColor Green

# Count total pages
$manifestLines = (Get-Content $manifestPath | Measure-Object -Line).Lines - 1  # Subtract header
Write-Host "  Total pages to process: $manifestLines" -ForegroundColor Cyan

# ============================================================================
# STEP 3: PASS A - Heuristic pairing (LLM OFF, guaranteed output)
# ============================================================================
Write-Host ""
Write-Host "[3/5] PASS A - Heuristic Pairing (LLM OFF)..." -ForegroundColor Yellow
Write-Host "  Strategy: Fast layout-based pairing, no LLM failures" -ForegroundColor Gray

$outputDirA = "reports\prod_$ts"
$progressCsvA = "$outputDirA\progress.csv"
$clientCsvA = "$outputDirA\client_translations.csv"

& .\.venv\Scripts\python.exe tools/run_manifest.py `
    --manifest "$manifestPath" `
    --profile profiles\akkadian_strict.json `
    --engines paddle `
    --prefer-text-layer `
    --pairing heuristic `
    --llm-off `
    --resume-safe `
    --skip-completed `
    --progress-csv "$progressCsvA" `
    --output-root "$outputDirA" `
    --status-bar `
    --akkadian-threshold 0.60 `
    --require-diacritic-or-marker `
    --min-diacritics 1 `
    --min-syllabic-tokens 3 `
    --min-syllabic-ratio 0.5 `
    --markers-strict `
    --client-csv "$clientCsvA"

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Pass A failed!" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Pass A complete!" -ForegroundColor Green

# Export consolidated client CSV
Write-Host "  Exporting client CSV (UTF-8 BOM)..." -ForegroundColor Gray

& .\.venv\Scripts\python.exe tools/export_translations.py `
    --inputs "$outputDirA" `
    --out "$clientCsvA" `
    --dedupe `
    --strip-layout `
    --excel-friendly

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Export failed!" -ForegroundColor Red
} else {
    Write-Host "✓ Client CSV: $clientCsvA" -ForegroundColor Green
    
    # Show stats
    $csvLines = (Get-Content $clientCsvA | Measure-Object -Line).Lines - 1
    Write-Host "  Translation pairs exported: $csvLines" -ForegroundColor Cyan
}

# ============================================================================
# STEP 4: PASS B - LLM retry on unpaired pages (optional, quality lift)
# ============================================================================
Write-Host ""
Write-Host "[4/5] PASS B - LLM Retry (Strict JSON, Unpaired Only)..." -ForegroundColor Yellow
Write-Host "  Strategy: Use LLM to improve pairing quality on failed pages" -ForegroundColor Gray

$outputDirB = "reports\prod_${ts}_llm"
$progressCsvB = "$outputDirB\progress.csv"
$clientCsvB = "$outputDirB\client_translations.csv"

& .\.venv\Scripts\python.exe tools/run_manifest.py `
    --manifest "$manifestPath" `
    --profile profiles\akkadian_strict.json `
    --engines paddle `
    --prefer-text-layer `
    --llm-on `
    --llm-json-strict `
    --llm-max-retries 2 `
    --only-unpaired `
    --resume-safe `
    --skip-completed `
    --progress-csv "$progressCsvB" `
    --output-root "$outputDirB" `
    --status-bar `
    --akkadian-threshold 0.60 `
    --require-diacritic-or-marker `
    --min-diacritics 1 `
    --min-syllabic-tokens 3 `
    --min-syllabic-ratio 0.5 `
    --markers-strict `
    --client-csv "$clientCsvB"

if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠ Pass B failed (non-critical, Pass A output still valid)" -ForegroundColor Yellow
} else {
    Write-Host "✓ Pass B complete!" -ForegroundColor Green
    
    # Export final client CSV
    Write-Host "  Exporting final client CSV (UTF-8 BOM)..." -ForegroundColor Gray
    
    & .\.venv\Scripts\python.exe tools/export_translations.py `
        --inputs "$outputDirB" `
        --out "$clientCsvB" `
        --dedupe `
        --strip-layout `
        --excel-friendly
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Final Client CSV: $clientCsvB" -ForegroundColor Green
        
        # Show stats
        $csvLines = (Get-Content $clientCsvB | Measure-Object -Line).Lines - 1
        Write-Host "  Translation pairs exported: $csvLines" -ForegroundColor Cyan
    }
}

# ============================================================================
# STEP 5: Generate summary
# ============================================================================
Write-Host ""
Write-Host "[5/5] Generating summary..." -ForegroundColor Yellow

$summaryPath = "$outputDirA\SUMMARY.md"
$summaryContent = @"
# Production Run Summary
**Date**: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
**Source**: $sourceDir
**Manifest**: $manifestPath

## Pass A - Heuristic Pairing (LLM OFF)
- **Output**: $outputDirA
- **Client CSV**: $clientCsvA
- **Strategy**: Layout-based heuristic pairing, no LLM
- **Pages processed**: (see progress.csv)

## Pass B - LLM Retry (Optional)
- **Output**: $outputDirB
- **Client CSV**: $clientCsvB
- **Strategy**: LLM correction with strict JSON mode (retry=2)
- **Target**: Unpaired pages from Pass A

## Deliverables
1. **Client CSV (UTF-8 BOM)**: Side-by-side Akkadian ⇄ Translation
   - Pass A: $clientCsvA
   - Pass B: $clientCsvB (if completed)

2. **Progress Tracking**: 
   - Pass A: $progressCsvA
   - Pass B: $progressCsvB

3. **Per-PDF Outputs**: See subdirectories in output roots

## Configuration
- **Profile**: profiles\akkadian_strict.json
- **Detection**: threshold=0.60, syllabic gates, markers strict
- **Text Extraction**: Prefer embedded text layer (26x faster)
- **OCR Fallback**: PaddleOCR v5 (when text layer empty)

## Next Steps
1. Review client CSV in Excel (UTF-8 BOM ensures proper diacritics)
2. Spot-check Akkadian blocks for false positives
3. Verify translation pairing quality
4. Re-run Pass B if needed with adjusted guardrails

---
Generated by: prod_run.ps1
"@

Set-Content -Path $summaryPath -Value $summaryContent -Encoding UTF8
Write-Host "✓ Summary written: $summaryPath" -ForegroundColor Green

# ============================================================================
# FINAL STATUS
# ============================================================================
Write-Host ""
Write-Host $separator -ForegroundColor Cyan
Write-Host "PRODUCTION RUN COMPLETE" -ForegroundColor Cyan
Write-Host $separator -ForegroundColor Cyan
Write-Host ""
Write-Host "OUTPUTS:" -ForegroundColor Yellow
Write-Host "  [GUARANTEED] Pass A Client CSV: $clientCsvA" -ForegroundColor Green
if (Test-Path $clientCsvB) {
    Write-Host "  [IMPROVED]   Pass B Client CSV: $clientCsvB" -ForegroundColor Green
}
Write-Host ""
Write-Host "  Summary: $summaryPath" -ForegroundColor Cyan
Write-Host "  Manifest: $manifestPath" -ForegroundColor Cyan
Write-Host ""
Write-Host "[SUCCESS] Pipeline execution successful!" -ForegroundColor Green
Write-Host ""
