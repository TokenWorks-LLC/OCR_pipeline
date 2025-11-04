# Secondary Sources Pipeline Runner
# Processes all PDFs in the Secondary Sources folder with existing flags

Write-Host "`nSECONDARY SOURCES FULL PIPELINE RUN" -ForegroundColor Cyan
Write-Host "====================================`n" -ForegroundColor Cyan

# Activate venv
.\.venv\Scripts\Activate.ps1

# Timestamped run root
$ts = Get-Date -Format "yyyyMMdd_HHmm"
$RUNROOT = "reports\secondary_sources_$ts"
New-Item -ItemType Directory -Force -Path $RUNROOT | Out-Null

Write-Host "Output directory: $RUNROOT`n" -ForegroundColor Green

# Locate Secondary Sources folder
Write-Host "Locating Secondary Sources folder..." -ForegroundColor Cyan

$roots = @('G:\', 'H:\', 'I:\', 'J:\')
$Secondary = $null

foreach ($r in $roots) {
    $cand = Join-Path $r 'Shared drives\Secondary Sources'
    if (Test-Path $cand) {
        $Secondary = $cand
        break
    }
    
    $cand2 = Join-Path $r 'My Drive\Secondary Sources'
    if (Test-Path $cand2) {
        $Secondary = $cand2
        break
    }
}

if (-not $Secondary) {
    Write-Host "WARNING: Google Drive folder not found" -ForegroundColor Yellow
    
    $localData = "data\drive\Secondary Sources"
    if (Test-Path $localData) {
        Write-Host "Found local copy: $localData" -ForegroundColor Cyan
        $Secondary = $localData
        Write-Host "Using local data" -ForegroundColor Green
    } else {
        Write-Host "ERROR: No data source found" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "Found: $Secondary" -ForegroundColor Green
}

# Check for manifest
Write-Host "`nChecking for manifest..." -ForegroundColor Cyan

$MANIFEST = "data\drive\manifest_full.txt"

if (Test-Path $MANIFEST) {
    $lines = Get-Content $MANIFEST
    $pageCount = $lines.Count
    $uniquePdfs = ($lines | ForEach-Object { ($_ -split "`t")[0] } | Select-Object -Unique).Count
    
    Write-Host "Using existing manifest: $MANIFEST" -ForegroundColor Green
    Write-Host "  Pages: $pageCount" -ForegroundColor Gray
    Write-Host "  PDFs:  $uniquePdfs" -ForegroundColor Gray
} else {
    Write-Host "ERROR: Manifest not found: $MANIFEST" -ForegroundColor Red
    exit 1
}

# Run the pipeline
Write-Host "`nSTARTING PIPELINE" -ForegroundColor Cyan
Write-Host "=================" -ForegroundColor Cyan

$PROGRESS = "$RUNROOT\progress.csv"
$CLIENT   = "$RUNROOT\client_translations.csv"

Write-Host "`nConfiguration:" -ForegroundColor White
Write-Host "  Manifest:  $MANIFEST" -ForegroundColor Gray
Write-Host "  Profile:   profiles\akkadian_strict.json" -ForegroundColor Gray
Write-Host "  Engine:    paddle" -ForegroundColor Gray
Write-Host "  Memory:    max-pages-in-mem=1" -ForegroundColor Gray
Write-Host "  Progress:  $PROGRESS" -ForegroundColor Gray
Write-Host "  Client:    $CLIENT" -ForegroundColor Gray
Write-Host "`nGuardrails:" -ForegroundColor White
Write-Host "  threshold: 0.60" -ForegroundColor Gray
Write-Host "  min_syllabic_tokens: 3" -ForegroundColor Gray
Write-Host "  min_syllabic_ratio: 0.5" -ForegroundColor Gray
Write-Host "  markers_strict: true" -ForegroundColor Gray
Write-Host "`nLaunching...`n" -ForegroundColor Green

python tools/run_manifest.py `
    --manifest $MANIFEST `
    --profile profiles\akkadian_strict.json `
    --engines paddle `
    --max-pages-in-mem 1 `
    --resume `
    --progress-csv $PROGRESS `
    --client-csv $CLIENT `
    --output-dir $RUNROOT `
    --live-progress `
    --akkadian-threshold 0.60 `
    --require-diacritic-or-marker `
    --min-diacritics 1 `
    --min-syllabic-tokens 3 `
    --min-syllabic-ratio 0.5 `
    --markers-strict

$exitCode = $LASTEXITCODE

Write-Host "`n=====================================" -ForegroundColor Cyan

if ($exitCode -ne 0) {
    Write-Host "PIPELINE FAILED (Exit Code: $exitCode)" -ForegroundColor Red
    exit $exitCode
}

Write-Host "RUN COMPLETE" -ForegroundColor Green
Write-Host "====================================`n" -ForegroundColor Cyan

Write-Host "Output Files:" -ForegroundColor Cyan
Write-Host "  Progress:  $PROGRESS" -ForegroundColor Gray
Write-Host "  Client:    $CLIENT" -ForegroundColor Gray
Write-Host "  Outputs:   $RUNROOT`n" -ForegroundColor Gray

Write-Host "Quick Stats:" -ForegroundColor Cyan

if (Test-Path $PROGRESS) {
    $progressData = Import-Csv $PROGRESS
    $totalProcessed = $progressData.Count - 1
    $successful = ($progressData | Where-Object { $_.success -eq 'True' }).Count
    $failed = ($progressData | Where-Object { $_.success -eq 'False' }).Count
    
    Write-Host "  Pages processed: $totalProcessed" -ForegroundColor Gray
    Write-Host "  Successful: $successful" -ForegroundColor Green
    if ($failed -gt 0) {
        Write-Host "  Failed: $failed" -ForegroundColor Yellow
    }
}

if (Test-Path $CLIENT) {
    $clientRows = (Get-Content $CLIENT | Measure-Object -Line).Lines - 1
    Write-Host "  Akkadian detections: $clientRows`n" -ForegroundColor Gray
} else {
    Write-Host "  Client CSV: Not created (no detections)`n" -ForegroundColor Yellow
}

Write-Host "====================================`n" -ForegroundColor Cyan
