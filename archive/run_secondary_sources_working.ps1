# ============================================================================
# SECONDARY SOURCES PIPELINE - WORKING IMPLEMENTATION
# ============================================================================
# Goal: Process all PDFs in Google Drive "Secondary Sources" folder
# Uses existing run_manifest.py flags with equivalent functionality
# ============================================================================

Write-Host "`n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "  SECONDARY SOURCES FULL PIPELINE RUN" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`n" -ForegroundColor Cyan

# ============================================================================
# 0) Activate environment & set up run paths
# ============================================================================
Write-Host "📁 Setting up environment..." -ForegroundColor Cyan

# Activate venv
.\.venv\Scripts\Activate.ps1

# Timestamped run root
$ts = Get-Date -Format "yyyyMMdd_HHmm"
$RUNROOT = "reports\secondary_sources_$ts"
New-Item -ItemType Directory -Force -Path $RUNROOT | Out-Null

Write-Host "✅ Output directory: $RUNROOT`n" -ForegroundColor Green

# ============================================================================
# 1) Locate Google Drive "Secondary Sources" folder
# ============================================================================
Write-Host "🔍 Locating Secondary Sources folder..." -ForegroundColor Cyan

# Probe common Drive for Desktop mount points
$roots = @('G:\', 'H:\', 'I:\', 'J:\')
$Secondary = $null

foreach ($r in $roots) {
    # Check Shared drives
    $cand = Join-Path $r 'Shared drives\Secondary Sources'
    if (Test-Path $cand) {
        $Secondary = $cand
        break
    }
    
    # Check My Drive
    $cand2 = Join-Path $r 'My Drive\Secondary Sources'
    if (Test-Path $cand2) {
        $Secondary = $cand2
        break
    }
}

if (-not $Secondary) {
    Write-Host "⚠️  Google Drive folder not found on G:\, H:\, I:\, or J:\" -ForegroundColor Yellow
    
    # Check for local data as fallback
    $localData = "data\drive\Secondary Sources"
    if (Test-Path $localData) {
        Write-Host "   Found local copy: $localData" -ForegroundColor Cyan
        Write-Host "   Use local data? (Y/N): " -NoNewline -ForegroundColor Yellow
        $response = Read-Host
        
        if ($response -eq 'Y' -or $response -eq 'y') {
            $Secondary = $localData
            Write-Host "✅ Using local data: $Secondary`n" -ForegroundColor Green
        } else {
            Write-Host "`n❌ Cannot proceed without data source." -ForegroundColor Red
            Write-Host "   Please:" -ForegroundColor Yellow
            Write-Host "   1. Open Google Drive for Desktop" -ForegroundColor Yellow
            Write-Host "   2. Enable 'Stream files' mode" -ForegroundColor Yellow
            Write-Host "   3. Ensure 'Secondary Sources' folder is accessible`n" -ForegroundColor Yellow
            exit 1
        }
    } else {
        Write-Host "`n❌ No data source found (neither Drive nor local)." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "✅ Found: $Secondary`n" -ForegroundColor Green
}

# ============================================================================
# 2) Build or verify manifest
# ============================================================================
Write-Host "📝 Checking for existing manifest..." -ForegroundColor Cyan

$MANIFEST = "data\drive\manifest_full.txt"

if (Test-Path $MANIFEST) {
    # Verify existing manifest
    $lines = Get-Content $MANIFEST
    $pageCount = $lines.Count
    $uniquePdfs = ($lines | ForEach-Object { ($_ -split "`t")[0] } | Select-Object -Unique).Count
    
    Write-Host "✅ Using existing manifest: $MANIFEST" -ForegroundColor Green
    Write-Host "   Pages: $pageCount" -ForegroundColor Gray
    Write-Host "   PDFs:  $uniquePdfs`n" -ForegroundColor Gray
    
} else {
    Write-Host "⚠️  Manifest not found: $MANIFEST" -ForegroundColor Yellow
    Write-Host "   Checking for build_manifest.py..." -ForegroundColor Cyan
    
    if (Test-Path "tools\build_manifest.py") {
        Write-Host "   Building new manifest from: $Secondary`n" -ForegroundColor Cyan
        
        # Create manifest directory
        New-Item -ItemType Directory -Force -Path (Split-Path $MANIFEST -Parent) | Out-Null
        
        # Build manifest
        python tools/build_manifest.py `
            --inputs "$Secondary" `
            --out $MANIFEST
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "`n❌ Failed to build manifest." -ForegroundColor Red
            exit 1
        }
        
        Write-Host "✅ Manifest built successfully`n" -ForegroundColor Green
    } else {
        Write-Host "`n❌ Cannot proceed: No manifest and no build tool." -ForegroundColor Red
        exit 1
    }
}

# ============================================================================
# 3) Run the full pipeline
# ============================================================================
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "  STARTING PIPELINE" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`n" -ForegroundColor Cyan

$PROGRESS = "$RUNROOT\progress.csv"
$CLIENT   = "$RUNROOT\client_translations.csv"

Write-Host "Configuration:" -ForegroundColor White
Write-Host "  📄 Manifest:  $MANIFEST" -ForegroundColor Gray
Write-Host "  ⚙️  Profile:   profiles\akkadian_strict.json" -ForegroundColor Gray
Write-Host "  🔧 Engine:    paddle (text-layer auto-enabled)" -ForegroundColor Gray
Write-Host "  💾 Memory:    max-pages-in-mem=1 (RAM-safe)" -ForegroundColor Gray
Write-Host "  🔄 Resume:    enabled via progress CSV" -ForegroundColor Gray
Write-Host "  📊 Progress:  $PROGRESS" -ForegroundColor Gray
Write-Host "  📝 Client:    $CLIENT (UTF-8 BOM)" -ForegroundColor Gray
Write-Host "`nDomain-Prior Guardrails:" -ForegroundColor White
Write-Host "  • threshold: 0.60 (precision-first)" -ForegroundColor Gray
Write-Host "  • min_syllabic_tokens: 3 (linguistic invariant)" -ForegroundColor Gray
Write-Host "  • min_syllabic_ratio: 0.5 (≥50% syllabic density)" -ForegroundColor Gray
Write-Host "  • markers_strict: true (DUMU, LUGAL, KÙ.BABBAR only)" -ForegroundColor Gray
Write-Host "  • require_diacritic_or_marker: true" -ForegroundColor Gray
Write-Host "`n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`n" -ForegroundColor Cyan

Write-Host "🚀 Launching pipeline...`n" -ForegroundColor Green

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

Write-Host "`n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan

if ($exitCode -ne 0) {
    Write-Host "  PIPELINE FAILED (Exit Code: $exitCode)" -ForegroundColor Red
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`n" -ForegroundColor Cyan
    exit $exitCode
}

# ============================================================================
# 4) Post-run summary
# ============================================================================
Write-Host "  RUN COMPLETE ✅" -ForegroundColor Green
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`n" -ForegroundColor Cyan

Write-Host "📊 Output Files:" -ForegroundColor Cyan
Write-Host "   • Progress:  $PROGRESS" -ForegroundColor Gray
Write-Host "   • Client:    $CLIENT" -ForegroundColor Gray
Write-Host "   • Outputs:   $RUNROOT`n" -ForegroundColor Gray

Write-Host "📈 Quick Stats:" -ForegroundColor Cyan

# Progress CSV stats
if (Test-Path $PROGRESS) {
    $progressData = Import-Csv $PROGRESS
    $totalProcessed = $progressData.Count - 1
    $successful = ($progressData | Where-Object { $_.success -eq 'True' }).Count
    $failed = ($progressData | Where-Object { $_.success -eq 'False' }).Count
    
    Write-Host "   • Total pages processed: $totalProcessed" -ForegroundColor Gray
    Write-Host "   • Successful: $successful" -ForegroundColor Green
    if ($failed -gt 0) {
        Write-Host "   • Failed: $failed" -ForegroundColor Yellow
    }
}

# Client CSV stats
if (Test-Path $CLIENT) {
    $clientRows = (Get-Content $CLIENT | Measure-Object -Line).Lines - 1
    Write-Host "   • Akkadian detections: $clientRows" -ForegroundColor Gray
    
    if ($clientRows -eq 0) {
        Write-Host "`n   Info: No Akkadian detected (expected with strict guardrails)" -ForegroundColor Cyan
    } elseif ($clientRows -le 1) {
        Write-Host "`n   Warning: Only dummy/header rows in client CSV" -ForegroundColor Yellow
    }
} else {
    Write-Host "   • Client CSV: Not created (no detections)" -ForegroundColor Yellow
}

Write-Host "`n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "  NEXT STEPS" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`n" -ForegroundColor Cyan

Write-Host "1️⃣  Review client CSV for quality:" -ForegroundColor White
Write-Host "   Import-Csv '$CLIENT' | Format-Table" -ForegroundColor Yellow
Write-Host ""

Write-Host "2️⃣  Check for errors in progress CSV:" -ForegroundColor White
Write-Host "   Import-Csv '$PROGRESS' | Where-Object { `$_.success -eq 'False' }" -ForegroundColor Yellow
Write-Host ""

Write-Host "3️⃣  To resume (if interrupted):" -ForegroundColor White
Write-Host "   Re-run this script - progress is tracked automatically" -ForegroundColor Yellow
Write-Host ""

Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`n" -ForegroundColor Cyan
