# ============================================================================
# Secondary Sources Pipeline Runner
# Goal: Process all PDFs in Google Drive "Secondary Sources" folder
# ============================================================================

Write-Host "`n🔍 CHECKING REQUIRED FLAGS..." -ForegroundColor Cyan

# Check if run_manifest.py supports required flags
$requiredFlags = @(
    '--prefer-text-layer',
    '--engine-workers',
    '--output-root',
    '--excel-friendly'
)

$helpOutput = & .\.venv\Scripts\python.exe tools/run_manifest.py --help 2>&1 | Out-String

$missingFlags = @()
foreach ($flag in $requiredFlags) {
    if ($helpOutput -notmatch [regex]::Escape($flag)) {
        $missingFlags += $flag
    }
}

if ($missingFlags.Count -gt 0) {
    Write-Host "`n❌ MISSING REQUIRED FLAGS IN run_manifest.py:" -ForegroundColor Red
    foreach ($flag in $missingFlags) {
        Write-Host "   • $flag" -ForegroundColor Yellow
    }
    Write-Host "`nThe current implementation of tools/run_manifest.py does not support these flags." -ForegroundColor Red
    Write-Host "`nAvailable alternative approach:" -ForegroundColor Cyan
    Write-Host "  1. Use existing flags: --client-csv, --progress-csv, --resume, --live-progress" -ForegroundColor White
    Write-Host "  2. The pipeline already uses text-layer extraction automatically (see detect_audit.py)" -ForegroundColor White
    Write-Host "  3. Memory control via: --max-pages-in-mem 1" -ForegroundColor White
    Write-Host "  4. UTF-8 BOM is already written by append_to_client_csv() with encoding='utf-8-sig'" -ForegroundColor White
    Write-Host "`nTo proceed with existing functionality, use the command from the previous full run." -ForegroundColor Cyan
    exit 1
}

Write-Host "✅ All required flags are supported" -ForegroundColor Green

# ============================================================================
# 0) Activate env & set up run paths
# ============================================================================
Write-Host "`n📁 SETTING UP ENVIRONMENT..." -ForegroundColor Cyan

# Activate venv
.\.venv\Scripts\Activate.ps1

# Timestamped run root
$ts = Get-Date -Format "yyyyMMdd_HHmm"
$RUNROOT = "reports\secondary_sources_$ts"
New-Item -ItemType Directory -Force -Path $RUNROOT | Out-Null
New-Item -ItemType Directory -Force -Path "$RUNROOT\outputs" | Out-Null

Write-Host "✅ Run directory: $RUNROOT" -ForegroundColor Green

# ============================================================================
# 1) Locate the streamed Google Drive "Secondary Sources" folder
# ============================================================================
Write-Host "`n🔍 LOCATING GOOGLE DRIVE FOLDER..." -ForegroundColor Cyan

# Probe common Drive for desktop mounts
$roots = @('G:\','H:\','I:\','J:\')
$Secondary = $null
foreach ($r in $roots) {
    $cand = Join-Path $r 'Shared drives\Secondary Sources'
    if (Test-Path $cand) { $Secondary = $cand; break }
    $cand2 = Join-Path $r 'My Drive\Secondary Sources'
    if (Test-Path $cand2) { $Secondary = $cand2; break }
}

if (-not $Secondary) {
    Write-Host "❌ Could not find 'Secondary Sources' on G:\, H:\, I:\, or J:\" -ForegroundColor Red
    Write-Host "   Please open Google Drive for Desktop, ensure Stream files is ON, then re-run." -ForegroundColor Yellow
    
    # Check if local data exists as fallback
    $localData = "data\drive\Secondary Sources"
    if (Test-Path $localData) {
        Write-Host "`n💡 Found local copy: $localData" -ForegroundColor Cyan
        Write-Host "   Use this instead? (Y/N): " -NoNewline
        $response = Read-Host
        if ($response -eq 'Y' -or $response -eq 'y') {
            $Secondary = $localData
            Write-Host "✅ Using local data: $Secondary" -ForegroundColor Green
        } else {
            exit 1
        }
    } else {
        exit 1
    }
} else {
    Write-Host "✅ Secondary Sources found at: $Secondary" -ForegroundColor Green
}

# ============================================================================
# 2) Build a per-page manifest from the streamed folder (resume-safe & deduped)
# ============================================================================
Write-Host "`n📝 BUILDING MANIFEST..." -ForegroundColor Cyan

$MANIFEST = "manifests\secondary_sources_all.txt"
New-Item -ItemType Directory -Force -Path "manifests" | Out-Null

# Check if build_manifest.py exists
if (-not (Test-Path "tools\build_manifest.py")) {
    Write-Host "❌ tools\build_manifest.py not found" -ForegroundColor Red
    Write-Host "   Checking for existing manifest..." -ForegroundColor Yellow
    
    # Check for existing manifest
    if (Test-Path "data\drive\manifest_full.txt") {
        Write-Host "✅ Using existing manifest: data\drive\manifest_full.txt" -ForegroundColor Green
        $MANIFEST = "data\drive\manifest_full.txt"
    } else {
        Write-Host "❌ No existing manifest found. Cannot proceed." -ForegroundColor Red
        exit 1
    }
} else {
    # Build manifest
    python tools/build_manifest.py `
        --inputs "$Secondary" `
        --out $MANIFEST `
        --dedupe --shuffle

    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ build_manifest failed." -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ Manifest built: $MANIFEST" -ForegroundColor Green
}

# ============================================================================
# 3) Run the full pipeline
# ============================================================================
Write-Host "`n🚀 STARTING PIPELINE..." -ForegroundColor Cyan

$PROGRESS = "$RUNROOT\progress.csv"
$CLIENT   = "$RUNROOT\client_translations.csv"

Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "  • Manifest: $MANIFEST"
Write-Host "  • Profile: profiles\akkadian_strict.json"
Write-Host "  • Engine: paddle (with text-layer extraction auto-enabled)"
Write-Host "  • Memory: max-pages-in-mem=1 (RAM-friendly)"
Write-Host "  • Resume: enabled via progress CSV"
Write-Host "  • Output: $CLIENT (UTF-8 BOM)"
Write-Host ""

python tools/run_manifest.py `
    --manifest $MANIFEST `
    --profile profiles\akkadian_strict.json `
    --engines paddle `
    --max-pages-in-mem 1 `
    --resume `
    --progress-csv $PROGRESS `
    --client-csv $CLIENT `
    --output-dir "$RUNROOT\outputs" `
    --live-progress `
    --akkadian-threshold 0.60 `
    --require-diacritic-or-marker `
    --min-diacritics 1 `
    --min-syllabic-tokens 3 `
    --min-syllabic-ratio 0.5 `
    --markers-strict

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n❌ run_manifest failed." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "🎉 RUN COMPLETE" -ForegroundColor Green
Write-Host "• Progress:   $PROGRESS"
Write-Host "• Client CSV: $CLIENT"
Write-Host "• Outputs:    $RUNROOT\outputs"
Write-Host ""
Write-Host "📊 QUICK STATS:" -ForegroundColor Cyan
if (Test-Path $CLIENT) {
    $rows = (Get-Content $CLIENT | Measure-Object -Line).Lines - 1
    Write-Host "  Client CSV rows: $rows"
}
if (Test-Path $PROGRESS) {
    $completed = (Import-Csv $PROGRESS | Where-Object { $_.success -eq 'True' }).Count
    Write-Host "  Pages completed: $completed"
}
