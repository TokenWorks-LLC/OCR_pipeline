# OCR Pipeline Project Context

## 🎯 Current Status
- **Environment**: `C:\Users\angel\Desktop\TokenWorks LLC\ocr_env`
- **Project**: `C:\Users\angel\Desktop\TokenWorks LLC\OCR_pipeline`
- **Branch**: `evaluation_and_metrics`

## ✅ Completed Features
1. **Cost of Compute Metrics**:
   - Word counting for modern languages
   - Token counting (text elements for Akkadian)
   - CPU monitoring (fixed psutil interval issue)
   - Memory monitoring
   - Time per word/token calculations

2. **Analysis System**:
   - `src/simple_analysis.py` - Main analysis engine
   - `run_analysis_menu.py` - Interactive menu
   - `src/baseline_accuracy.py` - Baseline assessment
   - Charts and reports generation

3. **Evaluation Pipeline**:
   - `run_eval_incremental.py` - Fixed word/token aggregation
   - `production/comprehensive_pipeline.py` - Added resource monitoring
   - Fixed CPU tracking (was 0%, now shows 950% on 6-core system)

## 🔧 Key Fixes Made
- **CPU Monitoring**: Fixed psutil.cpu_percent() to use interval parameter
- **Word Counting**: Added aggregation from page-level to document-level
- **Resource Tracking**: Memory and CPU monitoring during processing
- **Dependencies**: Installed matplotlib, pandas for analysis

## 📊 Current Results
- **Basic Mode**: 10 docs, 52 pages, 96.2% success, 15,858 words
- **CPU Usage**: 950% (1.58 cores on 6-core system)
- **Memory Usage**: 1,995 MB average
- **Time per Word**: 0.0812s
- **Time per Token**: 0.5447s

## 🚀 Quick Start Commands
```bash
# Activate environment
C:\Users\angel\Desktop\TokenWorks LLC\ocr_env\Scripts\Activate.ps1

# Navigate to project
cd "C:\Users\angel\Desktop\TokenWorks LLC\OCR_pipeline"

# Run analysis
python run_analysis_menu.py

# Run evaluations
python run_eval_incremental.py -c config_eval_basic.json
python run_eval_incremental.py -c config_eval_akkadian.json
```

## 🔍 Next Steps
1. Run Akkadian evaluation with new metrics
2. Compare Basic vs Akkadian with full cost analysis
3. Consider enabling GPU processing (paddle_use_gpu: True)
4. Add GPU monitoring if needed

## 📁 Key Files
- `production/comprehensive_pipeline.py` - Main OCR pipeline
- `run_eval_incremental.py` - Evaluation runner
- `src/simple_analysis.py` - Analysis engine
- `run_analysis_menu.py` - Interactive menu
- `config_eval_basic.json` - Basic mode config
- `config_eval_akkadian.json` - Akkadian mode config

## 💬 Chat Persistence Issue
- Cursor chats not saving between sessions
- Check Settings → General → Chat History
- Update Cursor to latest version
- Check %APPDATA%\Cursor\User\workspaceStorage\

