@echo off
echo 🚀 OCR Pipeline Quick Setup
echo ============================

REM Navigate to project directory
cd /d "C:\Users\angel\Desktop\TokenWorks LLC\OCR_pipeline"

REM Activate Python environment
call "C:\Users\angel\Desktop\TokenWorks LLC\ocr_env\Scripts\activate.bat"

REM Verify environment
python --version
echo.
echo ✅ Environment activated!
echo 📁 Working directory: %CD%
echo.
echo Available commands:
echo   python run_analysis_menu.py    - Run analysis menu
echo   python run_eval_incremental.py -c config_eval_basic.json - Run basic eval
echo   python run_eval_incremental.py -c config_eval_akkadian.json - Run Akkadian eval
echo.
echo Press any key to continue...
pause > nul

