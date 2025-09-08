@echo off
REM OCR Pipeline - Windows Batch Script
REM This script provides easy access to common OCR pipeline operations

echo.
echo ============================================================
echo            OCR Pipeline - Windows Helper Script
echo ============================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Please install Python 3.8 or higher and add it to your PATH
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "run_pipeline.py" (
    echo ERROR: run_pipeline.py not found
    echo Please run this script from the OCR_pipeline directory
    pause
    exit /b 1
)

:menu
echo Please select an option:
echo.
echo 1. Setup OCR Pipeline (first-time setup)
echo 2. Run Quick Start Examples
echo 3. Process files with default config
echo 4. Process files with custom config
echo 5. Validate configuration
echo 6. View help
echo 7. Exit
echo.
set /p choice="Enter your choice (1-7): "

if "%choice%"=="1" goto setup
if "%choice%"=="2" goto quickstart  
if "%choice%"=="3" goto default_run
if "%choice%"=="4" goto custom_run
if "%choice%"=="5" goto validate
if "%choice%"=="6" goto help
if "%choice%"=="7" goto exit
echo Invalid choice. Please try again.
goto menu

:setup
echo.
echo Running setup...
python setup.py
pause
goto menu

:quickstart
echo.
echo Running quick start examples...
python quick_start.py
pause
goto menu

:default_run
echo.
echo Processing files with default configuration...
if not exist "config.json" (
    echo ERROR: config.json not found
    echo Please run setup first (option 1)
    pause
    goto menu
)
python run_pipeline.py
pause
goto menu

:custom_run
echo.
set /p config_file="Enter path to config file: "
if not exist "%config_file%" (
    echo ERROR: Config file not found: %config_file%
    pause
    goto menu
)
echo Processing with custom config: %config_file%
python run_pipeline.py -c "%config_file%"
pause
goto menu

:validate
echo.
if not exist "config.json" (
    echo ERROR: config.json not found
    echo Please run setup first (option 1)
    pause
    goto menu
)
echo Validating configuration...
python run_pipeline.py --validate-only
pause
goto menu

:help
echo.
echo ============================================================
echo                    OCR Pipeline Help
echo ============================================================
echo.
echo This OCR pipeline processes PDF files and images to extract text.
echo.
echo Quick Setup:
echo   1. Run setup (option 1) to install dependencies and create directories
echo   2. Place your PDF/image files in the data/input directory
echo   3. Run processing (option 3) to process all files
echo.
echo Configuration:
echo   - Edit config.json to customize processing options
echo   - Supported formats: PDF, PNG, JPG, JPEG, TIFF, BMP
echo   - Enable/disable AI text correction, Akkadian extraction, etc.
echo.
echo Advanced Features:
echo   - AI-powered text correction using LLM models
echo   - Specialized Akkadian text extraction for academic research
echo   - Multi-language support (English, Turkish, German, French, Italian)
echo   - Batch processing with parallel execution
echo.
echo Output:
echo   - CSV files with extracted text and metadata
echo   - HTML overlays showing text detection regions
echo   - PDF reports for Akkadian translations (if applicable)
echo   - Processing logs and statistics
echo.
echo For more information, see README.md
echo.
pause
goto menu

:exit
echo.
echo Thank you for using OCR Pipeline!
exit /b 0
