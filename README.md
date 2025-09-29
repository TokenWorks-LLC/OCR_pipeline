# 📄 OCR Pipeline - Complete Developer's Guide

> **🚀 A production-ready document processing system with AI-powered OCR, multi-language support, and specialized academic text extraction capabilities.**
> 
> **⚡ NEW:** Optimized for Apple Silicon (M1/M2/M3) with 8-70 second Docker builds!

## 🎯 Perfect For
- **Document digitization** projects with high-volume requirements
- **Academic research** involving historical texts and translations  
- **Multi-language** document processing workflows
- **AI-enhanced** OCR accuracy with LLM text correction
- **Batch processing** of PDFs, images, and mixed document types

## 📋 Table of Contents

- [🚀 Quick Start](#quick-start)
- [📖 Project Overview](#project-overview)
- [📁 Directory Structure](#directory-structure)
- [🔧 Key Files Explained](#key-files-explained)
- [⚙️ Configuration Guide](#configuration-guide)
- [💻 Development Setup](#development-setup)
- [🐳 Docker Setup & Deployment](#docker-setup--deployment)
- [📚 Usage Examples](#usage-examples)
- [📖 API Reference](#api-reference)
- [🏗️ Architecture Overview](#architecture-overview)
- [🤝 Contributing](#contributing)
- [🚨 Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

> **💡 New to OCR pipelines?** Start with the Docker approach - it handles all dependencies automatically!

### 🎬 30-Second Demo
```bash
# 1. Get the project
git clone <your-repo-url> && cd OCR_pipeline

# 2. For Apple Silicon Macs (M1/M2/M3) - Super Fast! ⚡
docker build -f Dockerfile.arm64 -t ocr-pipeline .

# 3. For Intel/AMD systems
docker build -t ocr-pipeline .

# 4. Test it! (processes sample files)
docker run --rm -v "$PWD":/workspace -w /workspace ocr-pipeline python run_pipeline.py --input-dir data/samples
```

### 👥 Choose Your Path

#### 🎯 **End Users** (Just want to process documents)
```bash
# One-time setup
python setup.py                    # Installs everything + validates system

# Configure your preferences  
nano config.json                   # Edit: input folders, languages, AI features

# Add files and go!
cp your_files/* data/input/         # Put your PDFs/images here  
python run_pipeline.py             # Magic happens ✨
ls data/output/                     # Your results are here!
```

#### 💻 **Developers** (Want to understand/modify the code)
```bash
# Get familiar with the codebase
python quick_start.py               # Interactive tutorials + examples

# Set up development environment  
pip install -r requirements.txt     # Install Python dependencies
python setup.py                     # Validate everything works

# Explore the architecture
ls src/                             # 17 focused core modules
ls production/                      # Production-ready pipeline
cat config.json                     # Configuration options
```

#### 🐳 **Docker Users** (Recommended for consistency + Apple Silicon)
```bash
# Apple Silicon (M1/M2/M3) - FAST BUILD ⚡ (8-70 seconds!)
docker build -f Dockerfile.arm64 -t ocr-pipeline .

# Intel/AMD64 systems
docker build -t ocr-pipeline .

# Run interactively (best for development)
docker run --rm -it -v "$PWD":/workspace -w /workspace ocr-pipeline bash

# Run directly (best for automation)  
docker run --rm -v "$PWD":/workspace -w /workspace ocr-pipeline python run_pipeline.py

# Advanced: Use docker-compose for easier workflow
docker compose build && docker compose run --rm ocr "data/input/your-file.pdf"
```

> **💡 Need more Docker details?** See `README_docker.md` for comprehensive Docker setup, docker-compose workflows, Makefile shortcuts, and platform-specific troubleshooting.

### ⚡ **Why Our Setup is Developer-Friendly**

- **🎯 Single entry point**: `run_pipeline.py` does everything
- **⚙️ No complex config**: Edit one JSON file, get productive immediately  
- **🔄 Fast iteration**: Docker builds in seconds, not minutes
- **📋 Self-validating**: Built-in health checks catch issues early
- **📚 Interactive learning**: `quick_start.py` teaches you the codebase
- **🧪 Safe testing**: `--dry-run` and `--validate-only` flags for safe exploration

### 🚀 **NEW: Apple Silicon Optimization (Sept 2025)**

> **🍎 Mac M1/M2/M3 Users**: We eliminated the infamous PyMuPDF build failures! 
> 
> **Before**: 30+ minute builds that often failed with SWIG/compilation errors  
> **Now**: ✅ **8-70 second builds** ✅ **100% success rate** ✅ **Binary wheels only**  
> 
> Just use: `docker build -f Dockerfile.arm64 -t ocr-pipeline .`

**🎯 What We Fixed:**
- PyMuPDF compilation from source → Pre-built binary wheels  
- OpenCV 45+ minute builds → Direct pip install (4.10.0.84)  
- PaddlePaddle architecture issues → Official ARM64 wheel index  
- Unpredictable build times → Consistent performance with layer caching

---

## ⏱️ Getting Started in 5 Minutes

### 🚀 **The Absolute Fastest Way** (Docker)
```bash
# 1. Get the code
git clone <your-repo-url> && cd OCR_pipeline

# 2. Build (choose your system)
docker build -f Dockerfile.arm64 -t ocr-pipeline .    # Apple Silicon (M1/M2/M3)
docker build -t ocr-pipeline .                        # Intel/AMD64

# 3. Test with sample files  
docker run --rm -v "$PWD":/workspace -w /workspace ocr-pipeline \
  python run_pipeline.py --input-dir data/samples

# 4. Check results
ls data/output/  # Your processed files are here! 🎉
```

**🐍 The Python-Native Way** (More control)
```bash
# 1. Get the code & dependencies
git clone <your-repo-url> && cd OCR_pipeline
pip install -r requirements.txt

# 2. Validate installation (recommended)
python validate_requirements.py        # Check all packages work correctly

# 3. One-time setup (creates folders, validates system)
python setup.py

# 4. Try it out!
cp /path/to/your/document.pdf data/input/  # Add your file
python run_pipeline.py                     # Process it

# 5. Check results  
ls data/output/  # Processed text files here! 🎉
```

### Docker Users
```bash
# Intel/AMD64 (Windows, Intel Macs, Linux)
docker build -t tokenworks-ocr:latest .
docker run --rm -v "$PWD":/app -w /app tokenworks-ocr:latest python run_pipeline.py

# Apple Silicon (M1/M2/M3 Macs)  
docker build -f Dockerfile.arm64 -t tokenworks-ocr:latest .
docker run --rm -v "$PWD":/app -w /app tokenworks-ocr:latest python run_pipeline.py

# See "Docker Setup & Deployment" section for full details
```

### ✅ **Verify Everything Works**
```bash
# Validate all packages installed correctly
python validate_requirements.py         # Comprehensive package check with versions

# Quick health check (should show all green checkmarks)
python -c "from src.healthcheck import run_health_check; run_health_check()"

# Test core functionality
python run_pipeline.py --validate-only  # Checks config without processing
python run_pipeline.py --dry-run         # Shows what would be processed
```

### 🛠️ **Customize for Your Needs**
```bash
# Edit the main configuration
nano config.json        # Set input/output folders, enable AI, etc.

# Learn the codebase interactively  
python quick_start.py    # Interactive tutorials + examples

# Explore what's available
ls src/                  # 17 core modules for OCR processing
ls data/samples/         # Example files to test with
```

**🎉 That's it!** You now have a working OCR pipeline. Check the sections below for advanced features and customization.

---

## 📖 Project Overview

### 🎯 **What This Pipeline Does**

This OCR pipeline transforms **any document** into **structured, searchable, AI-corrected text** with these powerful capabilities:

- **📄 Multi-format processing**: PDFs, PNG, JPG, JPEG, TIFF, BMP - handles them all
- **🧠 AI-powered accuracy**: LLM-based text correction using Ollama/Mistral (optional)
- **🏛️ Academic specialization**: Akkadian text extraction for archaeological research  
- **🌍 Multi-language support**: English, Turkish, German, French, Italian
- **⚡ Batch processing**: Handle 100s of documents efficiently with parallel processing
- **🚀 Production-ready**: Enterprise-grade architecture with comprehensive error handling
- **🐳 Docker-optimized**: Especially fast on Apple Silicon (M1/M2/M3) systems

### 🔍 **Real-World Use Cases**

- **📚 Digital libraries**: Batch digitize historical documents and books
- **🏛️ Academic research**: Extract and translate ancient texts (specialized Akkadian support)  
- **💼 Business documents**: Process invoices, contracts, forms at scale
- **🗃️ Archive digitization**: Convert paper records to searchable databases
- **📋 Document analysis**: Extract structured data from unstructured sources

### Recent Optimization (v2.0)

This project has been **significantly streamlined** for better performance and maintainability:

- **Unified Entry Point**: Single `run_pipeline.py` replaces multiple legacy scripts
- **Centralized Configuration**: All settings managed through `config.json`
- **Reduced File Count**: Cleaned from 50+ files to 17 essential modules (66% reduction)
- **Space Efficient**: Removed 2-3GB tessdata directory (using PaddleOCR's built-in models)
- **Production Focus**: Single optimized pipeline in `production/comprehensive_pipeline.py`
- **Clean Architecture**: No redundant files, clear module responsibilities

**Benefits**: Faster startup, easier deployment, reduced complexity, improved maintainability.

---

## Directory Structure

```
OCR_pipeline/
├── Configuration & Entry Points
│   ├── config.json                 # Main configuration file (EDIT THIS!)
│   ├── run_pipeline.py            # Main entry point (START HERE!)
│   ├── setup.py                   # First-time setup script
│   ├── quick_start.py             # Interactive examples and tutorials
│   ├── run_ocr.bat                # Windows GUI-like interface
│   └── requirements.txt           # Python dependencies
│
├── Core Source Code
│   └── src/                       # Core processing modules (17 essential files)
│       ├── Intelligence Layer
│       │   ├── pipeline.py        # Main processing orchestration
│       │   ├── llm_correction.py  # AI text correction engine
│       │   ├── akkadian_extract.py # Academic text extraction
│       │   └── lang_and_extract.py # Language detection
│       │
│       ├── Vision & OCR Layer
│       │   ├── ocr_utils.py       # OCR engine interfaces
│       │   ├── preprocess.py      # Image preprocessing
│       │   ├── orientation.py     # Image orientation detection
│       │   └── reading_order.py   # Text flow detection
│       │
│       ├── Data & Output Layer
│       │   ├── csv_writer.py      # CSV output formatting
│       │   ├── aggregated_csv.py  # Advanced CSV operations
│       │   ├── pdf_utils.py       # PDF handling utilities
│       │   └── translations_pdf.py # Academic PDF reports
│       │
│       ├── System & Config Layer
│       │   ├── config.py          # Configuration management
│       │   ├── cli.py             # Command-line utilities
│       │   ├── healthcheck.py     # System diagnostics
│       │   ├── preflight.py       # Pre-run validation
│       │   └── alignment_report.py # Quality assurance
│
├── Production Pipeline
│   └── production/                # Production-ready implementation
│       └── comprehensive_pipeline.py # Main production pipeline (CORE!)
│
├── Data Directories
│   └── data/
│       ├── input/                 # Default input directory (PUT FILES HERE!)
│       ├── input_pdfs/           # Alternative PDF input directory
│       ├── output/               # Processing results (CHECK HERE!)
│       └── samples/              # Test files and examples
│
├── Testing & Quality
│   └── tests/                    # Unit tests and test data
│
├── Documentation
│   └── doc/                      # Additional documentation
│       ├── README.md             # General documentation
│       ├── AKKADIAN_FEATURE.md   # Akkadian processing details
│       ├── AUTHORS               # Project contributors
│       └── LICENSE               # License information
│
└── Environment
    ├── .env/                     # Python virtual environment
    └── .gitignore               # Git ignore rules
```

---

## Key Files Explained

### **Start Here Files**

#### `run_pipeline.py` - **Main Entry Point**
- **Purpose**: The primary way to run the OCR pipeline
- **Usage**: `python run_pipeline.py`
- **Features**: 
  - Reads configuration from `config.json`
  - Supports single file or batch processing
  - Handles all supported file formats
  - Provides comprehensive error handling and logging

#### `config.json` - **Configuration Hub**
- **Purpose**: Central configuration for all processing options
- **Edit This To**: 
  - Set input/output directories
  - Choose file formats to process
  - Enable/disable AI features
  - Configure language settings
  - Adjust performance parameters

#### `setup.py` - **First-Time Setup**
- **Purpose**: Automated environment setup and dependency checking
- **Run Once**: Installs packages, creates directories, validates system
- **Features**: Interactive setup with health checks

### **Core Processing Files**

#### `src/pipeline.py` - **Processing Orchestrator**
```python
# Main image processing coordination
def process_image(img, config):
    # 1. Preprocessing (rotation, enhancement)
    # 2. OCR execution (PaddleOCR/Tesseract)
    # 3. Text extraction and ordering
    # 4. Language detection
    # 5. Quality validation
```

#### `production/comprehensive_pipeline.py` - **Production Engine**
```python
# Full-featured production pipeline
class ComprehensivePipeline:
    def process_pdf(pdf_path, output_dir):
        # Enterprise-grade PDF processing
    
    def process_image(image_path, output_dir):
        # Single image processing
```

#### `src/llm_correction.py` - **AI Text Enhancement**
```python
# LLM-powered text correction
class LLMCorrector:
    def correct_text(text, language):
        # Uses Ollama/Mistral to fix OCR errors
```

### **Data Processing Files**

#### `src/csv_writer.py` - **Output Formatting**
- Creates structured CSV files with text, confidence scores, and metadata
- Handles different output formats (line-by-line vs. page-aggregated)

#### `src/akkadian_extract.py` - **Academic Specialization**
- Detects Akkadian transliterations in academic texts
- Extracts translation mappings
- Generates academic PDF reports

### **Utility Files**

#### `src/ocr_utils.py` - **OCR Engine Interface**
- Abstracts PaddleOCR and Tesseract implementations
- Provides unified OCR interface
- Handles different confidence scoring methods

#### `src/pdf_utils.py` - **PDF Processing**
- PDF to image conversion
- Multi-page handling
- Format validation

---

## ⚙️ Configuration Guide

### 🎯 **TL;DR: Essential Settings**

**First time?** Only edit these in `config.json`:
```json
{
  "input": {
    "input_directory": "./data/input",      // 📁 Where your files are
    "supported_formats": [".pdf", ".png", ".jpg"]  // 📄 What files to process
  },
  "output": {
    "output_directory": "./data/output"     // 💾 Where results go
  },
  "llm": {
    "enable_correction": false              // 🤖 Start with false, enable later
  }
}
```

**That's it!** Run `python run_pipeline.py` and you're processing documents.

### 🛠️ **Full Configuration Reference**

Our config system is designed to be **powerful but simple**. Here's the complete structure:

```json
{
  "input": {
    "input_directory": "./data/input",           // Where to find files
    "supported_formats": [".pdf", ".png", ".jpg"], // What files to process
    "process_all_files": true,                   // Process all vs. single file
    "specific_file": "",                         // Path for single file mode
    "recursive_search": false                    // Search subdirectories
  },
  "output": {
    "output_directory": "./data/output",         // Where to save results
    "create_subdirectories": true,               // Organize by input file
    "timestamp_outputs": true                    // Add timestamps to folders
  },
  "ocr": {
    "engine": "paddleocr",                       // OCR engine to use
    "dpi": 300,                                  // Image resolution
    "languages": ["en", "tr", "de"],             // Languages to detect
    "enable_text_correction": true,              // Use AI correction
    "confidence_threshold": 0.5                  // Minimum text confidence
  },
  "llm": {
    "enable_correction": true,                   // Enable AI features
    "provider": "ollama",                        // LLM provider
    "model": "mistral:latest",                   // AI model to use
    "base_url": "http://localhost:11434"         // Ollama server URL
  }
}
```

### ⏱️ **Common Configuration Scenarios**

**Copy-paste these into your `config.json` for instant productivity:**

#### **Scenario 1: "Just Process My PDFs"** (Beginner-friendly)
```json
{
  "input": {
    "input_directory": "./my_pdfs",
    "supported_formats": [".pdf"],
    "process_all_files": true
  },
  "ocr": {
    "engine": "paddleocr",
    "languages": ["en"]
  },
  "llm": {
    "enable_correction": false  // Fast processing, no AI
  }
}
```
**Perfect for**: Quick digitization, testing the pipeline, batch PDF conversion

#### **Scenario 2: "High-Quality AI Processing"** (Best results)
```json
{
  "input": {
    "input_directory": "./important_docs",
    "supported_formats": [".pdf", ".png", ".jpg"]
  },
  "ocr": {
    "engine": "paddleocr",
    "languages": ["en", "de", "fr"],
    "confidence_threshold": 0.7
  },
  "llm": {
    "enable_correction": true,
    "provider": "ollama",
    "model": "mistral:latest"
  }
}
```
**Perfect for**: Academic papers, important business documents, multi-language content

#### **Scenario 3: "Academic Research"** (Specialized features)
```json
{
  "input": {
    "input_directory": "./research_papers",
    "supported_formats": [".pdf"]
  },
  "ocr": {
    "languages": ["en", "de"]
  },
  "akkadian": {
    "enabled": true,
    "generate_reports": true
  },
  "llm": {
    "enable_correction": true,
    "model": "mistral:latest"
  }
}
```
**Perfect for**: Historical documents, archaeological texts, academic transcription

#### **Scenario 2: High-Quality Academic Processing**
```json
{
  "ocr": {
    "dpi": 600,                    // Higher quality
    "confidence_threshold": 0.8    // Stricter quality
  },
  "akkadian": {
    "enable_extraction": true,     // Enable academic features
    "generate_pdf_report": true
  },
  "llm": {
    "enable_correction": true,     // Full AI enhancement
    "timeout": 60                  // Longer timeout for quality
  }
}
```

#### **Scenario 3: Single Image Processing**
```json
{
  "input": {
    "process_all_files": false,
    "specific_file": "./document.jpg"
  },
  "output": {
    "create_subdirectories": false
  }
}
```

---

## 💻 Development Setup

### 🎯 **New Developer Onboarding**

**First time contributing?** Follow this path:

#### **Step 1: Get Your Bearings** (5 minutes)
```bash
# Clone and explore
git clone <repo-url> && cd OCR_pipeline

# What am I looking at?  
python quick_start.py           # Interactive tour of the codebase
ls src/                         # 17 core modules (the main logic)
cat config.json                 # The main configuration file
cat run_pipeline.py             # The main entry point
```

#### **Step 2: Set Up Your Environment** (Choose one)

**🐳 Option A: Docker (Recommended - No dependency hassles)**
```bash
# Build dev container (use your platform)
docker build -f Dockerfile.arm64 -t ocr-dev .    # Apple Silicon
docker build -t ocr-dev .                        # Intel/AMD64

# Start development environment  
docker run --rm -it -v "$PWD":/workspace -w /workspace ocr-dev bash

# Inside container - you're ready to code!
python run_pipeline.py --validate-only          # Test everything works
python -c "from src.healthcheck import run_health_check; run_health_check()"
```

**🐍 Option B: Local Python (More control, more setup)**
```bash
# Create isolated environment (highly recommended)
python -m venv .env && source .env/bin/activate  # Linux/Mac
# OR: .env\Scripts\activate                       # Windows

# Install everything
pip install -r requirements.txt
python setup.py                                  # Automated setup + validation

# Verify it works
python run_pipeline.py --validate-only
```

#### **Step 3: Learn the Architecture** (10 minutes)
```bash
# Understand the key files
echo "Main entry point:" && head -20 run_pipeline.py
echo "Core pipeline logic:" && head -20 src/pipeline.py  
echo "Production implementation:" && head -20 production/comprehensive_pipeline.py
echo "Configuration structure:" && head -20 config.json

# See it in action
python run_pipeline.py --input-dir data/samples --dry-run  # Preview without processing
python run_pipeline.py --input-dir data/samples            # Actually process samples
```

#### **Step 4: Make Your First Change** (15 minutes)
```bash
# Start with something simple
# Example: Add your name to the contributors

# 1. Make a small change
echo "# Added by [Your Name]" >> src/pipeline.py

# 2. Test that nothing breaks  
python run_pipeline.py --validate-only

# 3. Test with real data
python run_pipeline.py --input-dir data/samples

# 4. If all good, you're ready to make real contributions!
```

### 🧪 **Development Workflow & Best Practices**

#### **🔄 Daily Development Loop**
```bash
# 1. Start your environment
source .env/bin/activate    # Or start Docker container

# 2. Before making changes - ensure everything works
python run_pipeline.py --validate-only
python -c "from src.healthcheck import run_health_check; run_health_check()"

# 3. Make your changes
# ... edit files in src/, production/, etc. ...

# 4. Test continuously  
python run_pipeline.py --dry-run          # Preview mode  
python run_pipeline.py --input-dir data/samples --max-files 1  # Test 1 file
python run_pipeline.py --input-dir data/samples  # Full test

# 5. Check for issues
python -c "from src.healthcheck import run_health_check; run_health_check()"
```

#### **🎯 Key Development Files**

**Start here for most changes:**
- `src/pipeline.py` - 🧠 Main orchestration logic
- `production/comprehensive_pipeline.py` - 🚀 Production implementation  
- `config.json` - ⚙️ Configuration schema & defaults
- `run_pipeline.py` - 🎯 Main entry point & CLI handling

**Advanced modifications:**
- `src/ocr_utils.py` - OCR engine interfaces (PaddleOCR/Tesseract)
- `src/llm_correction.py` - AI text correction with Ollama/Mistral  
- `src/akkadian_extract.py` - Academic text extraction features
- `src/preprocess.py` - Image preprocessing & enhancement

#### **🔧 Development Tools & Commands**

**Health checks & validation:**
```bash
python -c "from src.healthcheck import run_health_check; run_health_check()"  # Full system check
python -c "import src.pipeline; print('✅ Core modules OK')"
python -c "import production.comprehensive_pipeline; print('✅ Production module OK')"
```

**Testing & debugging:**
```bash  
python run_pipeline.py --dry-run --verbose          # Preview + detailed logs
python run_pipeline.py --input-dir data/samples --max-files 1  # Test single file
python run_pipeline.py --input-dir data/samples     # Full sample test
```

**Module-specific testing:**
```bash
python -c "import src.ocr_utils; print('OCR module OK')"
python -c "import src.pipeline; print('Pipeline module OK')"
python -c "from src.pdf_utils import validate_pdf; print('PDF utils OK')"
```

### 🚨 **Common Development Issues & Solutions**

#### **Issue 1: "Import Error" or "Module Not Found"**
```bash
# Check Python path and virtual environment
python -c "import sys; print('\n'.join(sys.path))"
which python    # Should be in your .env/ directory if using venv

# Fix: Make sure you're in your virtual environment
source .env/bin/activate    # Linux/Mac
# OR .env\Scripts\activate  # Windows
```

#### **Issue 2: "No files found to process"**
```bash
# Check your directory structure
ls -la data/input/      # Should have files here
ls -la data/samples/    # Sample files to test with

# Fix: Copy test files
cp data/samples/* data/input/
```

#### **Issue 3: OCR not working / "PaddleOCR failed"**
```bash
# Test OCR components individually
python -c "import paddleocr; print('PaddleOCR OK')"
python -c "from src.ocr_utils import test_ocr; test_ocr()"

# Fix: Reinstall PaddlePaddle 3.x ecosystem dependencies
pip install --force-reinstall "paddlepaddle>=3.0.0,<4.0.0" "paddleocr>=3.2.0" opencv-python-headless

# If still issues, ensure PaddleX dependencies are installed
pip install --force-reinstall "PyYAML>=6.0" "typing-extensions>=4.12"
```

#### **Issue 4: Docker build issues on Apple Silicon**
```bash
# Use the ARM64-optimized Dockerfile
docker build -f Dockerfile.arm64 -t ocr-dev .

# If still issues, clean build
docker build --no-cache -f Dockerfile.arm64 -t ocr-dev .

# Check for success indicators in build logs
docker build -f Dockerfile.arm64 -t ocr-dev 2>&1 | grep -E "(cached|wheel)"
```

#### **Issue 5: Configuration not taking effect**
```bash
# Verify JSON syntax
python -c "import json; print(json.load(open('config.json')))"

# Test configuration loading
python -c "from src.config import load_config; print(load_config())"

# Check for hidden characters/encoding issues
file config.json    # Should show UTF-8 text
```

### 💡 **Pro Development Tips**

1. **Use `--dry-run` liberally** - Preview what will happen without processing
2. **Test with `data/samples/`** - Always test changes with known-good files first  
3. **Check health after changes** - `python -c "from src.healthcheck import run_health_check; run_health_check()"`
4. **Start Docker container once** - Keep it running, work inside it for consistency
5. **Use verbose logging** - Add `--verbose` to see what's happening under the hood
```bash
# Format code (if you have black installed)
black src/ production/

# Check imports and dependencies
python -c "import src.pipeline; print('✅ Core modules OK')"
python -c "import production.comprehensive_pipeline; print('✅ Production module OK')"
```

### 🎯 **Key Development Areas**

#### **OCR Engine Improvements** (`src/ocr_utils.py`)
- Add new OCR engines
- Improve text extraction accuracy
- Optimize preprocessing

#### **AI Integration** (`src/llm_correction.py`)
- Enhance text correction algorithms
- Add new AI providers
- Improve prompt engineering

#### **Academic Features** (`src/akkadian_extract.py`)
- Extend language support
- Improve archaeological text detection
- Add new academic output formats

#### **Pipeline Optimization** (`production/comprehensive_pipeline.py`)
- Performance improvements
- Memory optimization
- Batch processing enhancements

### 🐛 **Common Development Issues**

#### **Import Errors**
```bash
# Fix Python path issues
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
# OR
python -m src.module_name  # Use module syntax
```

#### **Dependency Conflicts**
```bash
# Use Docker to avoid local conflicts
docker build -f Dockerfile.arm64 -t ocr-dev .
docker run --rm -it -v "$PWD":/workspace -w /workspace ocr-dev

# OR reset virtual environment
deactivate
rm -rf .env
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

#### **Performance Testing**
```bash
# Profile processing time
time python run_pipeline.py --input-dir data/samples

# Memory usage monitoring
python -m memory_profiler run_pipeline.py  # If memory_profiler installed
```

### 📚 **Learning Resources**

#### **Understanding the Architecture**
1. Read `production/comprehensive_pipeline.py` - Main processing logic
2. Check `src/pipeline.py` - Core orchestration
3. Review `config.json` - Configuration options
4. Explore `doc/` - Additional documentation

#### **Testing with Sample Data**
```bash
# Use provided samples
ls data/samples/
python run_pipeline.py --input-dir data/samples

# Add your own test files
cp your_test.pdf data/input/
python run_pipeline.py
```

---

## Usage Examples

### Command Line Usage

```bash
# Basic processing with default config
python run_pipeline.py

# Use custom configuration
python run_pipeline.py -c my_config.json

# Validate configuration only
python run_pipeline.py --validate-only

# Preview what would be processed
python run_pipeline.py --dry-run

# Get help
python run_pipeline.py --help
```

### Windows Users
```batch
# Interactive menu interface
run_ocr.bat

# Follow the numbered menu options:
# 1. Setup OCR Pipeline
# 2. Run Quick Start Examples
# 3. Process files with default config
# etc.
```

### Python API Usage

```python
import sys
sys.path.append('./production')
from comprehensive_pipeline import ComprehensivePipeline, PipelineConfig

# Create configuration
config = PipelineConfig(
    llm_provider='ollama',
    enable_akkadian_extraction=True,
    dpi=300
)

# Initialize pipeline
pipeline = ComprehensivePipeline(config)

# Process a PDF
result = pipeline.process_pdf(
    pdf_path="./document.pdf",
    output_dir="./results"
)

# Process an image
result = pipeline.process_image(
    image_path="./image.jpg",
    output_dir="./results"
)

print(f"Processed {result['pages_processed']} pages")
print(f"Output: {result['output_csv']}")
```

---

## Architecture Overview

### Processing Flow

```
Input Files → Preprocessing → OCR → AI Enhancement → Output
```

#### Detailed Flow:
1. **Input Validation** (`preflight.py`)
   - File format checking
   - System readiness validation
   - Dependency verification

2. **Image Preprocessing** (`preprocess.py`)
   - Orientation correction
   - Quality enhancement
   - Noise reduction

3. **OCR Processing** (`ocr_utils.py`)
   - Text detection and recognition
   - Confidence scoring
   - Multi-engine support

4. **Text Enhancement** (`llm_correction.py`)
   - AI-powered error correction
   - Language-aware improvements
   - Context-based fixes

5. **Specialized Processing** (`akkadian_extract.py`)
   - Academic text detection
   - Translation extraction
   - Research report generation

6. **Output Generation** (`csv_writer.py`)
   - Structured data formatting
   - Multiple output formats
   - Quality reports

### Key Design Patterns

#### **Configuration-Driven Architecture**
- All behavior controlled by `config.json`
- No code changes needed for different use cases
- Runtime configuration validation

#### **Modular Processing Pipeline**
- Each stage is independently testable
- Easy to add new processing steps
- Clear separation of concerns

#### **Multi-Engine Support**
- Abstract OCR interface
- Fallback mechanisms
- Performance optimization

---

## API Reference

### Main Pipeline Classes

#### `ComprehensivePipeline`
```python
class ComprehensivePipeline:
    def __init__(config: PipelineConfig):
        """Initialize with configuration"""
    
    def process_pdf(pdf_path: str, output_dir: str) -> Dict:
        """Process PDF file and return results"""
    
    def process_image(image_path: str, output_dir: str) -> Dict:
        """Process single image and return results"""
```

#### `PipelineConfig`
```python
@dataclass
class PipelineConfig:
    llm_provider: str = "ollama"
    llm_model: str = "mistral:latest"
    dpi: int = 300
    enable_reading_order: bool = True
    enable_llm_correction: bool = True
    enable_akkadian_extraction: bool = True
    # ... more configuration options
```

### Utility Functions

#### OCR Processing
```python
from src.ocr_utils import ocr_paddle_lines, ocr_tesseract_lines

# Process image with PaddleOCR
lines = ocr_paddle_lines(image_array)

# Process with Tesseract
lines = ocr_tesseract_lines(image_array, langs="eng+tur")
```

#### Text Correction
```python
from src.llm_correction import LLMCorrector

corrector = LLMCorrector(provider="ollama", model="mistral:latest")
result = corrector.correct_text("text with errors", language="english")
```

---

## 🤝 Contributing

### 🌟 **Welcome, Fellow Developers!**

We've **streamlined this codebase** to be contributor-friendly. Whether you're fixing a typo or adding major features, here's how to jump in effectively:

### 🎯 **Find Your Contribution Style**

#### 🚀 **"Quick Wins"** (Perfect for first-time contributors)
- **📝 Documentation fixes** - Typos, unclear explanations, missing examples
- **⚙️ Configuration examples** - Add new scenarios to `config.json` examples  
- **🔧 Error messages** - Make error messages more helpful and actionable
- **🧪 Test cases** - Add tests for existing functions (see `tests/` directory)
- **🔍 Code comments** - Add explanatory comments to complex functions

*Time commitment: 15-60 minutes • Impact: High • Difficulty: Low*

#### 💡 **"Feature Enhancer"** (Great for regular contributors)  
- **🔌 New OCR engines** - Integrate additional OCR providers in `src/ocr_utils.py`
- **🖼️ Preprocessing improvements** - Enhance image processing in `src/preprocess.py`  
- **🌍 Language support** - Extend multi-language capabilities in `src/lang_and_extract.py`
- **📄 Output formats** - Add new export formats in `src/csv_writer.py`
- **⚡ Performance optimizations** - Speed up batch processing, memory usage

*Time commitment: 2-8 hours • Impact: High • Difficulty: Medium*

#### 🧠 **"Architecture Contributor"** (For experienced developers)
- **🤖 AI/LLM providers** - Add new AI correction backends in `src/llm_correction.py`
- **🏛️ Academic features** - Extend specialized text extraction in `src/akkadian_extract.py`  
- **🐳 Container optimization** - Further improve ARM64 Docker builds
- **⚡ Production pipeline** - Enhance `production/comprehensive_pipeline.py`
- **🏗️ System architecture** - Major refactoring, new processing paradigms

*Time commitment: 1-4 weeks • Impact: Very High • Difficulty: High*

### 📋 **Step-by-Step Contribution Process**

#### **🔧 Setup Your Development Environment** (5 minutes)
```bash
# 1. Fork & clone
git clone https://github.com/YOUR-USERNAME/OCR_pipeline.git
cd OCR_pipeline

# 2. Set up environment (choose one)
# Docker (recommended):
docker build -f Dockerfile.arm64 -t ocr-dev .    # Apple Silicon
docker build -t ocr-dev .                        # Intel/AMD64

# Start development environment  
docker run --rm -it -v "$PWD":/workspace -w /workspace ocr-dev bash

# OR Local Python:
python -m venv .env && source .env/bin/activate
pip install -r requirements.txt && python setup.py
```

#### **🎯 Understand the Codebase** (10 minutes)
```bash
# Get the lay of the land
python quick_start.py                    # Interactive tour
ls src/                                  # 17 core modules
cat config.json                          # Main configuration
python run_pipeline.py --validate-only  # Test everything works
```

#### **✏️ Make Your Changes** (Varies by contribution)
```bash
# Key files for different types of changes:
src/pipeline.py              # Core processing logic
src/ocr_utils.py            # OCR engine integrations  
src/llm_correction.py       # AI text correction
production/comprehensive_pipeline.py  # Production implementation
config.json                 # Configuration schema
```

#### **🧪 Test Your Changes** (5-10 minutes)
```bash
# Essential testing before submitting:
python run_pipeline.py --validate-only                    # Config validation
python run_pipeline.py --dry-run --input-dir data/samples # Preview mode
python run_pipeline.py --input-dir data/samples           # Full test
python -c "from src.healthcheck import run_health_check; run_health_check()"  # Health check
```

#### **📤 Submit Your Contribution**
```bash
# Standard Git workflow
git add .
git commit -m "feat: add [your contribution description]"
git push origin your-branch-name
# Create Pull Request on GitHub
```

### 🛠️ **Code Guidelines & Best Practices**

#### **📁 File Organization Principles**
- **Core processing logic** → `src/` (17 focused, single-purpose modules)
- **Production implementations** → `production/`  
- **User-facing scripts** → Root directory (`run_pipeline.py`, `setup.py`, etc.)
- **Configuration** → `config.json` (single source of truth)
- **Documentation** → `doc/` and this `README.md`

#### **💻 Coding Standards**
```python
# Follow these patterns for consistency:
def new_function(config: dict, input_path: str) -> dict:
    """
    Clear, concise docstring explaining what this function does.
    
    Args:
        config: Configuration dictionary from config.json
        input_path: Path to input file or directory
        
    Returns:
        dict: Status and results in standardized format
    """
    try:
        # Implementation here
        result = process_something(input_path, config)
        return {
            "status": "success", 
            "data": result,
            "message": "Processing completed successfully"
        }
    except Exception as e:
        return {
            "status": "error", 
            "error": str(e),
            "message": f"Failed to process {input_path}"
        }
```

#### **⚙️ Configuration Changes**
When adding new features, extend `config.json` thoughtfully:
```json
{
  "new_feature": {
    "enabled": false,              // Default to disabled for safety
    "parameter": "default_value",   // Sensible defaults
    "advanced_options": {
      "timeout": 30,
      "retries": 3
    }
  }
}
```

### 🚀 **Deployment Readiness Checklist**

Before deploying, ensure you've validated:

**✅ System Requirements:**
```bash
# Check your system
uname -m                    # Verify architecture (x86_64 or arm64)
docker --version            # Ensure Docker 20.10+ installed
python --version            # Ensure Python 3.8+ for native installs

# Verify resources
docker system info | grep "Total Memory"  # 4GB+ RAM recommended
df -h                      # 10GB+ free disk space
```

**✅ Platform-Specific Setup:**
```bash
# Apple Silicon (M1/M2/M3) - Use optimized build:
docker build -f Dockerfile.arm64 -t ocr-pipeline .

# Intel/AMD64 (Windows, Intel Macs, Linux) - Use standard build:  
docker build -t ocr-pipeline .

# Windows users - Use PowerShell for best compatibility:
# PowerShell: docker run --rm -v "${PWD}:/workspace" -w /workspace ocr-pipeline python run_pipeline.py
```

**✅ Functionality Validation:**
```bash
# Quick health check (run after installation):
python -c "from src.healthcheck import run_health_check; run_health_check()"

# Test Docker container:
docker run --rm ocr-pipeline python -c "import cv2, fitz, paddleocr; print('✅ All dependencies working')"

# Validate with sample file:
docker run --rm -v "$PWD:/workspace" -w /workspace ocr-pipeline python run_pipeline.py --dry-run
```

**✅ Common Issues Resolved:**
- ✅ **PyMuPDF compilation errors** → Fixed with binary-only wheels in ARM64 builds
- ✅ **Platform architecture mismatches** → Clear Dockerfile selection guide
- ✅ **Windows path separator issues** → PowerShell examples with proper path handling  
- ✅ **Corporate firewall problems** → Proxy configuration examples
- ✅ **Permission denied on Linux** → Docker group setup instructions
- ✅ **Memory issues** → Resource allocation and batch size recommendations

> **🔧 Need Help?** Check the comprehensive [Troubleshooting](#troubleshooting) section above, or see `README_docker.md` for Docker-specific deployment guides across Windows, macOS, and Linux.
