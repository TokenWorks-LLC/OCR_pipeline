# OCR Pipeline - Developer's Guide

> **A comprehensive document processing system with AI-powered OCR, multi-language support, and specialized academic text extraction capabilities.**

## Table of Contents

- [Quick Start](#quick-start)
- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Key Files Explained](#key-files-explained)
- [Configuration Guide](#configuration-guide)
- [Development Setup](#development-setup)
- [Usage Examples](#usage-examples)
- [Evaluation Modes](#evaluation-modes)
- [Gold Pages System](#gold-pages-system)
- [Analysis System](#analysis-system)
- [Smart LLM Triggering](#smart-llm-triggering)
- [API Reference](#api-reference)
- [Architecture Overview](#architecture-overview)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### For End Users
```bash
# 1. Setup (first time only)
python setup.py

# 2. Configure your processing (edit config.json)
# 3. Add files to data/input/ directory
# 4. Run processing
python run_pipeline.py
```

### For Developers
```bash
# 1. Clone and setup development environment
pip install -r requirements.txt

# 2. Run interactive setup
python quick_start.py

# 3. Explore the codebase structure below
```

---

## Project Overview

This OCR pipeline is designed for **academic and professional document processing** with these key capabilities:

- **Multi-format processing**: PDFs, PNG, JPG, JPEG, TIFF, BMP
- **AI-powered accuracy**: LLM-based text correction using Ollama/Mistral
- **Academic specialization**: Akkadian text extraction for archaeological research
- **Multi-language support**: English, Turkish, German, French, Italian
- **Batch processing**: Handle large document collections efficiently
- **Production-ready**: Enterprise-grade architecture with comprehensive error handling

### Recent Optimization (v2.0)

This project has been **significantly streamlined** for better performance and maintainability:

- **✅ Unified Entry Point**: Single `run_pipeline.py` replaces multiple legacy scripts
- **✅ Centralized Configuration**: All settings managed through `config.json`
- **✅ Reduced File Count**: Cleaned from 50+ files to 17 essential modules (66% reduction)
- **✅ Space Efficient**: Removed 2-3GB tessdata directory (using PaddleOCR's built-in models)
- **✅ Production Focus**: Single optimized pipeline in `production/comprehensive_pipeline.py`
- **✅ Clean Architecture**: No redundant files, clear module responsibilities

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

## Configuration Guide

### Basic Configuration Structure

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

### **Common Configuration Scenarios**

#### **Scenario 1: Basic PDF Processing**
```json
{
  "input": {
    "input_directory": "./my_pdfs",
    "supported_formats": [".pdf"],
    "process_all_files": true
  },
  "llm": {
    "enable_correction": false  // Disable AI for speed
  }
}
```

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

## Development Setup

### Prerequisites
- **Python 3.8+** (Required)
- **Git** (For version control)
- **Ollama** (Optional, for AI features)

### Installation Steps

1. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd OCR_pipeline
   ```

2. **Setup Environment**
   ```bash
   # Automated setup (recommended)
   python setup.py
   
   # OR Manual setup
   pip install -r requirements.txt
   ```

3. **Verify Installation**
   ```bash
   python run_pipeline.py --validate-only
   ```

### Development Workflow

1. **Explore Examples**
   ```bash
   python quick_start.py  # Interactive tutorials
   ```

2. **Test Configuration**
   ```bash
   python run_pipeline.py --dry-run  # Preview without processing
   ```

3. **Run Processing**
   ```bash
   python run_pipeline.py  # Full processing
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

## Evaluation Modes

The OCR pipeline includes three specialized evaluation modes for testing and benchmarking different processing configurations.

### Available Evaluation Modes

#### **1. Basic Mode** (`config_eval_basic.json`)
- **Purpose**: Standard OCR processing without advanced features
- **Features**: 
  - Basic OCR with PaddleOCR
  - Confidence threshold: 0.7
  - LLM correction: Disabled
  - Benchmark recording: Enabled
- **Use Case**: Baseline performance measurement

#### **2. Advanced Mode** (`config_eval_advanced.json`)
- **Purpose**: Full-featured processing with LLM correction
- **Features**:
  - OCR with confidence threshold: 0.7
  - LLM correction: Enabled
  - Reading order detection: Enabled
  - Akkadian extraction: Disabled
  - Benchmark recording: Enabled
- **Use Case**: Quality-focused processing

#### **3. Akkadian Mode** (`config_eval_akkadian.json`)
- **Purpose**: Academic research with specialized Akkadian text extraction
- **Features**:
  - OCR with confidence threshold: 0.7
  - LLM correction: Enabled
  - Akkadian extraction: Enabled
  - Translation PDF generation: Enabled
  - Akkadian confidence threshold: 0.8
  - Benchmark recording: Enabled
- **Use Case**: Archaeological and linguistic research

### Running Evaluations

```bash
# Basic evaluation
python run_eval_incremental.py -c config_eval_basic.json

# Advanced evaluation
python run_eval_incremental.py -c config_eval_advanced.json

# Akkadian evaluation
python run_eval_incremental.py -c config_eval_akkadian.json

# Smart LLM evaluation (optimized performance)
python run_eval_smart_llm.py -c config_eval_akkadian_smart.json
```

### Evaluation Output Structure

```
data/eval_output/
├── eval_research_results_basic/          # Basic mode results
├── eval_research_results_advanced/       # Advanced mode results
├── eval_research_results_akkadian/       # Akkadian mode results
└── eval_research_results_akkadian_smart/ # Smart LLM results
    ├── document_name/
    │   ├── comprehensive_report.json     # Detailed results
    │   ├── comprehensive_results.csv     # CSV data
    │   └── translations_report.pdf       # Akkadian translations (if enabled)
    └── processing_report_YYYYMMDD_HHMMSS.json
```

---

## Gold Pages System

The Gold Pages system provides ground truth integration for accuracy measurement and LLM correction evaluation. This system allows you to measure the effectiveness of OCR and LLM corrections against manually verified reference data.

### What Are Gold Pages?

Gold Pages are **manually verified ground truth data** containing:
- **Akkadian-translation pairs**: Expert-verified text and translations
- **Confidence scores**: Quality ratings for each entry
- **Document references**: Source document and page information
- **Verification metadata**: Who verified the data and when

### Gold Pages Data Structure

```json
{
  "metadata": {
    "total_entries": 150,
    "last_updated": "2024-01-15T10:30:00",
    "version": "1.0"
  },
  "entries": [
    {
      "document_id": "akkadian_doc_001",
      "page_number": 1,
      "akkadian_text": "lugal",
      "translation_text": "king",
      "confidence_score": 0.95,
      "verified_by": "expert_linguist",
      "created_date": "2024-01-15T10:30:00",
      "notes": "Common Akkadian term"
    }
  ]
}
```

### How to Use Gold Pages

#### **Step 1: Add Ground Truth Data**
```python
from src.gold_pages_manager import create_gold_pages_manager

# Initialize Gold Pages Manager
manager = create_gold_pages_manager("./data/gold_pages")

# Add your intern's ground truth data
manager.add_gold_page(
    document_id="doc_001",
    page_number=1,
    akkadian_text="lugal",
    translation_text="king",
    confidence_score=0.95,
    verified_by="expert_linguist",
    notes="Common Akkadian term"
)
```

#### **Step 2: Run Gold Pages Evaluation**
```bash
# V2 Gold Pages + Smart LLM evaluation (RECOMMENDED)
python run_eval_gold_pages_v2.py -c config_eval_gold_pages_akkadian_v2.json

# Basic Gold Pages evaluation (legacy)
python run_eval_gold_pages.py -c config_eval_gold_pages_akkadian.json
```

#### **Step 3: Analyze Results**
```bash
# Run analysis menu
python run_analysis_menu.py

# Select options:
# 1. Run Summary Analysis
# 2. Run Comprehensive Analysis
# 3. View Available Evaluations
```

### Gold Pages Configuration

```json
{
  "gold_pages": {
    "enable_gold_pages": true,
    "gold_pages_directory": "./data/gold_pages",
    "enable_accuracy_measurement": true,
    "enable_before_after_comparison": true,
    "success_thresholds": {
      "min_improvement": 0.05,
      "max_time_increase": 0.5,
      "cost_per_accuracy_point": 0.10
    }
  }
}
```

### What You Get from Gold Pages

#### **Accuracy Measurements**
- **Character-level accuracy**: Precise character-by-character comparison
- **Word-level accuracy**: Word recognition accuracy metrics
- **Line-level accuracy**: Text line recognition accuracy
- **Before/after LLM comparison**: Measure LLM improvement effectiveness

#### **Success Metrics**
- **Minimum improvement**: +5% accuracy improvement required
- **Maximum time increase**: +50% processing time limit
- **Cost per accuracy point**: <$0.10 cost efficiency threshold
- **ROI justification**: Data-driven cost-benefit analysis

#### **Generated Reports**
- `gold_pages_analysis_report.json`: Detailed accuracy data
- `accuracy_measurements.json`: Before/after comparison metrics
- `gold_pages_summary.md`: Human-readable analysis report
- Visual charts and graphs showing improvements

### Integration Points

1. **Data Storage**: `./data/gold_pages/gold_pages.json`
2. **Evaluation Configs**: `config_eval_gold_pages_*.json`
3. **Evaluation Runners**: `run_eval_gold_pages.py`
4. **Analysis**: `run_analysis_menu.py`

**Important**: Gold Pages only affect evaluation mode - the main OCR pipeline remains unchanged!

### Intern Integration Guide

When your intern completes the ground truth data, follow these steps to integrate it:

#### **Step 1: Add Gold Pages Data**
```python
from src.gold_pages_manager import create_gold_pages_manager

# Initialize Gold Pages Manager
manager = create_gold_pages_manager("./data/gold_pages")

# Add each ground truth entry
manager.add_gold_page(
    document_id="doc_001",
    page_number=1,
    akkadian_text="lugal",
    translation_text="king",
    confidence_score=0.95,
    verified_by="intern_name",
    notes="Verified by intern on [date]"
)
```

#### **Step 2: Validate Data**
```python
# Check data integrity
validation = manager.validate_gold_pages()
print(f"Valid: {validation['is_valid']}")
print(f"Total entries: {validation['statistics']['total_entries']}")
```

#### **Step 3: Run V2 Evaluation**
```bash
# Run V2 evaluation with Gold Pages data
python run_eval_gold_pages_v2.py -c config_eval_gold_pages_akkadian_v2.json
```

#### **Step 4: Analyze Results**
```bash
# Run analysis to see accuracy improvements
python run_analysis_menu.py
```

#### **Expected Results**
- Character/word/line accuracy measurements
- Before/after LLM comparison
- Cost-benefit analysis
- Success threshold compliance

---

## Analysis System

A comprehensive analysis and comparison system for OCR pipeline evaluation results, providing detailed performance metrics, visualizations, and baseline accuracy assessment.

### Quick Start

```bash
# Install analysis dependencies
pip install -r requirements_analysis.txt

# Run analysis menu
python run_analysis_menu.py

# Or use integrated menus
python quick_start.py  # → Option 5
run_ocr.bat           # → Option 6
```

### Analysis Features

#### **Summary Analysis**
- **Mode Comparison**: Compare Basic vs Akkadian vs Advanced modes
- **Performance Metrics**: Success rate, confidence, processing time
- **Visual Charts**: Bar charts, heatmaps, distribution plots
- **Detailed Reports**: Markdown and JSON output formats

#### **Baseline Accuracy Assessment**
- **Character-Level Accuracy**: Precise character-by-character comparison
- **Word-Level Accuracy**: Word recognition accuracy metrics
- **Line-Level Accuracy**: Text line recognition accuracy
- **Ground Truth Management**: Store and manage reference data

#### **Interactive Menu System**
- **Mode Selection**: Choose which evaluation modes to compare
- **Custom Comparisons**: Select specific modes for detailed analysis
- **Quick Analysis**: Analyze all available modes at once
- **Baseline Generation**: Create baseline accuracy guidelines

### Analysis Outputs

#### **Visualizations**
- **Mode Comparison Chart**: Bar charts comparing success rates, confidence, and processing time
- **Confidence Distribution**: Histogram showing OCR confidence score distribution
- **Performance Heatmap**: Heatmap of all performance metrics across modes

#### **Reports**
- **Detailed Markdown Report**: Comprehensive analysis with recommendations
- **JSON Summary**: Machine-readable summary for programmatic access
- **Baseline Assessment**: Guidelines for establishing OCR accuracy baselines

#### **Metrics Tracked**
- **Success Rate**: Percentage of successfully processed pages
- **Average Confidence**: Mean OCR confidence score
- **Processing Time**: Time per page and total processing time
- **Text Elements**: Number of text elements extracted per page
- **Corrections Made**: Number of LLM corrections applied
- **Akkadian Translations**: Number of specialized translations found

### Baseline Accuracy Implementation

#### **Setting Up Ground Truth**
```bash
# Create ground truth directory
mkdir -p ./data/ground_truth

# Add ground truth files
./data/ground_truth/
├── document1_page_001.txt
├── document1_page_001_metadata.json
├── document1_page_002.txt
└── document1_page_002_metadata.json
```

#### **Using Ground Truth Manager**
```python
from src.baseline_accuracy import GroundTruthManager

gt_manager = GroundTruthManager()

# Save ground truth
gt_manager.save_ground_truth(
    document_id="document1",
    page_num=1,
    ground_truth_text="Reference text here"
)

# Load ground truth
gt_text = gt_manager.load_ground_truth("document1", 1)
```

#### **Accuracy Thresholds**
| Mode | Character Accuracy | Word Accuracy | Line Accuracy |
|------|-------------------|---------------|---------------|
| Basic | >95% | >90% | >85% |
| Akkadian | >92% | >87% | >80% |
| Advanced | >97% | >93% | >88% |

### Analysis File Structure

```
OCR_pipeline/
├── src/
│   ├── summary_analysis.py      # Main analysis engine
│   └── baseline_accuracy.py     # Accuracy calculation functions
├── run_analysis_menu.py         # Interactive analysis menu
├── demo_analysis.py             # Demo and examples
├── requirements_analysis.txt    # Analysis dependencies
└── data/
    ├── eval_output/             # Evaluation results
    │   ├── eval_research_results_basic/
    │   ├── eval_research_results_akkadian/
    │   └── eval_research_results_advanced/
    └── analysis_output/         # Analysis results
        ├── mode_comparison.png
        ├── confidence_distribution.png
        ├── metrics_heatmap.png
        ├── detailed_analysis_report.md
        └── analysis_summary.json
```

---

## Smart LLM Triggering

An optimization system that intelligently triggers LLM correction based on confidence scores and language type, significantly reducing processing time while maintaining quality.

### How Smart LLM Triggering Works

#### **For Akkadian Text**
- **Always sends to LLM** (regardless of confidence)
- **Reason**: Specialized language needs expert correction
- **Detection**: Uses regex patterns for Akkadian characters (ā, ē, ī, ū, š, ṣ, ṭ, ḫ)

#### **For Modern Languages (EN, TR, DE, FR)**
- **Only sends to LLM if confidence < 0.8**
- **Skips LLM if confidence ≥ 0.8**
- **Reason**: High-confidence modern text is already good

### Expected Performance Impact

- **LLM Calls**: ~40-60% reduction (only Akkadian + low-confidence modern)
- **Processing Time**: ~40-60% reduction
- **Quality**: Same or better (high-confidence text stays unchanged)

### Smart LLM Configuration

```json
{
  "llm": {
    "enable_correction": true,
    "smart_triggering": true,
    "modern_lang_max_confidence": 0.8,
    "akkadian_always_correct": true
  }
}
```

### Running Smart LLM Evaluation

```bash
# Run smart LLM evaluation
python run_eval_smart_llm.py -c config_eval_akkadian_smart.json

# Run performance comparison
python test_smart_llm_performance.py
```

### Smart LLM Output

The smart evaluation shows detailed statistics:
```
🧠 Applying smart LLM correction to page 1...
   ✅ Smart LLM: 15 changed, 25 skipped
```

This means:
- **15 lines** were corrected by LLM (Akkadian + low-confidence modern)
- **25 lines** were skipped (high-confidence modern text)
- **Total**: 40 lines processed, 37.5% LLM call reduction

### Smart LLM Files

- `src/smart_llm_correction.py`: Smart LLM triggering logic
- `config_eval_akkadian_smart.json`: Smart LLM evaluation config
- `run_eval_smart_llm.py`: Smart LLM evaluation runner
- `test_smart_llm_performance.py`: Performance testing script

**Important**: Smart LLM triggering only affects evaluation mode - the main OCR pipeline remains unchanged!

---

## V2 Evaluation Mode (Complete Solution)

The V2 evaluation mode combines Smart LLM triggering with Gold Pages integration for maximum performance and accuracy measurement.

### V2 Features

- **Smart LLM Triggering**: Only Akkadian + low-confidence text sent to LLM
- **Gold Pages Integration**: Ground truth accuracy measurement
- **Enhanced Analysis**: Smart LLM metrics in reports
- **Performance Optimization**: 40-60% LLM call reduction

### V2 Configuration Files

- `config_eval_gold_pages_akkadian_v2.json` - Akkadian + Smart LLM + Gold Pages
- `config_eval_gold_pages_basic_v2.json` - Basic + Smart LLM + Gold Pages
- `config_eval_gold_pages_advanced_v2.json` - Advanced + Smart LLM + Gold Pages

### Running V2 Evaluation

```bash
# V2 Akkadian evaluation (recommended)
python run_eval_gold_pages_v2.py -c config_eval_gold_pages_akkadian_v2.json

# V2 Basic evaluation
python run_eval_gold_pages_v2.py -c config_eval_gold_pages_basic_v2.json

# V2 Advanced evaluation
python run_eval_gold_pages_v2.py -c config_eval_gold_pages_advanced_v2.json
```

### V2 Output Structure

```
data/eval_output/
├── eval_research_results_gold_pages_akkadian_v2/    # V2 Akkadian results
├── eval_research_results_gold_pages_basic_v2/       # V2 Basic results
└── eval_research_results_gold_pages_advanced_v2/    # V2 Advanced results
    ├── document_name/
    │   ├── comprehensive_report.json                # Detailed results with Smart LLM metrics
    │   ├── comprehensive_results.csv                # CSV data
    │   └── translations_report.pdf                  # Akkadian translations (if enabled)
    └── processing_report_YYYYMMDD_HHMMSS.json
```

### V2 Analysis Integration

V2 results are automatically included in analysis reports with:
- Smart LLM efficiency metrics
- LLM call reduction percentages
- Gold Pages accuracy measurements
- Before/after LLM comparisons
- Cost-benefit analysis

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

## Contributing

### For New Developers

1. **Start Small**: Begin with configuration changes or documentation improvements
2. **Understand the Flow**: Read through `pipeline.py` and `comprehensive_pipeline.py`
3. **Test Thoroughly**: Use `python run_pipeline.py --dry-run` for safe testing
4. **Follow Patterns**: Maintain the modular, configuration-driven architecture

### Development Guidelines

#### **Code Organization**
- **Core logic** → `src/` directory
- **Production implementations** → `production/` directory
- **User interfaces** → Root directory
- **Tests** → `tests/` directory

#### **Adding New Features**
1. Add configuration options to `config.json` schema
2. Implement core logic in appropriate `src/` module
3. Update `comprehensive_pipeline.py` to use new feature
4. Add tests and documentation
5. Update this README

#### **Testing New Features**
```bash
# Validate configuration changes
python run_pipeline.py --validate-only

# Test with small files first
python run_pipeline.py --dry-run

# Check specific modules
python -m pytest tests/
```

---

## Troubleshooting

### Common Issues

#### **Configuration Errors**
```bash
# Problem: "Config file not found"
# Solution: Ensure config.json exists in project root
python setup.py  # Creates default config

# Problem: "Invalid JSON in config"
# Solution: Validate JSON syntax
python -m json.tool config.json
```

#### **Import Errors**
```bash
# Problem: "Module not found"
# Solution: Install dependencies
pip install -r requirements.txt

# Problem: "PaddleOCR not working"
# Solution: Check installation
python -c "import paddleocr; print('OK')"
```

#### **Processing Errors**
```bash
# Problem: "No files found to process"
# Solution: Check input directory and file formats
ls data/input/  # Verify files exist

# Problem: "LLM connection failed"
# Solution: Check Ollama service
curl http://localhost:11434/api/tags
```

#### **Performance Issues**
```json
// Problem: Processing too slow
// Solution: Adjust performance settings in config.json
{
  "processing": {
    "batch_size": 5,        // Reduce batch size
    "max_workers": 2        // Reduce parallel processing
  },
  "llm": {
    "timeout": 60,          // Increase timeout
    "max_concurrent_corrections": 1  // Reduce LLM load
  }
}
```

### Debug Mode

```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG
python run_pipeline.py

# Check processing logs
tail -f pipeline.log

# Health check
python -c "from src.healthcheck import run_health_check; run_health_check()"
```

### Getting Help

1. **Check Logs**: Look at `pipeline.log` for detailed error messages
2. **Run Health Check**: `python setup.py` includes system diagnostics
3. **Test Configuration**: Use `--validate-only` and `--dry-run` flags
4. **Start Simple**: Try processing a single small file first

---

## Performance Guidelines

### Recommended Settings by Use Case

| Use Case | DPI | LLM | Batch Size | Expected Speed |
|----------|-----|-----|------------|----------------|
| **Quick Processing** | 150 | Disabled | 20 | ~1 page/sec |
| **Standard Quality** | 300 | Enabled | 10 | ~10 sec/page |
| **High Quality** | 600 | Enabled | 5 | ~30 sec/page |
| **Academic Research** | 600 | Enabled + Akkadian | 3 | ~60 sec/page |

### System Requirements

- **Minimum**: 4GB RAM, 2 CPU cores
- **Recommended**: 8GB RAM, 4 CPU cores
- **High Performance**: 16GB RAM, 8 CPU cores, SSD storage
- **GPU Support**: Optional for PaddleOCR acceleration

---

**Happy Processing!**

> For additional help, check the `doc/` directory or run `python quick_start.py` for interactive tutorials.
