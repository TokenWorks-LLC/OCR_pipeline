# Gold Pages Integration Guide

## 🏆 How Gold Pages Work

### **Data Flow Diagram**

```
📄 PDF Documents
    ↓
🔍 OCR Pipeline (PaddleOCR)
    ↓
📊 Text Extraction
    ↓
🏆 Gold Pages Comparison ←── 📁 Ground Truth Data
    ↓
📈 Accuracy Measurement
    ↓
📊 Before/After Analysis
    ↓
📋 Success Metrics Report
```

## 📁 Directory Structure

```
OCR_pipeline/
├── data/
│   ├── gold_pages/                    # ← GOLD PAGES DATA GOES HERE
│   │   ├── gold_pages.json           # Main ground truth data
│   │   └── gold_pages_backup.json    # Backup
│   ├── input_pdfs/                   # Your PDF documents
│   └── eval_output/                  # Evaluation results
│       ├── eval_research_results_akkadian/
│       └── eval_research_results_gold_pages_akkadian/
├── src/
│   ├── gold_pages_manager.py         # Gold Pages data management
│   ├── accuracy_measurement.py       # Before/after comparison
│   └── cost_benefit_analyzer.py      # ROI tracking
└── config_eval_gold_pages_*.json     # Gold Pages evaluation configs
```

## 🔄 Integration Points

### **1. Data Storage** 📁
- **Location**: `./data/gold_pages/gold_pages.json`
- **Format**: JSON with Akkadian-translation pairs
- **Structure**: Document ID, page number, Akkadian text, translation, confidence

### **2. Evaluation Integration** ⚙️
- **Configs**: `config_eval_gold_pages_*.json`
- **Runners**: `run_eval_gold_pages.py`
- **Smart LLM**: `run_eval_smart_llm.py`

### **3. Analysis Integration** 📊
- **Menu**: `run_analysis_menu.py`
- **Reports**: Generated in `./data/analysis_output/`
- **Metrics**: Accuracy improvements, cost-benefit analysis

## 🎯 When Gold Pages Are Ready

### **Step 1: Add Ground Truth Data**
```python
from src.gold_pages_manager import create_gold_pages_manager

manager = create_gold_pages_manager()

# Add your intern's ground truth data
manager.add_gold_page(
    document_id="doc_001",
    page_number=1,
    akkadian_text="lugal",
    translation_text="king",
    confidence_score=0.95,
    verified_by="expert_linguist"
)
```

### **Step 2: Run Gold Pages Evaluation**
```bash
# Run evaluation with Gold Pages
python run_eval_gold_pages.py -c config_eval_gold_pages_akkadian.json

# Or with smart LLM triggering
python run_eval_smart_llm.py -c config_eval_gold_pages_akkadian.json
```

### **Step 3: Analyze Results**
```bash
# Run analysis menu
python run_analysis_menu.py

# Select options:
# 1. Run Summary Analysis
# 2. Run Comprehensive Analysis
# 3. View Available Evaluations
```

## 📊 What You Get

### **Accuracy Measurements**
- Character-level accuracy
- Word-level accuracy  
- Line-level accuracy
- Before/after LLM comparison

### **Success Metrics**
- Minimum improvement: +5%
- Maximum time increase: +50%
- Cost per accuracy point: <$0.10
- ROI justification

### **Reports Generated**
- `gold_pages_analysis_report.json`
- `accuracy_measurements.json`
- `gold_pages_summary.md`
- Visual charts and graphs

## 🔧 Configuration Options

### **Gold Pages Config**
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

## 🚀 Ready to Use

The Gold Pages system is **completely ready** and waiting for your intern's ground truth data. Once you have the data:

1. **Add it** using the Gold Pages Manager
2. **Run evaluation** with Gold Pages configs
3. **Analyze results** to measure improvements
4. **Justify LLM costs** with concrete data

**No main pipeline changes needed** - everything works through the evaluation mode!
