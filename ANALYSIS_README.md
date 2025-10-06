# OCR Pipeline Analysis System

A comprehensive analysis and comparison system for OCR pipeline evaluation results, providing detailed performance metrics, visualizations, and baseline accuracy assessment.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_analysis.txt
```

### 2. Run Analysis Menu
```bash
python run_analysis_menu.py
```

### 3. Or Use Integrated Menus
- **Quick Start**: `python quick_start.py` → Option 5
- **Windows Batch**: `run_ocr.bat` → Option 6

## 📊 Features

### Summary Analysis
- **Mode Comparison**: Compare Basic vs Akkadian vs Advanced modes
- **Performance Metrics**: Success rate, confidence, processing time
- **Visual Charts**: Bar charts, heatmaps, distribution plots
- **Detailed Reports**: Markdown and JSON output formats

### Baseline Accuracy Assessment
- **Character-Level Accuracy**: Precise character-by-character comparison
- **Word-Level Accuracy**: Word recognition accuracy metrics
- **Line-Level Accuracy**: Text line recognition accuracy
- **Ground Truth Management**: Store and manage reference data

### Interactive Menu System
- **Mode Selection**: Choose which evaluation modes to compare
- **Custom Comparisons**: Select specific modes for detailed analysis
- **Quick Analysis**: Analyze all available modes at once
- **Baseline Generation**: Create baseline accuracy guidelines

## 📁 File Structure

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

## 🔧 Usage Examples

### Command Line Analysis
```bash
# Compare two specific evaluation modes
python src/summary_analysis.py \
    ./data/eval_output/eval_research_results_basic \
    ./data/eval_output/eval_research_results_akkadian

# Analyze all modes in a directory
python src/summary_analysis.py ./data/eval_output/eval_research_results_*
```

### Programmatic Usage
```python
from src.summary_analysis import SummaryAnalyzer

# Initialize analyzer
analyzer = SummaryAnalyzer("./data/analysis_output")

# Run analysis
eval_dirs = [
    "./data/eval_output/eval_research_results_basic",
    "./data/eval_output/eval_research_results_akkadian"
]
results = analyzer.run_analysis(eval_dirs)

print(f"Report: {results['markdown_report']}")
print(f"Charts: {results['charts']}")
```

### Baseline Accuracy Calculation
```python
from src.baseline_accuracy import BaselineAccuracyCalculator

calculator = BaselineAccuracyCalculator()

# Calculate accuracy metrics
ocr_text = "Your OCR result text"
ground_truth = "Reference ground truth text"

metrics = calculator.calculate_overall_accuracy(ocr_text, ground_truth)
print(f"Overall accuracy: {metrics['overall_score']:.3f}")
```

## 📈 Analysis Outputs

### Visualizations
- **Mode Comparison Chart**: Bar charts comparing success rates, confidence, and processing time
- **Confidence Distribution**: Histogram showing OCR confidence score distribution
- **Performance Heatmap**: Heatmap of all performance metrics across modes

### Reports
- **Detailed Markdown Report**: Comprehensive analysis with recommendations
- **JSON Summary**: Machine-readable summary for programmatic access
- **Baseline Assessment**: Guidelines for establishing OCR accuracy baselines

### Metrics Tracked
- **Success Rate**: Percentage of successfully processed pages
- **Average Confidence**: Mean OCR confidence score
- **Processing Time**: Time per page and total processing time
- **Text Elements**: Number of text elements extracted per page
- **Corrections Made**: Number of LLM corrections applied
- **Akkadian Translations**: Number of specialized translations found

## 🎯 Baseline Accuracy Implementation

### Setting Up Ground Truth
1. **Create Ground Truth Directory**:
   ```bash
   mkdir -p ./data/ground_truth
   ```

2. **Add Ground Truth Files**:
   ```
   ./data/ground_truth/
   ├── document1_page_001.txt
   ├── document1_page_001_metadata.json
   ├── document1_page_002.txt
   └── document1_page_002_metadata.json
   ```

3. **Use Ground Truth Manager**:
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

### Accuracy Thresholds
| Mode | Character Accuracy | Word Accuracy | Line Accuracy |
|------|-------------------|---------------|---------------|
| Basic | >95% | >90% | >85% |
| Akkadian | >92% | >87% | >80% |
| Advanced | >97% | >93% | >88% |

## 🔍 Troubleshooting

### Common Issues

1. **No Evaluation Data Found**
   - Ensure evaluation runs have completed
   - Check that `./data/eval_output/` contains evaluation results
   - Run evaluations first: `python run_pipeline.py -c config_eval_basic.json --eval-mode`

2. **Missing Dependencies**
   - Install analysis requirements: `pip install -r requirements_analysis.txt`
   - Ensure matplotlib, pandas, and seaborn are installed

3. **Analysis Fails**
   - Check that evaluation directories contain `comprehensive_report.json` files
   - Verify file permissions and encoding
   - Check the analysis log: `./data/analysis_output/analysis.log`

### Debug Mode
```bash
# Run with verbose logging
python run_analysis_menu.py --verbose

# Check analysis log
tail -f ./data/analysis_output/analysis.log
```

## 🚀 Advanced Features

### Custom Analysis
```python
# Custom analysis with specific metrics
analyzer = SummaryAnalyzer("./custom_output")
analyzer.analysis_results = custom_metrics
results = analyzer.create_visualizations(comparison_data)
```

### Integration with Evaluation Pipeline
```python
# Add to your evaluation pipeline
from src.baseline_accuracy import create_baseline_from_evaluation

# After running evaluation
baseline = create_baseline_from_evaluation(eval_results)
```

### Automated Reporting
```python
# Schedule regular analysis
import schedule

def run_daily_analysis():
    analyzer = SummaryAnalyzer()
    results = analyzer.run_analysis(eval_dirs)
    # Send results via email, Slack, etc.

schedule.every().day.at("09:00").do(run_daily_analysis)
```

## 📚 API Reference

### SummaryAnalyzer Class
- `load_evaluation_data(eval_dirs)`: Load evaluation results
- `calculate_baseline_metrics(data)`: Calculate performance metrics
- `compare_modes(mode_data)`: Compare different evaluation modes
- `create_visualizations(comparison_data)`: Generate charts
- `generate_detailed_report(comparison_data)`: Create markdown report
- `run_analysis(eval_dirs)`: Run complete analysis

### BaselineAccuracyCalculator Class
- `calculate_character_accuracy(ocr_text, ground_truth)`: Character-level metrics
- `calculate_word_accuracy(ocr_text, ground_truth)`: Word-level metrics
- `calculate_line_accuracy(ocr_text, ground_truth)`: Line-level metrics
- `calculate_overall_accuracy(ocr_text, ground_truth)`: Comprehensive metrics
- `compare_with_baseline(current, baseline)`: Compare against baseline

### GroundTruthManager Class
- `save_ground_truth(doc_id, page_num, text, metadata)`: Save reference text
- `load_ground_truth(doc_id, page_num)`: Load reference text
- `list_available_ground_truth()`: List all available ground truth data

## 🤝 Contributing

1. **Adding New Metrics**: Extend the `calculate_baseline_metrics` method
2. **New Visualizations**: Add chart generation methods to `create_visualizations`
3. **Custom Reports**: Extend the report generation functions
4. **Integration**: Add new menu options or command-line arguments

## 📄 License

This analysis system is part of the OCR Pipeline project and follows the same license terms.

