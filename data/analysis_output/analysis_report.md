# OCR Pipeline Evaluation Analysis Report
Generated: 2025-09-24 20:03:50
============================================================

## Executive Summary

### EVAL_RESEARCH_RESULTS_AKKADIAN Mode
- **Documents Processed**: 11
- **Total Pages**: 84
- **Success Rate**: 96.4%
- **Average Confidence**: 0.933
- **Processing Time**: 3596.4s

## Cost of Compute Analysis

### Understanding Resource Usage

**CPU Usage Explained:**
- **100%** = One CPU core working at full capacity
- **Higher percentages** = Multiple CPU cores working simultaneously
- **Very high percentages** = All cores + hyperthreading + multiple processes
- *Higher percentages mean more efficient use of your computer's processing power*

**Memory Usage Explained:**
- Shows how much RAM (memory) the OCR process uses
- *Average* means typical usage throughout the entire run, not peak usage
- Higher memory usage can indicate processing complex documents

**Why This Matters:**
- **High CPU usage** = Software is working efficiently
- **Low CPU usage** = Software might be waiting or not optimized
- **Memory usage** = Shows how much system resources are needed

### EVAL_RESEARCH_RESULTS_AKKADIAN Mode - Resource Usage
- **Text Elements**: 4787 total, 57.0 per page
- **Word Count**: 15462 total, 184.1 per page
- **Token Count**: 2468 total, 29.4 per page
- **Time per Text Element**: 0.7513s
- **Time per Word**: 0.2326s
- **Time per Token**: 1.4572s
- **Average CPU Usage**: 976.9% (using all CPU cores + hyperthreading + subprocesses)
- **Average Memory Usage**: 2225.3 MB

*Note: 'Average' means the typical usage throughout the entire processing run, not peak usage*

#### Cost Efficiency Analysis
- **Processing Speed**: ❌ **Slow** - Needs optimization
- **Throughput**: 4.3 words/second, 0.7 tokens/second
- **CPU Efficiency**: 🔥 **High CPU usage** - Consider optimization
- **Memory Efficiency**: 🧠 **High memory usage** - Consider memory optimization

## Recommendations

✅ **Excellent success rate** (96.4%) - eval_research_results_akkadian mode is performing well
🔥 **High CPU usage** (976.9%) - Consider optimizing for better efficiency
🧠 **High memory usage** (2225MB) - Consider memory optimization
⚠️ **Slow processing speed** (0.2326s/word) - Consider optimization

## Notes
- **Token Count**: Uses text_elements count (suitable for Akkadian/limited language detection)
- **Word Count**: Only calculated for modern languages (English, German, French, Turkish, etc.)
- **CPU Usage**: 100% = 1 CPU core. Higher % = multiple cores + hyperthreading + subprocesses
- **Memory Usage**: Average RAM consumption during processing (not peak usage)
- **Resource Monitoring**: Tracked throughout entire processing run, not just peak moments
- **Cost of Compute**: Time per word/token shows processing efficiency