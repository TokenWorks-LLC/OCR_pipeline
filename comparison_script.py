import pandas as pd
import json
import os

baseline_path = "reports/eval_gold_pages_only_current_eval_final/evaluation_summary.csv"
current_path = "reports/eval_gold_pages_only_current_eval_rotlayout/evaluation_summary.csv"
metadata_path = "reports/eval_gold_pages_only_current_eval_rotlayout/run_metadata.json"
client_page_text_path = "reports/eval_gold_pages_only_current_rotlayout/client_page_text.csv"
progress_path = "reports/eval_gold_pages_only_current_rotlayout/progress.csv"

# 1) Key metrics
cols = ['cer_mean', 'wer_mean', 'normalized_cer_mean', 'normalized_wer_mean', 'empty_output_rate', 'timeout_rate', 'missing_predictions', 'matched_pages']
baseline = pd.read_csv(baseline_path).iloc[0][cols]
current = pd.read_csv(current_path).iloc[0][cols]
delta = current - baseline
metrics_df = pd.DataFrame({'baseline': baseline, 'current': current, 'delta': delta})
print("--- Key Metrics ---")
print(metrics_df)
print("\n")

# 2) extraction_method, status, used_text_layer counts
progress_df = pd.read_csv(progress_path)
print("--- Extraction Method Counts ---")
print(progress_df['extraction_method'].value_counts() if 'extraction_method' in progress_df.columns else "N/A")
print("\n--- Status Counts ---")
print(progress_df['status'].value_counts() if 'status' in progress_df.columns else "N/A")
print("\n--- Used Text Layer Counts ---")
print(progress_df['used_text_layer'].value_counts() if 'used_text_layer' in progress_df.columns else "N/A")
print("\n")

# 3) client_page_text analysis
client_df = pd.read_csv(client_page_text_path)
for col in ['detected_orientation_class', 'detected_layout_type']:
    if col in client_df.columns:
        non_empty_count = client_df[col].notna().sum()
        val_counts = client_df[col].value_counts()
        print(f"--- {col} Analysis ---")
        print(f"Non-empty count: {non_empty_count}")
        print("Value counts:")
        print(val_counts)
        print("\n")

# 4) run_metadata baseline_comparison
with open(metadata_path, 'r') as f:
    metadata = json.load(f)
baseline_comp = metadata.get('baseline_comparison', {})
improved = baseline_comp.get('improved_pages', [])
worsened = baseline_comp.get('worsened_pages', [])
print("--- Baseline Comparison Arrays ---")
print(f"Improved pages empty: {len(improved) == 0}")
print(f"Worsened pages empty: {len(worsened) == 0}")

