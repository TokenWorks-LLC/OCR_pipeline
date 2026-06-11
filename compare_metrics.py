import pandas as pd
import numpy as np

# Load CSVs
df_fast = pd.read_csv('reports/eval_gold_pages_only_postchange_fast_eval_v2/per_page_metrics.csv')
df_paddle_eval = pd.read_csv('reports/eval_gold_pages_only_postchange_two_pass_controlled_paddle_v3_eval/per_page_metrics.csv')
df_ensemble_eval = pd.read_csv('reports/eval_gold_pages_only_postchange_two_pass_controlled_ensemble_budgeted_v1_eval/per_page_metrics.csv')

df_paddle_text = pd.read_csv('reports/eval_gold_pages_only_postchange_two_pass_controlled_paddle_v3/client_page_text.csv')
df_ensemble_text = pd.read_csv('reports/eval_gold_pages_only_postchange_two_pass_controlled_ensemble_budgeted_v1/client_page_text.csv')

# Construct page_key for text dataframes
for df in [df_paddle_text, df_ensemble_text]:
    df['page_key'] = df['pdf_name'] + '_page_' + df['page'].astype(str)

# Ensure final_selection_reason exists
if 'final_selection_reason' not in df_paddle_text.columns:
    df_paddle_text['final_selection_reason'] = None

# Merge eval metrics
# Columns in per_page_metrics.csv: page_key,pdf_name,page,status,cer,wer,runtime_ms,ocr_text_length
merged = df_fast.rename(columns={
    'status': 'status_fast', 'cer': 'cer_fast', 'wer': 'wer_fast', 
    'runtime_ms': 'runtime_fast_ms', 'ocr_text_length': 'ocr_text_length_fast'
})

m_paddle = df_paddle_eval[['page_key', 'status', 'cer', 'wer', 'runtime_ms', 'ocr_text_length']].rename(columns={
    'status': 'status_paddle', 'cer': 'cer_paddle', 'wer': 'wer_paddle', 
    'runtime_ms': 'runtime_paddle_ms', 'ocr_text_length': 'ocr_text_length_paddle'
})

m_ensemble = df_ensemble_eval[['page_key', 'status', 'cer', 'wer', 'runtime_ms', 'ocr_text_length']].rename(columns={
    'status': 'status_ensemble', 'cer': 'cer_ensemble', 'wer': 'wer_ensemble', 
    'runtime_ms': 'runtime_ensemble_ms', 'ocr_text_length': 'ocr_text_length_ensemble'
})

merged = pd.merge(merged, m_paddle, on='page_key', how='outer')
merged = pd.merge(merged, m_ensemble, on='page_key', how='outer')

# Enrich with provenance
p_cols = ['page_key', 'final_output_source', 'final_selection_reason', 'second_pass_status', 'second_pass_runtime_ms', 'total_page_runtime_ms']

p_paddle = df_paddle_text[p_cols].rename(columns={
    'final_output_source': 'final_output_source_paddle',
    'final_selection_reason': 'final_selection_reason_paddle',
    'second_pass_status': 'second_pass_status_paddle',
    'second_pass_runtime_ms': 'second_pass_runtime_ms_paddle',
    'total_page_runtime_ms': 'total_page_runtime_ms_paddle'
})

p_ensemble = df_ensemble_text[p_cols].rename(columns={
    'final_output_source': 'final_output_source_ensemble',
    'final_selection_reason': 'final_selection_reason_ensemble',
    'second_pass_status': 'second_pass_status_ensemble',
    'second_pass_runtime_ms': 'second_pass_runtime_ms_ensemble',
    'total_page_runtime_ms': 'total_page_runtime_ms_ensemble'
})

merged = pd.merge(merged, p_paddle, on='page_key', how='left')
merged = pd.merge(merged, p_ensemble, on='page_key', how='left')

# Calculate deltas
merged['cer_delta_ensemble_vs_paddle'] = merged['cer_ensemble'] - merged['cer_paddle']
merged['wer_delta_ensemble_vs_paddle'] = merged['wer_ensemble'] - merged['wer_paddle']
merged['runtime_delta_ensemble_vs_paddle'] = merged['total_page_runtime_ms_ensemble'] - merged['total_page_runtime_ms_paddle']

# Select final columns and save
final_cols = [
    'page_key', 'pdf_name', 'page', 'status_fast', 'status_paddle', 'status_ensemble',
    'cer_fast', 'cer_paddle', 'cer_ensemble', 'wer_fast', 'wer_paddle', 'wer_ensemble',
    'runtime_fast_ms', 'runtime_paddle_ms', 'runtime_ensemble_ms',
    'ocr_text_length_fast', 'ocr_text_length_paddle', 'ocr_text_length_ensemble',
    'final_output_source_paddle', 'final_selection_reason_paddle', 'second_pass_status_paddle', 'second_pass_runtime_ms_paddle', 'total_page_runtime_ms_paddle',
    'final_output_source_ensemble', 'final_selection_reason_ensemble', 'second_pass_status_ensemble', 'second_pass_runtime_ms_ensemble', 'total_page_runtime_ms_ensemble',
    'cer_delta_ensemble_vs_paddle', 'wer_delta_ensemble_vs_paddle', 'runtime_delta_ensemble_vs_paddle'
]
# For safety, ensure pdf_name and page are taken from where they exist (some rows might only be in one file)
# The merge with df_fast should provide them for most
merged = merged[final_cols]
output_path = 'reports/eval_gold_pages_only_postchange_two_pass_controlled_ensemble_budgeted_v1_eval/page_level_paddle_vs_ensemble.csv'
merged.to_csv(output_path, index=False)

# 1) Output row count
print(f"1) Total merged rows: {len(merged)}")

# 2) Rows for specific page keys
print("\n2) Specified page keys:")
print(merged[merged['page_key'].isin(['akt_7a_page_35', 'akt_4b_2006_page_52'])].to_string())

# 3) Failing pages (status != 'SUCCESS')
failing_paddle = merged[merged['status_paddle'] != 'SUCCESS']
failing_ensemble = merged[merged['status_ensemble'] != 'SUCCESS']
print(f"\n3) Paddle failing ({len(failing_paddle)}):", failing_paddle['page_key'].tolist())
print(f"   Ensemble failing ({len(failing_ensemble)}):", failing_ensemble['page_key'].tolist())

# 4) Top 5 improves CER vs paddle
print("\n4) Top 5 CER Improvements (Ensemble vs Paddle, negative is better):")
print(merged.sort_values('cer_delta_ensemble_vs_paddle').head(5)[['page_key', 'cer_paddle', 'cer_ensemble', 'cer_delta_ensemble_vs_paddle']].to_string())

# 5) Top 5 regresses CER vs paddle
print("\n5) Top 5 CER Regressions (Ensemble vs Paddle, positive is worse):")
print(merged.sort_values('cer_delta_ensemble_vs_paddle', ascending=False).head(5)[['page_key', 'cer_paddle', 'cer_ensemble', 'cer_delta_ensemble_vs_paddle']].to_string())

# 6) Longer but noisier (ensemble text length > paddle text length but CER is worse)
longer_noisier = merged[(merged['ocr_text_length_ensemble'] > merged['ocr_text_length_paddle']) & (merged['cer_delta_ensemble_vs_paddle'] > 0)]
print("\n6) Up to 5 'longer but noisier' pages (ensemble length > paddle length but CER is worse):")
print(longer_noisier.head(5)[['page_key', 'ocr_text_length_paddle', 'ocr_text_length_ensemble', 'cer_paddle', 'cer_ensemble', 'cer_delta_ensemble_vs_paddle']].to_string())
