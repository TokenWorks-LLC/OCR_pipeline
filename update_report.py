import pandas as pd
import numpy as np

def load_csv(path):
    return pd.read_csv(path, dtype=str)

def safe_to_numeric(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# Paths
dir_fast = "reports/eval_gold_pages_only_postchange_fast_eval"
dir_paddle = "reports/eval_gold_pages_only_postchange_two_pass_controlled_paddle_v3_eval"
dir_ensemble = "reports/eval_gold_pages_only_postchange_two_pass_controlled_ensemble_budgeted_v1_eval"

prov_paddle_path = "reports/eval_gold_pages_only_postchange_two_pass_controlled_paddle_v3/client_page_text.csv"
prov_ensemble_path = "reports/eval_gold_pages_only_postchange_two_pass_controlled_ensemble_budgeted_v1/client_page_text.csv"

# Load per_page_metrics
df_fast = load_csv(f"{dir_fast}/per_page_metrics.csv")
df_paddle = load_csv(f"{dir_paddle}/per_page_metrics.csv")
df_ensemble = load_csv(f"{dir_ensemble}/per_page_metrics.csv")

# Load provenance
prov_paddle = load_csv(prov_paddle_path)
prov_ensemble = load_csv(prov_ensemble_path)

def process_prov(df):
    if 'page_id' in df.columns:
        df = df[df['page_id'].notna() & (df['page_id'] != "")]
        return df.drop_duplicates(subset=['page_id'], keep='last').set_index('page_id')
    return pd.DataFrame()

prov_paddle_clean = process_prov(prov_paddle)
prov_ensemble_clean = process_prov(prov_ensemble)

# Merge provenance onto per_page tables by page_key
df_paddle = df_paddle.merge(prov_paddle_clean, left_on='page_key', right_index=True, how='left', suffixes=('', '_prov'))
df_ensemble = df_ensemble.merge(prov_ensemble_clean, left_on='page_key', right_index=True, how='left', suffixes=('', '_prov'))

# Metrics and provenance columns to include
cols_metrics = ['status', 'cer', 'wer', 'runtime_ms', 'ocr_text_length']
cols_prov = ['final_output_source', 'final_selection_reason', 'second_pass_status', 
             'second_pass_runtime_ms', 'total_page_runtime_ms', 'fallback_reason', 'fallback_engine']

# Rename function
def prepare_df(df, prefix, is_fast=False):
    cols_to_keep = []
    rename_map = {}
    
    if is_fast:
        cols_to_keep.extend(['page_key', 'pdf_name', 'page'])
    else:
        cols_to_keep.append('page_key')

    for col in cols_metrics + cols_prov:
        target_name = f"{col}_{prefix}"
        if col in df.columns:
            rename_map[col] = target_name
            cols_to_keep.append(target_name)
        else:
            df[target_name] = np.nan
            cols_to_keep.append(target_name)
            
    return df.rename(columns=rename_map)[cols_to_keep]

df_fast_prep = prepare_df(df_fast, "fast", is_fast=True)
df_paddle_prep = prepare_df(df_paddle, "paddle")
df_ensemble_prep = prepare_df(df_ensemble, "ensemble")

# Merge
final_df = df_fast_prep.merge(df_paddle_prep, on='page_key', how='left').merge(df_ensemble_prep, on='page_key', how='left')

# Robust numeric conversion
numeric_cols = [
    'cer_fast', 'cer_paddle', 'cer_ensemble', 
    'wer_fast', 'wer_paddle', 'wer_ensemble',
    'runtime_ms_fast', 'runtime_ms_paddle', 'runtime_ms_ensemble',
    'ocr_text_length_fast', 'ocr_text_length_paddle', 'ocr_text_length_ensemble',
    'second_pass_runtime_ms_paddle', 'total_page_runtime_ms_paddle',
    'second_pass_runtime_ms_ensemble', 'total_page_runtime_ms_ensemble'
]
final_df = safe_to_numeric(final_df, numeric_cols)

# Deltas (using the correct prefixed names)
final_df['cer_delta_ensemble_vs_paddle'] = final_df['cer_ensemble'] - final_df['cer_paddle']
final_df['wer_delta_ensemble_vs_paddle'] = final_df['wer_ensemble'] - final_df['wer_paddle']
final_df['runtime_delta_ensemble_vs_paddle'] = final_df['runtime_ms_ensemble'] - final_df['runtime_ms_paddle']

# Final order
final_cols = [
    'page_key', 'pdf_name', 'page', 'status_fast', 'status_paddle', 'status_ensemble',
    'cer_fast', 'cer_paddle', 'cer_ensemble', 'wer_fast', 'wer_paddle', 'wer_ensemble',
    'runtime_ms_fast', 'runtime_ms_paddle', 'runtime_ms_ensemble',
    'ocr_text_length_fast', 'ocr_text_length_paddle', 'ocr_text_length_ensemble',
    'final_output_source_paddle', 'final_selection_reason_paddle', 'second_pass_status_paddle', 'second_pass_runtime_ms_paddle', 'total_page_runtime_ms_paddle', 'fallback_reason_paddle', 'fallback_engine_paddle',
    'final_output_source_ensemble', 'final_selection_reason_ensemble', 'second_pass_status_ensemble', 'second_pass_runtime_ms_ensemble', 'total_page_runtime_ms_ensemble', 'fallback_reason_ensemble', 'fallback_engine_ensemble',
    'cer_delta_ensemble_vs_paddle', 'wer_delta_ensemble_vs_paddle', 'runtime_delta_ensemble_vs_paddle'
]
# Fix naming mismatch in final_cols (requested was runtime_fast_ms etc., but my script created runtime_ms_fast)
rename_final = {
    'runtime_ms_fast': 'runtime_fast_ms',
    'runtime_ms_paddle': 'runtime_paddle_ms',
    'runtime_ms_ensemble': 'runtime_ensemble_ms'
}
final_df = final_df.rename(columns=rename_final)
final_cols_fixed = [c.replace('runtime_ms_', 'runtime_').replace('_ms', '_ms') if 'runtime_ms' in c else c for c in final_cols] # Wait, I'll just hardcode it.

final_cols = [
    'page_key', 'pdf_name', 'page', 'status_fast', 'status_paddle', 'status_ensemble',
    'cer_fast', 'cer_paddle', 'cer_ensemble', 'wer_fast', 'wer_paddle', 'wer_ensemble',
    'runtime_fast_ms', 'runtime_paddle_ms', 'runtime_ensemble_ms',
    'ocr_text_length_fast', 'ocr_text_length_paddle', 'ocr_text_length_ensemble',
    'final_output_source_paddle', 'final_selection_reason_paddle', 'second_pass_status_paddle', 'second_pass_runtime_ms_paddle', 'total_page_runtime_ms_paddle', 'fallback_reason_paddle', 'fallback_engine_paddle',
    'final_output_source_ensemble', 'final_selection_reason_ensemble', 'second_pass_status_ensemble', 'second_pass_runtime_ms_ensemble', 'total_page_runtime_ms_ensemble', 'fallback_reason_ensemble', 'fallback_engine_ensemble',
    'cer_delta_ensemble_vs_paddle', 'wer_delta_ensemble_vs_paddle', 'runtime_delta_ensemble_vs_paddle'
]

final_df = final_df[final_cols]

# Save
output_path = "reports/eval_gold_pages_only_postchange_two_pass_controlled_ensemble_budgeted_v1_eval/page_level_paddle_vs_ensemble.csv"
final_df.to_csv(output_path, index=False)

# Print results
print(f"Final row count: {len(final_df)}")
print(f"Non-empty final_output_source_ensemble count: {final_df['final_output_source_ensemble'].notna().sum()}")
print("Value counts for final_output_source_ensemble:")
print(final_df['final_output_source_ensemble'].value_counts())
print("\nRows for akt_7a_page_35 and adams_1982_property_rights_page_3:")
print(final_df[final_df['page_key'].isin(['akt_7a_page_35', 'adams_1982_property_rights_page_3'])].to_string())
