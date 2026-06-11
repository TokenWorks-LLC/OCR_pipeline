import pandas as pd
import os

def normalize_page_key(row):
    if pd.notna(row.get('page_id')) and str(row['page_id']).strip() != '':
        return str(row['page_id']).strip()
    
    stem = os.path.splitext(row['pdf_name'])[0].lower()
    page = int(row['page'])
    if page == 1:
        return stem
    else:
        return f"{stem}_page_{page}"

# Paths
fast_metrics_path = "reports/eval_gold_pages_only_postchange_fast_eval_v2/per_page_metrics.csv"
paddle_metrics_path = "reports/eval_gold_pages_only_postchange_two_pass_controlled_paddle_eval/per_page_metrics.csv"
ensemble_metrics_path = "reports/eval_gold_pages_only_postchange_two_pass_controlled_ensemble_budgeted_v1_eval/per_page_metrics.csv"

paddle_client_path = "reports/eval_gold_pages_only_postchange_two_pass_controlled_paddle/client_page_text.csv"
ensemble_client_path = "reports/eval_gold_pages_only_postchange_two_pass_controlled_ensemble_budgeted_v1/client_page_text.csv"

output_path = "reports/eval_gold_pages_only_postchange_two_pass_controlled_ensemble_budgeted_v1_eval/page_level_paddle_vs_ensemble.csv"

# Load data
df_fast = pd.read_csv(fast_metrics_path)
df_paddle_metrics = pd.read_csv(paddle_metrics_path)
df_ensemble_metrics = pd.read_csv(ensemble_metrics_path)

df_paddle_client = pd.read_csv(paddle_client_path)
df_ensemble_client = pd.read_csv(ensemble_client_path)

# Drop page_text to save memory
if 'page_text' in df_paddle_client.columns:
    df_paddle_client = df_paddle_client.drop(columns=['page_text'])
if 'page_text' in df_ensemble_client.columns:
    df_ensemble_client = df_ensemble_client.drop(columns=['page_text'])

# Create keys
df_fast['page_key'] = df_fast.apply(normalize_page_key, axis=1)
df_paddle_metrics['page_key'] = df_paddle_metrics.apply(normalize_page_key, axis=1)
df_ensemble_metrics['page_key'] = df_ensemble_metrics.apply(normalize_page_key, axis=1)
df_paddle_client['page_key'] = df_paddle_client.apply(normalize_page_key, axis=1)
df_ensemble_client['page_key'] = df_ensemble_client.apply(normalize_page_key, axis=1)

# Provenance fields to merge
prov_fields = [
    'final_output_source', 'final_selection_reason', 
    'second_pass_status', 'second_pass_runtime_ms', 
    'total_page_runtime_ms', 'fallback_reason', 'fallback_engine'
]

# Merge provenance onto metrics
def merge_prov(df_metrics, df_client, fields):
    # Only keep fields that exist in df_client
    existing_fields = ['page_key'] + [f for f in fields if f in df_client.columns]
    return pd.merge(
        df_metrics, 
        df_client[existing_fields], 
        on='page_key', 
        how='left'
    )

df_paddle = merge_prov(df_paddle_metrics, df_paddle_client, prov_fields)
df_ensemble = merge_prov(df_ensemble_metrics, df_ensemble_client, prov_fields)

# Prepare for final merge
df_fast = df_fast.set_index('page_key')
df_paddle = df_paddle.set_index('page_key')
df_ensemble = df_ensemble.set_index('page_key')

# Identify common columns (mostly metrics)
common_cols = list(set(df_fast.columns) & set(df_paddle.columns) & set(df_ensemble.columns))
# Filter out some non-metric columns if they exist
common_cols = [c for c in common_cols if c not in ['pdf_name', 'page', 'page_id'] and not any(f in c for f in prov_fields)]

# Build the merged dataframe
df_final = df_ensemble[['pdf_name', 'page']].copy()

for col in common_cols:
    df_final[f'{col}_fast'] = df_fast[col]
    df_final[f'{col}_paddle'] = df_paddle[col]
    df_final[f'{col}_ensemble'] = df_ensemble[col]

# Add provenance fields
for field in prov_fields:
    if field in df_paddle.columns:
        df_final[f'{field}_paddle'] = df_paddle[field]
    if field in df_ensemble.columns:
        df_final[f'{field}_ensemble'] = df_ensemble[field]

df_final.to_csv(output_path)

# 1) number of rows with non-empty final_output_source_ensemble
non_empty_prov = df_final[df_final['final_output_source_ensemble'].notna() & (df_final['final_output_source_ensemble'] != '')]
print(f"Number of rows with non-empty final_output_source_ensemble: {len(non_empty_prov)}")

# 2) values/counts of final_output_source_ensemble
print("\nValues/counts of final_output_source_ensemble:")
print(df_final['final_output_source_ensemble'].value_counts(dropna=False))

# 3) row snippets for akt_7a_page_35 and adams_1982_property_rights_page_3 showing ensemble provenance fields
print("\nRow snippets:")
target_keys = ['akt_7a_page_35', 'adams_1982_property_rights_page_3']
# Update cols_to_show based on what actually exists in the final dataframe
cols_to_show = ['pdf_name', 'page'] + [f'{f}_ensemble' for f in prov_fields if f'{f}_ensemble' in df_final.columns]
for key in target_keys:
    if key in df_final.index:
        print(f"\nKey: {key}")
        print(df_final.loc[[key], cols_to_show].to_string())
    else:
        print(f"\nKey: {key} not found in index")
