import pandas as pd
import os

# 1) Read reports/adaptive_render_benchmark_matrix.csv and reports/adaptive_render_per_page_metrics.csv
matrix_path = 'reports/adaptive_render_benchmark_matrix.csv'
per_page_path = 'reports/adaptive_render_per_page_metrics.csv'

matrix_df = pd.read_csv(matrix_path)
per_page_df = pd.read_csv(per_page_path)

# Ensure failed/empty are boolean
per_page_df['failed'] = per_page_df['failed'].astype(bool)
per_page_df['empty'] = per_page_df['empty'].astype(bool)

# 2) writes reports/adaptive_render_benchmark_rerun_after_integrity_fix.csv 
# as a copy of the matrix with an added column rerun_after_integrity_fix=true
matrix_rerun_df = matrix_df.copy()
matrix_rerun_df['rerun_after_integrity_fix'] = True
matrix_rerun_csv_path = 'reports/adaptive_render_benchmark_rerun_after_integrity_fix.csv'
matrix_rerun_df.to_csv(matrix_rerun_csv_path, index=False)

# 3) computes per-split and per-dataset CER/WER mean and failed/empty rates by profile_id from per-page rows
def aggregate_metrics(df, group_cols):
    agg = df.groupby(group_cols).agg({
        'CER': 'mean',
        'WER': 'mean',
        'failed': 'mean',
        'empty': 'mean'
    }).reset_index()
    agg.rename(columns={'failed': 'failed_rate', 'empty': 'empty_rate'}, inplace=True)
    return agg

split_metrics = aggregate_metrics(per_page_df, ['profile_id', 'split_kind'])
dataset_metrics = aggregate_metrics(per_page_df, ['profile_id', 'dataset_id'])

# 4) writes reports/adaptive_render_benchmark_rerun_after_integrity_fix.md 
# including overall matrix table plus per-split and per-dataset sections.
md_path = 'reports/adaptive_render_benchmark_rerun_after_integrity_fix.md'
with open(md_path, 'w') as f:
    f.write("# Adaptive Render Benchmark Rerun After Integrity Fix\n\n")
    
    f.write("## Overall Matrix\n")
    f.write(matrix_rerun_df.to_markdown(index=False))
    f.write("\n\n")
    
    f.write("## Metrics Per Split\n")
    f.write(split_metrics.to_markdown(index=False))
    f.write("\n\n")
    
    f.write("## Metrics Per Dataset\n")
    f.write(dataset_metrics.to_markdown(index=False))
    f.write("\n\n")

# Summary output
print(f"File created: {matrix_rerun_csv_path} (Rows: {len(matrix_rerun_df)})")
print(f"File created: {md_path}")
print(f"Rows in split_metrics: {len(split_metrics)}")
print(f"Rows in dataset_metrics: {len(dataset_metrics)}")

