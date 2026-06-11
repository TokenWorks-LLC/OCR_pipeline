import pandas as pd
import pathlib

# Load the CSV files
client_df = pd.read_csv('reports/real_gold_eval_runs/smoke_50/run/client_page_text.csv')
per_page_df = pd.read_csv('reports/source_render_fix_per_page_metrics.csv')

# Build by_pdf_stem from client csv using page_text
# Note: client_df has 'pdf_name'. Let's check if 'pdf_name' or something else should be the stem.
# Looking at the requirement: "ocr_input_path stem exists in by_pdf_stem"
# Also need "number of client rows with non-empty page_text"

client_df['page_text'] = client_df['page_text'].fillna('')
client_non_empty_text = client_df[client_df['page_text'].str.strip() != '']
print(f"Number of client rows with non-empty page_text: {len(client_non_empty_text)}")

# Create by_pdf_stem mapping
# We'll use the stem of pdf_name as the key
def get_stem(path_str):
    if pd.isna(path_str): return None
    return pathlib.Path(path_str).stem

client_df['pdf_stem'] = client_df['pdf_name'].apply(get_stem)
by_pdf_stem = set(client_df['pdf_stem'].dropna().unique())

# Analyze per_page_df
per_page_df['ocr_input_stem'] = per_page_df['ocr_input_path'].apply(get_stem)

matched_mask = per_page_df['ocr_input_stem'].isin(by_pdf_stem)
matched_rows = per_page_df[matched_mask]
print(f"Number of per_page rows whose ocr_input_path stem exists in by_pdf_stem: {len(matched_rows)}")

# Dataset breakdown of matched rows
if 'dataset_id' in matched_rows.columns:
    print("\nDataset breakdown of matched rows:")
    print(matched_rows['dataset_id'].value_counts())
else:
    print("\n'dataset_id' column not found in per_page_df")

# Print 5 example stems from per_page not matched
not_matched = per_page_df[~matched_mask]
print("\n5 example stems from per_page not matched:")
print(not_matched['ocr_input_stem'].head(5).tolist())

