import pandas as pd

df = pd.read_csv('reports/eval_gold_pages_only_postchange_two_pass_controlled_ensemble_budgeted_v1_eval/page_level_paddle_vs_ensemble.csv')

print(f"Row count: {len(df)}")

print("\nFull rows for specific page_keys:")
print(df[df['page_key'].isin(['akt_7a_page_35', 'akt_4b_2006_page_52'])].to_string())

paddle_failing = df[df['status_paddle'] != 'success']
print(f"\nPaddle still-failing count: {len(paddle_failing)}")
print(f"Paddle failing page_keys: {paddle_failing['page_key'].tolist()}")

ensemble_failing = df[df['status_ensemble'] != 'success']
print(f"\nEnsemble still-failing count: {len(ensemble_failing)}")
print(f"Ensemble failing page_keys: {ensemble_failing['page_key'].tolist()}")

print("\nTop 5 best cer_delta_ensemble_vs_paddle (most negative):")
print(df.nsmallest(5, 'cer_delta_ensemble_vs_paddle')[['page_key', 'cer_delta_ensemble_vs_paddle']])

print("\nTop 5 worst cer_delta_ensemble_vs_paddle (most positive):")
print(df.nlargest(5, 'cer_delta_ensemble_vs_paddle')[['page_key', 'cer_delta_ensemble_vs_paddle']])

print("\nRows where ensemble length > paddle length and ensemble CER > paddle CER (cer_delta > 0):")
longer_and_worse = df[(df['ocr_text_length_ensemble'] > df['ocr_text_length_paddle']) & (df['cer_delta_ensemble_vs_paddle'] > 0)]
print(longer_and_worse[['page_key', 'ocr_text_length_paddle', 'ocr_text_length_ensemble', 'cer_paddle', 'cer_ensemble', 'cer_delta_ensemble_vs_paddle']].head(5))
