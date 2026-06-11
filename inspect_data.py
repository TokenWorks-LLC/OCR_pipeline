import json
import os

split_file = 'data/gold_registry/splits/regression_26.jsonl'
manifest_file = 'data/gold_registry/gold_manifest.jsonl'

split_page_ids = set()
if os.path.exists(split_file):
    with open(split_file, 'r') as f:
        for line in f:
            d = json.loads(line)
            if 'page_id' in d: split_page_ids.add(d['page_id'])

results = []
all_formats = {}
path_fields = ['local_image_path', 'local_pdf_path', 'ground_truth_text_path', 'ground_truth_layout_path']

with open(manifest_file, 'r') as f:
    for line in f:
        e = json.loads(line)
        if e.get('dataset_id') == 'local_gold_pages':
            fmt = e.get('annotation_format', 'N/A')
            all_formats[fmt] = all_formats.get(fmt, 0) + 1
        if e.get('split') == 'regression_26' or e.get('page_id') in split_page_ids:
            row = [e.get('page_id'), e.get('document_id'), e.get('source_file'), e.get('annotation_format')]
            exists_vals = []
            for pf in path_fields:
                p = e.get(pf)
                exists = os.path.exists(p) if p else False
                exists_vals.append("Y" if exists else "N")
            results.append(row + exists_vals)

headers = ['page_id', 'doc_id', 'source', 'fmt', 'img?', 'pdf?', 'txt?', 'lay?']
print(f"{' | '.join(headers)}")
for r in results:
    print(f"{' | '.join(str(x) for x in r)}")

print("\nAnnotation formats for local_gold_pages:", all_formats)
