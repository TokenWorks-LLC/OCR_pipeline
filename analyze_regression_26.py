import json
import os

split_file = 'data/gold_registry/splits/regression_26.jsonl'
manifest_file = 'data/gold_registry/gold_manifest.jsonl'

split_page_ids = []
if os.path.exists(split_file):
    with open(split_file, 'r') as f:
        for line in f:
            d = json.loads(line)
            if 'page_id' in d:
                split_page_ids.append(d['page_id'])

manifest_data = {}
if os.path.exists(manifest_file):
    with open(manifest_file, 'r') as f:
        for line in f:
            d = json.loads(line)
            if 'page_id' in d:
                manifest_data[d['page_id']] = d

supported_formats = {'JSON_BOXES', 'PAGE_XML', 'ALTO_XML'}

manifest_found = 0
manifest_missing = 0
local_pdf_exists = 0
gt_text_exists = 0
unsupported_annotation_format_count = 0
missing_ids = []

print("page_id | manifest_found | doc_id | dataset_id | source_file (exists) | image (exists) | pdf (exists) | text (exists) | layout (exists) | format")
print("-" * 150)

for pid in split_page_ids:
    entry = manifest_data.get(pid)
    if entry:
        manifest_found += 1
        doc_id = entry.get('document_id', 'N/A')
        dataset_id = entry.get('dataset_id', 'N/A')
        source_file = entry.get('source_file', '')
        local_image_path = entry.get('local_image_path', '')
        local_pdf_path = entry.get('local_pdf_path', '')
        gt_text_path = entry.get('ground_truth_text_path', '')
        gt_layout_path = entry.get('ground_truth_layout_path', '')
        fmt = entry.get('annotation_format', 'N/A')

        s_exists = os.path.exists(source_file) if source_file else False
        i_exists = os.path.exists(local_image_path) if local_image_path else False
        p_exists = os.path.exists(local_pdf_path) if local_pdf_path else False
        t_exists = os.path.exists(gt_text_path) if gt_text_path else False
        l_exists = os.path.exists(gt_layout_path) if gt_layout_path else False

        if p_exists: local_pdf_exists += 1
        if t_exists: gt_text_exists += 1
        if fmt not in supported_formats: unsupported_annotation_format_count += 1

        print(f"{pid} | Y | {doc_id} | {dataset_id} | {source_file} ({s_exists}) | {local_image_path} ({i_exists}) | {local_pdf_path} ({p_exists}) | {gt_text_path} ({t_exists}) | {gt_layout_path} ({l_exists}) | {fmt}")
    else:
        manifest_missing += 1
        missing_ids.append(pid)
        print(f"{pid} | N | - | - | - | - | - | - | - | -")

print("\nSummary:")
print(f"manifest_found: {manifest_found}")
print(f"manifest_missing: {manifest_missing}")
print(f"local_pdf_exists: {local_pdf_exists}")
print(f"gt_text_exists: {gt_text_exists}")
print(f"unsupported_annotation_format_count: {unsupported_annotation_format_count}")
print(f"manifest_missing page_ids: {missing_ids}")
