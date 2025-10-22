#!/usr/bin/env python3
from huggingface_hub import model_info

info = model_info('deepseek-ai/DeepSeek-OCR')
print(f"Model: {info.id}")
print(f"Tags: {info.tags}")
print(f"Private: {info.private}")

siblings = info.siblings
print(f"\nFiles: {len(siblings)}")

total_size = 0
for s in siblings:
    size = s.size or 0
    total_size += size
    print(f"  {s.rfilename}: {size/1024**3:.1f}GB")

print(f"\nTotal size: {total_size/1024**3:.1f}GB")
print(f"Downloads: {getattr(info, 'downloads', 'N/A')}")
