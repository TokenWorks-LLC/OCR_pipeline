'''
Demo script
Testing Akkadian llm Corrections
@author Maryan Le
TokenWorks LLC
'''
# jiwer (just another word error rate) is a python library used to compute the word error rate
# metric used for speech recognition, OCR or macchine translation systems
from jiwer import wer
from src.llm_correction import create_llm_corrector

# 5 OCR spans with typos
# text copied from CHATGPTa
ocr_lines_with_typos = [
    "The king dwelt in the palce of Ninveh",     # typo: "palce" → "palace", "Ninveh" → "Nineveh"
    "He offerred many giftss to the gods",       # typo: "offerred", "giftss"
    "A grand templ stood by the river",          # typo: "templ" → "temple"
    "Scribs wrote on clay tablets",              # typo: "Scribs" → "Scribes"
    "The bttle was fierce and long",             # typo: "bttle" → "battle"
]

# Akkadian text without typos
# text copied from CHATGPT
true_version = [
    "The king dwelt in the palace of Nineveh",
    "He offered many gifts to the gods",
    "A grand temple stood by the river",
    "Scribes wrote on clay tablets",
    "The battle was fierce and long"
]

# create an llm corrector 
# then run the ocr lines with typos through the llm
llm_corrector = create_llm_corrector()
results = llm_corrector.correct_multiple_texts(ocr_lines_with_typos)

# Results is a list of objects, 
# each object contains data, including corrected_text
# loop through each res in results and collected only the corrected text

correct_lines = [res.corrected_text for res in results]

# Compute the WER
raw_wer = wer(true_version, ocr_lines_with_typos)
corrected_wer = wer(true_version, correct_lines)


# Output the results
print("Word Error Rate (WER)")
print(f"Before llm correction: {raw_wer:.2%}")
print(f"After llm correction: {corrected_wer:.2%}")
print()

print("Original vs. Corrected Text")
for i in range(5):
    print(f"OCR            : {ocr_lines_with_typos[i]}")
    print(f"Corrected      : {correct_lines[i]}")
    print(f"True Version   : {true_version[i]}")
    print()

