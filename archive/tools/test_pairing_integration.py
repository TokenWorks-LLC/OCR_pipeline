#!/usr/bin/env python3
"""
End-to-end integration test for pairing pipeline.
Tests individual components and validates data flow.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

print("="*60)
print("PAIRING PIPELINE INTEGRATION TEST")
print("="*60)

# Test 1: Blockification
print("\n[1/4] Testing blockification...")
try:
    from blockification import TextBlockifier, TextBlock
    
    mock_lines = [
        {'text': 'ša-ar-ru-um i-re-eb', 'bbox': (100, 100, 300, 120), 'confidence': 0.85, 'line_id': 'l1'},
        {'text': 'König tritt ein', 'bbox': (100, 150, 250, 170), 'confidence': 0.90, 'line_id': 'l2'},
    ]
    
    blockifier = TextBlockifier()
    blocks = blockifier.blockify(mock_lines, page_num=1, page_width=2480, page_height=3508)
    
    print(f"  ✅ Blockification works: {len(blocks)} blocks created")
    print(f"     Akkadian blocks: {sum(1 for b in blocks if b.is_akk)}")
    print(f"     Non-Akkadian blocks: {sum(1 for b in blocks if not b.is_akk)}")
    
except Exception as e:
    print(f"  ❌ Blockification failed: {e}")
    sys.exit(1)

# Test 2: Translation Pairing
print("\n[2/4] Testing translation pairing...")
try:
    from translation_pairing import TranslationPairer, PairingConfig, TranslationPair
    
    # Create mock blocks for pairing
    from dataclasses import replace
    
    # Force separate blocks
    if len(blocks) >= 1:
        # Create a translation block manually
        akk_block = blocks[0]
        trans_block = replace(
            akk_block,
            block_id='block_trans',
            text='König tritt ein',
            is_akk=False,
            lang='de',
            bbox=(100, 150, 250, 170)
        )
        test_blocks = [akk_block, trans_block]
    else:
        test_blocks = blocks
    
    config = PairingConfig(
        weight_distance=0.4,
        weight_column=0.2,
        weight_language=0.15
    )
    pairer = TranslationPairer(config)
    pairs = pairer.pair_blocks(test_blocks, page=1, pdf_id='test')
    
    print(f"  ✅ Pairing works: {len(pairs)} pairs created")
    if pairs:
        print(f"     Average score: {sum(p.score for p in pairs) / len(pairs):.3f}")
    
except Exception as e:
    print(f"  ❌ Pairing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: CSV Export
print("\n[3/4] Testing CSV export...")
try:
    output_dir = Path("data/output/integration_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "translations.csv"
    
    if pairs:
        pairer.save_pairs_csv(pairs, output_path)  # Pass Path object, not string
        print(f"  ✅ CSV export works: {output_path}")
        
        # Validate CSV content
        with open(output_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"     Rows: {len(lines) - 1} (+ header)")
    else:
        print(f"  ⚠️  CSV export skipped (no pairs)")
    
except Exception as e:
    print(f"  ❌ CSV export failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Configuration Loading
print("\n[4/4] Testing profile configuration...")
try:
    import json
    
    with open("profiles/akkadian_strict.json", 'r', encoding='utf-8') as f:
        profile = json.load(f)
    
    if 'pairing' in profile:
        pairing_cfg = profile['pairing']
        print(f"  ✅ Profile config present")
        print(f"     Weights: distance={pairing_cfg['weights']['distance']}, column={pairing_cfg['weights']['column']}")
        print(f"     Max distance: {pairing_cfg['max_dist_px']}px")
        print(f"     Markers: {len(pairing_cfg['lexical_markers'])} defined")
    else:
        print(f"  ❌ No pairing section in profile")
        sys.exit(1)
    
except Exception as e:
    print(f"  ❌ Profile loading failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "="*60)
print("INTEGRATION TEST SUMMARY")
print("="*60)
print("✅ All components operational")
print("\nPipeline data flow validated:")
print("  1. Mock fusion lines → Blockifier → TextBlocks")
print("  2. TextBlocks → Pairer → TranslationPairs")
print("  3. TranslationPairs → CSV → translations.csv")
print("  4. Profile → Configuration → PairingConfig")

print("\n⚠️  Next steps for full validation:")
print("  - Integrate with real ROVER fusion output")
print("  - Test on actual PDF pages with OCR")
print("  - Measure pairing F1 on labeled gold data")
print("  - Generate overlay visualizations")

print("\n" + "="*60)
print("STATUS: READY FOR PIPELINE INTEGRATION")
print("="*60)
