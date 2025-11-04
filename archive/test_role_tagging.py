#!/usr/bin/env python3
"""Test role tagging for reference patterns."""

import sys
sys.path.insert(0, 'src')

from block_roles import BlockRoleTagger

# Test cases
test_cases = [
    {'text': 'HW s. 124 a.\nevcuttur. ubalit II pres. Formdadr.', 'name': 'HW s.'},
    {'text': 'Kt. j/k 430\nö.y.', 'name': 'Kt. j/k'},
]

tagger = BlockRoleTagger()

for test in test_cases:
    tagged = tagger.tag_block(test)
    print(f"\n=== {test['name']} ===")
    print(f"Text: {test['text'][:50]}")
    print(f"Role: {tagged['role']}")
    print(f"Confidence: {tagged['role_confidence']}")
    print(f"Reasons: {tagged['role_reasons']}")
    print(f"Expected: reference_meta")
    print(f"Status: {'✅ PASS' if tagged['role'].value == 'reference_meta' else '❌ FAIL'}")

