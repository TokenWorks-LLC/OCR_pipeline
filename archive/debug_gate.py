"""Debug gate enforcement."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lang_and_akkadian import is_akkadian_transliteration

test_text = "a-na-ku i-ma-at"

# With gate disabled
no_gate = {"threshold": 0.20, "require_diacritic_or_marker": False}
is_akk_no, score_no = is_akkadian_transliteration(test_text, config=no_gate)

# With gate enabled  
gate = {"threshold": 0.20, "require_diacritic_or_marker": True}
is_akk_gate, score_gate = is_akkadian_transliteration(test_text, config=gate)

print(f"Text: '{test_text}'")
print(f"\nWithout gate: is_akk={is_akk_no}, score={score_no:.3f}")
print(f"With gate:    is_akk={is_akk_gate}, score={score_gate:.3f}")
print(f"\nGate enforced: {is_akk_no and not is_akk_gate}")
