import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "tools"))
from run_page_text import main
if __name__ == "__main__":
    sys.exit(main())