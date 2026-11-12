from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from piml_color_recipe.preprocessing.audit_raw_data import run_audit


if __name__ == "__main__":
    run_audit(PROJECT_ROOT)
