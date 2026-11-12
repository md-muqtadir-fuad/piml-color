#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Make src-first package import work when running from project root without editable install.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from piml_color_recipe.preprocessing.validation_splits import SCRIPT_VERSION, build_validation_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Create leakage-safe validation splits for PIML color recipe modelling.")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of GroupKFold folds for recipe_no and target_id splits.")
    args = parser.parse_args()
    build_validation_splits(project_root=PROJECT_ROOT, n_splits=args.n_splits)


if __name__ == "__main__":
    main()
