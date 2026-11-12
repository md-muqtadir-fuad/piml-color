"""Build the dye -> hue-family mapping used by the analysis scripts.

Reads the cleaned long recipe table (data/interim/recipe_dye_long.csv, from script 01) and the dye-vector columns of
the supervised feature table (data/processed/features_supervised.csv, from script 03). The hue family of each dye is
inferred from keywords in its name (yellow/gold, red/ruby/rubine/rose/carmine, blue/navy/ocean/turquoise, black).

Output: data/global_naming/global_dye_mapping_active_<n>.csv (n = number of dye columns; the analysis scripts read
the file for n = 36).
Run: python scripts/06_build_dye_hue_mapping.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from piml_color_recipe.preprocessing.master_table import _safe_dye_col  # noqa: E402

HUE_KEYWORDS = {
    "yellow": ("yellow", "gold"),
    "red": ("red", "ruby", "rubine", "rose", "carmine"),
    "blue": ("blue", "navy", "ocean", "turquoise"),
    "black": ("black",),
}


def infer_hue(name: str) -> str:
    words = name.lower().replace("-", " ").split()
    for hue, keys in HUE_KEYWORDS.items():
        if any(w.startswith(k) for w in words for k in keys):
            return hue
    return "other"


def main() -> None:
    long = pd.read_csv(ROOT / "data/interim/recipe_dye_long.csv")
    feats = pd.read_csv(ROOT / "data/processed/features_supervised.csv", nrows=1)
    dye_cols = [c.removeprefix("vec__") for c in feats.columns
                if c.startswith("vec__dye_") and c.endswith("_pct") and "count" not in c and "total" not in c]
    present = long[long["dye_present"].astype(bool)]
    names = sorted(present["dye_name"].dropna().astype(str).unique())
    by_col = {_safe_dye_col(n): n for n in names}
    rows = []
    for k, col in enumerate(dye_cols, 1):
        name = by_col.get(col)
        if name is None:
            raise SystemExit(f"no dye name found for column {col}")
        recipes = present.loc[present["dye_name"] == name, "recipe_no"].nunique()
        rows.append({"global_dye_name": f"Dye {k:02d}", "dye_name_clean": name, "dye_column": col,
                     "original_brand_inferred": name.split()[0], "original_hue_inferred": infer_hue(name),
                     "n_recipes_present": int(recipes)})
    out_dir = ROOT / "data" / "global_naming"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"global_dye_mapping_active_{len(rows)}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    hues = pd.Series([r["original_hue_inferred"] for r in rows]).value_counts().to_dict()
    print(f"wrote {path.relative_to(ROOT)}: {len(rows)} dyes {hues}")


if __name__ == "__main__":
    main()
