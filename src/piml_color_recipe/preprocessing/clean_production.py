from __future__ import annotations

from typing import Any
import re

import numpy as np
import pandas as pd

from piml_color_recipe.preprocessing.normalization import (
    clean_text,
    looks_like_pantone,
    looks_like_recipe,
    normalize_code,
    normalize_column_names,
    normalize_target_id,
    to_numeric_value,
)
from piml_color_recipe.utils.validation import require_columns

NUMERIC_PRODUCTION_COLUMNS = [
    "modal",
    "viscose",
    "cotton",
    "polyester",
    "lycra",
    "gsm",
    "sample_weight",
    "salt",
    "soda_ash",
    "total_trial",
]
FIBER_COLUMNS = ["modal", "viscose", "cotton", "polyester", "lycra"]


def parse_l2r(value: Any) -> float:
    text = clean_text(value).lower().replace(" ", "")
    m = re.match(r"^(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)$", text)
    if not m:
        return float("nan")
    left, right = float(m.group(1)), float(m.group(2))
    return right / left if left else float("nan")


def clean_production(production_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    df = normalize_column_names(production_raw)
    require_columns(df, ["target_id", "recipe_no"], "production.csv")

    out = df.copy()
    out["target_id_raw"] = out["target_id"]
    out["recipe_no_raw"] = out["recipe_no"]
    out["target_id"] = out["target_id"].map(normalize_target_id)
    out["recipe_no"] = out["recipe_no"].map(normalize_code)

    for col in ["kniting_struc1", "kniting_struc2", "fabric", "shade", "l2r"]:
        if col in out.columns:
            out[col] = out[col].map(clean_text)

    for col in NUMERIC_PRODUCTION_COLUMNS:
        if col in out.columns:
            out[col] = out[col].map(to_numeric_value)

    fiber_cols = [c for c in FIBER_COLUMNS if c in out.columns]
    if fiber_cols:
        out["fiber_total_pct"] = out[fiber_cols].sum(axis=1, min_count=1)
        out["dominant_fiber"] = out[fiber_cols].idxmax(axis=1)
        out["is_lycra_blend"] = out.get("lycra", pd.Series(0, index=out.index)).fillna(0) > 0
        out["is_polyester_blend"] = out.get("polyester", pd.Series(0, index=out.index)).fillna(0) > 0
    else:
        out["fiber_total_pct"] = np.nan
        out["dominant_fiber"] = ""
        out["is_lycra_blend"] = False
        out["is_polyester_blend"] = False

    salt = out.get("salt", pd.Series(np.nan, index=out.index))
    soda = out.get("soda_ash", pd.Series(np.nan, index=out.index))
    gsm = out.get("gsm", pd.Series(np.nan, index=out.index)).replace(0, np.nan)
    out["chemical_load"] = salt.fillna(0) + soda.fillna(0)
    out["salt_soda_ratio"] = salt / soda.replace(0, np.nan)
    out["chemical_load_per_gsm"] = out["chemical_load"] / gsm
    out["l2r_numeric"] = out["l2r"].map(parse_l2r) if "l2r" in out.columns else np.nan

    out["recipe_no_valid"] = out["recipe_no"].map(looks_like_recipe)
    out["target_id_valid"] = out["target_id"].map(looks_like_pantone)
    out["fiber_total_valid"] = out["fiber_total_pct"].between(99.5, 100.5) | out["fiber_total_pct"].isna()
    out["duplicate_recipe_no"] = out["recipe_no"].duplicated(keep=False)

    diagnostics = {
        "rows": int(len(out)),
        "invalid_recipe_no": int((~out["recipe_no_valid"]).sum()),
        "invalid_target_id_warning": int((~out["target_id_valid"]).sum()),
        "fiber_total_not_100_warning": int((~out["fiber_total_valid"]).sum()),
        "duplicate_recipe_no_warning": int(out["duplicate_recipe_no"].sum()),
    }
    return out, diagnostics
