from __future__ import annotations

import numpy as np
import pandas as pd

from .columns import safe_numeric


def add_recipe_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    features: list[str] = []

    candidates = [
        "dye_count",
        "total_dye_pct",
        "vec__dye_count_vectorized",
        "vec__total_dye_pct_vectorized",
        "has_cmc_quality",
        "has_production",
        "has_target_reflectance",
        "has_target_ks",
        "has_dye_vector",
    ]
    for col in candidates:
        if col in out.columns:
            out[col] = safe_numeric(out[col])
            features.append(col)

    dye_total = "vec__total_dye_pct_vectorized" if "vec__total_dye_pct_vectorized" in out.columns else "total_dye_pct"
    if dye_total in out.columns and "production__gsm" in out.columns:
        out["feat_recipe_total_dye_per_gsm"] = safe_numeric(out[dye_total]) / (safe_numeric(out["production__gsm"]) + 1e-6)
        features.append("feat_recipe_total_dye_per_gsm")

    if dye_total in out.columns and "production__chemical_load" in out.columns:
        out["feat_recipe_dye_to_chemical_load_ratio"] = safe_numeric(out[dye_total]) / (safe_numeric(out["production__chemical_load"]) + 1e-6)
        features.append("feat_recipe_dye_to_chemical_load_ratio")

    if "vec__dye_count_vectorized" in out.columns and "dye_count" in out.columns:
        out["feat_recipe_dye_count_agreement"] = (safe_numeric(out["vec__dye_count_vectorized"]) == safe_numeric(out["dye_count"])).astype(int)
        features.append("feat_recipe_dye_count_agreement")

    for col in features:
        out[col] = safe_numeric(out[col]).replace([np.inf, -np.inf], np.nan)
    return out, list(dict.fromkeys(features))
