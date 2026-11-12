from __future__ import annotations

import numpy as np
import pandas as pd

from .columns import safe_numeric


def detect_dye_vector_columns(df: pd.DataFrame) -> list[str]:
    return sorted([
        c for c in df.columns
        if c.startswith("vec__dye_") and c.endswith("_pct")
    ])


def add_dye_family_features(df: pd.DataFrame, dye_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    features: list[str] = []
    if not dye_cols:
        return out, features

    dye_numeric = out[dye_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    family_patterns = {
        "feat_dye_brand_a_pct": ["brand_a"],
        "feat_dye_brand_b_pct": ["brand_b"],
        "feat_dye_brand_c_pct": ["brand_c"],
        "feat_dye_brand_d_pct": ["brand_d"],
        "feat_dye_hue_yellow_pct": ["yellow", "gold"],
        "feat_dye_hue_red_pct": ["red", "ruby", "rubine", "rose", "carmine"],
        "feat_dye_hue_blue_pct": ["blue", "navy", "ocean"],
        "feat_dye_hue_black_pct": ["black"],
    }

    for new_col, keys in family_patterns.items():
        matched = [c for c in dye_cols if any(k in c.lower() for k in keys)]
        out[new_col] = dye_numeric[matched].sum(axis=1) if matched else 0.0
        features.append(new_col)

    hue_known = ["feat_dye_hue_yellow_pct", "feat_dye_hue_red_pct", "feat_dye_hue_blue_pct", "feat_dye_hue_black_pct"]
    total = dye_numeric.sum(axis=1)
    out["feat_dye_hue_other_pct"] = total - out[hue_known].sum(axis=1)
    out["feat_dye_nonzero_count_engineered"] = (dye_numeric > 0).sum(axis=1)
    out["feat_dye_max_component_pct"] = dye_numeric.max(axis=1)
    out["feat_dye_component_entropy"] = _row_entropy(dye_numeric)
    features.extend([
        "feat_dye_hue_other_pct",
        "feat_dye_nonzero_count_engineered",
        "feat_dye_max_component_pct",
        "feat_dye_component_entropy",
    ])

    out["feat_dye_red_blue_ratio"] = out["feat_dye_hue_red_pct"] / (out["feat_dye_hue_blue_pct"] + 1e-6)
    out["feat_dye_yellow_blue_ratio"] = out["feat_dye_hue_yellow_pct"] / (out["feat_dye_hue_blue_pct"] + 1e-6)
    out["feat_dye_yellow_red_ratio"] = out["feat_dye_hue_yellow_pct"] / (out["feat_dye_hue_red_pct"] + 1e-6)
    out["feat_dye_primary_balance_std"] = out[["feat_dye_hue_yellow_pct", "feat_dye_hue_red_pct", "feat_dye_hue_blue_pct"]].std(axis=1)
    features.extend([
        "feat_dye_red_blue_ratio",
        "feat_dye_yellow_blue_ratio",
        "feat_dye_yellow_red_ratio",
        "feat_dye_primary_balance_std",
    ])

    for col in features:
        out[col] = safe_numeric(out[col]).replace([np.inf, -np.inf], np.nan)
    return out, list(dict.fromkeys(features))


def _row_entropy(values: pd.DataFrame) -> pd.Series:
    arr = values.to_numpy(dtype=float)
    row_sum = arr.sum(axis=1, keepdims=True)
    prob = np.divide(arr, row_sum, out=np.zeros_like(arr), where=row_sum > 0)
    entropy = -np.sum(np.where(prob > 0, prob * np.log(prob + 1e-12), 0.0), axis=1)
    return pd.Series(entropy, index=values.index)
