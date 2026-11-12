from __future__ import annotations

import numpy as np
import pandas as pd

from .columns import safe_numeric, sanitize_feature_name


def add_process_numeric_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    features: list[str] = []

    candidates = [
        "production__gsm",
        "production__sample_weight",
        "production__salt",
        "production__soda_ash",
        "production__total_trial",
        "production__fiber_total_pct",
        "production__chemical_load",
        "production__salt_soda_ratio",
        "production__chemical_load_per_gsm",
        "production__l2r_numeric",
        "production__modal",
        "production__viscose",
        "production__cotton",
        "production__polyester",
        "production__lycra",
        "production__is_lycra_blend",
        "production__is_polyester_blend",
        "production__recipe_no_valid",
        "production__target_id_valid",
        "production__fiber_total_valid",
        "production__duplicate_recipe_no",
    ]
    for col in candidates:
        if col in out.columns:
            out[col] = safe_numeric(out[col])
            features.append(col)

    if {"production__salt", "production__soda_ash"}.issubset(out.columns):
        out["feat_process_salt_minus_soda"] = safe_numeric(out["production__salt"]) - safe_numeric(out["production__soda_ash"])
        out["feat_process_salt_plus_soda"] = safe_numeric(out["production__salt"]) + safe_numeric(out["production__soda_ash"])
        features.extend(["feat_process_salt_minus_soda", "feat_process_salt_plus_soda"])

    if {"production__salt", "production__gsm"}.issubset(out.columns):
        out["feat_process_salt_per_gsm"] = safe_numeric(out["production__salt"]) / (safe_numeric(out["production__gsm"]) + 1e-6)
        features.append("feat_process_salt_per_gsm")

    if {"production__soda_ash", "production__gsm"}.issubset(out.columns):
        out["feat_process_soda_per_gsm"] = safe_numeric(out["production__soda_ash"]) / (safe_numeric(out["production__gsm"]) + 1e-6)
        features.append("feat_process_soda_per_gsm")

    if {"production__cotton", "production__polyester"}.issubset(out.columns):
        out["feat_process_cotton_polyester_ratio"] = safe_numeric(out["production__cotton"]) / (safe_numeric(out["production__polyester"]) + 1e-6)
        features.append("feat_process_cotton_polyester_ratio")

    for col in features:
        out[col] = safe_numeric(out[col]).replace([np.inf, -np.inf], np.nan)
    return out, list(dict.fromkeys(features))


def add_process_categorical_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    cat_cols = [
        "production__fabric",
        "production__kniting_struc1",
        "production__kniting_struc2",
        "production__dominant_fiber",
    ]
    features: list[str] = []
    dummy_frames: list[pd.DataFrame] = []
    for col in cat_cols:
        if col not in out.columns:
            continue
        values = out[col].fillna("missing").astype(str).str.strip().replace({"": "missing"})
        # Keep a stable rare-category guard.
        if values.nunique(dropna=False) > 25:
            top = values.value_counts(dropna=False).head(24).index
            values = values.where(values.isin(top), "other_rare")
        dummies = pd.get_dummies(values, prefix=f"feat_cat_{sanitize_feature_name(col)}", dtype=int)
        dummy_frames.append(dummies)
        features.extend(dummies.columns.tolist())
    if dummy_frames:
        out = pd.concat([out] + dummy_frames, axis=1)
    return out, list(dict.fromkeys(features))
