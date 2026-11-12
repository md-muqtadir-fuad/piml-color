from __future__ import annotations

import re

import pandas as pd

from piml_color_recipe.preprocessing.normalization import clean_text, looks_like_pantone, normalize_column_names, normalize_target_id, to_numeric_value
from piml_color_recipe.utils.validation import require_columns


def stable_reflectance_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        m = re.match(r"reflectance_(\d+)nm", str(col))
        if m and 400 <= int(m.group(1)) <= 750:
            cols.append(col)
    return sorted(cols, key=lambda c: int(re.search(r"(\d+)", c).group(1)))


def clean_pantones(pantones_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, int]]:
    df = normalize_column_names(pantones_raw)
    require_columns(df, ["standard_name", "shade_name", "batch_name"], "pantones.csv")
    out = df.copy()

    refl_cols = stable_reflectance_columns(out)
    if not refl_cols:
        raise ValueError("No stable reflectance_400nm ... reflectance_750nm columns found in pantones.csv")

    for col in refl_cols:
        out[col] = out[col].map(to_numeric_value)

    for col in ["standard_name", "shade_name", "batch_name"]:
        if col in out.columns:
            out[col] = out[col].map(normalize_target_id)
    if "record_type" in out.columns:
        out["record_type"] = out["record_type"].map(clean_text)

    out["target_id"] = ""
    for col in ["batch_name", "shade_name", "standard_name"]:
        if col in out.columns:
            mask = (out["target_id"] == "") & out[col].notna() & (out[col].astype(str).str.strip() != "")
            out.loc[mask, "target_id"] = out.loc[mask, col]
    out["target_id"] = out["target_id"].map(normalize_target_id)

    out["target_id_valid"] = out["target_id"].map(looks_like_pantone)
    out["reflectance_complete_400_750"] = out[refl_cols].notna().all(axis=1)
    out["reflectance_missing_count_400_750"] = out[refl_cols].isna().sum(axis=1)
    out["reflectance_mean_400_750"] = out[refl_cols].mean(axis=1)

    rank = pd.Series(0, index=out.index, dtype=int)
    if "record_type" in out.columns:
        record_type = out["record_type"].str.upper()
        rank += record_type.eq("BATCH_DATA").astype(int) * 2
        rank += record_type.eq("STANDARD_DATA").astype(int)
    rank += out["reflectance_complete_400_750"].astype(int) * 4
    out["target_record_rank"] = rank

    unique = (
        out.sort_values(["target_id", "target_record_rank"], ascending=[True, False])
        .drop_duplicates("target_id", keep="first")
        .copy()
    )

    diagnostics = {
        "rows": int(len(out)),
        "unique_target_ids": int(out["target_id"].nunique(dropna=True)),
        "stable_reflectance_columns": int(len(refl_cols)),
        "complete_400_750_rows": int(out["reflectance_complete_400_750"].sum()),
        "invalid_target_id_warning": int((~out["target_id_valid"]).sum()),
        "unique_duplicate_target_ids_after_dedup": int(unique["target_id"].duplicated().sum()),
    }
    return out, unique, refl_cols, diagnostics
