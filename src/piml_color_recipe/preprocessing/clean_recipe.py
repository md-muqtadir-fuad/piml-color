from __future__ import annotations

from typing import Any

import pandas as pd

from piml_color_recipe.preprocessing.normalization import (
    clean_text,
    looks_like_pantone,
    looks_like_recipe,
    looks_like_sample,
    normalize_code,
    normalize_column_names,
    normalize_target_id,
    to_numeric_value,
)
from piml_color_recipe.utils.validation import require_columns

REQUIRED_RECIPE_COLUMNS = [
    "sample_id",
    "target_id",
    "recipe_no",
    "dye1",
    "dye1(%)",
    "dye2",
    "dye2(%)",
    "dye3",
    "dye3(%)",
]


def clean_recipe(recipe_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    df = normalize_column_names(recipe_raw)
    require_columns(df, REQUIRED_RECIPE_COLUMNS, "recipe.csv")

    out = df.copy()
    out["sample_id_raw"] = out["sample_id"]
    out["target_id_raw"] = out["target_id"]
    out["recipe_no_raw"] = out["recipe_no"]

    out["sample_id"] = out["sample_id"].map(normalize_code)
    out["recipe_no"] = out["recipe_no"].map(normalize_code)
    out["target_id"] = out["target_id"].map(normalize_target_id)

    for col in ["dye1", "dye2", "dye3"]:
        out[col] = out[col].map(clean_text)
        out[f"{col}_clean"] = out[col].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()

    for col in ["dye1(%)", "dye2(%)", "dye3(%)"]:
        out[col] = out[col].map(to_numeric_value)

    out["dye_count"] = 0
    for i in (1, 2, 3):
        out["dye_count"] += ((out[f"dye{i}_clean"].fillna("") != "") & (out[f"dye{i}(%)"].fillna(0) > 0)).astype(int)
    out["total_dye_pct"] = out[["dye1(%)", "dye2(%)", "dye3(%)"]].sum(axis=1, min_count=1)

    out["sample_id_valid"] = out["sample_id"].map(looks_like_sample)
    out["recipe_no_valid"] = out["recipe_no"].map(looks_like_recipe)
    out["target_id_valid"] = out["target_id"].map(looks_like_pantone)
    out["duplicate_sample_id"] = out["sample_id"].duplicated(keep=False)

    dye_rows: list[dict[str, Any]] = []
    for _, row in out.iterrows():
        for pos in (1, 2, 3):
            dye_name = row.get(f"dye{pos}_clean", "")
            dye_pct = row.get(f"dye{pos}(%)", pd.NA)
            if dye_name != "" or pd.notna(dye_pct):
                dye_rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "recipe_no": row["recipe_no"],
                        "target_id": row["target_id"],
                        "dye_position": pos,
                        "dye_name": dye_name,
                        "dye_pct": dye_pct,
                        "dye_present": bool(dye_name != "" and pd.notna(dye_pct) and float(dye_pct) > 0),
                    }
                )
    dye_long = pd.DataFrame(dye_rows)

    diagnostics = {
        "invalid_sample_id": int((~out["sample_id_valid"]).sum()),
        "invalid_recipe_no": int((~out["recipe_no_valid"]).sum()),
        "invalid_target_id_warning": int((~out["target_id_valid"]).sum()),
        "duplicate_sample_id_warning": int(out["duplicate_sample_id"].sum()),
        "dye_long_rows": int(len(dye_long)),
    }
    return out, dye_long, diagnostics
