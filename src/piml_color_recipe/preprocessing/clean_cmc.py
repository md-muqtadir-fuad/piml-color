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
from piml_color_recipe.preprocessing.schema_correction import select_best_cmc_layout
from piml_color_recipe.utils.validation import require_columns

REQUIRED_CMC_COLUMNS = [
    "sample_id",
    "target_id",
    "recipe_no",
    "illum",
    "dE",
    "pf_decision",
    "dL",
    "da",
    "db",
    "dc_csc",
    "dh_sh",
    "met_index",
]
NUMERIC_CMC_COLUMNS = ["dE", "dL", "da", "db", "dc_csc", "dh_sh", "met_index"]


def normalize_pf(value: Any) -> str:
    text = clean_text(value).lower()
    return {"passed": "pass", "p": "pass", "failed": "fail", "f": "fail"}.get(text, text)


def clean_cmc(cmc_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    df = normalize_column_names(cmc_raw)
    require_columns(df, REQUIRED_CMC_COLUMNS, "cmc.csv")

    out = df.copy()
    out["sample_id_original"] = out["sample_id"]
    out["target_id_original"] = out["target_id"]
    out["recipe_no_original"] = out["recipe_no"]

    cleaned_records: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []

    for idx, row in out.iterrows():
        selected, score = select_best_cmc_layout(
            row["sample_id_original"],
            row["target_id_original"],
            row["recipe_no_original"],
        )
        rec = row.to_dict()
        rec["sample_id"] = normalize_code(selected.sample_id)
        rec["target_id"] = normalize_target_id(selected.target_id)
        rec["recipe_no"] = normalize_code(selected.recipe_no)
        rec["schema_layout_selected"] = selected.layout_name
        rec["schema_validity_score"] = score
        rec["schema_corrected"] = bool(selected.layout_name != "normal")
        cleaned_records.append(rec)

        if selected.layout_name != "normal" or score < 3:
            audit_records.append(
                {
                    "row_index": idx,
                    "sl_no": row.get("sl_no", pd.NA),
                    "layout_selected": selected.layout_name,
                    "validity_score": score,
                    "sample_id_original": clean_text(row["sample_id_original"]),
                    "target_id_original": clean_text(row["target_id_original"]),
                    "recipe_no_original": clean_text(row["recipe_no_original"]),
                    "sample_id_clean": rec["sample_id"],
                    "target_id_clean": rec["target_id"],
                    "recipe_no_clean": rec["recipe_no"],
                }
            )

    cleaned = pd.DataFrame(cleaned_records)
    cleaned["illum"] = cleaned["illum"].map(lambda x: clean_text(x).lower())
    cleaned["pf_decision"] = cleaned["pf_decision"].map(normalize_pf)

    for col in NUMERIC_CMC_COLUMNS:
        cleaned[col] = cleaned[col].map(to_numeric_value)

    cleaned["sample_id_valid"] = cleaned["sample_id"].map(looks_like_sample)
    cleaned["recipe_no_valid"] = cleaned["recipe_no"].map(looks_like_recipe)
    cleaned["target_id_valid"] = cleaned["target_id"].map(looks_like_pantone)
    cleaned["illum_valid"] = cleaned["illum"].isin(["f11", "d65"])
    cleaned["duplicate_cmc_key"] = cleaned.duplicated(subset=["sample_id", "recipe_no", "target_id", "illum"], keep=False)

    audit = pd.DataFrame(audit_records)
    diagnostics = {
        "rows": int(len(cleaned)),
        "schema_corrected_rows": int(cleaned["schema_corrected"].sum()),
        "invalid_sample_id": int((~cleaned["sample_id_valid"]).sum()),
        "invalid_recipe_no": int((~cleaned["recipe_no_valid"]).sum()),
        "invalid_target_id": int((~cleaned["target_id_valid"]).sum()),
        "invalid_illum": int((~cleaned["illum_valid"]).sum()),
        "duplicate_cmc_key": int(cleaned["duplicate_cmc_key"].sum()),
    }
    return cleaned, audit, diagnostics
