from __future__ import annotations

import pandas as pd


def build_target_matching_audit(recipe: pd.DataFrame, production: pd.DataFrame, pantone_unique: pd.DataFrame) -> pd.DataFrame:
    recipe_targets = set(recipe["target_id"].dropna().astype(str)) if "target_id" in recipe.columns else set()
    production_targets = set(production["target_id"].dropna().astype(str)) if "target_id" in production.columns else set()
    pantone_targets = set(pantone_unique["target_id"].dropna().astype(str)) if "target_id" in pantone_unique.columns else set()
    all_targets = sorted(recipe_targets | production_targets)

    rows = []
    for target in all_targets:
        rows.append(
            {
                "target_id": target,
                "in_recipe": target in recipe_targets,
                "recipe_rows": int((recipe["target_id"] == target).sum()) if "target_id" in recipe.columns else 0,
                "in_production": target in production_targets,
                "production_rows": int((production["target_id"] == target).sum()) if "target_id" in production.columns else 0,
                "in_pantones": target in pantone_targets,
                "match_status": "matched" if target in pantone_targets else "unmatched",
            }
        )
    return pd.DataFrame(rows)
