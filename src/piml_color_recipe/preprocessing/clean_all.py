from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from piml_color_recipe.preprocessing.clean_cmc import clean_cmc
from piml_color_recipe.preprocessing.clean_pantones import clean_pantones
from piml_color_recipe.preprocessing.clean_production import clean_production
from piml_color_recipe.preprocessing.clean_recipe import clean_recipe
from piml_color_recipe.preprocessing.target_matching import build_target_matching_audit
from piml_color_recipe.utils.hashing import make_file_manifest
from piml_color_recipe.utils.io import read_csv_flexible
from piml_color_recipe.utils.run_context import PipelineRunContext
from piml_color_recipe.utils.validation import SelfCheckResult, assert_no_failures

SCRIPT_VERSION = "2026-06-22-v1-src-first-cleaning-final"
RAW_FILES = ["recipe.csv", "cmc.csv", "production.csv", "pantones.csv"]


def _read_raw_files(ctx: PipelineRunContext) -> dict[str, pd.DataFrame]:
    raw: dict[str, pd.DataFrame] = {}
    for fname in RAW_FILES:
        path = ctx.paths.raw / fname
        ctx.logger.info("Reading raw file: %s", path)
        df = read_csv_flexible(path)
        raw[fname] = df
        ctx.logger.info("Loaded %s: %s rows x %s cols", fname, len(df), len(df.columns))
    return raw


def _save_fig(path: Path, title: str | None = None) -> None:
    if title:
        plt.title(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def _write_cleaning_quality_tables(
    ctx: PipelineRunContext,
    recipe: pd.DataFrame,
    cmc: pd.DataFrame,
    production: pd.DataFrame,
    pantone: pd.DataFrame,
    target_audit: pd.DataFrame,
    diagnostics: dict[str, dict[str, Any]],
) -> None:
    summary_rows: list[dict[str, Any]] = [{"metric": "script_version", "value": SCRIPT_VERSION}]
    for section, values in diagnostics.items():
        for key, value in values.items():
            summary_rows.append({"metric": f"{section}_{key}", "value": value})
    summary_rows.extend(
        [
            {"metric": "target_matched", "value": int((target_audit["match_status"] == "matched").sum())},
            {"metric": "target_unmatched_warning", "value": int((target_audit["match_status"] == "unmatched").sum())},
        ]
    )
    ctx.write_df(pd.DataFrame(summary_rows), ctx.table_path("cleaning_summary.csv"))

    recipe_cols = [
        "sample_id",
        "recipe_no",
        "target_id",
        "sample_id_valid",
        "recipe_no_valid",
        "target_id_valid",
        "duplicate_sample_id",
        "total_dye_pct",
        "dye_count",
    ]
    ctx.write_df(recipe[[c for c in recipe_cols if c in recipe.columns]], ctx.table_path("recipe_cleaning_quality_flags.csv"))

    cmc_cols = [
        "sample_id",
        "recipe_no",
        "target_id",
        "illum",
        "schema_layout_selected",
        "schema_corrected",
        "schema_validity_score",
        "sample_id_valid",
        "recipe_no_valid",
        "target_id_valid",
        "illum_valid",
        "duplicate_cmc_key",
    ]
    ctx.write_df(cmc[[c for c in cmc_cols if c in cmc.columns]], ctx.table_path("cmc_cleaning_quality_flags.csv"))

    prod_cols = [
        "recipe_no",
        "target_id",
        "recipe_no_valid",
        "target_id_valid",
        "fiber_total_pct",
        "fiber_total_valid",
        "duplicate_recipe_no",
        "chemical_load",
        "l2r_numeric",
        "gsm",
    ]
    ctx.write_df(production[[c for c in prod_cols if c in production.columns]], ctx.table_path("production_cleaning_quality_flags.csv"))

    duplicate_targets = pantone[pantone["target_id"].duplicated(keep=False)].copy()
    dup_cols = [c for c in ["target_id", "record_type", "reflectance_complete_400_750", "target_record_rank"] if c in duplicate_targets.columns]
    if len(duplicate_targets):
        duplicate_targets = duplicate_targets[dup_cols].sort_values("target_id")
    else:
        duplicate_targets = pd.DataFrame(columns=dup_cols)
    ctx.write_df(duplicate_targets, ctx.table_path("pantone_duplicate_target_records.csv"))


def _make_cleaning_figures(
    ctx: PipelineRunContext,
    recipe: pd.DataFrame,
    cmc: pd.DataFrame,
    production: pd.DataFrame,
    target_audit: pd.DataFrame,
) -> None:
    plt.figure(figsize=(6, 4))
    cmc["schema_corrected"].value_counts(dropna=False).rename(index={False: "not corrected", True: "corrected"}).plot(kind="bar")
    plt.ylabel("Rows")
    _save_fig(ctx.figure_path("cmc_schema_correction_counts.png"), "CMC schema correction counts")
    ctx.logger.info("Wrote figure -> %s", ctx.figure_path("cmc_schema_correction_counts.png"))

    plt.figure(figsize=(6, 4))
    target_audit["match_status"].value_counts().plot(kind="bar")
    plt.ylabel("Unique targets")
    _save_fig(ctx.figure_path("target_match_status.png"), "Target match status against Pantone spectra")
    ctx.logger.info("Wrote figure -> %s", ctx.figure_path("target_match_status.png"))

    plt.figure(figsize=(7, 4))
    recipe["total_dye_pct"].dropna().plot(kind="hist", bins=25)
    plt.xlabel("Total dye %")
    plt.ylabel("Frequency")
    _save_fig(ctx.figure_path("recipe_total_dye_pct_distribution.png"), "Recipe total dye percentage distribution")
    ctx.logger.info("Wrote figure -> %s", ctx.figure_path("recipe_total_dye_pct_distribution.png"))

    plt.figure(figsize=(7, 4))
    for illum, group in cmc.groupby("illum"):
        plt.hist(group["dE"].dropna(), bins=25, alpha=0.45, label=str(illum))
    plt.xlabel("dE")
    plt.ylabel("Frequency")
    plt.legend()
    _save_fig(ctx.figure_path("cmc_de_distribution_by_illum.png"), "CMC dE distribution by illuminant")
    ctx.logger.info("Wrote figure -> %s", ctx.figure_path("cmc_de_distribution_by_illum.png"))

    if {"gsm", "chemical_load"}.issubset(production.columns):
        plt.figure(figsize=(7, 4))
        plt.scatter(production["gsm"], production["chemical_load"])
        plt.xlabel("GSM")
        plt.ylabel("Salt + soda ash")
        _save_fig(ctx.figure_path("production_chemical_load_vs_gsm.png"), "Production chemical load vs GSM")
        ctx.logger.info("Wrote figure -> %s", ctx.figure_path("production_chemical_load_vs_gsm.png"))


def _self_check(
    recipe: pd.DataFrame,
    cmc: pd.DataFrame,
    production: pd.DataFrame,
    pantone_unique: pd.DataFrame,
    target_audit: pd.DataFrame,
) -> SelfCheckResult:
    result = SelfCheckResult(stage="cleaning")
    result.add_count("recipe_rows", int(len(recipe)))
    result.add_count("cmc_rows", int(len(cmc)))
    result.add_count("production_rows", int(len(production)))
    result.add_count("pantone_unique_targets", int(pantone_unique["target_id"].nunique()))
    result.add_count("cmc_schema_corrected_rows", int(cmc["schema_corrected"].sum()))
    result.add_count("target_matched", int((target_audit["match_status"] == "matched").sum()))
    result.add_count("target_unmatched", int((target_audit["match_status"] == "unmatched").sum()))

    fatal_counts = {
        "recipe_invalid_sample_id": int((~recipe["sample_id_valid"]).sum()),
        "recipe_invalid_recipe_no": int((~recipe["recipe_no_valid"]).sum()),
        "cmc_invalid_sample_id": int((~cmc["sample_id_valid"]).sum()),
        "cmc_invalid_recipe_no": int((~cmc["recipe_no_valid"]).sum()),
        "cmc_invalid_target_id": int((~cmc["target_id_valid"]).sum()),
        "cmc_invalid_illum": int((~cmc["illum_valid"]).sum()),
        "production_invalid_recipe_no": int((~production["recipe_no_valid"]).sum()),
        "pantone_duplicate_unique_target_ids": int(pantone_unique["target_id"].duplicated().sum()),
    }
    warning_counts = {
        "recipe_invalid_target_id_warning": int((~recipe["target_id_valid"]).sum()),
        "recipe_duplicate_sample_id_warning": int(recipe["duplicate_sample_id"].sum()),
        "production_invalid_target_id_warning": int((~production["target_id_valid"]).sum()),
        "production_fiber_total_not_100_warning": int((~production["fiber_total_valid"]).sum()),
        "target_unmatched_warning": int((target_audit["match_status"] == "unmatched").sum()),
    }
    for key, value in fatal_counts.items():
        result.add_count(key, value)
        result.fail_if(value != 0, f"{key}={value}")
    for key, value in warning_counts.items():
        result.add_count(key, value)
        result.warn_if(value != 0, f"{key}={value}")

    result.fail_if(len(recipe) == 0, "recipe_cleaned is empty")
    result.fail_if(len(cmc) == 0, "cmc_cleaned is empty")
    result.fail_if(len(production) == 0, "production_cleaned is empty")
    result.fail_if(len(pantone_unique) == 0, "pantone_target_spectra_unique is empty")
    return result


def run_cleaning(project_root: str | Path | None = None) -> dict[str, Any]:
    ctx = PipelineRunContext(
        script_name="01_clean_all.py",
        stage="cleaning",
        project_root=project_root,
        script_version=SCRIPT_VERSION,
    )
    try:
        input_paths = [ctx.paths.raw / fname for fname in RAW_FILES]
        input_manifest = make_file_manifest(input_paths, root=ctx.paths.root)
        ctx.write_df(input_manifest, ctx.paths.manifests / "cleaning_input_file_hashes.csv")
        ctx.write_json(input_manifest.to_dict(orient="records"), ctx.paths.manifests / "cleaning_input_file_hashes.json")

        raw = _read_raw_files(ctx)

        recipe, dye_long, recipe_diag = clean_recipe(raw["recipe.csv"])
        ctx.logger.info("Recipe cleaning diagnostics: %s", recipe_diag)

        cmc, cmc_audit, cmc_diag = clean_cmc(raw["cmc.csv"])
        ctx.logger.info("CMC cleaning diagnostics: %s", cmc_diag)

        production, production_diag = clean_production(raw["production.csv"])
        ctx.logger.info("Production cleaning diagnostics: %s", production_diag)

        pantone, pantone_unique, refl_cols, pantone_diag = clean_pantones(raw["pantones.csv"])
        ctx.logger.info("Pantone cleaning diagnostics: %s", pantone_diag)
        ctx.logger.info("Stable reflectance columns: %s", refl_cols)

        target_audit = build_target_matching_audit(recipe, production, pantone_unique)
        ctx.logger.info(
            "Target matching: %s matched, %s unmatched across %s unique targets",
            int((target_audit["match_status"] == "matched").sum()),
            int((target_audit["match_status"] == "unmatched").sum()),
            len(target_audit),
        )

        ctx.write_df(recipe, ctx.paths.interim / "recipe_cleaned.csv")
        ctx.write_df(cmc, ctx.paths.interim / "cmc_schema_corrected.csv")
        ctx.write_df(cmc_audit, ctx.paths.interim / "cmc_schema_correction_audit.csv")
        ctx.write_df(cmc, ctx.paths.interim / "cmc_cleaned.csv")
        ctx.write_df(production, ctx.paths.interim / "production_cleaned.csv")
        ctx.write_df(pantone, ctx.paths.interim / "pantone_spectra_cleaned.csv")
        ctx.write_df(pantone_unique, ctx.paths.interim / "pantone_target_spectra_unique.csv")
        ctx.write_df(dye_long, ctx.paths.interim / "recipe_dye_long.csv")
        ctx.write_df(target_audit, ctx.paths.interim / "target_matching_audit.csv")

        diagnostics = {
            "recipe": recipe_diag,
            "cmc": cmc_diag,
            "production": production_diag,
            "pantone": pantone_diag,
        }
        _write_cleaning_quality_tables(ctx, recipe, cmc, production, pantone, target_audit, diagnostics)
        _make_cleaning_figures(ctx, recipe, cmc, production, target_audit)

        output_paths = sorted(
            list(ctx.paths.interim.glob("*.csv"))
            + list(ctx.tables_dir.glob("*.csv"))
            + list(ctx.figures_dir.glob("*.png"))
        )
        output_manifest = make_file_manifest(output_paths, root=ctx.paths.root)
        ctx.write_df(output_manifest, ctx.paths.manifests / "cleaning_output_file_hashes.csv")
        ctx.write_json(output_manifest.to_dict(orient="records"), ctx.paths.manifests / "cleaning_output_file_hashes.json")

        check = _self_check(recipe, cmc, production, pantone_unique, target_audit)
        ctx.log_self_check(check)
        assert_no_failures(check)

        summary = {
            "cmc_schema_corrected_rows": int(cmc["schema_corrected"].sum()),
            "cmc_invalid_sample_id": int((~cmc["sample_id_valid"]).sum()),
            "cmc_invalid_recipe_no": int((~cmc["recipe_no_valid"]).sum()),
            "target_matched": int((target_audit["match_status"] == "matched").sum()),
            "target_unmatched": int((target_audit["match_status"] == "unmatched").sum()),
            "stable_reflectance_columns": refl_cols,
            "output_files": int(len(output_manifest)),
        }
        ctx.finish(status="OK", summary=summary)
        return summary
    except Exception as exc:
        ctx.log_error(str(exc))
        ctx.finish(status="FAILED", summary={"error": str(exc)})
        raise
