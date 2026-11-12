from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from piml_color_recipe.utils.hashing import make_file_manifest
from piml_color_recipe.utils.io import read_csv_flexible, write_csv, write_json
from piml_color_recipe.utils.plotting import save_bar, save_hist
from piml_color_recipe.utils.run_context import PipelineRunContext
from piml_color_recipe.utils.validation import SelfCheckResult

SCRIPT_VERSION = "2026-06-22-v1-src-first-master-final"
EPS = 1e-6
QUALITY_COLUMNS = ["dE", "dL", "da", "db", "dc_csc", "dh_sh", "met_index"]
ILLUMINANTS = ["d65", "f11"]


@dataclass(frozen=True)
class MasterInputs:
    recipe: pd.DataFrame
    cmc: pd.DataFrame
    production: pd.DataFrame
    pantone: pd.DataFrame


def _log_read(ctx: PipelineRunContext, path: Path) -> pd.DataFrame:
    ctx.logger.info("Reading input file: %s", path)
    df = read_csv_flexible(path)
    ctx.logger.info("Loaded %s: %s rows x %s cols", path.name, len(df), len(df.columns))
    return df


def load_inputs(ctx: PipelineRunContext) -> MasterInputs:
    files = {
        "recipe": ctx.paths.interim / "recipe_cleaned.csv",
        "cmc": ctx.paths.interim / "cmc_cleaned.csv",
        "production": ctx.paths.interim / "production_cleaned.csv",
        "pantone": ctx.paths.interim / "pantone_target_spectra_unique.csv",
    }
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing cleaned/interim inputs. Run scripts/01_clean_all.py first. Missing: " + "; ".join(missing))

    manifest = make_file_manifest(list(files.values()), root=ctx.paths.root)
    ctx.write_df(manifest, ctx.paths.manifests / "master_input_file_hashes.csv")
    ctx.write_json(manifest.to_dict(orient="records"), ctx.paths.manifests / "master_input_file_hashes.json")

    return MasterInputs(
        recipe=_log_read(ctx, files["recipe"]),
        cmc=_log_read(ctx, files["cmc"]),
        production=_log_read(ctx, files["production"]),
        pantone=_log_read(ctx, files["pantone"]),
    )


def make_recipe_row_uid(recipe: pd.DataFrame) -> pd.DataFrame:
    df = recipe.copy()
    if "recipe_row_uid" not in df.columns:
        df.insert(0, "recipe_row_uid", [f"rr_{i + 1:04d}_{sid}" for i, sid in enumerate(df["sample_id"].astype(str))])
    return df


def build_cmc_wide(cmc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = cmc.copy()
    df["illum"] = df["illum"].astype(str).str.lower().str.strip()
    for col in QUALITY_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    key_cols = ["sample_id", "recipe_no", "target_id"]
    duplicated = df.duplicated(key_cols + ["illum"], keep=False)
    duplicate_audit = df.loc[duplicated, key_cols + ["illum"] + [c for c in QUALITY_COLUMNS if c in df.columns]].copy()

    # If duplicate CMC entries exist, average numeric metrics and keep first decision.
    agg_map: dict[str, Any] = {c: "mean" for c in QUALITY_COLUMNS if c in df.columns}
    if "pf_decision" in df.columns:
        agg_map["pf_decision"] = "first"
    grouped = df.groupby(key_cols + ["illum"], dropna=False, as_index=False).agg(agg_map)

    pieces = []
    for illum in ILLUMINANTS:
        sub = grouped[grouped["illum"] == illum].copy()
        keep = key_cols + [c for c in QUALITY_COLUMNS if c in sub.columns]
        if "pf_decision" in sub.columns:
            keep.append("pf_decision")
        sub = sub[keep]
        rename = {c: f"{c}_{illum}" for c in sub.columns if c not in key_cols}
        pieces.append(sub.rename(columns=rename))

    if not pieces:
        wide = pd.DataFrame(columns=key_cols)
    else:
        wide = pieces[0]
        for piece in pieces[1:]:
            wide = wide.merge(piece, on=key_cols, how="outer")

    # Derived multi-illuminant quality targets.
    if {"dE_d65", "dE_f11"}.issubset(wide.columns):
        wide["has_dE_d65"] = wide["dE_d65"].notna()
        wide["has_dE_f11"] = wide["dE_f11"].notna()
        wide["has_complete_dE"] = wide["has_dE_d65"] & wide["has_dE_f11"]
        wide["dE_mean"] = wide[["dE_d65", "dE_f11"]].mean(axis=1)
        wide["dE_max"] = wide[["dE_d65", "dE_f11"]].max(axis=1)
        wide["dE_min"] = wide[["dE_d65", "dE_f11"]].min(axis=1)
        wide["metamerism_gap"] = wide["dE_d65"] - wide["dE_f11"]
        wide["metamerism_gap_abs"] = wide["metamerism_gap"].abs()
        wide["quality_pass_both"] = False
        if {"pf_decision_d65", "pf_decision_f11"}.issubset(wide.columns):
            wide["quality_pass_both"] = (
                wide["pf_decision_d65"].astype(str).str.lower().eq("pass")
                & wide["pf_decision_f11"].astype(str).str.lower().eq("pass")
            )
    else:
        wide["has_complete_dE"] = False

    return wide, duplicate_audit


def _safe_dye_col(name: Any) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return f"dye_{s}_pct" if s else "dye_unknown_pct"


def build_dye_vector(recipe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_rows: list[dict[str, Any]] = []
    for _, row in recipe.iterrows():
        for pos in (1, 2, 3):
            dye_col = f"dye{pos}_clean" if f"dye{pos}_clean" in recipe.columns else f"dye{pos}"
            pct_col = f"dye{pos}(%)"
            dye_name = row.get(dye_col)
            dye_pct = pd.to_numeric(row.get(pct_col), errors="coerce")
            present = pd.notna(dye_name) and str(dye_name).strip() != "" and pd.notna(dye_pct) and float(dye_pct) != 0
            long_rows.append(
                {
                    "recipe_row_uid": row["recipe_row_uid"],
                    "sample_id": row.get("sample_id"),
                    "recipe_no": row.get("recipe_no"),
                    "target_id": row.get("target_id"),
                    "dye_position": pos,
                    "dye_name": str(dye_name).strip().lower() if pd.notna(dye_name) else "",
                    "dye_feature": _safe_dye_col(dye_name),
                    "dye_pct": float(dye_pct) if pd.notna(dye_pct) else 0.0,
                    "dye_present": bool(present),
                }
            )
    long = pd.DataFrame(long_rows)
    present = long[long["dye_present"]].copy()
    if present.empty:
        base = recipe[["recipe_row_uid", "sample_id", "recipe_no", "target_id"]].copy()
        base["dye_count_vectorized"] = 0
        base["total_dye_pct_vectorized"] = 0.0
        return base, long

    pivot = present.pivot_table(
        index=["recipe_row_uid", "sample_id", "recipe_no", "target_id"],
        columns="dye_feature",
        values="dye_pct",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()
    pivot.columns.name = None
    dye_cols = [c for c in pivot.columns if c.startswith("dye_") and c.endswith("_pct")]
    pivot["dye_count_vectorized"] = (pivot[dye_cols] > 0).sum(axis=1)
    pivot["total_dye_pct_vectorized"] = pivot[dye_cols].sum(axis=1)

    # Preserve recipe rows with no present dye values, if any.
    base = recipe[["recipe_row_uid", "sample_id", "recipe_no", "target_id"]].copy()
    out = base.merge(pivot, on=["recipe_row_uid", "sample_id", "recipe_no", "target_id"], how="left")
    for col in dye_cols + ["dye_count_vectorized", "total_dye_pct_vectorized"]:
        if col in out.columns:
            out[col] = out[col].fillna(0)
    return out, long


def stable_reflectance_columns(pantone: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in pantone.columns:
        m = re.fullmatch(r"reflectance_(\d+)nm", str(c))
        if m and 400 <= int(m.group(1)) <= 750:
            cols.append(c)
    return sorted(cols, key=lambda x: int(re.search(r"(\d+)", x).group(1)))


def build_spectral_matrices(pantone: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    refl_cols = stable_reflectance_columns(pantone)
    meta_cols = [c for c in ["target_id", "shade_name", "batch_name", "standard_name", "reflectance_complete_400_750", "reflectance_missing_count_400_750", "reflectance_mean_400_750", "target_record_rank"] if c in pantone.columns]
    reflectance = pantone[meta_cols + refl_cols].copy()
    for c in refl_cols:
        reflectance[c] = pd.to_numeric(reflectance[c], errors="coerce")

    ks = reflectance[meta_cols].copy()
    ks_cols: list[str] = []
    for c in refl_cols:
        nm = re.search(r"(\d+)", c).group(1)
        out_col = f"ks_{nm}nm"
        r = pd.to_numeric(reflectance[c], errors="coerce")
        # Pantone reflectance is stored as percent in these files. Convert to fraction.
        r = np.where(r > 1.5, r / 100.0, r)
        r = np.clip(r.astype(float), EPS, 0.999999)
        ks[out_col] = ((1.0 - r) ** 2) / (2.0 * r)
        ks_cols.append(out_col)

    return reflectance, ks, refl_cols, ks_cols


def build_master_tables(inputs: MasterInputs, ctx: PipelineRunContext) -> dict[str, pd.DataFrame]:
    recipe = make_recipe_row_uid(inputs.recipe)
    cmc_wide, cmc_duplicate_audit = build_cmc_wide(inputs.cmc)
    dye_vector, dye_long_with_uid = build_dye_vector(recipe)
    target_reflectance, target_ks, refl_cols, ks_cols = build_spectral_matrices(inputs.pantone)

    ctx.logger.info("Wide CMC quality table: %s rows x %s cols", len(cmc_wide), len(cmc_wide.columns))
    ctx.logger.info("Dye vector matrix: %s rows x %s cols", len(dye_vector), len(dye_vector.columns))
    ctx.logger.info("Target reflectance matrix: %s rows x %s cols", len(target_reflectance), len(target_reflectance.columns))
    ctx.logger.info("Target K/S matrix: %s rows x %s cols", len(target_ks), len(target_ks.columns))

    # Production is recipe-level. Prefer recipe_no merge; keep target mismatch as audit rather than fatal.
    prod_cols = [c for c in inputs.production.columns if c != "sl_no"]
    production = inputs.production[prod_cols].copy()
    production = production.drop_duplicates(subset=["recipe_no"], keep="first")

    master_quality = recipe.merge(cmc_wide, on=["sample_id", "recipe_no", "target_id"], how="left", indicator="_merge_cmc")
    master_quality["has_cmc_quality"] = master_quality["_merge_cmc"].eq("both")
    master_quality = master_quality.drop(columns=["_merge_cmc"])

    master_quality = master_quality.merge(production.add_prefix("production__"), left_on="recipe_no", right_on="production__recipe_no", how="left", indicator="_merge_production")
    master_quality["has_production"] = master_quality["_merge_production"].eq("both")
    master_quality = master_quality.drop(columns=["_merge_production"])

    master_spectral = master_quality.merge(target_reflectance.add_prefix("target_refl__"), left_on="target_id", right_on="target_refl__target_id", how="left", indicator="_merge_reflectance")
    master_spectral["has_target_reflectance"] = master_spectral["_merge_reflectance"].eq("both")
    master_spectral = master_spectral.drop(columns=["_merge_reflectance"])

    master_spectral = master_spectral.merge(target_ks.add_prefix("target_ks__"), left_on="target_id", right_on="target_ks__target_id", how="left", indicator="_merge_ks")
    master_spectral["has_target_ks"] = master_spectral["_merge_ks"].eq("both")
    master_spectral = master_spectral.drop(columns=["_merge_ks"])

    modelling = master_spectral.merge(dye_vector.add_prefix("vec__"), left_on="recipe_row_uid", right_on="vec__recipe_row_uid", how="left", indicator="_merge_dye_vector")
    modelling["has_dye_vector"] = modelling["_merge_dye_vector"].eq("both")
    modelling = modelling.drop(columns=["_merge_dye_vector"])

    # Full supervised row requires complete D65/F11 quality, production, target reflectance, and dye vector.
    if "has_complete_dE" not in modelling.columns:
        modelling["has_complete_dE"] = False
    else:
        modelling["has_complete_dE"] = modelling["has_complete_dE"].astype("boolean").fillna(False).astype(bool)
    modelling["is_full_supervised"] = (
        modelling["has_complete_dE"]
        & modelling["has_production"].fillna(False).astype(bool)
        & modelling["has_target_reflectance"].fillna(False).astype(bool)
        & modelling["has_target_ks"].fillna(False).astype(bool)
        & modelling["has_dye_vector"].fillna(False).astype(bool)
    )

    # Quality issue reasons for rows that are not fully supervised.
    reasons = []
    for _, r in modelling.iterrows():
        row_reasons = []
        if not bool(r.get("has_complete_dE", False)):
            row_reasons.append("missing_complete_d65_f11_quality")
        if not bool(r.get("has_production", False)):
            row_reasons.append("missing_production")
        if not bool(r.get("has_target_reflectance", False)):
            row_reasons.append("missing_target_reflectance")
        if not bool(r.get("has_target_ks", False)):
            row_reasons.append("missing_target_ks")
        if not bool(r.get("has_dye_vector", False)):
            row_reasons.append("missing_dye_vector")
        reasons.append(";".join(row_reasons))
    modelling["non_full_supervised_reason"] = reasons

    return {
        "master_recipe_quality": master_quality,
        "master_recipe_quality_spectral": master_spectral,
        "modelling_dataset": modelling,
        "dye_vector_matrix": dye_vector,
        "recipe_dye_long_with_uid": dye_long_with_uid,
        "target_reflectance_matrix": target_reflectance,
        "target_ks_matrix": target_ks,
        "cmc_quality_wide": cmc_wide,
        "cmc_duplicate_key_audit": cmc_duplicate_audit,
    }


def summarize_outputs(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    modelling = tables["modelling_dataset"]
    summary = {
        "recipe_rows": int(len(tables["master_recipe_quality"])),
        "master_recipe_quality_rows": int(len(tables["master_recipe_quality"])),
        "master_recipe_quality_spectral_rows": int(len(tables["master_recipe_quality_spectral"])),
        "modelling_dataset_rows": int(len(modelling)),
        "rows_with_cmc_quality": int(modelling["has_cmc_quality"].sum()) if "has_cmc_quality" in modelling else 0,
        "rows_with_complete_dE": int(modelling["has_complete_dE"].sum()) if "has_complete_dE" in modelling else 0,
        "rows_with_production": int(modelling["has_production"].sum()) if "has_production" in modelling else 0,
        "rows_with_target_reflectance": int(modelling["has_target_reflectance"].sum()) if "has_target_reflectance" in modelling else 0,
        "rows_with_target_ks": int(modelling["has_target_ks"].sum()) if "has_target_ks" in modelling else 0,
        "rows_with_dye_vector": int(modelling["has_dye_vector"].sum()) if "has_dye_vector" in modelling else 0,
        "full_supervised_rows": int(modelling["is_full_supervised"].sum()) if "is_full_supervised" in modelling else 0,
        "unique_recipe_no_full_supervised": int(modelling.loc[modelling["is_full_supervised"], "recipe_no"].nunique()) if "is_full_supervised" in modelling else 0,
        "unique_target_id_full_supervised": int(modelling.loc[modelling["is_full_supervised"], "target_id"].nunique()) if "is_full_supervised" in modelling else 0,
        "dye_vector_columns": int(sum(c.startswith("dye_") and c.endswith("_pct") for c in tables["dye_vector_matrix"].columns)),
        "target_reflectance_columns": int(sum(c.startswith("reflectance_") and c.endswith("nm") for c in tables["target_reflectance_matrix"].columns)),
        "target_ks_columns": int(sum(c.startswith("ks_") and c.endswith("nm") for c in tables["target_ks_matrix"].columns)),
        "cmc_duplicate_key_rows": int(len(tables["cmc_duplicate_key_audit"])),
    }
    return summary


def write_tables_and_figures(tables: dict[str, pd.DataFrame], ctx: PipelineRunContext) -> dict[str, Any]:
    processed = ctx.paths.processed
    outputs: list[Path] = []

    outputs.append(ctx.write_df(tables["master_recipe_quality"], processed / "master_recipe_quality.csv"))
    outputs.append(ctx.write_df(tables["master_recipe_quality_spectral"], processed / "master_recipe_quality_spectral.csv"))
    outputs.append(ctx.write_df(tables["dye_vector_matrix"], processed / "dye_vector_matrix.csv"))
    outputs.append(ctx.write_df(tables["target_reflectance_matrix"], processed / "target_reflectance_matrix.csv"))
    outputs.append(ctx.write_df(tables["target_ks_matrix"], processed / "target_ks_matrix.csv"))
    outputs.append(ctx.write_df(tables["modelling_dataset"], processed / "modelling_dataset.csv"))
    outputs.append(ctx.write_df(tables["cmc_quality_wide"], processed / "cmc_quality_wide.csv"))

    summary = summarize_outputs(tables)
    summary_df = pd.DataFrame([{"metric": k, "value": v} for k, v in summary.items()])
    outputs.append(ctx.write_df(summary_df, ctx.table_path("master_build_summary.csv")))

    modelling = tables["modelling_dataset"]
    availability_items = [
        "has_cmc_quality",
        "has_complete_dE",
        "has_production",
        "has_target_reflectance",
        "has_target_ks",
        "has_dye_vector",
        "is_full_supervised",
    ]
    availability = []
    for col in availability_items:
        if col in modelling.columns:
            availability.append({"component": col, "rows": int(modelling[col].fillna(False).astype(bool).sum()), "total_rows": int(len(modelling)), "rate": float(modelling[col].fillna(False).astype(bool).mean())})
    availability_df = pd.DataFrame(availability)
    outputs.append(ctx.write_df(availability_df, ctx.table_path("merge_availability_summary.csv")))

    non_full_cols = [c for c in ["recipe_row_uid", "sample_id", "recipe_no", "target_id", "has_cmc_quality", "has_complete_dE", "has_production", "has_target_reflectance", "has_target_ks", "has_dye_vector", "is_full_supervised", "non_full_supervised_reason"] if c in modelling.columns]
    non_full = modelling.loc[~modelling["is_full_supervised"].fillna(False).astype(bool), non_full_cols].copy()
    outputs.append(ctx.write_df(non_full, ctx.table_path("non_full_supervised_rows_audit.csv")))

    if "production__target_id" in modelling.columns:
        mismatch = modelling.loc[modelling["has_production"].fillna(False).astype(bool) & modelling["production__target_id"].notna() & (modelling["target_id"].astype(str) != modelling["production__target_id"].astype(str)), ["recipe_row_uid", "sample_id", "recipe_no", "target_id", "production__target_id"]].copy()
    else:
        mismatch = pd.DataFrame(columns=["recipe_row_uid", "sample_id", "recipe_no", "target_id", "production__target_id"])
    outputs.append(ctx.write_df(mismatch, ctx.table_path("production_target_mismatch_audit.csv")))

    dup_sample = modelling.loc[modelling.duplicated("sample_id", keep=False), ["recipe_row_uid", "sample_id", "recipe_no", "target_id", "total_dye_pct", "is_full_supervised"]].copy() if "sample_id" in modelling.columns else pd.DataFrame()
    outputs.append(ctx.write_df(dup_sample, ctx.table_path("duplicate_sample_id_rows_audit.csv")))

    target_summary = modelling.groupby("target_id", dropna=False).agg(
        rows=("recipe_row_uid", "count"),
        full_supervised_rows=("is_full_supervised", "sum"),
        recipes=("recipe_no", "nunique"),
        mean_dE_max=("dE_max", "mean") if "dE_max" in modelling.columns else ("recipe_row_uid", "count"),
        min_dE_max=("dE_max", "min") if "dE_max" in modelling.columns else ("recipe_row_uid", "count"),
    ).reset_index()
    outputs.append(ctx.write_df(target_summary, ctx.table_path("target_level_master_coverage.csv")))

    quality_cols = [c for c in ["dE_d65", "dE_f11", "dE_mean", "dE_max", "metamerism_gap_abs"] if c in modelling.columns]
    if quality_cols:
        qsum = modelling.loc[modelling["is_full_supervised"], quality_cols].describe().T.reset_index().rename(columns={"index": "metric"})
    else:
        qsum = pd.DataFrame(columns=["metric"])
    outputs.append(ctx.write_df(qsum, ctx.table_path("master_quality_metric_summary.csv")))

    dye_cols = [c for c in tables["dye_vector_matrix"].columns if c.startswith("dye_") and c.endswith("_pct")]
    dye_summary = pd.DataFrame({
        "dye_feature": dye_cols,
        "nonzero_rows": [(tables["dye_vector_matrix"][c] > 0).sum() for c in dye_cols],
        "mean_pct": [tables["dye_vector_matrix"][c].mean() for c in dye_cols],
        "max_pct": [tables["dye_vector_matrix"][c].max() for c in dye_cols],
        "total_pct": [tables["dye_vector_matrix"][c].sum() for c in dye_cols],
    }).sort_values(["nonzero_rows", "total_pct"], ascending=False)
    outputs.append(ctx.write_df(dye_summary, ctx.table_path("dye_vector_feature_summary.csv")))

    outputs.append(ctx.write_df(tables["cmc_duplicate_key_audit"], ctx.table_path("cmc_duplicate_key_audit.csv")))

    # Figures.
    if not availability_df.empty:
        p = save_bar(availability_df.set_index("component")["rows"], ctx.figure_path("master_component_availability.png"), "Master-table component availability", ylabel="Rows")
        ctx.logger.info("Wrote figure -> %s", p)
    if "dE_max" in modelling.columns:
        p = save_hist(modelling.loc[modelling["is_full_supervised"], "dE_max"].dropna(), ctx.figure_path("master_dE_max_distribution.png"), "Full-supervised dE_max distribution", "dE_max")
        ctx.logger.info("Wrote figure -> %s", p)
    if "metamerism_gap_abs" in modelling.columns:
        p = save_hist(modelling.loc[modelling["is_full_supervised"], "metamerism_gap_abs"].dropna(), ctx.figure_path("master_metamerism_gap_distribution.png"), "Full-supervised metamerism gap distribution", "|dE_D65 - dE_F11|", bins=25)
        ctx.logger.info("Wrote figure -> %s", p)

    # Scatter and mean spectrum need direct matplotlib.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if {"dE_d65", "dE_f11"}.issubset(modelling.columns):
            sub = modelling.loc[modelling["is_full_supervised"], ["dE_d65", "dE_f11"]].dropna()
            if len(sub):
                plt.figure(figsize=(6, 5))
                plt.scatter(sub["dE_d65"], sub["dE_f11"], alpha=0.75)
                lim = max(sub["dE_d65"].max(), sub["dE_f11"].max()) * 1.05
                plt.plot([0, lim], [0, lim], linestyle="--")
                plt.xlabel("dE D65")
                plt.ylabel("dE F11")
                plt.title("D65 vs F11 quality error")
                plt.tight_layout()
                p = ctx.figure_path("master_dE_d65_vs_f11.png")
                plt.savefig(p, dpi=200)
                plt.close()
                ctx.logger.info("Wrote figure -> %s", p)
        refl_cols = [c for c in modelling.columns if c.startswith("target_refl__reflectance_") and c.endswith("nm")]
        if refl_cols:
            mean_refl = modelling.loc[modelling["is_full_supervised"], refl_cols].mean(axis=0, skipna=True)
            x = [int(re.search(r"(\d+)", c).group(1)) for c in refl_cols]
            plt.figure(figsize=(8, 5))
            plt.plot(x, mean_refl.values)
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Mean reflectance (%)")
            plt.title("Mean matched target reflectance spectrum")
            plt.tight_layout()
            p = ctx.figure_path("matched_target_mean_reflectance_spectrum.png")
            plt.savefig(p, dpi=200)
            plt.close()
            ctx.logger.info("Wrote figure -> %s", p)
    except Exception as exc:  # figure generation must not corrupt the data build
        ctx.log_warning("Optional master figure generation failed", error=str(exc))

    manifest = make_file_manifest(outputs, root=ctx.paths.root)
    ctx.write_df(manifest, ctx.paths.manifests / "master_output_file_hashes.csv")
    ctx.write_json(manifest.to_dict(orient="records"), ctx.paths.manifests / "master_output_file_hashes.json")
    return summary


def self_check(tables: dict[str, pd.DataFrame], summary: dict[str, Any]) -> SelfCheckResult:
    modelling = tables["modelling_dataset"]
    checks = {
        "master_row_count_matches_recipe": int(len(tables["master_recipe_quality"]) == 259),
        "modelling_row_count_matches_recipe": int(len(modelling) == len(tables["master_recipe_quality"])),
        "recipe_row_uid_duplicates": int(modelling["recipe_row_uid"].duplicated().sum()) if "recipe_row_uid" in modelling.columns else -1,
        "cmc_quality_key_duplicates": int(tables["cmc_quality_wide"].duplicated(["sample_id", "recipe_no", "target_id"]).sum()) if all(c in tables["cmc_quality_wide"].columns for c in ["sample_id", "recipe_no", "target_id"]) else -1,
        "target_reflectance_target_duplicates": int(tables["target_reflectance_matrix"].duplicated("target_id").sum()) if "target_id" in tables["target_reflectance_matrix"].columns else -1,
        "target_ks_target_duplicates": int(tables["target_ks_matrix"].duplicated("target_id").sum()) if "target_id" in tables["target_ks_matrix"].columns else -1,
        "dye_vector_uid_duplicates": int(tables["dye_vector_matrix"].duplicated("recipe_row_uid").sum()) if "recipe_row_uid" in tables["dye_vector_matrix"].columns else -1,
        "full_supervised_rows": int(summary.get("full_supervised_rows", 0)),
        "rows_with_cmc_quality": int(summary.get("rows_with_cmc_quality", 0)),
        "rows_with_target_reflectance": int(summary.get("rows_with_target_reflectance", 0)),
        "dye_vector_columns": int(summary.get("dye_vector_columns", 0)),
        "target_reflectance_columns": int(summary.get("target_reflectance_columns", 0)),
        "target_ks_columns": int(summary.get("target_ks_columns", 0)),
    }
    failures: list[str] = []
    warnings: list[str] = []
    if checks["modelling_row_count_matches_recipe"] != 1:
        failures.append("modelling_dataset row count does not match recipe rows")
    if checks["recipe_row_uid_duplicates"] != 0:
        failures.append("duplicate recipe_row_uid values")
    if checks["cmc_quality_key_duplicates"] != 0:
        failures.append("duplicate wide CMC quality keys")
    if checks["target_reflectance_target_duplicates"] != 0:
        failures.append("duplicate target reflectance target IDs")
    if checks["target_ks_target_duplicates"] != 0:
        failures.append("duplicate target K/S target IDs")
    if checks["dye_vector_uid_duplicates"] != 0:
        failures.append("duplicate dye vector recipe_row_uid values")
    if checks["full_supervised_rows"] < 150:
        failures.append("too few full supervised rows for downstream modelling")
    if checks["target_reflectance_columns"] != 36 or checks["target_ks_columns"] != 36:
        failures.append("expected exactly 36 stable 400-750 nm reflectance/K/S columns")
    if checks["rows_with_target_reflectance"] < checks["full_supervised_rows"]:
        failures.append("full supervised row count exceeds target reflectance coverage")

    return SelfCheckResult(stage="master", checks=checks, warnings=warnings, failures=failures)


def run(project_root: str | Path | None = None) -> dict[str, Any]:
    ctx = PipelineRunContext(
        script_name="02_build_master_table.py",
        stage="master",
        project_root=project_root,
        script_version=SCRIPT_VERSION,
    )
    try:
        inputs = load_inputs(ctx)
        tables = build_master_tables(inputs, ctx)
        summary = write_tables_and_figures(tables, ctx)
        result = self_check(tables, summary)
        ctx.log_self_check(result)
        if not result.ok:
            raise AssertionError("Master table self-check failed: " + "; ".join(result.failures))
        ctx.finish(status="OK", summary=summary | {"self_check": result.to_dict()})
        return summary
    except Exception as exc:
        ctx.log_error("02_build_master_table.py failed", error=str(exc))
        ctx.finish(status="FAILED", summary={"error": str(exc)})
        raise
