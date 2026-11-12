from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from piml_color_recipe.features.columns import safe_numeric, sanitize_feature_name
from piml_color_recipe.features.dye_features import add_dye_family_features, detect_dye_vector_columns
from piml_color_recipe.features.feature_selection import run_multicollinearity_selection
from piml_color_recipe.features.process_features import add_process_categorical_features, add_process_numeric_features
from piml_color_recipe.features.recipe_features import add_recipe_features
from piml_color_recipe.features.spectral_features import (
    add_exploratory_pca,
    add_spectral_descriptors,
    detect_ks_columns,
    detect_reflectance_columns,
)
from piml_color_recipe.utils.hashing import make_file_manifest
from piml_color_recipe.utils.io import read_csv_flexible
from piml_color_recipe.utils.run_context import PipelineRunContext
from piml_color_recipe.utils.validation import SelfCheckResult, assert_no_failures

SCRIPT_VERSION = "2026-06-22-v2-src-first-features-multicollinearity-final"

QUALITY_TARGETS = [
    "dE_d65", "dE_f11", "dE_mean", "dE_max", "dE_min",
    "metamerism_gap", "metamerism_gap_abs",
    "met_index_d65", "met_index_f11",
    "dL_d65", "dL_f11", "da_d65", "da_f11", "db_d65", "db_f11",
    "dc_csc_d65", "dc_csc_f11", "dh_sh_d65", "dh_sh_f11",
]

FLAG_COLUMNS = [
    "is_full_supervised", "is_full_supervised_row",
    "has_cmc_quality", "has_production", "has_target_reflectance", "has_target_ks", "has_dye_vector",
]

KEY_COLUMNS = [
    "recipe_row_uid", "sl_no", "sample_id", "recipe_no", "target_id",
    "is_full_supervised", "is_full_supervised_row",
    "has_cmc_quality", "has_production", "has_target_reflectance", "has_target_ks", "has_dye_vector",
]


def run(project_root: str | Path | None = None) -> None:
    ctx = PipelineRunContext(
        script_name="03_build_features.py",
        stage="features",
        project_root=project_root,
        script_version=SCRIPT_VERSION,
    )
    try:
        summary = build_features(ctx)
        ctx.finish("OK", summary=summary)
    except Exception as exc:
        ctx.log_error(str(exc), exception_type=type(exc).__name__)
        ctx.finish("FAILED", summary={"error": str(exc), "exception_type": type(exc).__name__})
        raise


def build_features(ctx: PipelineRunContext) -> dict[str, Any]:
    input_path = ctx.paths.processed / "modelling_dataset.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing {input_path}. Run scripts/02_build_master_table.py first.")

    manifest = make_file_manifest([input_path], root=ctx.paths.root)
    ctx.write_df(manifest, ctx.paths.manifests / "features_input_file_hashes.csv")
    ctx.write_json(manifest.to_dict(orient="records"), ctx.paths.manifests / "features_input_file_hashes.json")

    df = read_csv_flexible(input_path)
    ctx.logger.info("Loaded %s: %s rows x %s cols", input_path.name, len(df), len(df.columns))

    df = normalize_flags(df)
    target_cols = [c for c in QUALITY_TARGETS if c in df.columns]
    for col in target_cols:
        df[col] = safe_numeric(df[col])

    refl_cols = detect_reflectance_columns(df)
    ks_cols = detect_ks_columns(df)
    dye_cols = detect_dye_vector_columns(df)
    ctx.logger.info(
        "Detected %d reflectance cols, %d K/S cols, %d dye vector cols, %d quality target cols",
        len(refl_cols), len(ks_cols), len(dye_cols), len(target_cols),
    )

    for col in refl_cols + ks_cols + dye_cols:
        df[col] = safe_numeric(df[col])

    df, recipe_cols = add_recipe_features(df)
    ctx.logger.info("Collected/added %d recipe-level features", len(recipe_cols))
    df, process_numeric_cols = add_process_numeric_features(df)
    ctx.logger.info("Collected/added %d process numeric features", len(process_numeric_cols))
    df, dye_family_cols = add_dye_family_features(df, dye_cols)
    ctx.logger.info("Added %d dye family/interaction features", len(dye_family_cols))
    df, spectral_descriptor_cols = add_spectral_descriptors(df, refl_cols, ks_cols)
    ctx.logger.info("Added %d spectral descriptor features", len(spectral_descriptor_cols))
    df, process_cat_cols = add_process_categorical_features(df)
    ctx.logger.info("Added %d one-hot process categorical features", len(process_cat_cols))
    df, pca_sets, pca_explained = add_exploratory_pca(df, refl_cols, ks_cols)
    for label, ratios in pca_explained.items():
        if ratios:
            ctx.logger.info("Added exploratory global PCA for %s; explained variance=%s", label, ratios)

    numeric_candidates = list(dict.fromkeys(
        recipe_cols + process_numeric_cols + dye_cols + dye_family_cols + spectral_descriptor_cols + process_cat_cols + refl_cols + ks_cols
    ))
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = safe_numeric(df[col]).replace([np.inf, -np.inf], np.nan)

    feature_sets = build_feature_sets(
        recipe_cols=recipe_cols,
        dye_cols=dye_cols,
        dye_family_cols=dye_family_cols,
        process_numeric_cols=process_numeric_cols,
        process_cat_cols=process_cat_cols,
        refl_cols=refl_cols,
        ks_cols=ks_cols,
        spectral_descriptor_cols=spectral_descriptor_cols,
        pca_sets=pca_sets,
    )
    default_feature_cols = list(dict.fromkeys(
        feature_sets["recipe_numeric"]
        + feature_sets["dye_vector"]
        + feature_sets["dye_family_interactions"]
        + feature_sets["process_numeric"]
        + feature_sets["process_categorical_onehot"]
        + feature_sets["spectral_reflectance_raw"]
        + feature_sets["spectral_ks_raw"]
        + feature_sets["spectral_descriptors"]
    ))
    default_feature_cols = [c for c in default_feature_cols if c in df.columns]

    supervised_mask = get_supervised_mask(df)
    multicollinearity = run_multicollinearity_selection(
        df=df,
        feature_cols=default_feature_cols,
        supervised_mask=supervised_mask,
        corr_threshold_primary=0.95,
        corr_threshold_strict=0.90,
        vif_threshold=10.0,
    )
    feature_sets = dict(feature_sets)
    feature_sets["linear_corr_pruned_095"] = multicollinearity.selected_feature_sets["linear_corr_pruned_095"]
    feature_sets["linear_corr_pruned_090_strict"] = multicollinearity.selected_feature_sets["linear_corr_pruned_090_strict"]
    feature_sets["linear_vif_pruned_10"] = multicollinearity.selected_feature_sets["linear_vif_pruned_10"]
    feature_sets["recommended_linear_safe"] = multicollinearity.selected_feature_sets["recommended_linear_safe"]
    feature_sets["recommended_tree_safe"] = multicollinearity.selected_feature_sets["recommended_tree_safe"]
    feature_sets["recommended_neural_safe"] = multicollinearity.selected_feature_sets["recommended_neural_safe"]
    ctx.logger.info(
        "Multicollinearity audit: high-corr pairs >=0.95=%d, recommended linear-safe corr-pruned features=%d, strict VIF-pruned features=%d",
        len(multicollinearity.high_corr_pairs),
        len(multicollinearity.selected_feature_sets.get("recommended_linear_safe", [])),
        len(multicollinearity.selected_features),
    )

    key_cols = [c for c in KEY_COLUMNS if c in df.columns]
    pca_cols = pca_sets.get("pca_reflectance_eda_global", []) + pca_sets.get("pca_ks_eda_global", [])
    all_feature_view_cols = list(dict.fromkeys(key_cols + target_cols + default_feature_cols + pca_cols))
    features_all = df[all_feature_view_cols].copy()
    features_supervised = features_all.loc[supervised_mask].copy()
    feature_matrix = df[list(dict.fromkeys(key_cols + default_feature_cols))].copy()
    target_matrix_supervised = df.loc[supervised_mask, list(dict.fromkeys(key_cols + target_cols))].copy()

    ctx.logger.info("features_all_rows: %s rows x %s cols", len(features_all), len(features_all.columns))
    ctx.logger.info("features_supervised: %s rows x %s cols", len(features_supervised), len(features_supervised.columns))
    ctx.logger.info("feature_matrix_unimputed: %s rows x %s cols, %s default features", len(feature_matrix), len(feature_matrix.columns), len(default_feature_cols))
    ctx.logger.info("target_matrix_supervised: %s rows x %s cols", len(target_matrix_supervised), len(target_matrix_supervised.columns))

    output_paths = write_outputs(ctx, features_all, features_supervised, feature_matrix, target_matrix_supervised, default_feature_cols, feature_sets, target_cols)
    write_multicollinearity_outputs(ctx, multicollinearity)
    write_audit_tables(ctx, df, features_supervised, default_feature_cols, feature_sets, target_cols)
    make_feature_figures(ctx, df, features_supervised, feature_sets)

    out_manifest = make_file_manifest(output_paths, root=ctx.paths.root)
    ctx.write_df(out_manifest, ctx.paths.manifests / "features_output_file_hashes.csv")
    ctx.write_json(out_manifest.to_dict(orient="records"), ctx.paths.manifests / "features_output_file_hashes.json")

    check = self_check(features_all, features_supervised, default_feature_cols, target_cols)
    ctx.log_self_check(check)
    assert_no_failures(check)

    return {
        "input_rows": int(len(df)),
        "features_all_rows": int(len(features_all)),
        "features_supervised_rows": int(len(features_supervised)),
        "default_feature_count": int(len(default_feature_cols)),
        "linear_safe_feature_count": int(len(multicollinearity.selected_feature_sets.get("recommended_linear_safe", []))),
        "strict_vif_pruned_feature_count": int(len(multicollinearity.selected_features)),
        "high_corr_pair_count_abs_spearman_ge_095": int(len(multicollinearity.high_corr_pairs)),
        "target_count": int(len(target_cols)),
        "reflectance_feature_count": int(len(refl_cols)),
        "ks_feature_count": int(len(ks_cols)),
        "dye_vector_feature_count": int(len(dye_cols)),
        "self_check": check.to_dict(),
        "note": "Global PCA columns are EDA-only; model scripts must fit PCA inside each CV fold if PCA is used.",
    }


def normalize_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Keep both names for backwards compatibility across older and src-first scripts.
    if "is_full_supervised" in out.columns and "is_full_supervised_row" not in out.columns:
        out["is_full_supervised_row"] = out["is_full_supervised"]
    if "is_full_supervised_row" in out.columns and "is_full_supervised" not in out.columns:
        out["is_full_supervised"] = out["is_full_supervised_row"]
    for col in FLAG_COLUMNS:
        if col in out.columns:
            out[col] = safe_numeric(out[col]).fillna(0).astype(int)
    return out


def get_supervised_mask(df: pd.DataFrame) -> pd.Series:
    if "is_full_supervised" in df.columns:
        return safe_numeric(df["is_full_supervised"]).fillna(0).astype(int).eq(1)
    if "is_full_supervised_row" in df.columns:
        return safe_numeric(df["is_full_supervised_row"]).fillna(0).astype(int).eq(1)
    required = [c for c in ["dE_d65", "dE_f11", "dE_max"] if c in df.columns]
    if required:
        return df[required].notna().all(axis=1)
    return pd.Series(False, index=df.index)


def build_feature_sets(**kwargs: list[str] | dict[str, list[str]]) -> dict[str, list[str]]:
    pca_sets = kwargs.pop("pca_sets")
    assert isinstance(pca_sets, dict)
    feature_sets = {
        "recipe_numeric": list(kwargs["recipe_cols"]),
        "dye_vector": list(kwargs["dye_cols"]),
        "dye_family_interactions": list(kwargs["dye_family_cols"]),
        "process_numeric": list(kwargs["process_numeric_cols"]),
        "process_categorical_onehot": list(kwargs["process_cat_cols"]),
        "spectral_reflectance_raw": list(kwargs["refl_cols"]),
        "spectral_ks_raw": list(kwargs["ks_cols"]),
        "spectral_descriptors": list(kwargs["spectral_descriptor_cols"]),
        "physics_compact": [
            "total_dye_pct",
            "feat_dye_max_component_pct",
            "feat_dye_component_entropy",
            "feat_dye_hue_yellow_pct",
            "feat_dye_hue_red_pct",
            "feat_dye_hue_blue_pct",
            "production__gsm",
            "production__salt",
            "production__soda_ash",
            "production__chemical_load",
            "feat_refl_mean",
            "feat_refl_range"
        ],
        "pca_reflectance_eda_global_not_for_cv_default": list(pca_sets.get("pca_reflectance_eda_global", [])),
        "pca_ks_eda_global_not_for_cv_default": list(pca_sets.get("pca_ks_eda_global", [])),
    }
    return {name: list(dict.fromkeys(cols)) for name, cols in feature_sets.items()}


def write_outputs(
    ctx: PipelineRunContext,
    features_all: pd.DataFrame,
    features_supervised: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    target_matrix_supervised: pd.DataFrame,
    feature_cols: list[str],
    feature_sets: dict[str, list[str]],
    target_cols: list[str],
) -> list[Path]:
    paths = [
        ctx.paths.processed / "features_all_rows.csv",
        ctx.paths.processed / "features_supervised.csv",
        ctx.paths.processed / "feature_matrix_unimputed.csv",
        ctx.paths.processed / "target_matrix_supervised.csv",
        ctx.paths.processed / "feature_columns.json",
        ctx.paths.processed / "feature_sets.json",
        ctx.paths.processed / "target_columns.json",
    ]
    ctx.write_df(features_all, paths[0])
    ctx.write_df(features_supervised, paths[1])
    ctx.write_df(feature_matrix, paths[2])
    ctx.write_df(target_matrix_supervised, paths[3])
    ctx.write_json(feature_cols, paths[4])
    ctx.write_json(feature_sets, paths[5])
    ctx.write_json(target_cols, paths[6])

    if "recipe_no" in features_supervised.columns:
        recipe_mean = features_supervised.groupby("recipe_no", as_index=False).mean(numeric_only=True)
        non_numeric = features_supervised.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            cols_to_keep = [c for c in non_numeric if c != "recipe_no"]
            if cols_to_keep:
                first_vals = features_supervised.groupby("recipe_no")[cols_to_keep].first().reset_index(drop=True)
                recipe_mean = pd.concat([recipe_mean, first_vals], axis=1)
        recipe_mean_path = ctx.paths.processed / "features_supervised_recipe_mean.csv"
        ctx.write_df(recipe_mean, recipe_mean_path)
        paths.append(recipe_mean_path)

    return paths



def write_multicollinearity_outputs(ctx: PipelineRunContext, result: Any) -> None:
    """Write multicollinearity diagnostics without changing the project folder structure."""
    ctx.write_df(result.summary, ctx.table_path("feature_selection_summary.csv"))
    ctx.write_df(result.high_corr_pairs, ctx.table_path("feature_high_correlation_pairs.csv"))
    ctx.write_df(result.drop_log, ctx.table_path("feature_multicollinearity_drop_log.csv"))
    ctx.write_df(result.vif_report, ctx.table_path("feature_vif_report.csv"))
    ctx.write_df(result.condition_report, ctx.table_path("feature_condition_index_report.csv"))
    ctx.write_json(result.selected_feature_sets, ctx.paths.processed / "selected_feature_sets.json")
    ctx.write_json(result.diagnostics, ctx.table_path("feature_selection_diagnostics.json"))

    # Compact figures for reviewer-facing diagnostics.
    summary = result.summary.copy()
    if not summary.empty:
        count_rows = summary[summary["metric"].astype(str).str.contains("feature_count", case=False, na=False)].copy()
        if not count_rows.empty:
            vals = pd.to_numeric(count_rows["value"], errors="coerce")
            plt.figure(figsize=(9, 4))
            plt.bar(count_rows["metric"].astype(str), vals)
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("Feature count")
            plt.title("Multicollinearity-aware feature set sizes")
            _save_current_fig(ctx.figure_path("feature_selection_feature_counts.png"), ctx)

    if not result.vif_report.empty and {"feature", "vif"}.issubset(result.vif_report.columns):
        top = result.vif_report.head(25).copy()
        top["vif"] = pd.to_numeric(top["vif"], errors="coerce")
        plt.figure(figsize=(9, max(4, len(top) * 0.25)))
        plt.barh(top["feature"].astype(str)[::-1], top["vif"][::-1])
        plt.xlabel("VIF")
        plt.title("Top VIF values after pruning")
        _save_current_fig(ctx.figure_path("feature_vif_top_after_pruning.png"), ctx)


def write_audit_tables(
    ctx: PipelineRunContext,
    df: pd.DataFrame,
    supervised: pd.DataFrame,
    feature_cols: list[str],
    feature_sets: dict[str, list[str]],
    target_cols: list[str],
) -> None:
    summary = pd.DataFrame([
        {"metric": "script_version", "value": SCRIPT_VERSION},
        {"metric": "input_rows", "value": int(len(df))},
        {"metric": "input_columns", "value": int(len(df.columns))},
        {"metric": "supervised_rows", "value": int(len(supervised))},
        {"metric": "default_feature_count", "value": int(len(feature_cols))},
        {"metric": "target_count", "value": int(len(target_cols))},
        {"metric": "reflectance_raw_feature_count", "value": int(len(feature_sets.get("spectral_reflectance_raw", [])))},
        {"metric": "ks_raw_feature_count", "value": int(len(feature_sets.get("spectral_ks_raw", [])))},
        {"metric": "dye_vector_feature_count", "value": int(len(feature_sets.get("dye_vector", [])))},
        {"metric": "process_numeric_feature_count", "value": int(len(feature_sets.get("process_numeric", [])))},
        {"metric": "process_categorical_onehot_feature_count", "value": int(len(feature_sets.get("process_categorical_onehot", [])))},
        {"metric": "spectral_descriptor_feature_count", "value": int(len(feature_sets.get("spectral_descriptors", [])))},
    ])
    ctx.write_df(summary, ctx.table_path("feature_build_summary.csv"))

    group_counts = pd.DataFrame([{"feature_group": k, "n_features": len(v)} for k, v in feature_sets.items()]).sort_values("feature_group")
    ctx.write_df(group_counts, ctx.table_path("feature_group_counts.csv"))

    ctx.write_df(feature_missingness(df, feature_cols), ctx.table_path("feature_missingness_summary.csv"))
    ctx.write_df(pd.DataFrame({"feature": feature_cols}), ctx.table_path("default_feature_columns.csv"))
    ctx.write_df(pd.DataFrame({"target": target_cols}), ctx.table_path("target_columns.csv"))
    ctx.write_df(leakage_exclusion_audit(df, feature_cols), ctx.table_path("feature_leakage_exclusion_audit.csv"))

    if target_cols:
        target_summary = supervised[target_cols].apply(pd.to_numeric, errors="coerce").describe().T.reset_index().rename(columns={"index": "target"})
        ctx.write_df(target_summary, ctx.table_path("supervised_target_summary.csv"))

    availability_rows = []
    for col in FLAG_COLUMNS:
        if col in df.columns:
            vals = safe_numeric(df[col]).fillna(0).astype(int)
            availability_rows.append({"availability_flag": col, "n_true": int(vals.sum()), "pct_true": round(float(vals.mean() * 100), 3)})
    ctx.write_df(pd.DataFrame(availability_rows), ctx.table_path("feature_input_availability_summary.csv"))


def feature_missingness(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    n = max(len(df), 1)
    for col in feature_cols:
        if col in df.columns:
            miss = int(df[col].isna().sum())
            nunique = int(df[col].nunique(dropna=True))
            dtype = str(df[col].dtype)
        else:
            miss, nunique, dtype = len(df), 0, "missing_col"
        rows.append({
            "feature": col,
            "missing_count": miss,
            "missing_pct": round(100 * miss / n, 3),
            "n_unique_non_missing": nunique,
            "dtype": dtype,
        })
    return pd.DataFrame(rows).sort_values(["missing_pct", "feature"], ascending=[False, True])


def leakage_exclusion_audit(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    feature_set = set(feature_cols)
    leakage_patterns = [
        "sample_id", "recipe_no", "target_id", "standard_name", "shade_name", "batch_name",
        "pf_decision", "quality_pass", "quality_fail", "dE", "dL", "da", "db", "dc_csc", "dh_sh", "met_index", "metamerism",
    ]
    rows = []
    for col in df.columns:
        if col in feature_set:
            continue
        low = col.lower()
        for pattern in leakage_patterns:
            if pattern.lower() in low:
                rows.append({"excluded_column": col, "reason": f"matches leakage-sensitive pattern '{pattern}'"})
                break
    return pd.DataFrame(rows)


def make_feature_figures(ctx: PipelineRunContext, df: pd.DataFrame, supervised: pd.DataFrame, feature_sets: dict[str, list[str]]) -> None:
    # 1. Feature group counts
    counts = pd.DataFrame([{"feature_group": k, "n_features": len(v)} for k, v in feature_sets.items()]).sort_values("feature_group")
    plt.figure(figsize=(10, 5))
    plt.bar(counts["feature_group"].astype(str), counts["n_features"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Number of features")
    plt.title("Feature count by group")
    _save_current_fig(ctx.figure_path("feature_group_counts.png"), ctx)

    # 2. Target distributions
    for target in [c for c in ["dE_d65", "dE_f11", "dE_max", "metamerism_gap_abs"] if c in supervised.columns]:
        vals = safe_numeric(supervised[target]).dropna()
        plt.figure(figsize=(7, 4))
        plt.hist(vals, bins=min(20, max(5, int(math.sqrt(max(len(vals), 1))))))
        plt.xlabel(target)
        plt.ylabel("Count")
        plt.title(f"Supervised target distribution: {target}")
        _save_current_fig(ctx.figure_path(f"target_distribution_{sanitize_feature_name(target)}.png"), ctx)

    # 3. D65 vs F11
    if {"dE_d65", "dE_f11"}.issubset(supervised.columns):
        x = safe_numeric(supervised["dE_d65"])
        y = safe_numeric(supervised["dE_f11"])
        plt.figure(figsize=(6, 6))
        plt.scatter(x, y, alpha=0.75)
        plt.xlabel("dE D65")
        plt.ylabel("dE F11")
        plt.title("dE D65 vs F11 in supervised rows")
        lim = np.nanmax(np.vstack([x.to_numpy(dtype=float), y.to_numpy(dtype=float)]))
        if np.isfinite(lim):
            plt.plot([0, lim], [0, lim], linestyle="--", linewidth=1)
        _save_current_fig(ctx.figure_path("supervised_dE_d65_vs_f11.png"), ctx)

    # 4. PCA scatter
    if {"feat_refl_pca1_eda_global", "feat_refl_pca2_eda_global", "dE_max"}.issubset(supervised.columns):
        plt.figure(figsize=(7, 5))
        sc = plt.scatter(supervised["feat_refl_pca1_eda_global"], supervised["feat_refl_pca2_eda_global"], c=safe_numeric(supervised["dE_max"]), alpha=0.75)
        plt.colorbar(sc, label="dE_max")
        plt.xlabel("Reflectance PCA1 (EDA global)")
        plt.ylabel("Reflectance PCA2 (EDA global)")
        plt.title("Exploratory target spectral PCA")
        _save_current_fig(ctx.figure_path("eda_reflectance_pca_scatter_by_dEmax.png"), ctx)

    # 5. Core correlations
    corr_candidates = [
        "dE_max", "dE_mean", "metamerism_gap_abs", "total_dye_pct", "vec__total_dye_pct_vectorized",
        "production__gsm", "production__salt", "production__soda_ash",
        "feat_refl_mean", "feat_refl_range", "feat_ks_mean", "feat_ks_range",
        "feat_dye_hue_yellow_pct", "feat_dye_hue_red_pct", "feat_dye_hue_blue_pct",
    ]
    corr_cols = [c for c in corr_candidates if c in supervised.columns]
    if len(corr_cols) >= 4:
        corr = supervised[corr_cols].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
        plt.figure(figsize=(max(8, len(corr_cols) * 0.55), max(6, len(corr_cols) * 0.45)))
        im = plt.imshow(corr.to_numpy(), aspect="auto", vmin=-1, vmax=1)
        plt.colorbar(im, label="Spearman correlation")
        plt.xticks(range(len(corr_cols)), corr_cols, rotation=60, ha="right", fontsize=8)
        plt.yticks(range(len(corr_cols)), corr_cols, fontsize=8)
        plt.title("Core feature/target Spearman correlations")
        _save_current_fig(ctx.figure_path("core_feature_target_spearman_correlation.png"), ctx)


def _save_current_fig(path: Path, ctx: PipelineRunContext) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    ctx.logger.info("Wrote figure -> %s", path)


def self_check(features_all: pd.DataFrame, supervised: pd.DataFrame, feature_cols: list[str], target_cols: list[str]) -> SelfCheckResult:
    result = SelfCheckResult(stage="features")
    result.add_count("features_all_rows", int(len(features_all)))
    result.add_count("features_supervised_rows", int(len(supervised)))
    result.add_count("n_feature_columns", int(len(feature_cols)))
    result.add_count("n_target_columns", int(len(target_cols)))
    result.add_count("duplicate_recipe_row_uid", int(features_all["recipe_row_uid"].duplicated().sum()) if "recipe_row_uid" in features_all.columns else -1)
    result.add_count("missing_recipe_row_uid", int(features_all["recipe_row_uid"].isna().sum()) if "recipe_row_uid" in features_all.columns else -1)

    if feature_cols:
        x = features_all[feature_cols].apply(pd.to_numeric, errors="coerce")
        result.add_count("nonfinite_feature_values", int(np.isinf(x.to_numpy(dtype=float)).sum()))
    else:
        result.add_count("nonfinite_feature_values", -1)

    for target in ["dE_max", "dE_d65", "dE_f11"]:
        if target in supervised.columns:
            result.add_count(f"missing_{target}_supervised", int(supervised[target].isna().sum()))

    result.fail_if(features_all.empty, "features_all_rows is empty")
    result.fail_if(supervised.empty, "features_supervised is empty")
    result.fail_if(len(feature_cols) < 20, f"too few default features: {len(feature_cols)}")
    result.fail_if(len(target_cols) < 3, f"too few quality targets: {len(target_cols)}")
    result.fail_if(result.checks.get("duplicate_recipe_row_uid", 0) != 0, "duplicate recipe_row_uid in features_all")
    result.fail_if(result.checks.get("missing_recipe_row_uid", 0) != 0, "missing recipe_row_uid in features_all")
    result.fail_if(result.checks.get("nonfinite_feature_values", 0) != 0, "non-finite feature values found")
    for target in ["dE_max", "dE_d65", "dE_f11"]:
        key = f"missing_{target}_supervised"
        if key in result.checks:
            result.fail_if(result.checks[key] != 0, f"{target} has missing values in supervised rows")
    return result
