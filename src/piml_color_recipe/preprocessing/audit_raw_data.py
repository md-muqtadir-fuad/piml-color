from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from piml_color_recipe.preprocessing.normalization import (
    looks_like_pantone,
    looks_like_recipe,
    looks_like_sample,
    normalize_target_id,
    to_numeric_series,
)
from piml_color_recipe.utils import PipelineRunContext, SelfCheckResult, assert_no_failures, make_file_manifest, read_csv_flexible
from piml_color_recipe.utils.plotting import save_bar, save_hist

SCRIPT_VERSION = "2026-06-22-v1-src-first-raw-audit"
REQUIRED_RAW_FILES = ["recipe.csv", "cmc.csv", "production.csv", "pantones.csv"]


def detect_reflectance_columns(df: pd.DataFrame) -> list[str]:
    cols: list[tuple[int, str]] = []
    for c in df.columns:
        m = re.match(r"^reflectance_(\d+)nm$", str(c).strip(), flags=re.IGNORECASE)
        if m:
            cols.append((int(m.group(1)), c))
    return [c for _, c in sorted(cols)]


def read_raw_tables(ctx: PipelineRunContext) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for filename in REQUIRED_RAW_FILES:
        path = ctx.paths.raw / filename
        ctx.logger.info("Reading raw file: %s", path)
        tables[filename] = read_csv_flexible(path)
        ctx.logger.info("Loaded %s: %s rows x %s cols", filename, len(tables[filename]), len(tables[filename].columns))
    return tables


def build_columns_summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for filename, df in tables.items():
        for col in df.columns:
            s = df[col]
            rows.append(
                {
                    "file": filename,
                    "column": col,
                    "dtype": str(s.dtype),
                    "rows": len(df),
                    "non_null": int(s.notna().sum()),
                    "missing": int(s.isna().sum()),
                    "missing_pct": round(float(s.isna().mean() * 100), 4),
                    "unique_non_null": int(s.nunique(dropna=True)),
                    "example_1": _safe_example(s, 0),
                    "example_2": _safe_example(s, 1),
                    "example_3": _safe_example(s, 2),
                }
            )
    return pd.DataFrame(rows)


def _safe_example(s: pd.Series, idx: int) -> Any:
    vals = s.dropna().astype(str).head(idx + 1).tolist()
    return vals[idx] if len(vals) > idx else None


def build_duplicate_key_summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    specs = {
        "recipe.csv": [["sample_id"], ["recipe_no"], ["target_id"], ["sample_id", "recipe_no", "target_id"]],
        "cmc.csv": [["sample_id"], ["sample_id", "target_id", "recipe_no"], ["sample_id", "recipe_no", "illumination"]],
        "production.csv": [["recipe_no"], ["target_id"], ["recipe_no", "target_id"]],
        "pantones.csv": [["shade_name"], ["batch_name"], ["standard_name"]],
    }
    rows: list[dict[str, Any]] = []
    for filename, keys_list in specs.items():
        df = tables.get(filename)
        if df is None:
            continue
        for keys in keys_list:
            available = [k for k in keys if k in df.columns]
            if len(available) != len(keys):
                rows.append({"file": filename, "key_columns": "+".join(keys), "available": False, "duplicate_rows": None, "duplicate_groups": None})
                continue
            dup_mask = df.duplicated(subset=keys, keep=False)
            duplicate_groups = int(df.loc[dup_mask, keys].drop_duplicates().shape[0]) if dup_mask.any() else 0
            rows.append(
                {
                    "file": filename,
                    "key_columns": "+".join(keys),
                    "available": True,
                    "duplicate_rows": int(dup_mask.sum()),
                    "duplicate_groups": duplicate_groups,
                }
            )
    return pd.DataFrame(rows)


def audit_recipe_dyes(recipe: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i in (1, 2, 3):
        dye_col = f"dye{i}"
        pct_col = f"dye{i}(%)"
        if dye_col not in recipe.columns:
            continue
        pct = to_numeric_series(recipe[pct_col]) if pct_col in recipe.columns else pd.Series(np.nan, index=recipe.index)
        tmp = pd.DataFrame({"dye_position": i, "dye_name": recipe[dye_col].astype(str).str.strip(), "dye_pct": pct})
        tmp = tmp[tmp["dye_name"].notna() & (tmp["dye_name"].astype(str).str.lower() != "nan") & (tmp["dye_name"].astype(str).str.len() > 0)]
        for dye, g in tmp.groupby("dye_name", dropna=False):
            rows.append(
                {
                    "dye_name": dye,
                    "count": int(len(g)),
                    "mean_pct": round(float(g["dye_pct"].mean()), 6) if g["dye_pct"].notna().any() else None,
                    "min_pct": round(float(g["dye_pct"].min()), 6) if g["dye_pct"].notna().any() else None,
                    "max_pct": round(float(g["dye_pct"].max()), 6) if g["dye_pct"].notna().any() else None,
                    "positions_seen": ",".join(map(str, sorted(g["dye_position"].unique()))),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.groupby("dye_name", as_index=False).agg(
        count=("count", "sum"),
        mean_pct=("mean_pct", "mean"),
        min_pct=("min_pct", "min"),
        max_pct=("max_pct", "max"),
        positions_seen=("positions_seen", lambda x: ",".join(sorted(set(",".join(map(str, x)).split(","))))),
    )
    return out.sort_values(["count", "dye_name"], ascending=[False, True])


def audit_cmc_schema_candidates(cmc: pd.DataFrame) -> pd.DataFrame:
    if not {"sample_id", "target_id", "recipe_no"}.issubset(cmc.columns):
        return pd.DataFrame()
    df = cmc.copy()
    df["sample_as_recipe"] = df["sample_id"].map(looks_like_recipe)
    df["recipe_as_sample"] = df["recipe_no"].map(looks_like_sample)
    df["target_as_pantone"] = df["target_id"].map(looks_like_pantone)
    df["candidate_shift_sample_recipe_swap"] = df["sample_as_recipe"] & df["recipe_as_sample"] & df["target_as_pantone"]
    cols = ["sample_id", "target_id", "recipe_no", "sample_as_recipe", "recipe_as_sample", "target_as_pantone", "candidate_shift_sample_recipe_swap"]
    if "illumination" in df.columns:
        cols.append("illumination")
    if "dE" in df.columns:
        cols.append("dE")
    return df.loc[df["candidate_shift_sample_recipe_swap"], cols].copy()


def audit_cmc_quality(cmc: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    numeric_candidates = ["dE", "dL", "da", "db", "dc", "dh", "csc", "sh"]
    group_cols = ["illumination"] if "illumination" in cmc.columns else []
    for col in numeric_candidates:
        if col not in cmc.columns:
            continue
        s = to_numeric_series(cmc[col])
        if group_cols:
            for illum, idx in cmc.groupby("illumination", dropna=False).groups.items():
                vals = s.loc[idx].dropna()
                rows.append(_summary_row(col, vals, illumination=illum))
        vals = s.dropna()
        rows.append(_summary_row(col, vals, illumination="ALL"))
    return pd.DataFrame(rows)


def _summary_row(metric: str, vals: pd.Series, illumination: Any) -> dict[str, Any]:
    return {
        "metric": metric,
        "illumination": illumination,
        "count": int(vals.shape[0]),
        "mean": round(float(vals.mean()), 6) if len(vals) else None,
        "std": round(float(vals.std()), 6) if len(vals) > 1 else None,
        "min": round(float(vals.min()), 6) if len(vals) else None,
        "p25": round(float(vals.quantile(0.25)), 6) if len(vals) else None,
        "median": round(float(vals.median()), 6) if len(vals) else None,
        "p75": round(float(vals.quantile(0.75)), 6) if len(vals) else None,
        "max": round(float(vals.max()), 6) if len(vals) else None,
    }


def audit_pantone_reflectance(pantones: pd.DataFrame) -> pd.DataFrame:
    refl_cols = detect_reflectance_columns(pantones)
    rows: list[dict[str, Any]] = []
    for col in refl_cols:
        nm = int(re.search(r"(\d+)nm", col).group(1))
        vals = to_numeric_series(pantones[col])
        rows.append(
            {
                "wavelength_nm": nm,
                "column": col,
                "rows": len(pantones),
                "non_null": int(vals.notna().sum()),
                "missing": int(vals.isna().sum()),
                "missing_pct": round(float(vals.isna().mean() * 100), 4),
                "min": round(float(vals.min()), 6) if vals.notna().any() else None,
                "max": round(float(vals.max()), 6) if vals.notna().any() else None,
            }
        )
    return pd.DataFrame(rows)


def audit_target_matching(recipe: pd.DataFrame, production: pd.DataFrame, pantones: pd.DataFrame) -> pd.DataFrame:
    target_values: set[str] = set()
    for df, col in [(recipe, "target_id"), (production, "target_id")]:
        if col in df.columns:
            target_values.update(normalize_target_id(x) for x in df[col].dropna().unique())
    pantone_values: set[str] = set()
    for col in ["shade_name", "batch_name", "standard_name", "target_id"]:
        if col in pantones.columns:
            pantone_values.update(normalize_target_id(x) for x in pantones[col].dropna().unique())
    rows = []
    for target in sorted(v for v in target_values if v):
        rows.append({"target_id_norm": target, "matched_in_pantones_preclean": target in pantone_values})
    return pd.DataFrame(rows)


def audit_production_fibers(production: pd.DataFrame) -> pd.DataFrame:
    fiber_cols = [c for c in production.columns if str(c).lower().strip() in {"cotton", "polyester", "viscose", "modal", "lycra"}]
    rows: list[dict[str, Any]] = []
    for col in fiber_cols:
        vals = to_numeric_series(production[col])
        rows.append(
            {
                "fiber_column": col,
                "non_null": int(vals.notna().sum()),
                "mean_pct": round(float(vals.mean()), 6) if vals.notna().any() else None,
                "min_pct": round(float(vals.min()), 6) if vals.notna().any() else None,
                "max_pct": round(float(vals.max()), 6) if vals.notna().any() else None,
            }
        )
    return pd.DataFrame(rows)


def run_audit(project_root: str | Path | None = None) -> dict[str, Any]:
    ctx = PipelineRunContext(
        script_name="00_audit_raw_data.py",
        stage="audit",
        project_root=project_root,
        script_version=SCRIPT_VERSION,
    )
    try:
        raw_paths = [ctx.paths.raw / f for f in REQUIRED_RAW_FILES]
        manifest = make_file_manifest(raw_paths, root=ctx.paths.root)
        ctx.write_df(manifest, ctx.table_path("raw_file_manifest.csv"))
        ctx.write_json(manifest.to_dict(orient="records"), ctx.table_path("raw_file_manifest.json"))

        missing_files = manifest.loc[~manifest["exists"], "path"].tolist()
        if missing_files:
            raise FileNotFoundError(f"Missing required raw files: {missing_files}")

        tables = read_raw_tables(ctx)

        overview = pd.DataFrame(
            [
                {"file": filename, "rows": len(df), "columns": len(df.columns), "duplicate_full_rows": int(df.duplicated().sum())}
                for filename, df in tables.items()
            ]
        )
        columns_summary = build_columns_summary(tables)
        duplicate_summary = build_duplicate_key_summary(tables)
        dye_frequency = audit_recipe_dyes(tables["recipe.csv"])
        cmc_shift = audit_cmc_schema_candidates(tables["cmc.csv"])
        cmc_quality = audit_cmc_quality(tables["cmc.csv"])
        pantone_refl = audit_pantone_reflectance(tables["pantones.csv"])
        target_match = audit_target_matching(tables["recipe.csv"], tables["production.csv"], tables["pantones.csv"])
        fiber_summary = audit_production_fibers(tables["production.csv"])

        ctx.write_df(overview, ctx.table_path("audit_overview.csv"))
        ctx.write_df(columns_summary, ctx.table_path("columns_summary.csv"))
        ctx.write_df(duplicate_summary, ctx.table_path("duplicate_key_summary.csv"))
        ctx.write_df(dye_frequency, ctx.table_path("recipe_dye_frequency.csv"))
        ctx.write_df(cmc_shift, ctx.table_path("cmc_schema_shift_candidates.csv"))
        ctx.write_df(cmc_quality, ctx.table_path("cmc_quality_metric_summary.csv"))
        ctx.write_df(pantone_refl, ctx.table_path("pantone_reflectance_completeness.csv"))
        ctx.write_df(target_match, ctx.table_path("target_matching_preclean_audit.csv"))
        ctx.write_df(fiber_summary, ctx.table_path("production_fiber_summary.csv"))

        # Figures with safe defaults.
        save_bar(overview.set_index("file")["rows"], ctx.figure_path("01_rows_by_file.png"), "Rows by raw file")
        missing_top = columns_summary.sort_values("missing_pct", ascending=False).head(25).set_index("file" + ":" + columns_summary["column"].astype(str) if False else "column")
        # Above cannot combine indexes cleanly for all pandas versions; construct explicit label.
        missing_plot = columns_summary.copy()
        missing_plot["label"] = missing_plot["file"].str.replace(".csv", "", regex=False) + ":" + missing_plot["column"].astype(str)
        save_bar(missing_plot.sort_values("missing_pct", ascending=False).head(25).set_index("label")["missing_pct"], ctx.figure_path("02_missingness_top_columns.png"), "Top raw-column missingness", ylabel="Missing %")
        if not dye_frequency.empty:
            save_bar(dye_frequency.set_index("dye_name")["count"], ctx.figure_path("03_recipe_dye_frequency_top20.png"), "Top recipe dyes", top_n=20)
        if "illumination" in tables["cmc.csv"].columns:
            save_bar(tables["cmc.csv"]["illumination"].value_counts(), ctx.figure_path("04_cmc_illumination_counts.png"), "CMC illumination counts")
        if "pass_fail" in tables["cmc.csv"].columns:
            save_bar(tables["cmc.csv"]["pass_fail"].astype(str).str.lower().value_counts(), ctx.figure_path("05_cmc_pass_fail_counts.png"), "CMC pass/fail counts")
        if not pantone_refl.empty:
            save_bar(pantone_refl.set_index("wavelength_nm")["missing_pct"], ctx.figure_path("06_pantone_reflectance_missingness.png"), "Pantone reflectance missingness", xlabel="Wavelength nm", ylabel="Missing %")
        if not target_match.empty:
            save_bar(target_match["matched_in_pantones_preclean"].value_counts().rename(index={True: "matched", False: "unmatched"}), ctx.figure_path("07_target_match_rate.png"), "Preclean target match status")
        if "dE" in tables["cmc.csv"].columns:
            dE = to_numeric_series(tables["cmc.csv"]["dE"]).dropna()
            if len(dE):
                save_hist(dE, ctx.figure_path("08_cmc_de_distribution.png"), "CMC dE distribution", "dE")

        check = SelfCheckResult(stage="audit")
        check.add_count("n_raw_files", len(tables))
        check.add_count("missing_required_files", len(missing_files))
        check.add_count("empty_raw_tables", int(sum(df.empty for df in tables.values())))
        check.add_count("cmc_schema_shift_candidates", int(len(cmc_shift)))
        check.add_count("unique_recipe_targets", int(tables["recipe.csv"]["target_id"].nunique(dropna=True)) if "target_id" in tables["recipe.csv"].columns else -1)
        check.add_count("preclean_target_match_count", int(target_match["matched_in_pantones_preclean"].sum()) if not target_match.empty else 0)
        check.fail_if(len(missing_files) > 0, f"Missing raw files: {missing_files}")
        check.fail_if(any(df.empty for df in tables.values()), "At least one required raw table is empty")
        ctx.log_self_check(check)
        assert_no_failures(check)

        summary = {
            "raw_files": overview.to_dict(orient="records"),
            "cmc_schema_shift_candidates": int(len(cmc_shift)),
            "target_match_preclean": {
                "unique_targets": int(len(target_match)),
                "matched": int(target_match["matched_in_pantones_preclean"].sum()) if not target_match.empty else 0,
                "unmatched": int((~target_match["matched_in_pantones_preclean"]).sum()) if not target_match.empty else 0,
            },
            "reflectance_columns": len(detect_reflectance_columns(tables["pantones.csv"])),
        }
        ctx.finish(status="OK", summary=summary)
        return summary
    except Exception as exc:
        ctx.log_error(str(exc))
        ctx.finish(status="FAILED", summary={"error": str(exc)})
        raise
