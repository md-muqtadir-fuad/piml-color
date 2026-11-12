from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from piml_color_recipe.utils.hashing import make_file_manifest
from piml_color_recipe.utils.io import read_csv_flexible
from piml_color_recipe.utils.run_context import PipelineRunContext
from piml_color_recipe.utils.validation import SelfCheckResult, assert_no_failures, require_columns

SCRIPT_VERSION = "2026-06-22-v1-src-first-validation-splits-final"
RANDOM_SEED = 42
DEFAULT_N_SPLITS = 5


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if not np.isfinite(obj):
            return None
        return float(obj)
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if pd.isna(obj) if not isinstance(obj, (list, tuple, dict, np.ndarray)) else False:
        return None
    return obj


def _safe_series_stats(series: pd.Series) -> dict[str, Any]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"n": 0, "mean": None, "std": None, "min": None, "median": None, "max": None}
    return {
        "n": int(s.size),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if s.size > 1 else 0.0,
        "min": float(s.min()),
        "median": float(s.median()),
        "max": float(s.max()),
    }


def _target_summary(df: pd.DataFrame) -> dict[str, Any]:
    targets = [c for c in ["dE_d65", "dE_f11", "dE_max", "metamerism_gap_abs"] if c in df.columns]
    return {c: _safe_series_stats(df[c]) for c in targets}


def _make_de_bins(y: pd.Series, n_bins: int = 4) -> pd.Series:
    y_num = pd.to_numeric(y, errors="coerce")
    try:
        out = pd.qcut(y_num, q=n_bins, labels=False, duplicates="drop")
        return out.astype("Int64")
    except Exception:
        return pd.Series(np.zeros(len(y_num), dtype=int), index=y.index, dtype="Int64")


def _load_features_supervised(ctx: PipelineRunContext) -> pd.DataFrame:
    path = ctx.paths.processed / "features_supervised.csv"
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}. Run 03_build_features.py first.")
    df = read_csv_flexible(path)
    ctx.logger.info("Loaded %s: %s rows x %s cols", path, len(df), len(df.columns))
    return df.copy()


def _validate_features_supervised(df: pd.DataFrame) -> None:
    required = ["recipe_row_uid", "recipe_no", "target_id", "dE_max", "dE_d65", "dE_f11"]
    require_columns(df, required, "features_supervised.csv")
    if df.empty:
        raise ValueError("features_supervised.csv is empty.")
    if df["recipe_row_uid"].isna().any():
        raise ValueError("recipe_row_uid has missing values; split assignment would be unsafe.")
    if df["recipe_row_uid"].astype(str).duplicated().any():
        examples = df.loc[df["recipe_row_uid"].astype(str).duplicated(), "recipe_row_uid"].astype(str).head(10).tolist()
        raise ValueError(f"Duplicate recipe_row_uid values found: {examples}")
    for c in ["recipe_no", "target_id"]:
        if df[c].isna().any():
            raise ValueError(f"{c} has missing values; group split would be unsafe.")
    for c in ["dE_max", "dE_d65", "dE_f11"]:
        v = pd.to_numeric(df[c], errors="coerce")
        if v.isna().any() or not np.isfinite(v.to_numpy()).all():
            raise ValueError(f"{c} contains missing or non-finite values in supervised rows.")


def _group_kfold_indices(df: pd.DataFrame, group_col: str, n_splits: int = DEFAULT_N_SPLITS) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = df[group_col].astype(str).to_numpy()
    n_groups = int(pd.Series(groups).nunique())
    n_splits = max(2, min(int(n_splits), n_groups))
    try:
        from sklearn.model_selection import GroupKFold
        gkf = GroupKFold(n_splits=n_splits)
        return [(tr, va) for tr, va in gkf.split(df, pd.to_numeric(df["dE_max"], errors="coerce"), groups=groups)]
    except Exception:
        # Greedy deterministic fallback. Largest groups are assigned to the smallest validation fold.
        group_sizes = df.groupby(group_col, dropna=False).size().sort_values(ascending=False)
        fold_groups: list[list[str]] = [[] for _ in range(n_splits)]
        fold_counts = [0 for _ in range(n_splits)]
        for group, size in group_sizes.items():
            fold_id = int(np.argmin(fold_counts))
            fold_groups[fold_id].append(str(group))
            fold_counts[fold_id] += int(size)
        splits = []
        group_series = df[group_col].astype(str)
        for fold_id in range(n_splits):
            val_mask = group_series.isin(fold_groups[fold_id]).to_numpy()
            val_idx = np.flatnonzero(val_mask)
            train_idx = np.flatnonzero(~val_mask)
            splits.append((train_idx, val_idx))
        return splits


def _make_split_payload(
    df: pd.DataFrame,
    split_name: str,
    group_col: str,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    fold_payloads: list[dict[str, Any]] = []
    for fold_id, (train_idx, val_idx) in enumerate(splits, start=1):
        train = df.iloc[train_idx].copy()
        val = df.iloc[val_idx].copy()
        train_groups = sorted(train[group_col].astype(str).unique().tolist())
        val_groups = sorted(val[group_col].astype(str).unique().tolist())
        fold_payloads.append(
            {
                "split_name": split_name,
                "fold_id": int(fold_id),
                "group_col": group_col,
                "n_train": int(len(train)),
                "n_val": int(len(val)),
                "n_train_groups": int(len(train_groups)),
                "n_val_groups": int(len(val_groups)),
                "train_row_uids": train["recipe_row_uid"].astype(str).tolist(),
                "val_row_uids": val["recipe_row_uid"].astype(str).tolist(),
                "train_groups": train_groups,
                "val_groups": val_groups,
                "target_summary_train": _target_summary(train),
                "target_summary_val": _target_summary(val),
            }
        )
    return {
        "split_name": split_name,
        "group_col": group_col,
        "n_rows": int(len(df)),
        "n_folds": int(len(fold_payloads)),
        "random_seed": RANDOM_SEED,
        "folds": fold_payloads,
    }


def _make_leave_one_target_payload(df: pd.DataFrame) -> dict[str, Any]:
    groups = sorted(df["target_id"].astype(str).unique().tolist())
    folds: list[dict[str, Any]] = []
    target_series = df["target_id"].astype(str)
    for fold_id, target in enumerate(groups, start=1):
        val_mask = (target_series == target).to_numpy()
        val = df.loc[val_mask].copy()
        train = df.loc[~val_mask].copy()
        folds.append(
            {
                "split_name": "leave_one_target_out",
                "fold_id": int(fold_id),
                "group_col": "target_id",
                "left_out_target_id": target,
                "n_train": int(len(train)),
                "n_val": int(len(val)),
                "n_train_groups": int(train["target_id"].astype(str).nunique()),
                "n_val_groups": int(val["target_id"].astype(str).nunique()),
                "train_row_uids": train["recipe_row_uid"].astype(str).tolist(),
                "val_row_uids": val["recipe_row_uid"].astype(str).tolist(),
                "train_groups": sorted(train["target_id"].astype(str).unique().tolist()),
                "val_groups": [target],
                "target_summary_train": _target_summary(train),
                "target_summary_val": _target_summary(val),
            }
        )
    return {
        "split_name": "leave_one_target_out",
        "group_col": "target_id",
        "n_rows": int(len(df)),
        "n_folds": int(len(folds)),
        "random_seed": RANDOM_SEED,
        "folds": folds,
    }


def _check_split_payload(df: pd.DataFrame, payload: dict[str, Any]) -> dict[str, Any]:
    all_uids = set(df["recipe_row_uid"].astype(str).tolist())
    validation_uids: list[str] = []
    issue_rows: list[dict[str, Any]] = []
    checks = {
        "folds_with_train_val_uid_overlap": 0,
        "folds_with_group_leakage": 0,
        "folds_with_empty_train": 0,
        "folds_with_empty_val": 0,
        "rows_missing_from_validation": 0,
        "rows_validated_more_than_once": 0,
        "unknown_uids_in_split": 0,
    }
    for fold in payload["folds"]:
        train_uids = set(map(str, fold.get("train_row_uids", [])))
        val_uids = set(map(str, fold.get("val_row_uids", [])))
        train_groups = set(map(str, fold.get("train_groups", [])))
        val_groups = set(map(str, fold.get("val_groups", [])))
        uid_overlap = train_uids & val_uids
        group_overlap = train_groups & val_groups
        unknown = (train_uids | val_uids) - all_uids
        issue = {
            "split_name": payload["split_name"],
            "fold_id": fold["fold_id"],
            "uid_overlap_count": int(len(uid_overlap)),
            "group_overlap_count": int(len(group_overlap)),
            "unknown_uid_count": int(len(unknown)),
            "empty_train": int(len(train_uids) == 0),
            "empty_val": int(len(val_uids) == 0),
        }
        issue_rows.append(issue)
        checks["folds_with_train_val_uid_overlap"] += int(bool(uid_overlap))
        checks["folds_with_group_leakage"] += int(bool(group_overlap))
        checks["unknown_uids_in_split"] += int(len(unknown))
        checks["folds_with_empty_train"] += int(len(train_uids) == 0)
        checks["folds_with_empty_val"] += int(len(val_uids) == 0)
        validation_uids.extend(list(val_uids))
    val_counts = pd.Series(validation_uids).value_counts() if validation_uids else pd.Series(dtype=int)
    checks["rows_missing_from_validation"] = int(len(all_uids - set(validation_uids)))
    checks["rows_validated_more_than_once"] = int((val_counts > 1).sum())
    return {"checks": checks, "issue_rows": issue_rows}


def _fold_summary_rows(df: pd.DataFrame, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for fold in payload["folds"]:
        val_uids = set(map(str, fold["val_row_uids"]))
        val = df[df["recipe_row_uid"].astype(str).isin(val_uids)]
        row = {
            "split_name": payload["split_name"],
            "fold_id": int(fold["fold_id"]),
            "group_col": payload["group_col"],
            "n_train": int(fold["n_train"]),
            "n_val": int(fold["n_val"]),
            "n_train_groups": int(fold["n_train_groups"]),
            "n_val_groups": int(fold["n_val_groups"]),
            "val_dE_max_mean": float(pd.to_numeric(val["dE_max"], errors="coerce").mean()) if len(val) else None,
            "val_dE_max_std": float(pd.to_numeric(val["dE_max"], errors="coerce").std(ddof=1)) if len(val) > 1 else 0.0,
            "val_dE_max_min": float(pd.to_numeric(val["dE_max"], errors="coerce").min()) if len(val) else None,
            "val_dE_max_max": float(pd.to_numeric(val["dE_max"], errors="coerce").max()) if len(val) else None,
            "val_unique_targets": int(val["target_id"].astype(str).nunique()) if "target_id" in val else None,
            "val_unique_recipes": int(val["recipe_no"].astype(str).nunique()) if "recipe_no" in val else None,
        }
        if "dE_max_bin_diagnostic" in val.columns:
            for b, count in val["dE_max_bin_diagnostic"].value_counts(dropna=False).sort_index().items():
                row[f"val_dE_bin_{b}_count"] = int(count)
        rows.append(row)
    return rows


def _group_size_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["recipe_no", "target_id"]:
        sizes = df.groupby(col, dropna=False).size()
        rows.append(
            {
                "group_col": col,
                "n_groups": int(sizes.size),
                "min_group_size": int(sizes.min()),
                "median_group_size": float(sizes.median()),
                "mean_group_size": float(sizes.mean()),
                "max_group_size": int(sizes.max()),
                "singleton_groups": int((sizes == 1).sum()),
                "groups_with_at_least_3_rows": int((sizes >= 3).sum()),
            }
        )
    return pd.DataFrame(rows)


def _leave_one_target_summary(df: pd.DataFrame, payload: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for fold in payload["folds"]:
        val_uids = set(map(str, fold["val_row_uids"]))
        val = df[df["recipe_row_uid"].astype(str).isin(val_uids)]
        rows.append(
            {
                "fold_id": int(fold["fold_id"]),
                "target_id": str(fold.get("left_out_target_id", "")),
                "n_val": int(fold["n_val"]),
                "n_train": int(fold["n_train"]),
                "val_unique_recipes": int(val["recipe_no"].astype(str).nunique()),
                "val_dE_max_mean": float(pd.to_numeric(val["dE_max"], errors="coerce").mean()) if len(val) else None,
                "val_dE_max_min": float(pd.to_numeric(val["dE_max"], errors="coerce").min()) if len(val) else None,
                "val_dE_max_max": float(pd.to_numeric(val["dE_max"], errors="coerce").max()) if len(val) else None,
            }
        )
    return pd.DataFrame(rows)


def _split_row_assignments(df: pd.DataFrame, payloads: list[dict[str, Any]]) -> pd.DataFrame:
    base_cols = [c for c in ["recipe_row_uid", "sample_id", "sample_id_clean", "recipe_no", "target_id", "dE_max", "dE_d65", "dE_f11", "metamerism_gap_abs"] if c in df.columns]
    assignments = df[base_cols].copy()
    uid_to_index = {uid: i for i, uid in enumerate(assignments["recipe_row_uid"].astype(str).tolist())}
    for payload in payloads:
        col = f"{payload['split_name']}_fold"
        assignments[col] = pd.NA
        for fold in payload["folds"]:
            for uid in fold["val_row_uids"]:
                idx = uid_to_index.get(str(uid))
                if idx is not None:
                    assignments.at[idx, col] = int(fold["fold_id"])
    return assignments


def _plot_fold_row_counts(fold_summary: pd.DataFrame, split_name: str, out_path: Path) -> None:
    data = fold_summary[fold_summary["split_name"] == split_name].sort_values("fold_id")
    if data.empty:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.bar(data["fold_id"].astype(str), data["n_val"])
    plt.title(f"Validation rows by fold: {split_name}")
    plt.xlabel("Fold")
    plt.ylabel("Validation rows")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_fold_de_distribution(df: pd.DataFrame, assignments: pd.DataFrame, split_name: str, out_path: Path) -> None:
    fold_col = f"{split_name}_fold"
    if fold_col not in assignments.columns:
        return
    tmp = assignments[["recipe_row_uid", fold_col]].merge(df[["recipe_row_uid", "dE_max"]], on="recipe_row_uid", how="left")
    tmp = tmp.dropna(subset=[fold_col, "dE_max"])
    if tmp.empty:
        return
    folds = sorted(tmp[fold_col].astype(int).unique().tolist())
    data = [pd.to_numeric(tmp.loc[tmp[fold_col].astype(int) == f, "dE_max"], errors="coerce").dropna().tolist() for f in folds]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 4.8))
    plt.boxplot(data, showmeans=True)
    plt.xticks(range(1, len(folds) + 1), [str(f) for f in folds])
    plt.title(f"dE_max distribution by validation fold: {split_name}")
    plt.xlabel("Fold")
    plt.ylabel("dE_max")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_group_sizes(df: pd.DataFrame, group_col: str, out_path: Path) -> None:
    sizes = df.groupby(group_col, dropna=False).size().sort_values(ascending=False)
    if sizes.empty:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 4.8))
    plt.hist(sizes.to_numpy(), bins=min(20, max(5, int(sizes.max()))))
    plt.title(f"Group size distribution: {group_col}")
    plt.xlabel("Rows per group")
    plt.ylabel("Number of groups")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def build_validation_splits(project_root: str | Path | None = None, n_splits: int = DEFAULT_N_SPLITS) -> dict[str, Any]:
    ctx = PipelineRunContext(
        script_name="04_make_validation_splits.py",
        stage="splits",
        project_root=project_root,
        script_version=SCRIPT_VERSION,
    )
    try:
        input_manifest = make_file_manifest([ctx.paths.processed / "features_supervised.csv"], root=ctx.paths.root)
        ctx.write_df(input_manifest, ctx.paths.manifests / "splits_input_file_hashes.csv")
        ctx.write_json(input_manifest.to_dict(orient="records"), ctx.paths.manifests / "splits_input_file_hashes.json")

        df = _load_features_supervised(ctx)
        _validate_features_supervised(df)
        df = df.copy()
        df["recipe_row_uid"] = df["recipe_row_uid"].astype(str)
        df["recipe_no"] = df["recipe_no"].astype(str)
        df["target_id"] = df["target_id"].astype(str)
        df["dE_max_bin_diagnostic"] = _make_de_bins(df["dE_max"], n_bins=4)

        ctx.logger.info("Supervised rows: %s", len(df))
        ctx.logger.info("Unique recipe_no groups: %s", df["recipe_no"].nunique())
        ctx.logger.info("Unique target_id groups: %s", df["target_id"].nunique())

        recipe_payload = _make_split_payload(
            df,
            split_name="groupkfold_recipe_no",
            group_col="recipe_no",
            splits=_group_kfold_indices(df, "recipe_no", n_splits=n_splits),
        )
        target_payload = _make_split_payload(
            df,
            split_name="groupkfold_target_id",
            group_col="target_id",
            splits=_group_kfold_indices(df, "target_id", n_splits=n_splits),
        )
        loto_payload = _make_leave_one_target_payload(df)
        payloads = [recipe_payload, target_payload, loto_payload]

        ctx.write_json(_jsonable(recipe_payload), ctx.paths.splits / "groupkfold_recipe_no.json")
        ctx.write_json(_jsonable(target_payload), ctx.paths.splits / "groupkfold_target_id.json")
        ctx.write_json(_jsonable(loto_payload), ctx.paths.splits / "leave_one_target_out.json")

        leakage_rows = []
        self_check_rows = []
        result = SelfCheckResult(stage="splits")
        for payload in payloads:
            check = _check_split_payload(df, payload)
            row = {"split_name": payload["split_name"]} | check["checks"]
            self_check_rows.append(row)
            leakage_rows.extend(check["issue_rows"])
            ctx.logger.info("Self-check issue counts for %s: %s", payload["split_name"], check["checks"])
            for key, value in check["checks"].items():
                result.add_count(f"{payload['split_name']}__{key}", int(value))
            result.fail_if(check["checks"]["folds_with_train_val_uid_overlap"] > 0, f"UID leakage in {payload['split_name']}")
            result.fail_if(check["checks"]["folds_with_group_leakage"] > 0, f"Group leakage in {payload['split_name']}")
            result.fail_if(check["checks"]["folds_with_empty_train"] > 0, f"Empty train fold in {payload['split_name']}")
            result.fail_if(check["checks"]["folds_with_empty_val"] > 0, f"Empty validation fold in {payload['split_name']}")
            result.fail_if(check["checks"]["rows_missing_from_validation"] > 0, f"Rows missing from validation in {payload['split_name']}")
            result.fail_if(check["checks"]["rows_validated_more_than_once"] > 0, f"Rows validated more than once in {payload['split_name']}")
            result.fail_if(check["checks"]["unknown_uids_in_split"] > 0, f"Unknown UIDs in {payload['split_name']}")

        fold_summary = pd.DataFrame([r for payload in payloads for r in _fold_summary_rows(df, payload)])
        leakage_summary = pd.DataFrame(leakage_rows)
        split_self_check_summary = pd.DataFrame(self_check_rows)
        group_summary = _group_size_summary(df)
        loto_summary = _leave_one_target_summary(df, loto_payload)
        assignments = _split_row_assignments(df, payloads)

        build_summary = pd.DataFrame(
            [
                {"metric": "supervised_rows", "value": int(len(df))},
                {"metric": "unique_recipe_no_groups", "value": int(df["recipe_no"].nunique())},
                {"metric": "unique_target_id_groups", "value": int(df["target_id"].nunique())},
                {"metric": "groupkfold_recipe_no_folds", "value": int(recipe_payload["n_folds"])},
                {"metric": "groupkfold_target_id_folds", "value": int(target_payload["n_folds"])},
                {"metric": "leave_one_target_out_folds", "value": int(loto_payload["n_folds"])},
                {"metric": "main_split", "value": "groupkfold_recipe_no"},
                {"metric": "strict_split", "value": "groupkfold_target_id"},
                {"metric": "stress_test_split", "value": "leave_one_target_out"},
            ]
        )

        ctx.write_df(build_summary, ctx.table_path("split_build_summary.csv"))
        ctx.write_df(fold_summary, ctx.table_path("split_fold_summary.csv"))
        ctx.write_df(leakage_summary, ctx.table_path("leakage_check_summary.csv"))
        ctx.write_df(split_self_check_summary, ctx.table_path("split_self_check_summary.csv"))
        ctx.write_df(group_summary, ctx.table_path("group_size_summary.csv"))
        ctx.write_df(loto_summary, ctx.table_path("leave_one_target_out_summary.csv"))
        ctx.write_df(assignments, ctx.paths.splits / "split_row_assignments.csv")
        ctx.write_df(assignments, ctx.table_path("split_row_assignments.csv"))

        _plot_fold_row_counts(fold_summary, "groupkfold_recipe_no", ctx.figure_path("fold_row_counts_recipe_no.png"))
        _plot_fold_row_counts(fold_summary, "groupkfold_target_id", ctx.figure_path("fold_row_counts_target_id.png"))
        _plot_fold_de_distribution(df, assignments, "groupkfold_recipe_no", ctx.figure_path("fold_dEmax_distribution_recipe_no.png"))
        _plot_fold_de_distribution(df, assignments, "groupkfold_target_id", ctx.figure_path("fold_dEmax_distribution_target_id.png"))
        _plot_group_sizes(df, "recipe_no", ctx.figure_path("group_size_distribution_recipe_no.png"))
        _plot_group_sizes(df, "target_id", ctx.figure_path("group_size_distribution_target_id.png"))

        ctx.log_self_check(result)
        assert_no_failures(result)

        output_manifest = make_file_manifest(
            [
                ctx.paths.splits / "groupkfold_recipe_no.json",
                ctx.paths.splits / "groupkfold_target_id.json",
                ctx.paths.splits / "leave_one_target_out.json",
                ctx.paths.splits / "split_row_assignments.csv",
                ctx.table_path("split_build_summary.csv"),
                ctx.table_path("split_fold_summary.csv"),
                ctx.table_path("leakage_check_summary.csv"),
                ctx.table_path("split_self_check_summary.csv"),
                ctx.table_path("group_size_summary.csv"),
                ctx.table_path("leave_one_target_out_summary.csv"),
                ctx.figure_path("fold_row_counts_recipe_no.png"),
                ctx.figure_path("fold_row_counts_target_id.png"),
                ctx.figure_path("fold_dEmax_distribution_recipe_no.png"),
                ctx.figure_path("fold_dEmax_distribution_target_id.png"),
                ctx.figure_path("group_size_distribution_recipe_no.png"),
                ctx.figure_path("group_size_distribution_target_id.png"),
            ],
            root=ctx.paths.root,
        )
        ctx.write_df(output_manifest, ctx.paths.manifests / "splits_output_file_hashes.csv")
        ctx.write_json(output_manifest.to_dict(orient="records"), ctx.paths.manifests / "splits_output_file_hashes.json")

        summary = {
            "supervised_rows": int(len(df)),
            "unique_recipe_no_groups": int(df["recipe_no"].nunique()),
            "unique_target_id_groups": int(df["target_id"].nunique()),
            "groupkfold_recipe_no_folds": int(recipe_payload["n_folds"]),
            "groupkfold_target_id_folds": int(target_payload["n_folds"]),
            "leave_one_target_out_folds": int(loto_payload["n_folds"]),
            "self_check_ok": bool(result.ok),
        }
        ctx.finish(status="OK", summary=summary)
        return summary
    except Exception as exc:
        ctx.log_error(str(exc), error_type=type(exc).__name__)
        ctx.finish(status="FAILED", summary={"error": str(exc), "error_type": type(exc).__name__})
        raise
