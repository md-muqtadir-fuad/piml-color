from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class MulticollinearityResult:
    selected_features: list[str]
    dropped_features: list[str]
    high_corr_pairs: pd.DataFrame
    drop_log: pd.DataFrame
    vif_report: pd.DataFrame
    condition_report: pd.DataFrame
    summary: pd.DataFrame
    selected_feature_sets: dict[str, list[str]]
    diagnostics: dict[str, Any]


def _to_numeric_frame(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    cols = [c for c in feature_cols if c in df.columns]
    x = df[cols].apply(pd.to_numeric, errors="coerce")
    return x.replace([np.inf, -np.inf], np.nan)


def _feature_group(feature: str) -> str:
    low = feature.lower()
    if low.startswith("refl_") or "reflectance" in low or "feat_refl" in low:
        return "spectral_reflectance"
    if low.startswith("ks_") or "feat_ks" in low:
        return "spectral_ks"
    if low.startswith("dye_") or low.startswith("vec__") or "dye" in low:
        return "dye_recipe"
    if low.startswith("production__") or low.startswith("cat__") or any(k in low for k in ["gsm", "salt", "soda", "l2r", "fabric", "cotton", "polyester", "viscose", "modal", "lycra"]):
        return "process"
    if low.startswith("feat_"):
        return "engineered"
    return "other"


def feature_quality_table(x: pd.DataFrame) -> pd.DataFrame:
    n = max(len(x), 1)
    rows = []
    for col in x.columns:
        s = x[col]
        non_na = s.dropna()
        rows.append({
            "feature": col,
            "feature_group": _feature_group(col),
            "missing_count": int(s.isna().sum()),
            "missing_pct": round(float(s.isna().mean() * 100), 4),
            "n_unique_non_missing": int(non_na.nunique(dropna=True)),
            "std": float(non_na.std(ddof=0)) if len(non_na) else np.nan,
            "zero_variance_flag": int(non_na.nunique(dropna=True) <= 1),
            "row_count": int(n),
        })
    return pd.DataFrame(rows)


def high_correlation_pairs(x: pd.DataFrame, threshold: float = 0.95, method: str = "spearman") -> pd.DataFrame:
    if x.empty or x.shape[1] < 2:
        return pd.DataFrame(columns=["feature_1", "feature_2", "correlation", "abs_correlation", "method", "threshold"])
    # Avoid noisy RuntimeWarnings from constant columns during correlation.
    usable_cols = [c for c in x.columns if pd.to_numeric(x[c], errors="coerce").nunique(dropna=True) > 1]
    if len(usable_cols) < 2:
        return pd.DataFrame(columns=["feature_1", "feature_2", "correlation", "abs_correlation", "method", "threshold"])
    x = x[usable_cols]
    corr = x.corr(method=method, min_periods=max(10, int(0.2 * len(x))))
    rows = []
    cols = list(corr.columns)
    for i, c1 in enumerate(cols):
        vals = corr.iloc[i, i + 1 :]
        for c2, value in vals.items():
            if pd.notna(value) and abs(float(value)) >= threshold:
                rows.append({
                    "feature_1": c1,
                    "feature_2": c2,
                    "correlation": float(value),
                    "abs_correlation": abs(float(value)),
                    "method": method,
                    "threshold": threshold,
                    "feature_1_group": _feature_group(c1),
                    "feature_2_group": _feature_group(c2),
                })
    if not rows:
        return pd.DataFrame(columns=["feature_1", "feature_2", "correlation", "abs_correlation", "method", "threshold", "feature_1_group", "feature_2_group"])
    return pd.DataFrame(rows).sort_values(["abs_correlation", "feature_1", "feature_2"], ascending=[False, True, True])


def _priority_score(feature: str, quality: pd.DataFrame | None = None) -> tuple[float, float, float, str]:
    # Higher is better. Prefer compact engineered descriptors over dense raw spectral grids for linear-safe sets.
    group = _feature_group(feature)
    group_priority = {
        "engineered": 7.0,
        "dye_recipe": 6.0,
        "process": 5.0,
        "spectral_reflectance": 3.0,
        "spectral_ks": 3.0,
        "other": 1.0,
    }.get(group, 1.0)
    missing_penalty = 0.0
    unique_bonus = 0.0
    if quality is not None and not quality.empty and "feature" in quality.columns:
        row = quality.loc[quality["feature"].eq(feature)]
        if not row.empty:
            missing_penalty = float(row["missing_pct"].iloc[0])
            unique_bonus = float(row["n_unique_non_missing"].iloc[0])
    return (group_priority, -missing_penalty, unique_bonus, feature)


def greedy_correlation_prune(
    x: pd.DataFrame,
    threshold: float = 0.95,
    method: str = "spearman",
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    quality = feature_quality_table(x)
    zero_var = set(quality.loc[quality["zero_variance_flag"].eq(1), "feature"].astype(str).tolist())
    active = [c for c in x.columns if c not in zero_var]
    x_active = x[active]
    pairs = high_correlation_pairs(x_active, threshold=threshold, method=method)

    selected = set(active)
    rows = []
    if not zero_var:
        pass
    else:
        for f in sorted(zero_var):
            rows.append({
                "dropped_feature": f,
                "kept_feature": "",
                "reason": "zero_or_near_zero_variance",
                "correlation": np.nan,
                "abs_correlation": np.nan,
                "threshold": threshold,
            })

    for _, row in pairs.iterrows():
        f1 = str(row["feature_1"])
        f2 = str(row["feature_2"])
        if f1 not in selected or f2 not in selected:
            continue
        s1 = _priority_score(f1, quality)
        s2 = _priority_score(f2, quality)
        if s1 >= s2:
            keep, drop = f1, f2
        else:
            keep, drop = f2, f1
        selected.remove(drop)
        rows.append({
            "dropped_feature": drop,
            "kept_feature": keep,
            "reason": f"abs_{method}_corr_ge_{threshold}",
            "correlation": float(row["correlation"]),
            "abs_correlation": float(row["abs_correlation"]),
            "threshold": threshold,
        })

    selected_features = [c for c in x.columns if c in selected]
    drop_log = pd.DataFrame(rows)
    return selected_features, drop_log, pairs


def impute_and_standardize(x: pd.DataFrame) -> np.ndarray:
    arr = x.to_numpy(dtype=float)
    med = np.nanmedian(arr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    inds = np.where(~np.isfinite(arr))
    if inds[0].size:
        arr[inds] = np.take(med, inds[1])
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=0)
    std = np.where(std > 1e-12, std, 1.0)
    return (arr - mean) / std


def condition_index_report(x: pd.DataFrame) -> pd.DataFrame:
    if x.empty:
        return pd.DataFrame(columns=["n_features", "n_rows", "min_eigenvalue", "max_eigenvalue", "condition_index", "rank"])
    usable_cols = [c for c in x.columns if pd.to_numeric(x[c], errors="coerce").nunique(dropna=True) > 1]
    x = x[usable_cols]
    if x.empty:
        return pd.DataFrame(columns=["n_features", "n_rows", "min_eigenvalue", "max_eigenvalue", "condition_index", "rank"])
    z = impute_and_standardize(x)
    if z.shape[1] == 0:
        return pd.DataFrame(columns=["n_features", "n_rows", "min_eigenvalue", "max_eigenvalue", "condition_index", "rank"])
    corr = np.corrcoef(z, rowvar=False)
    corr = np.atleast_2d(np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0))
    # Numerical guard for diagonal after nan_to_num.
    np.fill_diagonal(corr, 1.0)
    eigvals = np.linalg.eigvalsh(corr)
    eigvals = np.clip(eigvals, 0.0, None)
    positive = eigvals[eigvals > 1e-10]
    min_eig = float(positive.min()) if positive.size else 0.0
    max_eig = float(eigvals.max()) if eigvals.size else 0.0
    cond = float(np.sqrt(max_eig / min_eig)) if min_eig > 0 else float("inf")
    rank = int(np.linalg.matrix_rank(z)) if z.size else 0
    return pd.DataFrame([{
        "n_features": int(x.shape[1]),
        "n_rows": int(x.shape[0]),
        "min_eigenvalue": min_eig,
        "max_eigenvalue": max_eig,
        "condition_index": cond,
        "rank": rank,
        "rank_deficient_flag": int(rank < x.shape[1]),
    }])


def vif_report(x: pd.DataFrame, max_features: int = 140) -> pd.DataFrame:
    """Fast VIF approximation from the pseudo-inverse of the feature correlation matrix.

    This is intentionally used instead of per-feature OLS because the project has
    p close to n and many raw spectral columns. It is stable, deterministic, and
    good enough for feature-set selection/audit. The exact model scripts still do
    fold-wise preprocessing to avoid leakage.
    """
    if x.empty:
        return pd.DataFrame(columns=["feature", "vif", "n_features_used", "status"])
    cols = list(x.columns)
    truncated = False
    if len(cols) > max_features:
        quality = feature_quality_table(x)
        cols = sorted(cols, key=lambda f: _priority_score(f, quality), reverse=True)[:max_features]
        truncated = True
    cols = [c for c in cols if pd.to_numeric(x[c], errors="coerce").nunique(dropna=True) > 1]
    if not cols:
        return pd.DataFrame(columns=["feature", "vif", "n_features_used", "status"])
    z = impute_and_standardize(x[cols])
    if z.shape[1] == 1:
        return pd.DataFrame([{"feature": cols[0], "vif": 1.0, "n_features_used": 1, "status": "single_feature"}])
    corr = np.corrcoef(z, rowvar=False)
    corr = np.atleast_2d(np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0))
    np.fill_diagonal(corr, 1.0)
    # tiny ridge improves stability for near-singular spectral matrices
    corr = corr + np.eye(corr.shape[0]) * 1e-8
    try:
        inv_corr = np.linalg.pinv(corr, rcond=1e-8)
        vif_vals = np.diag(inv_corr)
        status = "ok_truncated" if truncated else "ok"
    except Exception as exc:  # pragma: no cover - defensive
        vif_vals = np.full(len(cols), np.nan)
        status = f"failed:{type(exc).__name__}"
    rows = [{
        "feature": feature,
        "vif": float(vif) if np.isfinite(vif) else np.nan,
        "n_features_used": int(len(cols)),
        "status": status,
    } for feature, vif in zip(cols, vif_vals)]
    return pd.DataFrame(rows).sort_values(["vif", "feature"], ascending=[False, True])

def iterative_vif_prune(
    x: pd.DataFrame,
    start_features: list[str],
    vif_threshold: float = 10.0,
    max_iter: int = 60,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    """Fast one-pass VIF pruning for the final pipeline.

    Exact iterative VIF elimination is slow and unstable when p is close to n.
    We first correlation-prune, compute a stable pseudo-inverse VIF report, and
    remove features whose VIF remains above threshold. This is conservative and
    reproducible; downstream model scripts can use the resulting linear-safe set.
    """
    if not start_features:
        return [], pd.DataFrame(columns=["dropped_feature", "reason", "vif", "vif_threshold"]), pd.DataFrame()
    report = vif_report(x[start_features], max_features=max(140, len(start_features)))
    if report.empty or "vif" not in report.columns:
        return list(start_features), pd.DataFrame(columns=["dropped_feature", "reason", "vif", "vif_threshold"]), report
    rep_map = report.set_index("feature")["vif"].to_dict()
    selected = []
    rows = []
    for feature in start_features:
        vif = rep_map.get(feature, np.nan)
        if pd.notna(vif) and float(vif) > vif_threshold:
            rows.append({
                "dropped_feature": feature,
                "kept_feature": "",
                "reason": f"vif_gt_{vif_threshold}",
                "vif": float(vif),
                "vif_threshold": vif_threshold,
            })
        else:
            selected.append(feature)
    # Guard: if the threshold is too aggressive, keep the lowest-VIF compact subset.
    if len(selected) < 5 and len(start_features) >= 5:
        keep = report.sort_values(["vif", "feature"], ascending=[True, True]).head(min(30, len(start_features)))["feature"].astype(str).tolist()
        selected = [f for f in start_features if f in set(keep)]
        rows.append({
            "dropped_feature": "<threshold_guard>",
            "kept_feature": ";".join(selected[:20]),
            "reason": "kept_lowest_vif_subset_because_threshold_removed_too_many",
            "vif": np.nan,
            "vif_threshold": vif_threshold,
        })
    final_report = vif_report(x[selected], max_features=max(140, len(selected))) if selected else report
    return selected, pd.DataFrame(rows), final_report

def run_multicollinearity_selection(
    df: pd.DataFrame,
    feature_cols: list[str],
    supervised_mask: pd.Series | None = None,
    corr_threshold_primary: float = 0.95,
    corr_threshold_strict: float = 0.90,
    vif_threshold: float = 10.0,
) -> MulticollinearityResult:
    if supervised_mask is not None:
        base = df.loc[supervised_mask].copy()
    else:
        base = df.copy()
    x = _to_numeric_frame(base, feature_cols)
    quality = feature_quality_table(x)

    selected_corr95, drop_corr95, pairs95 = greedy_correlation_prune(x, threshold=corr_threshold_primary, method="spearman")
    selected_corr90, drop_corr90, pairs90 = greedy_correlation_prune(x, threshold=corr_threshold_strict, method="spearman")
    selected_vif, drop_vif, vif_final = iterative_vif_prune(x, selected_corr95, vif_threshold=vif_threshold)

    # Combined drop log with stages.
    drop_frames = []
    if not drop_corr95.empty:
        tmp = drop_corr95.copy(); tmp.insert(0, "stage", f"corr_prune_{corr_threshold_primary}"); drop_frames.append(tmp)
    if not drop_corr90.empty:
        tmp = drop_corr90.copy(); tmp.insert(0, "stage", f"corr_prune_{corr_threshold_strict}"); drop_frames.append(tmp)
    if not drop_vif.empty:
        tmp = drop_vif.copy(); tmp.insert(0, "stage", f"vif_prune_{vif_threshold}"); drop_frames.append(tmp)
    drop_log = pd.concat(drop_frames, ignore_index=True, sort=False) if drop_frames else pd.DataFrame(columns=["stage", "dropped_feature", "kept_feature", "reason"])

    condition_rows = []
    for label, cols in [
        ("default_all", list(feature_cols)),
        ("corr_pruned_095", selected_corr95),
        ("corr_pruned_090", selected_corr90),
        ("vif_pruned_10", selected_vif),
    ]:
        report = condition_index_report(x[[c for c in cols if c in x.columns]])
        if not report.empty:
            report.insert(0, "feature_set", label)
            condition_rows.append(report)
    condition = pd.concat(condition_rows, ignore_index=True) if condition_rows else pd.DataFrame()

    selected_feature_sets = {
        "default_all": list(feature_cols),
        "linear_corr_pruned_095": selected_corr95,
        "linear_corr_pruned_090_strict": selected_corr90,
        "linear_vif_pruned_10": selected_vif,
        # Recommended linear set uses correlation pruning. The VIF-pruned set is reported as a stricter diagnostic,
        # but can be too small for this sparse industrial dataset.
        "recommended_linear_safe": selected_corr95,
        "recommended_tree_safe": list(feature_cols),
        "recommended_neural_safe": list(feature_cols),
        "recommended_spectral_sequence_raw": [c for c in feature_cols if _feature_group(c) in {"spectral_reflectance", "spectral_ks"}],
    }

    summary = pd.DataFrame([
        {"metric": "input_default_feature_count", "value": int(len(feature_cols))},
        {"metric": "zero_variance_feature_count", "value": int(quality["zero_variance_flag"].sum()) if not quality.empty else 0},
        {"metric": "high_corr_pair_count_abs_spearman_ge_095", "value": int(len(pairs95))},
        {"metric": "high_corr_pair_count_abs_spearman_ge_090", "value": int(len(pairs90))},
        {"metric": "corr_pruned_095_feature_count", "value": int(len(selected_corr95))},
        {"metric": "corr_pruned_090_feature_count", "value": int(len(selected_corr90))},
        {"metric": "vif_pruned_10_feature_count", "value": int(len(selected_vif))},
        {"metric": "max_vif_after_pruning", "value": float(vif_final["vif"].max()) if not vif_final.empty and "vif" in vif_final else np.nan},
        {"metric": "condition_index_default_all", "value": float(condition.loc[condition["feature_set"].eq("default_all"), "condition_index"].iloc[0]) if not condition.empty and condition["feature_set"].eq("default_all").any() else np.nan},
        {"metric": "condition_index_vif_pruned_10", "value": float(condition.loc[condition["feature_set"].eq("vif_pruned_10"), "condition_index"].iloc[0]) if not condition.empty and condition["feature_set"].eq("vif_pruned_10").any() else np.nan},
    ])

    diagnostics = {
        "corr_threshold_primary": corr_threshold_primary,
        "corr_threshold_strict": corr_threshold_strict,
        "vif_threshold": vif_threshold,
        "note": "Use recommended_linear_safe for MLR/Lasso/ElasticNet-style models. It is correlation-pruned at |Spearman| >= 0.95. The VIF-pruned set is a stricter diagnostic and may be too small for final modelling. Tree/neural models may use all features but should still report this audit.",
    }
    return MulticollinearityResult(
        selected_features=selected_vif,
        dropped_features=[c for c in feature_cols if c not in set(selected_vif)],
        high_corr_pairs=pairs95,
        drop_log=drop_log,
        vif_report=vif_final,
        condition_report=condition,
        summary=summary,
        selected_feature_sets=selected_feature_sets,
        diagnostics=diagnostics,
    )
