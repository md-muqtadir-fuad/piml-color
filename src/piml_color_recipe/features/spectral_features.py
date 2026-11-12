from __future__ import annotations

import numpy as np
import pandas as pd

from .columns import nm_from_col, safe_numeric, sorted_nm_columns


def detect_reflectance_columns(df: pd.DataFrame) -> list[str]:
    cols = sorted_nm_columns(df.columns, "target_refl__reflectance")
    return [c for c in cols if c.endswith("nm")]


def detect_ks_columns(df: pd.DataFrame) -> list[str]:
    cols = sorted_nm_columns(df.columns, "target_ks__ks")
    return [c for c in cols if c.endswith("nm")]


def add_spectral_descriptors(df: pd.DataFrame, refl_cols: list[str], ks_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    features: list[str] = []
    if refl_cols:
        out, refl_features = _add_descriptors_for_block(out, refl_cols, "refl")
        features.extend(refl_features)
    if ks_cols:
        out, ks_features = _add_descriptors_for_block(out, ks_cols, "ks")
        features.extend(ks_features)
    return out, list(dict.fromkeys(features))


def _add_descriptors_for_block(df: pd.DataFrame, cols: list[str], label: str) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    features: list[str] = []
    block = out[cols].apply(pd.to_numeric, errors="coerce")
    wavelengths = np.array([nm_from_col(c) for c in cols], dtype=float)
    any_available = block.notna().any(axis=1)
    interp = block.T.interpolate(method="linear", limit_direction="both").T
    values = interp.to_numpy(dtype=float)

    # If an entire row is missing, interpolation leaves NaNs; fill with 0 only for numeric operations and mask later.
    values_safe = np.where(np.isfinite(values), values, 0.0)
    prefix = f"feat_{label}"

    descriptor_values = {
        f"{prefix}_mean": block.mean(axis=1, skipna=True),
        f"{prefix}_std": block.std(axis=1, skipna=True),
        f"{prefix}_min": block.min(axis=1, skipna=True),
        f"{prefix}_max": block.max(axis=1, skipna=True),
    }
    descriptor_values[f"{prefix}_range"] = descriptor_values[f"{prefix}_max"] - descriptor_values[f"{prefix}_min"]
    descriptor_values[f"{prefix}_area_400_750"] = pd.Series(np.trapezoid(values_safe, x=wavelengths, axis=1), index=out.index).where(any_available, np.nan)
    descriptor_values[f"{prefix}_slope_400_750"] = pd.Series((values_safe[:, -1] - values_safe[:, 0]) / (wavelengths[-1] - wavelengths[0]), index=out.index).where(any_available, np.nan)

    grad1 = np.gradient(values_safe, wavelengths, axis=1)
    grad2 = np.gradient(grad1, wavelengths, axis=1)
    descriptor_values[f"{prefix}_first_deriv_mean"] = pd.DataFrame(grad1, index=out.index).mean(axis=1).where(any_available, np.nan)
    descriptor_values[f"{prefix}_first_deriv_std"] = pd.DataFrame(grad1, index=out.index).std(axis=1).where(any_available, np.nan)
    descriptor_values[f"{prefix}_second_deriv_mean"] = pd.DataFrame(grad2, index=out.index).mean(axis=1).where(any_available, np.nan)
    descriptor_values[f"{prefix}_second_deriv_std"] = pd.DataFrame(grad2, index=out.index).std(axis=1).where(any_available, np.nan)

    bands = {
        "blue_400_490": (400, 490),
        "green_500_570": (500, 570),
        "yellow_orange_580_620": (580, 620),
        "red_630_750": (630, 750),
    }
    for band_name, (lo, hi) in bands.items():
        band_cols = [c for c in cols if lo <= nm_from_col(c) <= hi]
        descriptor_values[f"{prefix}_band_mean_{band_name}"] = block[band_cols].mean(axis=1, skipna=True) if band_cols else np.nan

    descriptor_df = pd.DataFrame(descriptor_values, index=out.index)
    out = pd.concat([out, descriptor_df], axis=1)
    features.extend(descriptor_df.columns.tolist())
    return out, features


def add_exploratory_pca(df: pd.DataFrame, refl_cols: list[str], ks_cols: list[str], max_components: int = 5) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, list[float]]]:
    """Add global PCA columns for EDA only, not for default CV feature set."""
    out = df.copy()
    pca_sets: dict[str, list[str]] = {"pca_reflectance_eda_global": [], "pca_ks_eda_global": []}
    explained: dict[str, list[float]] = {"refl": [], "ks": []}
    try:
        from sklearn.decomposition import PCA
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return out, pca_sets, explained

    def fit(cols: list[str], label: str) -> list[str]:
        nonlocal out
        if len(cols) < 2:
            return []
        x = out[cols].apply(pd.to_numeric, errors="coerce")
        usable = int(x.notna().any(axis=1).sum())
        if usable < 10:
            return []
        n_components = min(max_components, len(cols), usable)
        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), PCA(n_components=n_components, random_state=42))
        comps = pipe.fit_transform(x)
        pca_obj = pipe.named_steps["pca"]
        explained[label] = [round(float(v), 6) for v in pca_obj.explained_variance_ratio_]
        new = pd.DataFrame({f"feat_{label}_pca{i + 1}_eda_global": comps[:, i] for i in range(n_components)}, index=out.index)
        out = pd.concat([out, new], axis=1)
        return new.columns.tolist()

    pca_sets["pca_reflectance_eda_global"] = fit(refl_cols, "refl")
    pca_sets["pca_ks_eda_global"] = fit(ks_cols, "ks")
    return out, pca_sets, explained
