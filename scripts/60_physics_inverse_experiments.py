"""Physics-informed sparse inverse recipe experiments (reviewer-response study).

Each experiment id writes one block of rows to
outputs/physics_inverse_experiments/results.csv and per-row metrics to
outputs/physics_inverse_experiments/rows/.

Usage:
    python scripts/60_physics_inverse_experiments.py --exp E00
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from piml_color_recipe.inverse.km_physics import (  # noqa: E402
    delta_e_cmc,
    enumerate_subsets,
    fit_km,
    fit_km_joint,
    ks_to_reflectance,
    reflectance_to_lab,
    solve_subsets,
)

OUT_DIR = PROJECT_ROOT / "outputs" / "physics_inverse_experiments"
CACHE_DIR = OUT_DIR / "cache"
ROWS_DIR = OUT_DIR / "rows"
RESULTS_CSV = OUT_DIR / "results.csv"
ACTIVE = 0.01
SEED = 42
PROTOCOLS = {
    "grouped": ("groupkfold_recipe_no", "recipe_no"),
    "strict": ("groupkfold_target_id", "target_id"),
}
# Dataset-bookkeeping or post-outcome columns, not known when a new shade is formulated.
LEAKY_PROCESS = {
    "production__total_trial",
    "production__recipe_no_valid",
    "production__target_id_valid",
    "production__fiber_total_valid",
    "production__duplicate_recipe_no",
}
# Substrate and machine variables known before formulation. Salt and soda ash are excluded:
# the dyehouse sets them from the depth of shade, so they encode the answer (C01).
SUBSTRATE_COLS = [
    "production__gsm", "production__sample_weight", "production__fiber_total_pct", "production__l2r_numeric",
    "production__modal", "production__viscose", "production__cotton", "production__polyester", "production__lycra",
    "production__is_lycra_blend", "production__is_polyester_blend", "feat_process_cotton_polyester_ratio",
]


# --------------------------------------------------------------------------- data


class Data:
    def __init__(self) -> None:
        df = pd.read_csv(PROJECT_ROOT / "data/processed/features_supervised.csv", low_memory=False)
        self.df = df
        self.dye_cols = sorted(
            c for c in df.columns if c.startswith("vec__dye_") and c.endswith("_pct") and "count" not in c and "total" not in c
        )
        self.dye_names = [c.removeprefix("vec__dye_").removesuffix("_pct") for c in self.dye_cols]
        mapping = pd.read_csv(PROJECT_ROOT / "data/global_naming/global_dye_mapping_active_36.csv")
        hue = dict(zip(mapping["dye_column"], mapping["original_hue_inferred"]))
        self.dye_hue = np.array([str(hue.get(c.removeprefix("vec__"), "other")) for c in self.dye_cols])
        self.Y = df[self.dye_cols].fillna(0.0).to_numpy(float)
        self.R = df[[c for c in df.columns if c.startswith("target_refl__")]].to_numpy(float) / 100.0
        self.KS = df[[c for c in df.columns if c.startswith("target_ks__")]].to_numpy(float)
        self.lab_d65 = reflectance_to_lab(self.R, "D65")
        self.lab_f11 = reflectance_to_lab(self.R, "FL11")
        spectral = [c for c in df.columns if c.startswith(("target_refl__", "target_ks__"))]
        spectral += [c for c in df.columns if c.startswith(("feat_refl_", "feat_ks_")) and "pca" not in c]
        process = [
            c for c in df.columns
            if c.startswith("production__") and c not in LEAKY_PROCESS and pd.api.types.is_numeric_dtype(df[c])
        ]
        process += [c for c in df.columns if c.startswith(("feat_process_", "feat_cat_production_"))]
        substrate = [c for c in SUBSTRATE_COLS if c in df.columns]
        substrate += [c for c in df.columns if c.startswith("feat_cat_production_")]
        self.features = {
            "spectral": spectral,
            "spectral_process": spectral + process,  # includes salt/soda ash: leaks total dye (C01)
            "spectral_substrate": spectral + substrate,
        }
        self.families = sorted(set(self.dye_hue))
        self.F = np.column_stack([self.Y[:, self.dye_hue == g].sum(1) for g in self.families])
        uid_to_pos = {u: i for i, u in enumerate(df["recipe_row_uid"])}
        self.folds: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        for proto, (split, _) in PROTOCOLS.items():
            obj = json.loads((PROJECT_ROOT / f"data/splits/{split}.json").read_text(encoding="utf-8"))
            self.folds[proto] = [
                (np.array([uid_to_pos[u] for u in f["train_row_uids"]]), np.array([uid_to_pos[u] for u in f["val_row_uids"]]))
                for f in obj["folds"]
            ]

    def X(self, group: str) -> np.ndarray:
        return self.df[self.features[group]].apply(pd.to_numeric, errors="coerce").to_numpy(float)

    def clusters(self, proto: str) -> np.ndarray:
        return self.df[PROTOCOLS[proto][1]].astype(str).to_numpy()

    def strata(self, proto: str) -> np.ndarray:
        """1 = exact dye set seen in training, 2 = novel set of seen dyes, 3 = contains an unseen dye."""
        A = self.Y > ACTIVE
        sets = [frozenset(np.flatnonzero(r)) for r in A]
        out = np.zeros(len(A), dtype=int)
        for tr, va in self.folds[proto]:
            seen = A[tr].any(0)
            train_sets = {sets[i] for i in tr}
            for i in va:
                out[i] = 1 if sets[i] in train_sets else (2 if all(seen[j] for j in sets[i]) else 3)
        return out

    def to_family(self, P: np.ndarray) -> np.ndarray:
        return np.column_stack([P[:, self.dye_hue == g].sum(1) for g in self.families])


# ------------------------------------------------------------------------ metrics


def feasible(pred: np.ndarray, keep: int = 3, min_pct: float = ACTIVE) -> np.ndarray:
    """Published post-processing: threshold, keep top-k, restore the raw predicted total."""
    raw = np.maximum(np.nan_to_num(pred), 0.0)
    out = raw.copy()
    out[out < min_pct] = 0.0
    for i in range(len(out)):
        if np.count_nonzero(out[i]) > keep:
            drop = np.argsort(out[i])[::-1][keep:]
            out[i, drop] = 0.0
        if out[i].sum() <= 0 < raw[i].sum():
            top = np.argsort(raw[i])[::-1][:keep]
            out[i, top] = raw[i, top]
    scale = np.divide(raw.sum(1), out.sum(1), out=np.ones(len(out)), where=out.sum(1) > 0)
    return out * scale[:, None]


def row_metrics(Y: np.ndarray, P: np.ndarray, data: Data | None = None, km_models: dict[int, object] | None = None,
                row_fold: np.ndarray | None = None) -> pd.DataFrame:
    P = np.maximum(np.nan_to_num(P), 0.0)
    err = np.abs(P - Y)
    t, p = Y > ACTIVE, P > ACTIVE
    tp = (t & p).sum(1).astype(float)
    fp = (~t & p).sum(1).astype(float)
    fn = (t & ~p).sum(1).astype(float)
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    active_mae = np.array([err[i, t[i]].mean() if t[i].any() else np.nan for i in range(len(Y))])
    union = t | p
    union_mae = np.array([err[i, union[i]].mean() if union[i].any() else 0.0 for i in range(len(Y))])
    out = pd.DataFrame({
        "recipe_mae": err.mean(1),
        "active_mae": active_mae,
        "union_mae": union_mae,
        "total_abs_err": np.abs(P.sum(1) - Y.sum(1)),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "exact_set": (t == p).all(1).astype(float),
        "pred_count": p.sum(1).astype(float),
    })
    if data is not None:
        fam_t, fam_p = data.F, data.to_family(P)
        out["family_mae"] = np.abs(fam_p - fam_t).mean(1)
        ft, fp_ = fam_t > ACTIVE, fam_p > ACTIVE
        ftp = (ft & fp_).sum(1).astype(float)
        fprec = np.divide(ftp, fp_.sum(1), out=np.zeros_like(ftp), where=fp_.sum(1) > 0)
        frec = np.divide(ftp, ft.sum(1), out=np.zeros_like(ftp), where=ft.sum(1) > 0)
        out["family_f1"] = np.divide(2 * fprec * frec, fprec + frec, out=np.zeros_like(ftp), where=(fprec + frec) > 0)
    if data is not None and km_models is not None and row_fold is not None:
        de = np.full(len(Y), np.nan)
        for f, model in km_models.items():
            idx = np.flatnonzero(row_fold == f)
            refl = model.predict_reflectance(P[idx])
            de65 = delta_e_cmc(data.lab_d65[idx], reflectance_to_lab(refl, "D65"))
            de11 = delta_e_cmc(data.lab_f11[idx], reflectance_to_lab(refl, "FL11"))
            de[idx] = np.maximum(de65, de11)
        out["km_de_max"] = de
    return out


SUMMARY_COLS = ["recipe_mae", "active_mae", "union_mae", "total_abs_err", "precision", "recall", "f1", "exact_set",
                "pred_count", "family_mae", "family_f1", "km_de_max"]
STRATA_CSV_NAME = "results_by_stratum.csv"


def summarize(rows: pd.DataFrame) -> dict[str, float]:
    return {c: float(np.nanmean(rows[c])) for c in SUMMARY_COLS if c in rows}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, clusters: np.ndarray, n_boot: int = 2000, seed: int = SEED) -> tuple[float, float, float]:
    """Mean of (a - b) with a cluster bootstrap 95% CI."""
    diff = pd.Series(a - b).groupby(clusters).agg(["sum", "count"])
    s, c = diff["sum"].to_numpy(), diff["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n_boot, len(s)))
    boots = s[idx].sum(1) / c[idx].sum(1)
    return float(s.sum() / c.sum()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def record(exp: str, method: str, proto: str, rows: pd.DataFrame, P: np.ndarray, data: Data, note: str = "") -> dict:
    ROWS_DIR.mkdir(parents=True, exist_ok=True)
    rows = rows.copy()
    rows.insert(0, "recipe_row_uid", data.df["recipe_row_uid"].to_numpy())
    rows.to_csv(ROWS_DIR / f"{exp}__{method}__{proto}.csv", index=False)
    np.save(ROWS_DIR / f"{exp}__{method}__{proto}__pred.npy", P)
    res = {"exp": exp, "method": method, "protocol": proto, **summarize(rows), "note": note,
           "timestamp": time.strftime("%Y-%m-%d %H:%M")}
    _upsert(RESULTS_CSV, [res], exp, method, proto)
    strata = data.strata(proto)
    by_stratum = [
        {"exp": exp, "method": method, "protocol": proto, "stratum": s, "n": int((strata == s).sum()),
         **summarize(rows[strata == s])}
        for s in (1, 2, 3)
    ]
    _upsert(OUT_DIR / STRATA_CSV_NAME, by_stratum, exp, method, proto)
    return res


def _upsert(path: Path, new_rows: list[dict], exp: str, method: str, proto: str) -> None:
    prev = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not prev.empty:
        prev = prev[~((prev["exp"] == exp) & (prev["method"] == method) & (prev["protocol"] == proto))]
    pd.concat([prev, pd.DataFrame(new_rows)], ignore_index=True).to_csv(path, index=False)


def print_strata(exp: str, methods: list[str] | None = None, proto: str = "strict",
                 cols=("n", "f1", "union_mae", "active_mae", "family_mae", "family_f1", "total_abs_err")) -> None:
    t = pd.read_csv(OUT_DIR / STRATA_CSV_NAME)
    t = t[(t["exp"] == exp) & (t["protocol"] == proto)]
    if methods:
        t = t[t["method"].isin(methods)]
    print(f"--- {exp} by stratum ({proto})")
    print(t[["method", "stratum", *cols]].sort_values(["stratum", "method"]).to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))


def load_rows(exp: str, method: str, proto: str) -> pd.DataFrame:
    return pd.read_csv(ROWS_DIR / f"{exp}__{method}__{proto}.csv")


def print_table(results: list[dict]) -> None:
    t = pd.DataFrame(results)
    cols = ["protocol", "method"] + [c for c in SUMMARY_COLS if c in t]
    print(t[cols].sort_values(["protocol", "method"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


# --------------------------------------------------------------------- components


def row_fold_index(data: Data, proto: str) -> np.ndarray:
    rf = np.zeros(len(data.Y), dtype=int)
    for f, (_, va) in enumerate(data.folds[proto]):
        rf[va] = f
    return rf


def fold_km_models(data: Data, proto: str, **km_kwargs) -> dict[int, object]:
    return {f: fit_km(data.Y[tr], data.KS[tr], **km_kwargs) for f, (tr, _) in enumerate(data.folds[proto])}


def ml_regression_oof(data: Data, proto: str, family: str, group: str) -> np.ndarray:
    X = data.X(group)
    P = np.zeros_like(data.Y)
    for tr, va in data.folds[proto]:
        if family == "rf":
            est = RandomForestRegressor(n_estimators=200, max_features=0.5, min_samples_leaf=1, random_state=SEED, n_jobs=-1)
        else:
            est = ExtraTreesRegressor(n_estimators=120, max_features=1.0, min_samples_leaf=3, random_state=SEED, n_jobs=-1)
        model = make_pipeline(SimpleImputer(strategy="median"), est)
        model.fit(X[tr], data.Y[tr])
        P[va] = model.predict(X[va])
    return P


def dye_prior_oof(data: Data, proto: str, group: str = "spectral") -> np.ndarray:
    """Out-of-fold P(dye active) from a multi-label random forest."""
    X = data.X(group)
    A = (data.Y > ACTIVE).astype(int)
    prob = np.zeros_like(data.Y)
    for tr, va in data.folds[proto]:
        prob[va] = dye_prior_fit_predict(X[tr], A[tr], X[va])
    return prob


def dye_prior_fit_predict(X_tr: np.ndarray, A_tr: np.ndarray, X_va: np.ndarray, n_estimators: int = 300,
                          min_samples_leaf: int = 2) -> np.ndarray:
    prob = np.zeros((len(X_va), A_tr.shape[1]))
    seen = A_tr.any(0)
    clf = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features="sqrt",
                               random_state=SEED, n_jobs=-1),
    )
    clf.fit(X_tr, A_tr[:, seen])
    probas = clf.predict_proba(X_va)
    for pos, (col, pr) in enumerate(zip(np.flatnonzero(seen), probas)):
        classes = list(clf[-1].classes_[pos])
        prob[:, col] = pr[:, classes.index(1)] if 1 in classes else 0.0
    return prob


def subset_tables(data: Data, proto: str, tag: str, km_kwargs: dict, max_size: int = 3,
                  min_rows_usable: int = 1) -> dict[int, dict]:
    """Physics solutions for every subset and every validation row, cached per fold."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tables = {}
    for f, (tr, va) in enumerate(data.folds[proto]):
        path = CACHE_DIR / f"{tag}__{proto}__fold{f}.npz"
        if path.exists():
            z = np.load(path)
            tables[f] = {k: z[k] for k in z.files}
            continue
        model = make_km(data, tr, km_kwargs)
        usable = np.flatnonzero((data.Y[tr] > ACTIVE).sum(0) >= min_rows_usable)
        subsets = enumerate_subsets(usable, max_size)
        conc = np.zeros((len(va), len(subsets), max_size))
        d65 = np.zeros((len(va), len(subsets)))
        f11 = np.zeros_like(d65)
        c_max = float(data.Y[tr].max())
        t0 = time.time()
        for i, row in enumerate(va):
            sol = solve_subsets(model, data.KS[row], subsets, c_max=c_max)
            conc[i], d65[i], f11[i] = sol.conc, sol.de_d65, sol.de_f11
        print(f"  [{tag}/{proto}] fold {f}: {len(va)} rows x {len(subsets)} subsets in {time.time() - t0:.1f}s")
        tables[f] = {"val": va, "subsets": subsets, "conc": conc, "de_d65": d65, "de_f11": f11}
        np.savez_compressed(path, **tables[f])
    return tables


def select_from_tables(data: Data, tables: dict[int, dict], score_fn) -> np.ndarray:
    """score_fn(table, i) -> score per subset for validation row i (lower is better)."""
    P = np.zeros_like(data.Y)
    for tab in tables.values():
        subsets = tab["subsets"]
        for i, row in enumerate(tab["val"]):
            best = int(np.argmin(score_fn(tab, i, row)))
            for slot, dye in enumerate(subsets[best]):
                if dye >= 0:
                    P[row, dye] = tab["conc"][i, best, slot]
    return P


def subset_logit_sum(subsets: np.ndarray, prob_row: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    p = np.clip(prob_row, eps, 1 - eps)
    logit = np.append(np.log(p / (1 - p)), 0.0)  # index -1 -> padding contributes nothing
    return logit[subsets].sum(1)


# -------------------------------------------------------------------- experiments


def exp_e00(data: Data) -> list[dict]:
    """Baselines: trivial predictors and replicas of the published ML inverse models."""
    results = []
    for proto in PROTOCOLS:
        rf_idx = row_fold_index(data, proto)
        km = fold_km_models(data, proto)
        cands = {}
        cands["all_zeros"] = np.zeros_like(data.Y)
        mean_vec = np.zeros_like(data.Y)
        nn1 = np.zeros_like(data.Y)
        Xs = StandardScaler()
        Xspec = data.X("spectral")
        for tr, va in data.folds[proto]:
            mean_vec[va] = data.Y[tr].mean(0)
            sc = Xs.fit(np.nan_to_num(Xspec[tr]))
            a, b = sc.transform(np.nan_to_num(Xspec[va])), sc.transform(np.nan_to_num(Xspec[tr]))
            nn1[va] = data.Y[tr][np.argmin(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1), axis=1)]
        cands["train_mean_feasible"] = feasible(mean_vec)
        cands["nearest_historical_spectral"] = nn1
        cands["true_recipe_oracle"] = data.Y.copy()
        for fam in ("rf", "et"):
            for grp in ("spectral", "spectral_process"):
                cands[f"ml_{fam}_{grp}_feasible"] = feasible(ml_regression_oof(data, proto, fam, grp))
        for name, P in cands.items():
            rows = row_metrics(data.Y, P, data, km, rf_idx)
            results.append(record("E00", name, proto, rows, P, data))
    return results


def exp_e00b(data: Data) -> list[dict]:
    """Baselines rerun with salt / soda ash removed from the inputs (C01) and with stratified metrics."""
    tot = data.Y.sum(1)
    for c in ("production__sample_weight", "production__gsm", "production__l2r_numeric", "production__cotton"):
        x = pd.to_numeric(data.df[c], errors="coerce")
        print(f"  spearman({c}, total dye) = {x.corr(pd.Series(tot), method='spearman'):.3f}")
    results = []
    for proto in PROTOCOLS:
        rf_idx = row_fold_index(data, proto)
        km = fold_km_models(data, proto, uptake="langmuir", beta=0.3, weighting="relative")
        cands = {"all_zeros": np.zeros_like(data.Y), "true_recipe_oracle": data.Y.copy()}
        nn1 = np.zeros_like(data.Y)
        Xspec = np.nan_to_num(data.X("spectral"))
        for tr, va in data.folds[proto]:
            sc = StandardScaler().fit(Xspec[tr])
            a, b = sc.transform(Xspec[va]), sc.transform(Xspec[tr])
            nn1[va] = data.Y[tr][np.argmin(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1), axis=1)]
        cands["nearest_historical_spectral"] = nn1
        for fam in ("rf", "et"):
            for grp in ("spectral", "spectral_substrate", "spectral_process"):
                cands[f"ml_{fam}_{grp}_feasible"] = feasible(ml_regression_oof(data, proto, fam, grp))
        for name, P in cands.items():
            results.append(record("E00b", name, proto, row_metrics(data.Y, P, data, km, rf_idx), P, data))
    print_strata("E00b")
    return results


def km_true_recipe_de(data: Data, model, idx: np.ndarray) -> np.ndarray:
    refl = model.predict_reflectance(data.Y[idx])
    de65 = delta_e_cmc(data.lab_d65[idx], reflectance_to_lab(refl, "D65"))
    de11 = delta_e_cmc(data.lab_f11[idx], reflectance_to_lab(refl, "FL11"))
    return np.maximum(de65, de11)


E01_VARIANTS = {
    "lin_none": dict(weighting="none"),
    "lin_sqrt": dict(weighting="sqrt"),
    "lin_rel": dict(weighting="relative"),
    "lin_rel_nosub": dict(weighting="relative", fit_substrate=False),
    "lin_rel_ridge0.01": dict(weighting="relative", ridge=0.01),
    "lin_rel_ridge0.1": dict(weighting="relative", ridge=0.1),
    **{f"pow{g}_rel": dict(weighting="relative", uptake="power", beta=g) for g in (0.4, 0.5, 0.6, 0.7, 0.8)},
    **{f"lang{b}_rel": dict(weighting="relative", uptake="langmuir", beta=b) for b in (0.1, 0.3, 1.0, 3.0)},
}


def hue_band_upper(hue: np.ndarray, near_zero: float = 0.02, shoulder: float = 0.5) -> np.ndarray:
    """Upper bounds on unit K/S by hue family (bands fixed by the external review before any run):
    yellow: near zero at >= 560 nm, shoulder 520-550 nm; red: near zero at >= 630 nm, shoulder 600-620 nm."""
    from piml_color_recipe.inverse.km_physics import WAVELENGTHS as wl

    upper = np.full((len(hue), len(wl)), np.inf)
    for i, fam in enumerate(hue):
        if fam == "yellow":
            upper[i, wl >= 560] = near_zero
            upper[i, (wl >= 520) & (wl < 560)] = shoulder
        elif fam == "red":
            upper[i, wl >= 630] = near_zero
            upper[i, (wl >= 600) & (wl < 630)] = shoulder
    return upper


def make_km(data: Data, tr: np.ndarray, kw: dict):
    kw = dict(kw)
    mask = kw.pop("band_mask", None)
    if mask:
        kw["upper"] = hue_band_upper(data.dye_hue, **({} if mask is True else mask))
    if kw.pop("joint", False):
        return fit_km_joint(data.Y[tr], data.KS[tr], groups=data.dye_hue, **kw)
    return fit_km(data.Y[tr], data.KS[tr], groups=data.dye_hue, **kw)


E01B_VARIANTS = {
    f"{u}_s{s}_h{h}": dict(joint=True, weighting="relative", uptake=ut, beta=b, smooth=s, shrink=h)
    for u, ut, b in (("pow0.8", "power", 0.8), ("lang0.3", "langmuir", 0.3))
    for s in (0.0, 1.0, 10.0)
    for h in (0.0, 0.1, 1.0)
}


def _e01_fold(data: Data, kw: dict, tr: np.ndarray, va: np.ndarray):
    model = make_km(data, tr, kw)
    seen = (data.Y[tr] > ACTIVE).any(0)
    ok = np.array([(seen | ~(data.Y[r] > ACTIVE)).all() for r in va])
    return va, km_true_recipe_de(data, model, va), float(np.median(km_true_recipe_de(data, model, tr))), ok


def exp_e01(data: Data, variants: dict | None = None, tag: str = "E01") -> list[dict]:
    """KM forward-fit quality: colour error of the TRUE recipe under fold-fitted KM variants."""
    from joblib import Parallel, delayed

    variants = variants or E01_VARIANTS
    jobs = [(proto, name, f) for proto in PROTOCOLS for name in variants for f in range(len(data.folds[proto]))]
    res = Parallel(n_jobs=4, verbose=5)(
        delayed(_e01_fold)(data, variants[name], *data.folds[proto][f]) for proto, name, f in jobs
    )
    by_key: dict[tuple[str, str], list] = {}
    for (proto, name, _), r in zip(jobs, res):
        by_key.setdefault((proto, name), []).append(r)
    out = []
    for (proto, name), parts in by_key.items():
        de_val = np.zeros(len(data.Y))
        seen_all = np.zeros(len(data.Y), bool)
        de_train = []
        for va, de, de_tr, ok in parts:
            de_val[va], seen_all[va] = de, ok
            de_train.append(de_tr)
        if True:
            out.append({
                "protocol": proto, "variant": name,
                "val_median": np.median(de_val), "val_mean": de_val.mean(),
                "val_median_seen_dyes": np.median(de_val[seen_all]), "val_mean_seen_dyes": de_val[seen_all].mean(),
                "val_share_le_2": (de_val <= 2).mean(), "train_median": float(np.mean(de_train)),
                "n_seen": int(seen_all.sum()),
            })
    t = pd.DataFrame(out)
    t.to_csv(OUT_DIR / f"{tag}_km_forward_fit.csv", index=False)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return []


def exp_e01b(data: Data) -> list[dict]:
    """Joint KM fit with spectral smoothness and hue-family shrinkage."""
    return exp_e01(data, E01B_VARIANTS, "E01b")


_LANG = dict(weighting="relative", uptake="langmuir", beta=0.3)
E01C_VARIANTS = {
    "lang0.3": _LANG,
    "lang0.3_nosub": {**_LANG, "fit_substrate": False},
    **{f"lang0.3_shrink{h}": {**_LANG, "shrink": h} for h in (0.001, 0.003, 0.01, 0.03)},
    **{f"lang0.3_ridge{r}": {**_LANG, "ridge": r} for r in (0.0003, 0.001, 0.003)},
    **{f"lang0.3_joint_smooth{s}": {**_LANG, "joint": True, "smooth": s} for s in (0.01, 0.1)},
    **{f"lang{b}": {**_LANG, "beta": b} for b in (0.2, 0.4, 0.5)},
}


def exp_e01c(data: Data) -> list[dict]:
    """Weaker regularisation, substrate option and a finer Langmuir grid."""
    return exp_e01(data, E01C_VARIANTS, "E01c")


KM_DEFAULT = dict(weighting="relative", uptake="langmuir", beta=0.3, shrink=0.01)


def exp_e02(data: Data) -> list[dict]:
    """Classical computer colour matching: exhaustive <=3-dye search, pick the minimum KM colour difference."""
    results = []
    for proto in PROTOCOLS:
        rf_idx = row_fold_index(data, proto)
        km = {f: make_km(data, tr, KM_DEFAULT) for f, (tr, _) in enumerate(data.folds[proto])}
        tables = subset_tables(data, proto, "E02_lang0.3_shrink0.01_cmax", KM_DEFAULT)
        variants = {
            "ccm_min_de_max": lambda t, i, r: np.maximum(t["de_d65"][i], t["de_f11"][i]),
            "ccm_min_de_d65": lambda t, i, r: t["de_d65"][i],
        }
        for kappa in (0.25, 0.5, 1.0):  # parsimony: penalty per dye in CMC units (exploratory)
            variants[f"ccm_min_de_max_pen{kappa}"] = (
                lambda t, i, r, k=kappa: np.maximum(t["de_d65"][i], t["de_f11"][i]) + k * (t["subsets"] >= 0).sum(1)
            )
        for name, fn in variants.items():
            P = select_from_tables(data, tables, fn)
            results.append(record("E02", name, proto, row_metrics(data.Y, P, data, km, rf_idx), P, data))
        best_de = np.zeros(len(data.Y))
        for tab in tables.values():
            best_de[tab["val"]] = np.maximum(tab["de_d65"], tab["de_f11"]).min(1)
        print(f"  [{proto}] min achievable KM dE_max over all subsets: median {np.median(best_de):.3f}")
    print_strata("E02")
    return results


# ---------------------------------------------------------------- E03: hybrid, nested CV


def candidate_sets(A_tr: np.ndarray, hue: np.ndarray, max_size: int = 3,
                   extra: list[tuple] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Historical dye sets, their subsets, and single within-hue swaps (C01: covers every recoverable row)."""
    hist: dict[tuple, int] = {}
    for row in A_tr:
        s = tuple(np.flatnonzero(row))
        if 0 < len(s) <= max_size:
            hist[s] = hist.get(s, 0) + 1
    seen = np.flatnonzero(A_tr.any(0))
    cands = set(hist) | {tuple(sorted(int(d) for d in s)) for s in (extra or []) if 0 < len(s) <= max_size}
    for s in hist:
        for r in range(1, len(s)):
            cands.update(combinations(s, r))
        for pos, d_old in enumerate(s):
            for d_new in seen:
                if d_new not in s and hue[d_new] == hue[d_old]:
                    cands.add(tuple(sorted(s[:pos] + (int(d_new),) + s[pos + 1:])))
    cands = sorted(cands)
    arr = -np.ones((len(cands), max_size), dtype=int)
    counts = np.zeros(len(cands))
    for i, s in enumerate(cands):
        arr[i, : len(s)] = s
        counts[i] = hist.get(s, 0)
    return arr, counts


def inner_splits(data: Data, proto: str, tr: np.ndarray, n_splits: int = 4) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = data.clusters(proto)[tr]
    return [(tr[a], tr[b]) for a, b in GroupKFold(n_splits=n_splits).split(tr, groups=groups)]


KM_GRID = {f"b{b}_h{h}": dict(weighting="relative", uptake="langmuir", beta=b, shrink=h)
           for b in (0.2, 0.3, 0.5) for h in (0.0, 0.003, 0.01, 0.03)}


def select_km_inner(data: Data, proto: str, tr: np.ndarray, grid: dict | None = None) -> dict:
    """Choose the KM variant by inner-CV forward colour error of true recipes (seen-dye rows only)."""
    A = data.Y > ACTIVE
    grid = grid or KM_GRID
    scores = {}
    for name, kw in grid.items():
        errs = []
        for itr, iva in inner_splits(data, proto, tr):
            seen = A[itr].any(0)
            ok = np.array([(seen | ~A[r]).all() for r in iva])
            if ok.any():
                errs.append(km_true_recipe_de(data, make_km(data, itr, kw), iva[ok]))
        scores[name] = float(np.mean(np.concatenate(errs)))
    return grid[min(scores, key=scores.get)] | {"_name": min(scores, key=scores.get)}


def build_block(data: Data, proto: str, tr: np.ndarray, va: np.ndarray, km_kw: dict, prior_group: str,
                extra_sets: list[tuple] | None = None) -> dict:
    """Everything the selection rule needs for validation rows va, using only training rows tr."""
    A = data.Y > ACTIVE
    kw = {k: v for k, v in km_kw.items() if not k.startswith("_")}
    subsets, counts = candidate_sets(A[tr], data.dye_hue, extra=extra_sets)
    model = make_km(data, tr, kw)
    n_va, n_c = len(va), len(subsets)
    de = np.zeros((n_va, n_c))
    conc = np.zeros((n_va, n_c, subsets.shape[1]))
    c_max = float(data.Y[tr].max())
    for i, r in enumerate(va):
        sol = solve_subsets(model, data.KS[r], subsets, c_max=c_max)
        de[i], conc[i] = np.maximum(sol.de_d65, sol.de_f11), sol.conc
    X = data.X(prior_group)
    prob = dye_prior_fit_predict(X[tr], A[tr].astype(int), X[va])
    logit_sum = np.stack([subset_logit_sum(subsets, prob[i]) for i in range(n_va)])
    set_logp = np.log((counts + 0.5) / (counts.sum() + 0.5 * n_c))
    sizes = (subsets >= 0).sum(1)
    size_counts = np.array([(A[tr].sum(1) == k).sum() for k in range(subsets.shape[1] + 1)], dtype=float)
    card_logp = np.log((size_counts[sizes] + 1) / (size_counts.sum() + len(size_counts)))
    # Per-candidate outcome metrics, independent of the selection rule.
    true_sets = A[va]
    cand_mask = np.zeros((n_c, data.Y.shape[1]), bool)
    for c, s in enumerate(subsets):
        cand_mask[c, s[s >= 0]] = True
    tp = (true_sets[:, None, :] & cand_mask[None]).sum(-1)
    npred, ntrue = cand_mask.sum(1)[None, :], true_sets.sum(1)[:, None]
    prec = np.divide(tp, npred, out=np.zeros(tp.shape), where=npred > 0)
    rec = np.divide(tp, ntrue, out=np.zeros(tp.shape), where=ntrue > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(tp.shape), where=(prec + rec) > 0)
    full = np.zeros((n_va, n_c, data.Y.shape[1]))
    for slot in range(subsets.shape[1]):
        ok = np.flatnonzero(subsets[:, slot] >= 0)
        full[:, ok, subsets[ok, slot]] = conc[:, ok, slot]
    union = true_sets[:, None, :] | cand_mask[None]
    umae = (np.abs(full - data.Y[va][:, None, :]) * union).sum(-1) / union.sum(-1)
    return {"va": va, "subsets": subsets, "de": de, "conc": conc, "logit_sum": logit_sum, "set_logp": set_logp,
            "card_logp": card_logp, "f1": f1, "umae": umae, "prob": prob, "counts": counts, "cand_mask": cand_mask}


def score_block(b: dict, cfg: dict) -> np.ndarray:
    de = b["de"] if cfg["g"] == "lin" else np.log(b["de"] + 0.1)
    return (cfg["w_de"] * de - cfg["l_set"] * b["set_logp"][None, :] - cfg["l_dye"] * b["logit_sum"]
            - cfg["l_card"] * b["card_logp"][None, :])


def block_prediction(data: Data, b: dict, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    choice = np.argmin(score_block(b, cfg), axis=1)
    P = np.zeros((len(b["va"]), data.Y.shape[1]))
    for i, c in enumerate(choice):
        for slot, dye in enumerate(b["subsets"][c]):
            if dye >= 0:
                P[i, dye] = b["conc"][i, c, slot]
    return P, choice


CFG_GRID = [
    {"g": g, "w_de": w, "l_set": ls, "l_dye": ld, "l_card": lc}
    for g in ("lin", "log") for w in (0.0, 1.0) for ls in (0.0, 0.1, 0.3, 1.0, 3.0)
    for ld in (0.0, 0.1, 0.3, 1.0, 3.0) for lc in (0.0, 1.0)
    if not (w == 0 and ls == 0 and ld == 0 and lc == 0) and not (w == 0 and g == "log")
]
VARIANTS = {
    "hybrid": lambda c: True,
    "physics_candidates": lambda c: c["w_de"] == 1 and c["l_set"] == 0 and c["l_dye"] == 0 and c["l_card"] == 0,
    "prior_only": lambda c: c["w_de"] == 0,
    "hybrid_no_set_prior": lambda c: c["l_set"] == 0,
    "hybrid_no_dye_prior": lambda c: c["l_dye"] == 0,
}


def choose_cfg(data: Data, blocks: list[dict], allowed) -> dict:
    """Maximise mean F1 on pooled inner-validation rows; ties broken by lower union MAE."""
    best, best_key = None, None
    for cfg in CFG_GRID:
        if not allowed(cfg):
            continue
        f1s, umaes = [], []
        for b in blocks:
            choice = np.argmin(score_block(b, cfg), axis=1)
            rows = np.arange(len(choice))
            f1s.append(b["f1"][rows, choice])
            umaes.append(b["umae"][rows, choice])
        key = (round(float(np.mean(np.concatenate(f1s))), 6), -float(np.mean(np.concatenate(umaes))))
        if best_key is None or key > best_key:
            best, best_key = cfg, key
    return best


def run_hybrid(data: Data, proto: str, prior_group: str = "spectral_substrate", tag: str = "E03") -> dict[str, np.ndarray]:
    preds = {v: np.zeros_like(data.Y) for v in VARIANTS}
    preds["candidate_oracle"] = np.zeros_like(data.Y)
    log = []
    for f, (tr, va) in enumerate(data.folds[proto]):
        t0 = time.time()
        km_kw = select_km_inner(data, proto, tr)
        inner = [build_block(data, proto, itr, iva, km_kw, prior_group) for itr, iva in inner_splits(data, proto, tr)]
        outer = build_block(data, proto, tr, va, km_kw, prior_group)
        for v, allowed in VARIANTS.items():
            cfg = choose_cfg(data, inner, allowed)
            preds[v][va] = block_prediction(data, outer, cfg)[0]
            log.append({"protocol": proto, "fold": f, "variant": v, "km": km_kw["_name"], **cfg})
        best = outer["f1"].argmax(1)  # best candidate per row by F1: coverage ceiling of the candidate set
        preds["candidate_oracle"][va] = block_prediction(
            data, {**outer, "de": -np.eye(len(outer["subsets"]))[best]}, {"g": "lin", "w_de": 1.0, "l_set": 0, "l_dye": 0, "l_card": 0}
        )[0]
        print(f"  [{tag}/{proto}] fold {f}: km={km_kw['_name']} candidates={len(outer['subsets'])} {time.time() - t0:.0f}s")
    pd.DataFrame(log).to_csv(OUT_DIR / f"{tag}_{proto}_selected_configs.csv", index=False)
    return preds


def exp_e03(data: Data) -> list[dict]:
    """Hybrid physics + learned-prior selection over pruned candidate sets; all hyperparameters by inner CV."""
    results = []
    for proto in PROTOCOLS:
        rf_idx = row_fold_index(data, proto)
        km = {f: make_km(data, tr, KM_DEFAULT) for f, (tr, _) in enumerate(data.folds[proto])}
        for name, P in run_hybrid(data, proto).items():
            results.append(record("E03", name, proto, row_metrics(data.Y, P, data, km, rf_idx), P, data))
    print_strata("E03")
    for proto in PROTOCOLS:
        print(pd.read_csv(OUT_DIR / f"E03_{proto}_selected_configs.csv").to_string(index=False))
    return results


def chosen_vectors(data: Data, b: dict, choice: np.ndarray, conc_ml: np.ndarray, alpha: float) -> np.ndarray:
    """Recipe vectors for the chosen dye sets: alpha * physics + (1 - alpha) * ML concentrations.

    The ML amounts on the chosen dyes are rescaled to the ML model's own predicted total,
    as in the published post-processing.
    """
    P = np.zeros((len(choice), data.Y.shape[1]))
    for i, c in enumerate(choice):
        dyes = b["subsets"][c][b["subsets"][c] >= 0]
        phys = b["conc"][i, c, : len(dyes)]
        ml = np.maximum(conc_ml[i, dyes], 0.0)
        total = max(conc_ml[i].clip(min=0).sum(), 0.0)
        ml = ml * (total / ml.sum()) if ml.sum() > 0 else phys
        P[i, dyes] = alpha * phys + (1 - alpha) * ml
    return P


def ml_raw(data: Data, tr: np.ndarray, va: np.ndarray, group: str = "spectral_substrate") -> np.ndarray:
    X = data.X(group)
    est = make_pipeline(SimpleImputer(strategy="median"),
                        ExtraTreesRegressor(n_estimators=120, max_features=1.0, min_samples_leaf=3, random_state=SEED, n_jobs=-1))
    return est.fit(X[tr], data.Y[tr]).predict(X[va])


ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def exp_e04(data: Data) -> list[dict]:
    """Concentration source for the hybrid's chosen dye set: physics, ML, or an inner-CV blend."""
    results = []
    for proto in PROTOCOLS:
        rf_idx = row_fold_index(data, proto)
        km = {f: make_km(data, tr, KM_DEFAULT) for f, (tr, _) in enumerate(data.folds[proto])}
        preds = {k: np.zeros_like(data.Y) for k in ("hybrid_conc_physics", "hybrid_conc_ml", "hybrid_conc_blend")}
        log = []
        for f, (tr, va) in enumerate(data.folds[proto]):
            km_kw = select_km_inner(data, proto, tr)
            splits = inner_splits(data, proto, tr)
            inner = [build_block(data, proto, itr, iva, km_kw, "spectral_substrate") for itr, iva in splits]
            outer = build_block(data, proto, tr, va, km_kw, "spectral_substrate")
            cfg = choose_cfg(data, inner, VARIANTS["hybrid"])
            inner_ml = [ml_raw(data, itr, iva) for itr, iva in splits]
            alpha_err = {}
            for a in ALPHAS:
                errs = []
                for b, ml in zip(inner, inner_ml):
                    choice = np.argmin(score_block(b, cfg), axis=1)
                    errs.append(row_metrics(data.Y[b["va"]], chosen_vectors(data, b, choice, ml, a))["union_mae"].to_numpy())
                alpha_err[a] = float(np.mean(np.concatenate(errs)))
            alpha = min(alpha_err, key=alpha_err.get)
            choice = np.argmin(score_block(outer, cfg), axis=1)
            ml = ml_raw(data, tr, va)
            preds["hybrid_conc_physics"][va] = chosen_vectors(data, outer, choice, ml, 1.0)
            preds["hybrid_conc_ml"][va] = chosen_vectors(data, outer, choice, ml, 0.0)
            preds["hybrid_conc_blend"][va] = chosen_vectors(data, outer, choice, ml, alpha)
            log.append({"protocol": proto, "fold": f, "alpha": alpha, **{f"inner_umae_a{a}": e for a, e in alpha_err.items()}})
            print(f"  [E04/{proto}] fold {f}: alpha={alpha}")
        pd.DataFrame(log).to_csv(OUT_DIR / f"E04_{proto}_alpha.csv", index=False)
        for name, P in preds.items():
            results.append(record("E04", name, proto, row_metrics(data.Y, P, data, km, rf_idx), P, data))
    print_strata("E04")
    return results


# ------------------------------------------------- E05: learned ranking over candidate recipes

FEATURES_PHYSICS = ["de", "log_de", "de_hinge2"]
FEATURES_PRIOR = ["set_logp", "is_hist", "rf_logit_sum", "rf_min_logit", "ml_share", "ml_missing", "ml_mean_logc",
                  "card_logp", "size"]


def candidate_features(b: dict, ml: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """(n_rows, n_candidates, n_features) feature tensor for the choice model."""
    mask = b["cand_mask"].astype(float)  # (n_c, D)
    sizes = mask.sum(1)
    mlc = np.maximum(ml, 0.0)
    captured = mlc @ mask.T
    total = mlc.sum(1, keepdims=True)
    p = np.clip(b["prob"], 1e-3, 1 - 1e-3)
    logit = np.log(p / (1 - p))
    min_logit = np.where(mask[None] > 0, logit[:, None, :], np.inf).min(-1)
    n_va, n_c = b["de"].shape
    bc = lambda v: np.broadcast_to(v[None, :], (n_va, n_c))  # noqa: E731
    feats = {
        "de": b["de"], "log_de": np.log(b["de"] + 0.1), "de_hinge2": np.maximum(b["de"] - 2.0, 0.0),
        "set_logp": bc(b["set_logp"]), "is_hist": bc((b["counts"] > 0).astype(float)),
        "rf_logit_sum": b["logit_sum"], "rf_min_logit": min_logit,
        "ml_share": captured / (total + 1e-6), "ml_missing": total - captured,
        "ml_mean_logc": (np.log(mlc + 1e-3) @ mask.T) / sizes[None, :],
        "card_logp": bc(b["card_logp"]), "size": bc(sizes),
    }
    ml_mask = (feasible(ml) > ACTIVE).astype(float)  # the ML model's own top-3 set per row
    inter = ml_mask @ mask.T
    union = ml_mask.sum(1, keepdims=True) + sizes[None, :] - inter
    feats["ml_jaccard"] = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    feats["is_ml_top3"] = (feats["ml_jaccard"] >= 1 - 1e-9).astype(float)
    names = list(feats)
    return np.stack([feats[k] for k in names], -1), names


def ml_top_sets(ml: np.ndarray) -> list[tuple]:
    return [tuple(np.flatnonzero(p > ACTIVE)) for p in feasible(ml)]


def run_ranker2(data: Data, proto: str, use: list[str], tag: str) -> tuple[dict[str, np.ndarray], list[dict]]:
    """E05 ranker plus: each row's ML top-3 set added as a candidate, ML-overlap features, and
    concentrations blended between physics and ML with alpha chosen on inner folds."""
    preds = {k: np.zeros_like(data.Y) for k in ("blend", "conc_physics", "conc_ml")}
    log = []
    for f, (tr, va) in enumerate(data.folds[proto]):
        km_kw = select_km_inner(data, proto, tr)
        splits = inner_splits(data, proto, tr)
        inner_ml = [ml_raw(data, itr, iva) for itr, iva in splits]
        inner = [build_block(data, proto, itr, iva, km_kw, "spectral_substrate", extra_sets=ml_top_sets(ml))
                 for (itr, iva), ml in zip(splits, inner_ml)]
        inner_F = []
        for b, ml in zip(inner, inner_ml):
            F, names = candidate_features(b, ml)
            inner_F.append(F[..., [names.index(u) for u in use]])
        flat = np.concatenate([F.reshape(-1, F.shape[-1]) for F in inner_F])
        mu, sd = flat.mean(0), flat.std(0) + 1e-9
        inner_F = [(F - mu) / sd for F in inner_F]
        l2_score, l2_choices = {}, {}
        for l2 in (0.01, 0.1, 1.0, 10.0, 100.0):
            f1s, choices = [], []
            for h in range(len(inner)):
                w = fit_choice_model([(inner_F[j], _positive(inner[j])) for j in range(len(inner)) if j != h], l2)
                choice = (inner_F[h] @ w).argmax(1)
                f1s.append(inner[h]["f1"][np.arange(len(choice)), choice])
                choices.append(choice)
            l2_score[l2], l2_choices[l2] = float(np.mean(np.concatenate(f1s))), choices
        l2 = max(l2_score, key=l2_score.get)
        alpha_err = {}
        for a in ALPHAS:
            errs = [row_metrics(data.Y[b["va"]], chosen_vectors(data, b, c, ml, a))["union_mae"].to_numpy()
                    for b, c, ml in zip(inner, l2_choices[l2], inner_ml)]
            alpha_err[a] = float(np.mean(np.concatenate(errs)))
        alpha = min(alpha_err, key=alpha_err.get)
        w = fit_choice_model([(F, _positive(b)) for F, b in zip(inner_F, inner)], l2)
        outer_ml = ml_raw(data, tr, va)
        outer = build_block(data, proto, tr, va, km_kw, "spectral_substrate", extra_sets=ml_top_sets(outer_ml))
        F, names = candidate_features(outer, outer_ml)
        choice = (((F[..., [names.index(u) for u in use]]) - mu) / sd @ w).argmax(1)
        preds["blend"][va] = chosen_vectors(data, outer, choice, outer_ml, alpha)
        preds["conc_physics"][va] = chosen_vectors(data, outer, choice, outer_ml, 1.0)
        preds["conc_ml"][va] = chosen_vectors(data, outer, choice, outer_ml, 0.0)
        log.append({"protocol": proto, "fold": f, "variant": tag, "km": km_kw["_name"], "l2": l2, "alpha": alpha,
                    "inner_f1": l2_score[l2], **{f"umae_a{a}": v for a, v in alpha_err.items()}, **dict(zip(use, w))})
        print(f"  [E06/{proto}/{tag}] fold {f}: l2={l2} alpha={alpha} inner F1={l2_score[l2]:.3f}")
    return preds, log


ML_FEATURES = ["ml_jaccard", "is_ml_top3"]
E06_VARIANTS = {
    "ranker2_all": FEATURES_PHYSICS + FEATURES_PRIOR + ML_FEATURES,
    "ranker2_no_physics": FEATURES_PRIOR + ML_FEATURES,
}


def exp_e06(data: Data) -> list[dict]:
    """Ranker with ML candidate sets, overlap features and blended concentrations; nested CV."""
    results, logs = [], []
    for proto in PROTOCOLS:
        rf_idx = row_fold_index(data, proto)
        km = {f: make_km(data, tr, KM_DEFAULT) for f, (tr, _) in enumerate(data.folds[proto])}
        for name, use in E06_VARIANTS.items():
            preds, log = run_ranker2(data, proto, use, name)
            logs += log
            for suffix, P in preds.items():
                results.append(record("E06", f"{name}__{suffix}", proto, row_metrics(data.Y, P, data, km, rf_idx), P, data))
    pd.DataFrame(logs).to_csv(OUT_DIR / "E06_selected.csv", index=False)
    print_strata("E06")
    return results


def _positive(b: dict) -> np.ndarray:
    """Training target: the candidate with the best F1 for each row (ties: lowest union MAE); -1 if none overlaps."""
    key = b["f1"] - 1e-6 * b["umae"]
    pos = key.argmax(1)
    pos[b["f1"].max(1) <= 0] = -1
    return pos


def fit_choice_model(blocks: list[tuple[np.ndarray, np.ndarray]], l2: float) -> np.ndarray:
    """Conditional logit: P(candidate c | row) ∝ exp(F_c · w). Features are pre-standardised."""
    from scipy.optimize import minimize
    from scipy.special import logsumexp

    k = blocks[0][0].shape[-1]

    def nll(w: np.ndarray) -> tuple[float, np.ndarray]:
        total, grad = l2 * float(w @ w), 2 * l2 * w
        for F, pos in blocks:
            ok = pos >= 0
            Fo, po = F[ok], pos[ok]
            s = Fo @ w
            lse = logsumexp(s, axis=1)
            total += float((lse - s[np.arange(len(po)), po]).sum())
            prob = np.exp(s - lse[:, None])
            grad += (prob[..., None] * Fo).sum((0, 1)) - Fo[np.arange(len(po)), po].sum(0)
        return total, grad

    return minimize(nll, np.zeros(k), jac=True, method="L-BFGS-B").x


def fit_choice_model_ef1(blocks: list[tuple[np.ndarray, np.ndarray]], l2: float, w0: np.ndarray) -> np.ndarray:
    """Maximise expected F1 under the softmax choice distribution (aligns training with the metric)."""
    from scipy.optimize import minimize
    from scipy.special import softmax

    def obj(w: np.ndarray) -> tuple[float, np.ndarray]:
        total, grad = l2 * float(w @ w), 2 * l2 * w
        for F, f1 in blocks:
            p = softmax(F @ w, axis=1)
            ef1 = (p * f1).sum(1)
            total -= float(ef1.sum())
            mean_f = (p[..., None] * F).sum(1)
            grad -= ((p * f1)[..., None] * (F - mean_f[:, None, :])).sum((0, 1))
        return total, grad

    return minimize(obj, w0, jac=True, method="L-BFGS-B").x


BUNDLE_VERSIONS = {
    "v1": None,  # KM_GRID
    "v2_bandmask": {k: {**v, "band_mask": True} for k, v in KM_GRID.items()},
}
BUNDLE_VERSION = "v1"


def fold_bundle(data: Data, proto: str, f: int, version: str | None = None) -> dict:
    """KM choice, inner/outer candidate blocks and ML predictions for one outer fold (cached)."""
    import pickle

    version = version or BUNDLE_VERSION
    path = CACHE_DIR / f"bundle_{version}__{proto}__fold{f}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    tr, va = data.folds[proto][f]
    km_kw = select_km_inner(data, proto, tr, BUNDLE_VERSIONS[version])
    splits = inner_splits(data, proto, tr)
    inner_ml = [ml_raw(data, itr, iva) for itr, iva in splits]
    inner = [build_block(data, proto, itr, iva, km_kw, "spectral_substrate", extra_sets=ml_top_sets(ml))
             for (itr, iva), ml in zip(splits, inner_ml)]
    outer_ml = ml_raw(data, tr, va)
    outer = build_block(data, proto, tr, va, km_kw, "spectral_substrate", extra_sets=ml_top_sets(outer_ml))
    bundle = {"km": km_kw["_name"], "inner": inner, "inner_ml": inner_ml, "outer": outer, "outer_ml": outer_ml}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(bundle))
    return bundle


def run_ranker3(data: Data, proto: str, use: list[str], tag: str, objective: str = "ef1") -> tuple[dict[str, np.ndarray], list[dict]]:
    """As run_ranker2, on cached bundles, with the expected-F1 training objective."""
    preds = {k: np.zeros_like(data.Y) for k in ("blend", "conc_physics", "conc_ml")}
    log = []
    for f, (tr, va) in enumerate(data.folds[proto]):
        bd = fold_bundle(data, proto, f)
        inner, inner_ml = bd["inner"], bd["inner_ml"]
        inner_F = []
        for b, ml in zip(inner, inner_ml):
            F, names = candidate_features(b, ml)
            inner_F.append(F[..., [names.index(u) for u in use]])
        flat = np.concatenate([F.reshape(-1, F.shape[-1]) for F in inner_F])
        mu, sd = flat.mean(0), flat.std(0) + 1e-9
        inner_F = [(F - mu) / sd for F in inner_F]

        def fit(idx: list[int], l2: float) -> np.ndarray:
            w0 = fit_choice_model([(inner_F[j], _positive(inner[j])) for j in idx], l2)
            if objective == "likelihood":
                return w0
            return fit_choice_model_ef1([(inner_F[j], inner[j]["f1"]) for j in idx], l2, w0)

        l2_score, l2_choices = {}, {}
        for l2 in (0.01, 0.1, 1.0, 10.0, 100.0):
            f1s, choices = [], []
            for h in range(len(inner)):
                w = fit([j for j in range(len(inner)) if j != h], l2)
                choice = (inner_F[h] @ w).argmax(1)
                f1s.append(inner[h]["f1"][np.arange(len(choice)), choice])
                choices.append(choice)
            l2_score[l2], l2_choices[l2] = float(np.mean(np.concatenate(f1s))), choices
        l2 = max(l2_score, key=l2_score.get)
        alpha_err = {}
        for a in ALPHAS:
            errs = [row_metrics(data.Y[b["va"]], chosen_vectors(data, b, c, ml, a))["union_mae"].to_numpy()
                    for b, c, ml in zip(inner, l2_choices[l2], inner_ml)]
            alpha_err[a] = float(np.mean(np.concatenate(errs)))
        alpha = min(alpha_err, key=alpha_err.get)
        w = fit(list(range(len(inner))), l2)
        outer, outer_ml = bd["outer"], bd["outer_ml"]
        F, names = candidate_features(outer, outer_ml)
        choice = (((F[..., [names.index(u) for u in use]]) - mu) / sd @ w).argmax(1)
        preds["blend"][va] = chosen_vectors(data, outer, choice, outer_ml, alpha)
        preds["conc_physics"][va] = chosen_vectors(data, outer, choice, outer_ml, 1.0)
        preds["conc_ml"][va] = chosen_vectors(data, outer, choice, outer_ml, 0.0)
        log.append({"protocol": proto, "fold": f, "variant": tag, "objective": objective, "km": bd["km"], "l2": l2,
                    "alpha": alpha, "inner_f1": l2_score[l2], **dict(zip(use, w))})
        print(f"  [{tag}/{proto}] fold {f}: l2={l2} alpha={alpha} inner F1={l2_score[l2]:.3f}")
    return preds, log


def exp_e07(data: Data) -> list[dict]:
    """E06 with the ranker trained on expected F1 instead of best-candidate likelihood."""
    results, logs = [], []
    for proto in PROTOCOLS:
        rf_idx = row_fold_index(data, proto)
        km = {f: make_km(data, tr, KM_DEFAULT) for f, (tr, _) in enumerate(data.folds[proto])}
        for name, use in {"ef1_all": E06_VARIANTS["ranker2_all"], "ef1_no_physics": E06_VARIANTS["ranker2_no_physics"]}.items():
            preds, log = run_ranker3(data, proto, use, name)
            logs += log
            for suffix, P in preds.items():
                results.append(record("E07", f"{name}__{suffix}", proto, row_metrics(data.Y, P, data, km, rf_idx), P, data))
    pd.DataFrame(logs).to_csv(OUT_DIR / "E07_selected.csv", index=False)
    print_strata("E07")
    return results


# --------------------------------------- E08: KM-simulated augmentation for novel combinations


def raw_inputs(data: Data, rows: np.ndarray, ks: np.ndarray | None = None) -> np.ndarray:
    """Raw reflectance + K/S + substrate variables. ks overrides the spectra for synthetic rows."""
    ks = data.KS[rows] if ks is None else ks
    sub = data.df[[c for c in data.features["spectral_substrate"] if not c.startswith(("target_", "feat_refl_", "feat_ks_"))]]
    sub = sub.apply(pd.to_numeric, errors="coerce").to_numpy(float)[rows]
    return np.hstack([ks_to_reflectance(ks), ks, sub])


def synthesize(data: Data, tr: np.ndarray, model, min_rows: int = 3, depths=(0.7, 1.4)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic (parent row, recipe, K/S) by within-hue dye swaps and depth changes of real training recipes.

    The synthetic spectrum is the parent's measured K/S plus the KM-predicted change, so the KM
    model only has to be right about differences, not absolute colour.
    """
    A = data.Y > ACTIVE
    counts = A[tr].sum(0)
    parents, recipes = [], []
    for r in tr:
        dyes = np.flatnonzero(A[r])
        for d_old in dyes:
            for d_new in np.flatnonzero((counts >= min_rows) & (data.dye_hue == data.dye_hue[d_old])):
                if d_new in dyes:
                    continue
                c = data.Y[r].copy()
                c[d_new], c[d_old] = c[d_old], 0.0
                parents.append(r)
                recipes.append(c)
        for s in depths:
            parents.append(r)
            recipes.append(np.minimum(data.Y[r] * s, data.Y[tr].max()))
    parents, recipes = np.array(parents), np.array(recipes)
    ks = data.KS[parents] + model.predict_ks(recipes) - model.predict_ks(data.Y[parents])
    return parents, recipes, np.maximum(ks, 1e-4)


def et_fit_predict(X_tr, Y_tr, X_va, weights=None) -> np.ndarray:
    est = make_pipeline(SimpleImputer(strategy="median"),
                        ExtraTreesRegressor(n_estimators=120, max_features=1.0, min_samples_leaf=3, random_state=SEED, n_jobs=-1))
    est.fit(X_tr, Y_tr, extratreesregressor__sample_weight=weights)
    return est.predict(X_va)


def augmented_predict(data: Data, tr: np.ndarray, va: np.ndarray, w_syn: float, km_kw: dict) -> np.ndarray:
    X_real, Y_real = raw_inputs(data, tr), data.Y[tr]
    if w_syn <= 0:
        return et_fit_predict(X_real, Y_real, raw_inputs(data, va))
    model = make_km(data, tr, km_kw)
    parents, recipes, ks = synthesize(data, tr, model)
    X_syn = raw_inputs(data, parents, ks)
    # Scale so the synthetic rows carry w_syn of the real rows' total weight.
    wt = np.concatenate([np.ones(len(tr)), np.full(len(parents), w_syn * len(tr) / len(parents))])
    return et_fit_predict(np.vstack([X_real, X_syn]), np.vstack([Y_real, recipes]), raw_inputs(data, va), wt)


W_SYN = (0.0, 0.1, 0.25, 0.5, 1.0)


def exp_e08(data: Data) -> list[dict]:
    """ET inverse trained on real + KM-simulated recipes; synthetic weight chosen by inner CV (F1)."""
    results, log = [], []
    for proto in PROTOCOLS:
        rf_idx = row_fold_index(data, proto)
        km = {f: make_km(data, tr, KM_DEFAULT) for f, (tr, _) in enumerate(data.folds[proto])}
        preds = {k: np.zeros_like(data.Y) for k in ("aug_et_inner_cv", "aug_et_w0_raw_inputs", "aug_et_w0.5_fixed")}
        for f, (tr, va) in enumerate(data.folds[proto]):
            score = {}
            for w in W_SYN:
                f1s = []
                for itr, iva in inner_splits(data, proto, tr):
                    P = feasible(augmented_predict(data, itr, iva, w, KM_DEFAULT))
                    f1s.append(row_metrics(data.Y[iva], P)["f1"].to_numpy())
                score[w] = float(np.mean(np.concatenate(f1s)))
            w_best = max(score, key=score.get)
            preds["aug_et_inner_cv"][va] = feasible(augmented_predict(data, tr, va, w_best, KM_DEFAULT))
            preds["aug_et_w0_raw_inputs"][va] = feasible(augmented_predict(data, tr, va, 0.0, KM_DEFAULT))
            preds["aug_et_w0.5_fixed"][va] = feasible(augmented_predict(data, tr, va, 0.5, KM_DEFAULT))
            log.append({"protocol": proto, "fold": f, "w_syn": w_best, **{f"inner_f1_w{k}": v for k, v in score.items()}})
            print(f"  [E08/{proto}] fold {f}: w_syn={w_best} inner F1 by w: {score}")
        for name, P in preds.items():
            results.append(record("E08", name, proto, row_metrics(data.Y, P, data, km, rf_idx), P, data))
    pd.DataFrame(log).to_csv(OUT_DIR / "E08_selected.csv", index=False)
    print_strata("E08")
    return results


# --------------------------------------------- E09: confidence gate between ranker and ML


def _ranker_probs(F: np.ndarray, w: np.ndarray) -> np.ndarray:
    from scipy.special import softmax

    return softmax(F @ w, axis=1)


TAUS = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01)


def run_gate(data: Data, proto: str, use: list[str], tag: str) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Use the ranker's recipe when its top choice probability >= tau, else the ML top-3 recipe.

    tau (and the ranker's L2 and concentration blend alpha) are chosen on inner folds only.
    tau = 0 always trusts the ranker; tau > 1 always uses ML.
    """
    preds = {k: np.zeros_like(data.Y) for k in ("gate", "gate_ranker_part_only")}
    log = []
    for f, (tr, va) in enumerate(data.folds[proto]):
        bd = fold_bundle(data, proto, f)
        inner, inner_ml = bd["inner"], bd["inner_ml"]
        inner_F = []
        for b, ml in zip(inner, inner_ml):
            F, names = candidate_features(b, ml)
            inner_F.append(F[..., [names.index(u) for u in use]])
        flat = np.concatenate([F.reshape(-1, F.shape[-1]) for F in inner_F])
        mu, sd = flat.mean(0), flat.std(0) + 1e-9
        inner_F = [(F - mu) / sd for F in inner_F]
        # Leave-one-inner-fold-out ranker choices and confidences for each L2.
        l2_out = {}
        for l2 in (0.1, 1.0, 10.0, 100.0):
            parts = []
            for h in range(len(inner)):
                w = fit_choice_model([(inner_F[j], _positive(inner[j])) for j in range(len(inner)) if j != h], l2)
                prob = _ranker_probs(inner_F[h], w)
                parts.append((prob.argmax(1), prob.max(1)))
            l2_out[l2] = parts
        best = None
        for l2, parts in l2_out.items():
            for a in ALPHAS:
                for tau in TAUS:
                    f1s, umaes = [], []
                    for b, ml, (choice, conf) in zip(inner, inner_ml, parts):
                        Pr = chosen_vectors(data, b, choice, ml, a)
                        Pm = feasible(ml)
                        P = np.where((conf >= tau)[:, None], Pr, Pm)
                        m = row_metrics(data.Y[b["va"]], P)
                        f1s.append(m["f1"].to_numpy())
                        umaes.append(m["union_mae"].to_numpy())
                    key = (round(float(np.mean(np.concatenate(f1s))), 6), -float(np.mean(np.concatenate(umaes))))
                    if best is None or key > best[0]:
                        best = (key, l2, a, tau)
        _, l2, alpha, tau = best
        w = fit_choice_model([(F, _positive(b)) for F, b in zip(inner_F, inner)], l2)
        outer, outer_ml = bd["outer"], bd["outer_ml"]
        F, names = candidate_features(outer, outer_ml)
        prob = _ranker_probs((F[..., [names.index(u) for u in use]] - mu) / sd, w)
        choice, conf = prob.argmax(1), prob.max(1)
        Pr = chosen_vectors(data, outer, choice, outer_ml, alpha)
        Pm = feasible(outer_ml)
        use_ranker = conf >= tau
        preds["gate"][va] = np.where(use_ranker[:, None], Pr, Pm)
        preds["gate_ranker_part_only"][va] = Pr
        log.append({"protocol": proto, "fold": f, "variant": tag, "l2": l2, "alpha": alpha, "tau": tau,
                    "inner_f1": best[0][0], "share_ranker_outer": float(use_ranker.mean())})
        print(f"  [{tag}/{proto}] fold {f}: l2={l2} alpha={alpha} tau={tau} inner F1={best[0][0]:.3f} "
              f"ranker used on {use_ranker.mean():.0%} of outer rows")
    return preds, log


def exp_e09(data: Data) -> list[dict]:
    """Confidence gate: physics-informed ranker when confident, ML top-3 otherwise; nested CV."""
    results, logs = [], []
    for proto in PROTOCOLS:
        rf_idx = row_fold_index(data, proto)
        km = {f: make_km(data, tr, KM_DEFAULT) for f, (tr, _) in enumerate(data.folds[proto])}
        for name, use in {"gate_all": E06_VARIANTS["ranker2_all"], "gate_no_physics": E06_VARIANTS["ranker2_no_physics"]}.items():
            preds, log = run_gate(data, proto, use, name)
            logs += log
            for suffix, P in preds.items():
                results.append(record("E09", f"{name}__{suffix}", proto, row_metrics(data.Y, P, data, km, rf_idx), P, data))
    pd.DataFrame(logs).to_csv(OUT_DIR / "E09_selected.csv", index=False)
    print_strata("E09")
    return results


# ----------------------------------- E10: ranked shortlists (post hoc metric, fixed L2)

E10_VARIANTS = {
    "all": FEATURES_PHYSICS + FEATURES_PRIOR + ML_FEATURES,
    "no_physics": FEATURES_PRIOR + ML_FEATURES,
    "ml_only": ["ml_share", "ml_missing", "ml_mean_logc"] + ML_FEATURES,
    "physics_only": FEATURES_PHYSICS,
    "set_prior_only": ["set_logp", "is_hist", "card_logp", "size"],
    "physics_set_prior": FEATURES_PHYSICS + ["set_logp", "is_hist", "card_logp", "size"],
}
KS_LIST = (1, 3, 5)


def exp_e10(data: Data, l2: float = 10.0) -> list[dict]:
    """Hit@k and best-of-k F1 of ranked candidate shortlists. L2 fixed in advance, no inner selection."""
    A = data.Y > ACTIVE
    rows = []
    for proto in PROTOCOLS:
        strata = data.strata(proto)
        Xs = np.nan_to_num(data.X("spectral"))
        for f, (tr, va) in enumerate(data.folds[proto]):
            bd = fold_bundle(data, proto, f)
            inner, inner_ml, outer, outer_ml = bd["inner"], bd["inner_ml"], bd["outer"], bd["outer_ml"]
            F_out, names = candidate_features(outer, outer_ml)
            true_idx = np.array([next((c for c, s in enumerate(outer["subsets"]) if tuple(s[s >= 0]) == tuple(np.flatnonzero(A[r]))), -1)
                                 for r in va])
            for name, use in E10_VARIANTS.items():
                cols = [names.index(u) for u in use]
                inner_F = [candidate_features(b, ml)[0][..., cols] for b, ml in zip(inner, inner_ml)]
                flat = np.concatenate([F.reshape(-1, len(cols)) for F in inner_F])
                mu, sd = flat.mean(0), flat.std(0) + 1e-9
                w = fit_choice_model([((F - mu) / sd, _positive(b)) for F, b in zip(inner_F, inner)], l2)
                order = np.argsort(-((F_out[..., cols] - mu) / sd @ w), axis=1)
                for i, r in enumerate(va):
                    rec = {"protocol": proto, "method": f"ranker_{name}", "row": int(r), "stratum": int(strata[r])}
                    for k in KS_LIST:
                        top = order[i, :k]
                        rec[f"hit@{k}"] = float(true_idx[i] in top)
                        rec[f"bestf1@{k}"] = float(outer["f1"][i, top].max())
                    rows.append(rec)
            # Baseline: distinct dye sets of the nearest training shades by spectrum (historical retrieval).
            sc = StandardScaler().fit(Xs[tr])
            dist = ((sc.transform(Xs[va])[:, None, :] - sc.transform(Xs[tr])[None]) ** 2).sum(-1)
            for i, r in enumerate(va):
                sets = []
                for j in tr[np.argsort(dist[i])]:
                    s = tuple(np.flatnonzero(A[j]))
                    if s not in sets:
                        sets.append(s)
                    if len(sets) == max(KS_LIST):
                        break
                truth = set(np.flatnonzero(A[r]))
                f1s = [2 * len(truth & set(s)) / (len(truth) + len(s)) if (truth or s) else 0.0 for s in sets]
                rec = {"protocol": proto, "method": "knn_historical_sets", "row": int(r), "stratum": int(strata[r])}
                for k in KS_LIST:
                    rec[f"hit@{k}"] = float(tuple(sorted(truth)) in sets[:k])
                    rec[f"bestf1@{k}"] = float(max(f1s[:k]))
                rows.append(rec)
        print(f"  [E10/{proto}] done")
    t = pd.DataFrame(rows)
    t.to_csv(OUT_DIR / "E10_shortlist_rows.csv", index=False)
    metric_cols = [f"hit@{k}" for k in KS_LIST] + [f"bestf1@{k}" for k in KS_LIST]
    for proto in PROTOCOLS:
        s = t[t.protocol == proto]
        print(f"--- E10 {proto}: all rows")
        print(s.groupby("method")[metric_cols].mean().round(3).sort_values("hit@3", ascending=False).to_string())
        print(f"--- E10 {proto}: by stratum (hit@3 / bestf1@3)")
        print(s.groupby(["stratum", "method"])[["hit@1", "hit@3", "bestf1@1", "bestf1@3"]].mean().round(3).to_string())
    return []


# --------------------------- E11: fixed-hyperparameter ranker, full recipe output + sensitivity


def run_fixed_ranker(data: Data, proto: str, use: list[str], l2: float, alpha: float) -> np.ndarray:
    """Top-1 recipe from the choice model with L2 and the physics/ML concentration blend fixed in advance."""
    P = np.zeros_like(data.Y)
    for f, (tr, va) in enumerate(data.folds[proto]):
        bd = fold_bundle(data, proto, f)
        F_out, names = candidate_features(bd["outer"], bd["outer_ml"])
        cols = [names.index(u) for u in use]
        inner_F = [candidate_features(b, ml)[0][..., cols] for b, ml in zip(bd["inner"], bd["inner_ml"])]
        flat = np.concatenate([F.reshape(-1, len(cols)) for F in inner_F])
        mu, sd = flat.mean(0), flat.std(0) + 1e-9
        w = fit_choice_model([((F - mu) / sd, _positive(b)) for F, b in zip(inner_F, bd["inner"])], l2)
        choice = ((F_out[..., cols] - mu) / sd @ w).argmax(1)
        P[va] = chosen_vectors(data, bd["outer"], choice, bd["outer_ml"], alpha)
    return P


def exp_e11(data: Data) -> list[dict]:
    """Fixed settings (L2 = 10, alpha = 0.5) with sensitivity over L2 and alpha, and the no-physics ablation."""
    results = []
    settings = {
        "fixed_all_l2_10_a0.5": ("all", 10.0, 0.5),
        "fixed_no_physics_l2_10_a0.5": ("no_physics", 10.0, 0.5),
        "fixed_all_l2_1_a0.5": ("all", 1.0, 0.5),
        "fixed_all_l2_100_a0.5": ("all", 100.0, 0.5),
        "fixed_no_physics_l2_1_a0.5": ("no_physics", 1.0, 0.5),
        "fixed_no_physics_l2_100_a0.5": ("no_physics", 100.0, 0.5),
        "fixed_all_l2_10_a1.0": ("all", 10.0, 1.0),
        "fixed_all_l2_10_a0.0": ("all", 10.0, 0.0),
    }
    for proto in PROTOCOLS:
        rf_idx = row_fold_index(data, proto)
        km = {f: make_km(data, tr, KM_DEFAULT) for f, (tr, _) in enumerate(data.folds[proto])}
        for name, (feat, l2, alpha) in settings.items():
            P = run_fixed_ranker(data, proto, E10_VARIANTS[feat], l2, alpha)
            results.append(record("E11", name, proto, row_metrics(data.Y, P, data, km, rf_idx), P, data))
    print_strata("E11", cols=("n", "f1", "exact_set", "union_mae", "family_mae", "total_abs_err"))
    return results


# ---------------- E12: nested ranker with L2 chosen by held-out log-likelihood (smooth criterion)

L2_GRID = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


def heldout_nll(F: np.ndarray, pos: np.ndarray, w: np.ndarray) -> float:
    from scipy.special import logsumexp

    ok = pos >= 0
    s = F[ok] @ w
    return float((logsumexp(s, axis=1) - s[np.arange(ok.sum()), pos[ok]]).mean())


def run_nested_nll(data: Data, proto: str, use: list[str], tag: str) -> tuple[np.ndarray, list[dict]]:
    """L2 by leave-one-inner-fold-out held-out NLL; alpha by inner union MAE at that L2. Outer fold untouched."""
    P = np.zeros_like(data.Y)
    log = []
    for f, (tr, va) in enumerate(data.folds[proto]):
        bd = fold_bundle(data, proto, f)
        inner, inner_ml = bd["inner"], bd["inner_ml"]
        F_out, names = candidate_features(bd["outer"], bd["outer_ml"])
        cols = [names.index(u) for u in use]
        inner_F = [candidate_features(b, ml)[0][..., cols] for b, ml in zip(inner, inner_ml)]
        flat = np.concatenate([F.reshape(-1, len(cols)) for F in inner_F])
        mu, sd = flat.mean(0), flat.std(0) + 1e-9
        inner_F = [(F - mu) / sd for F in inner_F]
        pos = [_positive(b) for b in inner]
        nll, choices = {}, {}
        for l2 in L2_GRID:
            vals, ch = [], []
            for h in range(len(inner)):
                w = fit_choice_model([(inner_F[j], pos[j]) for j in range(len(inner)) if j != h], l2)
                vals.append(heldout_nll(inner_F[h], pos[h], w))
                ch.append((inner_F[h] @ w).argmax(1))
            nll[l2], choices[l2] = float(np.mean(vals)), ch
        l2 = min(nll, key=nll.get)
        alpha_err = {a: float(np.mean(np.concatenate([
            row_metrics(data.Y[b["va"]], chosen_vectors(data, b, c, ml, a))["union_mae"].to_numpy()
            for b, c, ml in zip(inner, choices[l2], inner_ml)]))) for a in ALPHAS}
        alpha = min(alpha_err, key=alpha_err.get)
        w = fit_choice_model(list(zip(inner_F, pos)), l2)
        choice = ((F_out[..., cols] - mu) / sd @ w).argmax(1)
        P[va] = chosen_vectors(data, bd["outer"], choice, bd["outer_ml"], alpha)
        log.append({"protocol": proto, "fold": f, "variant": tag, "l2": l2, "alpha": alpha,
                    **{f"nll_l2_{k}": v for k, v in nll.items()}, **dict(zip(use, w))})
        print(f"  [E12/{proto}/{tag}] fold {f}: l2={l2} alpha={alpha} held-out NLL by L2: "
              + " ".join(f"{k}:{v:.3f}" for k, v in nll.items()))
    return P, log


def exp_e12(data: Data) -> list[dict]:
    """Fully nested ranker (L2 by held-out NLL, alpha by inner union MAE), with the no-physics ablation."""
    results, logs = [], []
    for proto in PROTOCOLS:
        rf_idx = row_fold_index(data, proto)
        km = {f: make_km(data, tr, KM_DEFAULT) for f, (tr, _) in enumerate(data.folds[proto])}
        for name in ("all", "no_physics"):
            P, log = run_nested_nll(data, proto, E10_VARIANTS[name], name)
            logs += log
            results.append(record("E12", f"nested_nll_{name}", proto, row_metrics(data.Y, P, data, km, rf_idx), P, data))
    pd.DataFrame(logs).to_csv(OUT_DIR / "E12_selected.csv", index=False)
    print_strata("E12", cols=("n", "f1", "exact_set", "union_mae", "family_mae", "total_abs_err"))
    return results


# ------------- E13: external-review suggestions (local/pairwise priors, soft targets, L2 averaging)

SIGMA_E = 5.0  # CMC units, fixed by the external review before any run
GAMMA_SOFT = 6.0
L2_AVG = (1.0, 3.0, 10.0, 30.0, 100.0)


def prior_features(data: Data, b: dict, train_rows: np.ndarray) -> dict[str, np.ndarray]:
    """Compositional and colour-local set priors computed from the block's training rows only."""
    A = data.Y > ACTIVE
    At = A[train_rows].astype(float)
    n_i = At.sum(0)
    n_ij = At.T @ At
    mask = b["cand_mask"]
    n_c = len(mask)
    unary = np.array([np.log((n_i[m] + 0.5) / (len(train_rows) + 0.5)).mean() for m in mask])
    pair = np.zeros(n_c)
    for c, m in enumerate(mask):
        idx = np.flatnonzero(m)
        if len(idx) > 1:
            pair[c] = np.mean([np.log((n_ij[i, j] + 0.2) / np.sqrt((n_i[i] + 1) * (n_i[j] + 1)))
                               for a, i in enumerate(idx) for j in idx[a + 1:]])
    # Colour-local exact-set memory: historical uses of S weighted by CMC similarity of their target to the query.
    train_sets = [tuple(np.flatnonzero(A[r])) for r in train_rows]
    cand_index = {tuple(s[s >= 0]): c for c, s in enumerate(b["subsets"])}
    member = np.zeros((len(train_rows), n_c))
    for t, s in enumerate(train_sets):
        if s in cand_index:
            member[t, cand_index[s]] = 1.0
    local = np.zeros((len(b["va"]), n_c))
    for i, r in enumerate(b["va"]):
        de = delta_e_cmc(np.repeat(data.lab_d65[r][None], len(train_rows), 0), data.lab_d65[train_rows])
        local[i] = np.log(0.01 + np.exp(-de ** 2 / (2 * SIGMA_E ** 2)) @ member)
    bc = lambda v: np.broadcast_to(v[None, :], (len(b["va"]), n_c))  # noqa: E731
    return {"prior_unary": bc(unary), "prior_pair": bc(pair), "prior_local": local}


def block_features(data: Data, b: dict, ml: np.ndarray, train_rows: np.ndarray, use: list[str]) -> np.ndarray:
    F, names = candidate_features(b, ml)
    extra = prior_features(data, b, train_rows) if any(u.startswith("prior_") for u in use) else {}
    cols = [F[..., names.index(u)] if u in names else extra[u] for u in use]
    return np.stack(cols, -1)


def fit_soft_targets(blocks: list[tuple[np.ndarray, np.ndarray]], l2: float) -> np.ndarray:
    """Convex cross-entropy against soft targets q ∝ exp(gamma * F1) over each row's candidates."""
    from scipy.optimize import minimize
    from scipy.special import log_softmax, softmax

    prepared = []
    for F, f1 in blocks:
        ok = f1.max(1) > 0
        prepared.append((F[ok], softmax(GAMMA_SOFT * f1[ok], axis=1)))

    def obj(w: np.ndarray) -> tuple[float, np.ndarray]:
        total, grad = l2 * float(w @ w), 2 * l2 * w
        for F, q in prepared:
            logp = log_softmax(F @ w, axis=1)
            total -= float((q * logp).sum())
            grad += ((np.exp(logp) - q)[..., None] * F).sum((0, 1))
        return total, grad

    return minimize(obj, np.zeros(prepared[0][0].shape[-1]), jac=True, method="L-BFGS-B").x


def allowed_candidates(data: Data, b: dict, ml: np.ndarray, train_rows: np.ndarray) -> np.ndarray:
    """(n_rows, n_candidates) mask implementing Eq. 7 exactly: the training-fold pool (historical sets, their
    subsets, within-hue swaps) plus the row's OWN ML top-3 set. Cached blocks were built with the union of the ML
    sets of all rows of the block; this mask removes the other rows' sets (E16)."""
    A = data.Y > ACTIVE
    base, _ = candidate_sets(A[train_rows], data.dye_hue)
    base_keys = {tuple(int(d) for d in s[s >= 0]) for s in base}
    keys = [tuple(int(d) for d in s[s >= 0]) for s in b["subsets"]]
    index = {k: c for c, k in enumerate(keys)}
    in_base = np.array([k in base_keys for k in keys])
    allowed = np.repeat(in_base[None, :], len(ml), axis=0)
    for i, s in enumerate(ml_top_sets(ml)):
        key = tuple(sorted(int(d) for d in s))
        if key in index:
            allowed[i, index[key]] = True
    return allowed


def _positive_masked(b: dict, allowed: np.ndarray) -> np.ndarray:
    """As _positive, restricted to each row's allowed candidates."""
    key = np.where(allowed, b["f1"] - 1e-6 * b["umae"], -np.inf)
    pos = key.argmax(1)
    pos[np.where(allowed, b["f1"], 0.0).max(1) <= 0] = -1
    return pos


def _fix_set_logp(F: np.ndarray, use: list[str], b: dict, allowed: np.ndarray) -> np.ndarray:
    """P_hist (Eq. 9) normalised by each row's own candidate count |C_i|."""
    if "set_logp" in use:
        n_i = allowed.sum(1)[:, None]
        F = F.copy()
        F[..., use.index("set_logp")] = np.log((b["counts"][None, :] + 0.5) / (b["counts"].sum() + 0.5 * n_i))
    return F


def fit_choice_model_masked(blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray]], l2: float) -> np.ndarray:
    """Conditional logit over each row's allowed candidates only (softmax with disallowed scores at -inf)."""
    from scipy.optimize import minimize
    from scipy.special import logsumexp

    k = blocks[0][0].shape[-1]

    def nll(w: np.ndarray) -> tuple[float, np.ndarray]:
        total, grad = l2 * float(w @ w), 2 * l2 * w
        for F, pos, allowed in blocks:
            ok = pos >= 0
            Fo, po, ao = F[ok], pos[ok], allowed[ok]
            s = np.where(ao, Fo @ w, -np.inf)
            lse = logsumexp(s, axis=1)
            total += float((lse - s[np.arange(len(po)), po]).sum())
            prob = np.exp(s - lse[:, None])
            grad += (prob[..., None] * Fo).sum((0, 1)) - Fo[np.arange(len(po)), po].sum(0)
        return total, grad

    return minimize(nll, np.zeros(k), jac=True, method="L-BFGS-B").x


def fold_choice_probs(data: Data, proto: str, f: int, use: list[str]) -> tuple[dict, np.ndarray, np.ndarray]:
    """Final method for one outer fold (Eqs. 7-11): returns the bundle, the L2-averaged choice probabilities over
    each validation row's own candidates (0 elsewhere) and the allowed-candidate mask."""
    from scipy.special import softmax

    tr, va = data.folds[proto][f]
    bd = fold_bundle(data, proto, f)
    splits = inner_splits(data, proto, tr)
    inner = []
    for b, ml, (itr, _) in zip(bd["inner"], bd["inner_ml"], splits):
        al = allowed_candidates(data, b, ml, itr)
        inner.append((_fix_set_logp(block_features(data, b, ml, itr, use), use, b, al), _positive_masked(b, al), al))
    flat = np.concatenate([F[al] for F, _, al in inner])  # standardise over allowed candidates only
    mu, sd = flat.mean(0), flat.std(0) + 1e-9
    inner = [((F - mu) / sd, pos, al) for F, pos, al in inner]
    al_out = allowed_candidates(data, bd["outer"], bd["outer_ml"], tr)
    F_out = (_fix_set_logp(block_features(data, bd["outer"], bd["outer_ml"], tr, use), use, bd["outer"], al_out) - mu) / sd
    prob = np.zeros(F_out.shape[:2])
    for l2 in L2_AVG:
        w = fit_choice_model_masked(inner, l2)
        prob += softmax(np.where(al_out, F_out @ w, -np.inf), axis=1) / len(L2_AVG)
    return bd, prob, al_out


def run_averaged_ranker(data: Data, proto: str, use: list[str], soft: bool, alpha: float = 0.5,
                        own_ml_only: bool = True) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Ranker with probabilities averaged over a fixed L2 grid. Returns recipes, per-row Hit ranks, and blocks' order.

    own_ml_only=True (default, E16 onward) restricts every row to its own candidate set as in Eq. 7.
    own_ml_only=False reproduces E13/E15, where each row also saw the other validation rows' ML top-3 sets."""
    from scipy.special import softmax

    P = np.zeros_like(data.Y)
    hit_rank = np.full(len(data.Y), np.inf)
    if own_ml_only:
        if soft:
            raise NotImplementedError("soft targets are not used by the final method")
        A = data.Y > ACTIVE
        for f, (tr, va) in enumerate(data.folds[proto]):
            bd, prob, allowed = fold_choice_probs(data, proto, f, use)
            choice = prob.argmax(1)
            P[va] = chosen_vectors(data, bd["outer"], choice, bd["outer_ml"], alpha)
            subsets = bd["outer"]["subsets"]
            keys = [tuple(s[s >= 0]) for s in subsets]
            for i, r in enumerate(va):
                t = tuple(np.flatnonzero(A[r]))
                order = [c for c in np.argsort(-prob[i]) if allowed[i, c]]
                pos = [k for k, c in enumerate(order) if keys[c] == t]
                if pos:
                    hit_rank[r] = pos[0] + 1
        return P, hit_rank, []
    for f, (tr, va) in enumerate(data.folds[proto]):
        bd = fold_bundle(data, proto, f)
        splits = inner_splits(data, proto, tr)  # identical (deterministic) to the cached bundle's inner splits
        inner_F = [block_features(data, b, ml, itr, use) for b, ml, (itr, _) in zip(bd["inner"], bd["inner_ml"], splits)]
        flat = np.concatenate([F.reshape(-1, F.shape[-1]) for F in inner_F])
        mu, sd = flat.mean(0), flat.std(0) + 1e-9
        inner_F = [(F - mu) / sd for F in inner_F]
        F_out = (block_features(data, bd["outer"], bd["outer_ml"], tr, use) - mu) / sd
        prob = np.zeros(F_out.shape[:2])
        for l2 in L2_AVG:
            if soft:
                w = fit_soft_targets([(F, b["f1"]) for F, b in zip(inner_F, bd["inner"])], l2)
            else:
                w = fit_choice_model([(F, _positive(b)) for F, b in zip(inner_F, bd["inner"])], l2)
            prob += softmax(F_out @ w, axis=1) / len(L2_AVG)
        choice = prob.argmax(1)
        P[va] = chosen_vectors(data, bd["outer"], choice, bd["outer_ml"], alpha)
        A = data.Y > ACTIVE
        order = np.argsort(-prob, axis=1)
        for i, r in enumerate(va):
            t = tuple(np.flatnonzero(A[r]))
            pos = [k for k, c in enumerate(order[i]) if tuple(bd["outer"]["subsets"][c][bd["outer"]["subsets"][c] >= 0]) == t]
            if pos:
                hit_rank[r] = pos[0] + 1
    return P, hit_rank, []


BASE_ALL = FEATURES_PHYSICS + FEATURES_PRIOR + ML_FEATURES
NEW_PRIORS = ["prior_unary", "prior_pair", "prior_local"]
E13_VARIANTS = {
    "avg_base": (BASE_ALL, False),
    "avg_soft": (BASE_ALL, True),
    "avg_newprior": (BASE_ALL + NEW_PRIORS, False),
    "avg_full": (BASE_ALL + NEW_PRIORS, True),
    "avg_full_no_physics": ([u for u in BASE_ALL + NEW_PRIORS if u not in FEATURES_PHYSICS], True),
    "avg_newprior_no_physics": ([u for u in BASE_ALL + NEW_PRIORS if u not in FEATURES_PHYSICS], False),
}


def exp_e13(data: Data) -> list[dict]:
    """External-review suggestions, all constants fixed in advance; no inner hyperparameter selection."""
    results, hits = [], []
    for proto in PROTOCOLS:
        rf_idx = row_fold_index(data, proto)
        km = {f: make_km(data, tr, KM_DEFAULT) for f, (tr, _) in enumerate(data.folds[proto])}
        strata = data.strata(proto)
        for name, (use, soft) in E13_VARIANTS.items():
            P, rank, _ = run_averaged_ranker(data, proto, use, soft, own_ml_only=False)
            results.append(record("E13", name, proto, row_metrics(data.Y, P, data, km, rf_idx), P, data))
            for s in ("all", 1, 2):
                m = np.ones(len(rank), bool) if s == "all" else strata == s
                hits.append({"protocol": proto, "method": name, "stratum": s,
                             **{f"hit@{k}": float((rank[m] <= k).mean()) for k in (1, 3, 5)}})
            print(f"  [E13/{proto}] {name} done")
    h = pd.DataFrame(hits)
    h.to_csv(OUT_DIR / "E13_hits.csv", index=False)
    print(h.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print_strata("E13", cols=("n", "f1", "exact_set", "union_mae", "family_mae", "total_abs_err"))
    return results


def exp_e13b(data: Data) -> list[dict]:
    """Missing ablation: avg_base without physics features."""
    results = []
    for proto in PROTOCOLS:
        rf_idx = row_fold_index(data, proto)
        km = {f: make_km(data, tr, KM_DEFAULT) for f, (tr, _) in enumerate(data.folds[proto])}
        use = [u for u in BASE_ALL if u not in FEATURES_PHYSICS]
        P, rank, _ = run_averaged_ranker(data, proto, use, soft=False, own_ml_only=False)
        results.append(record("E13", "avg_base_no_physics", proto, row_metrics(data.Y, P, data, km, rf_idx), P, data))
        s = data.strata(proto)
        print(f"  [{proto}] hit@1/3/5 all: {[(rank <= k).mean().round(3) for k in (1, 3, 5)]} | stratum1: {[(rank[s == 1] <= k).mean().round(3) for k in (1, 3, 5)]}")
    return results


def exp_e16(data: Data) -> list[dict]:
    """Final method with Eq. 7 enforced (own ML top-3 set only), plus its no-physics ablation (review M2/M3)."""
    results, hits = [], []
    for proto in PROTOCOLS:
        rf_idx = row_fold_index(data, proto)
        km = {f: make_km(data, tr, KM_DEFAULT) for f, (tr, _) in enumerate(data.folds[proto])}
        strata = data.strata(proto)
        for name, use in (("avg_base", BASE_ALL), ("avg_base_no_physics", [u for u in BASE_ALL if u not in FEATURES_PHYSICS])):
            P, rank, _ = run_averaged_ranker(data, proto, use, soft=False, own_ml_only=True)
            results.append(record("E16", name, proto, row_metrics(data.Y, P, data, km, rf_idx), P, data))
            for s in ("all", 1, 2):
                m = np.ones(len(rank), bool) if s == "all" else strata == s
                hits.append({"protocol": proto, "method": name, "stratum": s,
                             **{f"hit@{k}": float((rank[m] <= k).mean()) for k in (1, 3, 5, 10)}})
            print(f"  [E16/{proto}] {name} done")
    h = pd.DataFrame(hits)
    h.to_csv(OUT_DIR / "E16_hits.csv", index=False)
    print(h.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return results


def exp_e15(data: Data) -> list[dict]:
    """E13's tuning-free rankers on candidate blocks rebuilt with the band-masked KM model (pre-specified)."""
    global BUNDLE_VERSION
    BUNDLE_VERSION = "v2_bandmask"
    results, hits = [], []
    variants = {
        "mask_avg_base": (BASE_ALL, False),
        "mask_avg_full": (BASE_ALL + NEW_PRIORS, True),
        "mask_avg_base_no_physics": ([u for u in BASE_ALL if u not in FEATURES_PHYSICS], False),
        "mask_avg_full_no_physics": ([u for u in BASE_ALL + NEW_PRIORS if u not in FEATURES_PHYSICS], True),
    }
    for proto in PROTOCOLS:
        rf_idx = row_fold_index(data, proto)
        km = {f: make_km(data, tr, KM_DEFAULT) for f, (tr, _) in enumerate(data.folds[proto])}
        strata = data.strata(proto)
        for f in range(len(data.folds[proto])):
            print(f"  [E15/{proto}] fold {f}: km={fold_bundle(data, proto, f)['km']}")
        for name, (use, soft) in variants.items():
            P, rank, _ = run_averaged_ranker(data, proto, use, soft, own_ml_only=False)
            results.append(record("E15", name, proto, row_metrics(data.Y, P, data, km, rf_idx), P, data))
            for s in ("all", 1, 2):
                m = np.ones(len(rank), bool) if s == "all" else strata == s
                hits.append({"protocol": proto, "method": name, "stratum": s,
                             **{f"hit@{k}": float((rank[m] <= k).mean()) for k in (1, 3, 5)}})
    h = pd.DataFrame(hits)
    h.to_csv(OUT_DIR / "E15_hits.csv", index=False)
    print(h.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print_strata("E15", cols=("n", "f1", "exact_set", "union_mae", "family_mae", "total_abs_err"))
    return results


# ------------------ T01/T02: ideas from the undergraduate thesis (position-wise dye formulation)


def position_labels(data: Data) -> np.ndarray:
    """(n_rows, 3) dye column index at recipe positions 1-3 (-1 = empty). Positions are hue roles:
    1 = yellow, 2 = red, 3 = blue in ~94 % of dyes (recipe_dye_long.csv)."""
    long = pd.read_csv(PROJECT_ROOT / "data/interim/recipe_dye_long.csv")
    mapping = pd.read_csv(PROJECT_ROOT / "data/global_naming/global_dye_mapping_active_36.csv")
    col_of = {n: data.dye_cols.index("vec__" + c) for n, c in zip(mapping["dye_name_clean"], mapping["dye_column"])
              if "vec__" + c in data.dye_cols}
    by_sample = {(s, int(p)): n for s, p, n, ok in long[["sample_id", "dye_position", "dye_name", "dye_present"]].itertuples(index=False) if ok}
    A = data.Y > ACTIVE
    hue_pos = {"yellow": 0, "red": 1, "blue": 2, "black": 2}
    lab = -np.ones((len(data.Y), 3), dtype=int)
    for i, s in enumerate(data.df["sample_id"]):
        for p in range(3):
            j = col_of.get(by_sample.get((s, p + 1)), -1)
            if j >= 0 and A[i, j]:
                lab[i, p] = j
        for j in np.flatnonzero(A[i]):  # active dyes the long table missed: place by hue role
            if j not in lab[i]:
                p = hue_pos.get(data.dye_hue[j], 2)
                free = [q for q in [p, 0, 1, 2] if lab[i, q] < 0]
                if free:
                    lab[i, free[0]] = j
    return lab


def slot_probs(X_tr: np.ndarray, lab_tr: np.ndarray, X_va: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per-position classifiers (thesis Section 3.1.5, here on spectra + substrate). Returns (classes, probs) per position."""
    out = []
    for p in range(lab_tr.shape[1]):
        y = lab_tr[:, p]
        classes = np.unique(y)
        if len(classes) == 1:
            out.append((classes, np.ones((len(X_va), 1))))
            continue
        clf = make_pipeline(SimpleImputer(strategy="median"),
                            ExtraTreesClassifier(n_estimators=300, max_features=1.0, min_samples_leaf=3, random_state=SEED, n_jobs=-1))
        clf.fit(X_tr, y)
        out.append((clf[-1].classes_, clf.predict_proba(X_va)))
    return out


def conditional_amounts(data: Data, X_tr, lab_tr, Y_tr, X_va, lab_va) -> np.ndarray:
    """Thesis Section 3.1.6: regress each position's concentration on inputs + one-hot dye identities
    (trained with true identities, applied with predicted ones)."""
    n_d = data.Y.shape[1]

    def design(X, lab):
        oh = np.zeros((len(X), 3 * (n_d + 1)))
        for p in range(3):
            oh[np.arange(len(X)), p * (n_d + 1) + np.where(lab[:, p] >= 0, lab[:, p], n_d)] = 1.0
        return np.hstack([X, oh])

    target = np.column_stack([np.where(lab_tr[:, p] >= 0, Y_tr[np.arange(len(Y_tr)), np.maximum(lab_tr[:, p], 0)], 0.0)
                              for p in range(3)])
    est = make_pipeline(SimpleImputer(strategy="median"),
                        ExtraTreesRegressor(n_estimators=300, max_features=1.0, min_samples_leaf=2, random_state=SEED, n_jobs=-1))
    est.fit(design(X_tr, lab_tr), target)
    amt = np.maximum(est.predict(design(X_va, lab_va)), 0.0)
    P = np.zeros((len(X_va), n_d))
    for p in range(3):
        ok = lab_va[:, p] >= 0
        np.add.at(P, (np.flatnonzero(ok), lab_va[ok, p]), amt[ok, p])
    return P


def exp_t01(data: Data) -> list[dict]:
    """Thesis two-stage model under strict/grouped CV: position-wise dye classification, then conditional amounts."""
    from sklearn.ensemble import ExtraTreesClassifier  # noqa: F401  (used in slot_probs)

    lab = position_labels(data)
    X = data.X("spectral_substrate")
    results = []
    for proto in PROTOCOLS:
        rf_idx = row_fold_index(data, proto)
        km = {f: make_km(data, tr, KM_DEFAULT) for f, (tr, _) in enumerate(data.folds[proto])}
        P_two_stage = np.zeros_like(data.Y)
        P_slot_ml_amounts = np.zeros_like(data.Y)
        for tr, va in data.folds[proto]:
            probs = slot_probs(X[tr], lab[tr], X[va])
            pred_lab = np.column_stack([cls[pr.argmax(1)] for cls, pr in probs])
            # The same dye cannot fill two positions: keep its most confident position.
            for i in range(len(va)):
                seen = {}
                for p in range(3):
                    j = pred_lab[i, p]
                    if j >= 0 and j in seen:
                        q = seen[j]
                        conf = lambda pp: probs[pp][1][i].max()  # noqa: E731
                        drop = p if conf(p) < conf(q) else q
                        pred_lab[i, drop] = -1
                        seen[j] = q if drop == p else p
                    elif j >= 0:
                        seen[j] = p
            P_two_stage[va] = conditional_amounts(data, X[tr], lab[tr], data.Y[tr], X[va], pred_lab)
            ml = ml_raw(data, tr, va)
            for i, r in enumerate(va):  # same dye choice, amounts from the 36-output ML regressor
                dyes = pred_lab[i][pred_lab[i] >= 0]
                if len(dyes):
                    v = np.maximum(ml[i, dyes], 0)
                    tot = max(ml[i].clip(min=0).sum(), 0)
                    P_slot_ml_amounts[r, dyes] = v * tot / v.sum() if v.sum() > 0 else tot / len(dyes)
        for name, P in (("thesis_two_stage", P_two_stage), ("thesis_slots_ml_amounts", P_slot_ml_amounts)):
            results.append(record("T01", name, proto, row_metrics(data.Y, P, data, km, rf_idx), P, data))
    print_strata("T01", cols=("n", "f1", "exact_set", "active_mae", "union_mae", "family_mae", "total_abs_err"))
    return results


def slot_prior_feature(data: Data, b: dict, probs: list[tuple[np.ndarray, np.ndarray]], lab_tr: np.ndarray) -> np.ndarray:
    """log P(S) under independent position classifiers: each dye of S goes to its modal training position."""
    n_d = data.Y.shape[1]
    modal = np.full(n_d, 2)
    for j in range(n_d):
        ps = np.flatnonzero((lab_tr == j).any(0))
        if len(ps):
            modal[j] = int(np.bincount(np.where(lab_tr == j)[1], minlength=3).argmax())
    eps = 1e-3
    logp = []
    for cls, pr in probs:
        idx = {int(c): k for k, c in enumerate(cls)}
        logp.append((idx, np.log(np.clip(pr, eps, 1.0))))
    n_va, n_c = len(b["va"]), len(b["subsets"])
    feat = np.zeros((n_va, n_c))
    for c, s in enumerate(b["subsets"]):
        dyes = [int(d) for d in s if d >= 0]
        assign = [-1, -1, -1]
        clash = 0
        for d in sorted(dyes, key=lambda d: modal[d]):
            p = modal[d]
            if assign[p] >= 0:
                free = [q for q in range(3) if assign[q] < 0]
                clash += 1
                p = free[0] if free else p
            assign[p] = d
        total = np.zeros(n_va)
        for p in range(3):
            idx, lp = logp[p]
            k = idx.get(assign[p])
            total += lp[:, k] if k is not None else np.log(eps)
        feat[:, c] = total - clash * 2.0
    return feat


def exp_t02(data: Data) -> list[dict]:
    """E13 tuning-free ranker + a position-factorised dye prior (compositional; targets novel combinations)."""
    from scipy.special import softmax

    lab = position_labels(data)
    X = data.X("spectral_substrate")
    results, hits = [], []
    variants = {
        "ranker_base_plus_slots": (BASE_ALL, True),
        "ranker_base": (BASE_ALL, False),
        "ranker_base_plus_slots_no_physics": ([u for u in BASE_ALL if u not in FEATURES_PHYSICS], True),
        "slots_only_with_physics": (FEATURES_PHYSICS, True),
    }
    for proto in PROTOCOLS:
        rf_idx = row_fold_index(data, proto)
        km = {f: make_km(data, tr, KM_DEFAULT) for f, (tr, _) in enumerate(data.folds[proto])}
        strata = data.strata(proto)
        cache = {}
        for f, (tr, va) in enumerate(data.folds[proto]):
            bd = fold_bundle(data, proto, f)
            splits = inner_splits(data, proto, tr)
            inner_slot = [slot_prior_feature(data, b, slot_probs(X[itr], lab[itr], X[iva]), lab[itr])
                          for b, (itr, iva) in zip(bd["inner"], splits)]
            outer_slot = slot_prior_feature(data, bd["outer"], slot_probs(X[tr], lab[tr], X[va]), lab[tr])
            cache[f] = (bd, splits, inner_slot, outer_slot)
        for name, (use, with_slots) in variants.items():
            P = np.zeros_like(data.Y)
            rank = np.full(len(data.Y), np.inf)
            for f, (tr, va) in enumerate(data.folds[proto]):
                bd, splits, inner_slot, outer_slot = cache[f]
                def feats(b, ml, rows, slot):
                    F = block_features(data, b, ml, rows, use)
                    return np.concatenate([F, slot[..., None]], -1) if with_slots else F
                inner_F = [feats(b, ml, itr, sl) for b, ml, (itr, _), sl in zip(bd["inner"], bd["inner_ml"], splits, inner_slot)]
                flat = np.concatenate([F.reshape(-1, F.shape[-1]) for F in inner_F])
                mu, sd = flat.mean(0), flat.std(0) + 1e-9
                inner_F = [(F - mu) / sd for F in inner_F]
                F_out = (feats(bd["outer"], bd["outer_ml"], tr, outer_slot) - mu) / sd
                prob = np.zeros(F_out.shape[:2])
                for l2 in L2_AVG:
                    w = fit_choice_model([(F, _positive(b)) for F, b in zip(inner_F, bd["inner"])], l2)
                    prob += softmax(F_out @ w, axis=1) / len(L2_AVG)
                choice = prob.argmax(1)
                P[va] = chosen_vectors(data, bd["outer"], choice, bd["outer_ml"], 0.5)
                A = data.Y > ACTIVE
                order = np.argsort(-prob, axis=1)
                subs = bd["outer"]["subsets"]
                for i, r in enumerate(va):
                    t = tuple(np.flatnonzero(A[r]))
                    pos = [k for k, c in enumerate(order[i]) if tuple(subs[c][subs[c] >= 0]) == t]
                    if pos:
                        rank[r] = pos[0] + 1
            results.append(record("T02", name, proto, row_metrics(data.Y, P, data, km, rf_idx), P, data))
            for s in ("all", 1, 2):
                m = np.ones(len(rank), bool) if s == "all" else strata == s
                hits.append({"protocol": proto, "method": name, "stratum": s,
                             **{f"hit@{k}": float((rank[m] <= k).mean()) for k in (1, 3, 5)}})
    print(pd.DataFrame(hits).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print_strata("T02", cols=("n", "f1", "exact_set", "active_mae", "union_mae", "family_mae", "total_abs_err"))
    return results


E14_VARIANTS = {
    "default": KM_DEFAULT,
    "band_mask": {**KM_DEFAULT, "band_mask": True},
    "band_mask_loose": {**KM_DEFAULT, "band_mask": {"near_zero": 0.1, "shoulder": 1.0}},  # sensitivity only
}


def exp_e14(data: Data) -> list[dict]:
    """Forward KM fit with hue-family absorption-band upper bounds (MCR-style selectivity constraints)."""
    return exp_e01(data, E14_VARIANTS, "E14")


def run_ranker(data: Data, proto: str, use: list[str], tag: str) -> tuple[np.ndarray, list[dict]]:
    P = np.zeros_like(data.Y)
    log = []
    for f, (tr, va) in enumerate(data.folds[proto]):
        km_kw = select_km_inner(data, proto, tr)
        splits = inner_splits(data, proto, tr)
        inner = [build_block(data, proto, itr, iva, km_kw, "spectral_substrate") for itr, iva in splits]
        inner_F = []
        for (itr, iva), b in zip(splits, inner):
            F, names = candidate_features(b, ml_raw(data, itr, iva))
            inner_F.append(F[..., [names.index(u) for u in use]])
        flat = np.concatenate([F.reshape(-1, F.shape[-1]) for F in inner_F])
        mu, sd = flat.mean(0), flat.std(0) + 1e-9
        inner_F = [(F - mu) / sd for F in inner_F]
        # L2 strength by leave-one-inner-fold-out F1.
        l2_score = {}
        for l2 in (0.01, 0.1, 1.0, 10.0, 100.0):
            f1s = []
            for h in range(len(inner)):
                w = fit_choice_model([(inner_F[j], _positive(inner[j])) for j in range(len(inner)) if j != h], l2)
                choice = (inner_F[h] @ w).argmax(1)
                f1s.append(inner[h]["f1"][np.arange(len(choice)), choice])
            l2_score[l2] = float(np.mean(np.concatenate(f1s)))
        l2 = max(l2_score, key=l2_score.get)
        w = fit_choice_model([(F, _positive(b)) for F, b in zip(inner_F, inner)], l2)
        outer = build_block(data, proto, tr, va, km_kw, "spectral_substrate")
        F, names = candidate_features(outer, ml_raw(data, tr, va))
        F = (F[..., [names.index(u) for u in use]] - mu) / sd
        choice = (F @ w).argmax(1)
        for i, c in enumerate(choice):
            for slot, dye in enumerate(outer["subsets"][c]):
                if dye >= 0:
                    P[va[i], dye] = outer["conc"][i, c, slot]
        log.append({"protocol": proto, "fold": f, "variant": tag, "km": km_kw["_name"], "l2": l2,
                    **{f"inner_f1_l2_{k}": v for k, v in l2_score.items()}, **dict(zip(use, w))})
        print(f"  [E05/{proto}/{tag}] fold {f}: l2={l2} inner F1={l2_score[l2]:.3f}")
    return P, log


E05_VARIANTS = {
    "ranker_all": FEATURES_PHYSICS + FEATURES_PRIOR,
    "ranker_no_physics": FEATURES_PRIOR,
    "ranker_physics_set_prior": FEATURES_PHYSICS + ["set_logp", "is_hist", "card_logp", "size"],
}


def exp_e05(data: Data) -> list[dict]:
    """Learned conditional-logit ranking over physics-solved candidate recipes; nested CV."""
    results, logs = [], []
    for proto in PROTOCOLS:
        rf_idx = row_fold_index(data, proto)
        km = {f: make_km(data, tr, KM_DEFAULT) for f, (tr, _) in enumerate(data.folds[proto])}
        for name, use in E05_VARIANTS.items():
            P, log = run_ranker(data, proto, use, name)
            logs += log
            results.append(record("E05", name, proto, row_metrics(data.Y, P, data, km, rf_idx), P, data))
    pd.DataFrame(logs).to_csv(OUT_DIR / "E05_selected.csv", index=False)
    print_strata("E05")
    return results


def compare(data: Data, a: str, b: str, proto: str,
            metrics=("f1", "union_mae", "active_mae", "family_mae", "total_abs_err")) -> pd.DataFrame:
    """Paired cluster-bootstrap differences (a - b); a and b are 'EXP:method'."""
    ra, rb = load_rows(*a.split(":"), proto), load_rows(*b.split(":"), proto)
    clusters, strata = data.clusters(proto), data.strata(proto)
    out = []
    for subset, mask in (("all", np.ones(len(ra), bool)), ("stratum1", strata == 1), ("stratum2", strata == 2),
                         ("stratum3", strata == 3), ("seen_dyes", strata < 3)):
        for m in metrics:
            x, y = ra[m].to_numpy(), rb[m].to_numpy()
            ok = mask & np.isfinite(x) & np.isfinite(y)
            mean, lo, hi = paired_bootstrap(x[ok], y[ok], clusters[ok])
            out.append({"subset": subset, "n": int(ok.sum()), "metric": m, "a": x[ok].mean(), "b": y[ok].mean(),
                        "diff": mean, "ci_lo": lo, "ci_hi": hi, "ci_excludes_0": lo > 0 or hi < 0})
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp")
    parser.add_argument("--compare", nargs=2, metavar=("EXP:METHOD_A", "EXP:METHOD_B"))
    parser.add_argument("--protocol", default="strict")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = Data()
    if args.compare:
        t = compare(data, *args.compare, args.protocol)
        t.to_csv(OUT_DIR / f"compare__{args.compare[0].replace(':', '_')}__vs__{args.compare[1].replace(':', '_')}__{args.protocol}.csv", index=False)
        print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        return
    fn = globals()[f"exp_{args.exp.lower()}"]
    t0 = time.time()
    results = fn(data)
    if results:
        print_table(results)
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
