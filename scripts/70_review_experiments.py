"""Experiments requested by the Scientific Reports referee review (2026-11-12).

B  Non-physics blend control: on the final method's selected dye sets, blend the Extra Trees amounts with a
   second, non-physical estimator (random forest) instead of the KM amounts. Tests whether the amount gain of
   the KM/ML blend exceeds what averaging two estimators gives.
S  KM positive control: targets simulated from the calibrated KM model itself (true recipe amounts), with
   smooth spectral noise scaled to a median colour error of 0, 1.0 and 2.7 CMC units; exhaustive matching is
   rerun. Tests whether dye sets are recoverable when the forward model is right.
R  Repeated strict CV: new random assignments of shades to five folds (and new tree seeds); the final method
   and the ML reference are refitted from scratch in each repeat.

The final method, the ML reference and all settings are those of 60_physics_inverse_experiments.py; nothing is
tuned here. Outputs: outputs/review_experiments/ (separate cache; the original caches are not touched).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location("pie", ROOT / "scripts" / "60_physics_inverse_experiments.py")
pie = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pie)

from scipy.ndimage import gaussian_filter1d  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402

from piml_color_recipe.inverse.km_physics import (  # noqa: E402
    delta_e_cmc, enumerate_subsets, ks_to_reflectance, reflectance_to_lab, solve_subsets,
)

OUT = ROOT / "outputs" / "review_experiments"
OUT.mkdir(parents=True, exist_ok=True)
METRICS = ["f1", "exact_set", "active_mae", "union_mae", "family_mae", "total_abs_err"]
B_FLIPS = 20000


def signflip(a: np.ndarray, b: np.ndarray, cl: np.ndarray, n: int = B_FLIPS, seed: int = 42) -> tuple[float, float]:
    """Paired cluster sign-flip test (main-text equation for Delta and p); rows with NaN are dropped."""
    ok = np.isfinite(a) & np.isfinite(b)
    d = pd.Series(a[ok] - b[ok]).groupby(cl[ok]).agg(["sum", "count"])
    D, m = d["sum"].to_numpy(), d["count"].to_numpy()
    delta = D.sum() / m.sum()
    eps = np.random.default_rng(seed).choice([-1.0, 1.0], size=(n, len(D)))
    null = np.abs(eps @ D / m.sum())
    return float(delta), float((1 + (null >= abs(delta) - 1e-15).sum()) / (1 + n))


def holm(p: list[float]) -> list[float]:
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (len(p) - k) * p[i])
        adj[i] = min(running, 1.0)
    return adj.tolist()


def rf_raw(data, tr: np.ndarray, va: np.ndarray, group: str = "spectral_substrate") -> np.ndarray:
    est = make_pipeline(SimpleImputer(strategy="median"),
                        RandomForestRegressor(n_estimators=200, max_features=0.5, min_samples_leaf=1,
                                              random_state=pie.SEED, n_jobs=-1))
    return est.fit(data.X(group)[tr], data.Y[tr]).predict(data.X(group)[va])


# ------------------------------------------------------------------ B: non-physics blend control
def exp_b(data) -> dict:
    out = {}
    for proto in ("strict", "grouped"):
        rf_idx = pie.row_fold_index(data, proto)
        km = {f: pie.make_km(data, tr, pie.KM_DEFAULT) for f, (tr, _) in enumerate(data.folds[proto])}
        P = {k: np.zeros_like(data.Y) for k in ("ml", "km_blend", "rf", "ml_rf_blend")}
        for f, (tr, va) in enumerate(data.folds[proto]):
            bd, prob, _ = pie.fold_choice_probs(data, proto, f, pie.BASE_ALL)
            choice = prob.argmax(1)
            et, rf = bd["outer_ml"], rf_raw(data, tr, va)
            P["ml"][va] = pie.chosen_vectors(data, bd["outer"], choice, et, 0.0)
            P["km_blend"][va] = pie.chosen_vectors(data, bd["outer"], choice, et, 0.5)
            P["rf"][va] = pie.chosen_vectors(data, bd["outer"], choice, rf, 0.0)
            P["ml_rf_blend"][va] = 0.5 * (P["ml"][va] + P["rf"][va])
        rows = {k: pie.row_metrics(data.Y, v, data, km, rf_idx) for k, v in P.items()}
        cl = data.clusters(proto)
        res = {"means": {k: {m: float(np.nanmean(r[m])) for m in ("active_mae", "union_mae", "family_mae", "total_abs_err")}
                         for k, r in rows.items()}}
        for a, b in (("km_blend", "ml_rf_blend"), ("ml_rf_blend", "ml"), ("km_blend", "ml")):
            tests = {m: signflip(rows[a][m].to_numpy(), rows[b][m].to_numpy(), cl)
                     for m in ("active_mae", "union_mae", "family_mae", "total_abs_err")}
            ph = holm([t[1] for t in tests.values()])
            res[f"{a}_vs_{b}"] = {m: {"delta": t[0], "p": t[1], "p_holm": h} for (m, t), h in zip(tests.items(), ph)}
        out[proto] = res
        print(f"[B/{proto}] means:", json.dumps(res["means"], indent=None))
        for key in [k for k in res if k != "means"]:
            print(f"   {key}: " + "; ".join(f"{m} {v['delta']:+.4f} (p_Holm {v['p_holm']:.3f})" for m, v in res[key].items()))
    (OUT / "B_blend_control.json").write_text(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------ S: KM positive control (simulation)
def _de_max(model, ks_a: np.ndarray, ks_b: np.ndarray) -> np.ndarray:
    ra, rb = ks_to_reflectance(ks_a), ks_to_reflectance(ks_b)
    return np.maximum(delta_e_cmc(reflectance_to_lab(ra, "D65"), reflectance_to_lab(rb, "D65")),
                      delta_e_cmc(reflectance_to_lab(ra, "FL11"), reflectance_to_lab(rb, "FL11")))


def exp_s(data, levels=(0.0, 1.0, 2.7), smooth_bins: float = 3.0) -> dict:
    proto = "strict"
    A = data.Y > pie.ACTIVE
    rng = np.random.default_rng(42)
    per_fold = []
    for f, (tr, va) in enumerate(data.folds[proto]):
        model = pie.make_km(data, tr, pie.KM_DEFAULT)
        usable = np.flatnonzero((data.Y[tr] > pie.ACTIVE).sum(0) >= 1)
        rows = [r for r in va if A[r].any() and set(np.flatnonzero(A[r])) <= set(usable)]
        clean = model.predict_ks(data.Y[rows])
        z = gaussian_filter1d(rng.standard_normal(clean.shape), smooth_bins, axis=1)
        z /= z.std(axis=1, keepdims=True)
        per_fold.append((f, tr, model, usable, np.array(rows), clean, z))
    all_clean = np.vstack([p[5] for p in per_fold])
    all_z = np.vstack([p[6] for p in per_fold])

    def median_de(sigma: float) -> float:
        return float(np.median(_de_max(None, all_clean, all_clean * np.exp(sigma * all_z))))

    sigmas = {}
    for lev in levels:
        if lev == 0:
            sigmas[lev] = 0.0
            continue
        lo, hi = 0.0, 3.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if median_de(mid) < lev else (lo, mid)
        sigmas[lev] = 0.5 * (lo + hi)
    print("[S] noise sigma per level:", {k: round(v, 4) for k, v in sigmas.items()},
          "achieved median dE:", {k: round(median_de(v), 3) for k, v in sigmas.items()})
    out = {"n_rows": int(sum(len(p[4]) for p in per_fold)), "sigma": sigmas, "levels": {}}
    for lev in levels:
        best_true, ranks, f1s, near = [], [], [], []
        t0 = time.time()
        for f, tr, model, usable, rows, clean, z in per_fold:
            subsets = enumerate_subsets(usable, 3)
            keys = [tuple(sorted(s[s >= 0])) for s in subsets]
            c_max = float(data.Y[tr].max())
            for i, r in enumerate(rows):
                target = clean[i] * np.exp(sigmas[lev] * z[i])
                sol = solve_subsets(model, target, subsets, c_max=c_max)
                de = np.maximum(sol.de_d65, sol.de_f11)
                ti = keys.index(tuple(sorted(np.flatnonzero(A[r]))))
                rank = 1 + int((de < de[ti] - 1e-9).sum())
                ranks.append(rank)
                best_true.append(rank == 1)
                near.append(float((de <= de.min() + 0.1).sum()))
                b = set(keys[int(np.argmin(de))])
                t = set(keys[ti])
                f1s.append(2 * len(b & t) / (len(b) + len(t)))
        out["levels"][lev] = {"true_best_share": float(np.mean(best_true)), "median_rank": float(np.median(ranks)),
                              "rank_q25_q75": [float(np.percentile(ranks, 25)), float(np.percentile(ranks, 75))],
                              "top10_share": float(np.mean(np.array(ranks) <= 10)), "f1_of_best": float(np.mean(f1s)),
                              "median_sets_within_0.1_of_best": float(np.median(near))}
        print(f"[S] level {lev}: {out['levels'][lev]} ({time.time() - t0:.0f}s)")
    (OUT / "S_km_simulation.json").write_text(json.dumps(out, indent=1, default=str))
    return out


# ------------------------------------------------------------------ R: repeated strict CV
def random_shade_folds(data, seed: int, k: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    shade = data.df["target_id"].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    names = rng.permutation(np.unique(shade))
    size = {s: int((shade == s).sum()) for s in names}
    fold_of, load = {}, np.zeros(k)
    for s in sorted(names, key=lambda s: -size[s] + rng.random() * 1e-3):  # largest first, random ties
        j = int(np.argmin(load + rng.random(k) * 1e-6))
        fold_of[s] = j
        load[j] += size[s]
    fid = np.array([fold_of[s] for s in shade])
    return [(np.flatnonzero(fid != j), np.flatnonzero(fid == j)) for j in range(k)]


def exp_r(data, n_rep: int = 10) -> pd.DataFrame:
    pie.CACHE_DIR = OUT / "cache"
    path = OUT / "R_repeated_cv.csv"
    done = pd.read_csv(path) if path.exists() else pd.DataFrame()
    rows = done.to_dict("records")
    for r in range(1, n_rep + 1):
        if not done.empty and r in set(done["repeat"]):
            continue
        t0 = time.time()
        proto = f"strict_r{r}"
        pie.PROTOCOLS[proto] = ("random_shade_folds", "target_id")
        data.folds[proto] = random_shade_folds(data, 1000 + r)
        pie.SEED = 42 + r
        rf_idx = pie.row_fold_index(data, proto)
        km = {f: pie.make_km(data, tr, pie.KM_DEFAULT) for f, (tr, _) in enumerate(data.folds[proto])}
        P_ml = pie.feasible(pie.ml_regression_oof(data, proto, "et", "spectral_substrate"))
        P_ours, _, _ = pie.run_averaged_ranker(data, proto, pie.BASE_ALL, soft=False, own_ml_only=True)
        m_ml = pie.row_metrics(data.Y, P_ml, data, km, rf_idx)
        m_ours = pie.row_metrics(data.Y, P_ours, data, km, rf_idx)
        cl = data.clusters(proto)
        tests = {m: signflip(m_ours[m].to_numpy(), m_ml[m].to_numpy(), cl) for m in METRICS}
        ph = holm([t[1] for t in tests.values()])
        rec = {"repeat": r, "seconds": round(time.time() - t0)}
        for (m, (d, p)), h in zip(tests.items(), ph):
            rec |= {f"{m}_ours": float(np.nanmean(m_ours[m])), f"{m}_ml": float(np.nanmean(m_ml[m])),
                    f"{m}_delta": d, f"{m}_p": p, f"{m}_p_holm": h}
        rows.append(rec)
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"[R] repeat {r}: famMAE {rec['family_mae_ours']:.3f} vs {rec['family_mae_ml']:.3f} "
              f"(p_Holm {rec['family_mae_p_holm']:.3f}); F1 {rec['f1_ours']:.3f} vs {rec['f1_ml']:.3f}; "
              f"{rec['seconds']}s", flush=True)
    df = pd.DataFrame(rows)
    summ = {m: {"delta_mean": float(df[f"{m}_delta"].mean()), "delta_min": float(df[f"{m}_delta"].min()),
                "delta_max": float(df[f"{m}_delta"].max()), "n_sig_holm": int((df[f"{m}_p_holm"] < 0.05).sum()),
                "ours_mean": float(df[f"{m}_ours"].mean()), "ml_mean": float(df[f"{m}_ml"].mean())} for m in METRICS}
    (OUT / "R_repeated_cv_summary.json").write_text(json.dumps({"n_repeats": len(df), **summ}, indent=1))
    print("[R] summary:", json.dumps(summ, indent=1))
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=["B", "S", "R"])
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    data = pie.Data()
    for k in args.only:
        t0 = time.time()
        {"B": exp_b, "S": exp_s, "R": lambda d: exp_r(d, args.repeats)}[k](data)
        print(f"== {k} finished in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
