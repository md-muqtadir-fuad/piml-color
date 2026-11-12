"""R01: robustness checks reported in the paper and its Supplementary Information.

Part A (independent of the final-method rerun):
  A1  colour matching with the true dye set refitted (M3): fitted true-set dE, rank of the true set, and the
      colour difference under a third illuminant (CIE A) for the best wrong set (metamerism check)
  A2  KM calibration gain with the per-fold, inner-CV-selected KM constants (M8.3, nested)
  A3  bootstrap variability of the calibrated unit K/S spectra (M8.2)
Part B (uses the E16 rows of the final method):
  B1  shade-weighted metrics, ours vs ML reference (M4)
  B2  stratum-wise F1 tests (M9)
  B3  blend vs KM amounts and the no-physics ranker (M2)
  B4  conformal coverage for all predicted dyes, and with a shade-level split (M5)
  B5  strength-weighted hue-family MAE (M6)
Writes outputs/physics_inverse_experiments/R01_review_checks.json.

Run: python scripts/69_review_checks.py [A|B|all]
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location("pie", ROOT / "scripts" / "60_physics_inverse_experiments.py")
pie = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pie)

from piml_color_recipe.inverse import km_physics as kmp  # noqa: E402

OUT = pie.OUT_DIR / "R01_review_checks.json"
D = pie.Data()
A = D.Y > pie.ACTIVE
RNG = np.random.default_rng(42)
_WA = kmp._tristimulus_weights("A")


def lab(refl: np.ndarray, ill: str) -> np.ndarray:
    if ill == "A":
        w, white = _WA
        return kmp.colour.XYZ_to_Lab((np.asarray(refl) @ w) / 100.0, white)
    return kmp.reflectance_to_lab(refl, ill)


def signflip(d: np.ndarray, cl: np.ndarray, n: int = 20000, weights: str = "rows") -> tuple[float, float]:
    """Paired sign-flip test on cluster sums; p includes the observed statistic ((b+1)/(B+1)).
    weights='rows': mean over rows (paper); 'shades': mean of per-shade means."""
    ok = np.isfinite(d)
    g = pd.Series(d[ok]).groupby(cl[ok]).agg(["sum", "count", "mean"])
    if weights == "rows":
        s, denom = g["sum"].to_numpy(), g["count"].sum()
    else:
        s, denom = g["mean"].to_numpy(), len(g)
    obs = s.sum() / denom
    null = (np.random.default_rng(42).choice([-1, 1], size=(n, len(s))) * s).sum(1) / denom
    b = int((np.abs(null) >= abs(obs) - 1e-12).sum())
    return float(obs), (b + 1) / (n + 1)


def seen_mask(proto: str) -> np.ndarray:
    seen = np.zeros(len(D.Y), bool)
    for tr, va in D.folds[proto]:
        s = A[tr].any(0)
        seen[va] = [(s | ~A[r]).all() for r in va]
    return seen


def km_for_fold(proto: str, f: int):
    import pickle
    name = pickle.loads((pie.CACHE_DIR / f"bundle_v1__{proto}__fold{f}.pkl").read_bytes())["km"]
    return name, pie.KM_GRID[name]


# ----------------------------------------------------------------------------------------------- part A
def a1_colour_matching() -> dict:
    """True set refitted vs every other set; ranks; dE under illuminant A for the best set."""
    seen = seen_mask("strict")
    rec = []
    for f, (tr, va) in enumerate(D.folds["strict"]):
        z = np.load(pie.CACHE_DIR / f"E02_lang0.3_shrink0.01_cmax__strict__fold{f}.npz")
        model = pie.make_km(D, tr, pie.KM_DEFAULT)
        keys = {tuple(s[s >= 0]): c for c, s in enumerate(z["subsets"])}
        de = np.maximum(z["de_d65"], z["de_f11"])
        for i, r in enumerate(z["val"]):
            t = tuple(np.flatnonzero(A[r]))
            if not seen[r] or t not in keys:
                continue
            c_true = keys[t]
            best = int(de[i].argmin())
            # reconstruct the best set and the refitted true set; colour difference under D65, F11 and A
            def recon(c):
                v = np.zeros(D.Y.shape[1])
                s = z["subsets"][c]
                v[s[s >= 0]] = z["conc"][i, c][: (s >= 0).sum()]
                return model.predict_reflectance(v[None])[0]
            tgt = D.R[r]
            out = {}
            for nm, c in (("best", best), ("true_fit", c_true)):
                rf = recon(c)
                out[nm] = {ill: float(kmp.delta_e_cmc(lab(tgt[None], ill), lab(rf[None], ill))[0]) for ill in ("D65", "FL11", "A")}
            rec.append({"row": int(r), "shade": D.clusters("strict")[r], "n_dyes": len(t),
                        "de_true_recorded": float(pie.km_true_recipe_de(D, model, np.array([r]))[0]),
                        "de_true_fit": float(de[i, c_true]), "de_best": float(de[i, best]),
                        "best_is_true": best == c_true, "rank_true": int((de[i] < de[i, c_true]).sum() + 1),
                        "n_sets": int(de.shape[1]),
                        "best_dA": out["best"]["A"], "true_fit_dA": out["true_fit"]["A"],
                        "best_dmax_d65_f11": max(out["best"]["D65"], out["best"]["FL11"])})
    t = pd.DataFrame(rec)
    wrong = t[~t.best_is_true]
    res = {
        "rows": int(len(t)), "shades": int(t.shade.nunique()),
        "share_better_than_recorded_true": float((t.de_best < t.de_true_recorded).mean()),
        "share_true_fit_not_best": float((~t.best_is_true).mean()),
        "share_strictly_better_than_true_fit": float((t.de_best < t.de_true_fit - 1e-9).mean()),
        "median_de_true_recorded": float(t.de_true_recorded.median()),
        "median_de_true_fit": float(t.de_true_fit.median()), "median_de_best": float(t.de_best.median()),
        "rank_true_median": float(t.rank_true.median()), "rank_true_q25_q75": [float(t.rank_true.quantile(.25)), float(t.rank_true.quantile(.75))],
        "n_sets_median": float(t.n_sets.median()),
        "share_true_in_top10": float((t.rank_true <= 10).mean()),
        "wrong_best_median_dE_A": float(wrong.best_dA.median()),
        "wrong_best_median_dE_D65F11": float(wrong.best_dmax_d65_f11.median()),
        "true_fit_median_dE_A": float(t.true_fit_dA.median()),
        "share_wrong_best_dA_gt_1": float((wrong.best_dA > 1.0).mean()),
        "share_wrong_best_dA_gt_dmax": float((wrong.best_dA > wrong.best_dmax_d65_f11).mean()),
    }
    t.to_csv(pie.OUT_DIR / "R01_A1_colour_matching_rows.csv", index=False)
    return res


def a2_nested_calibration() -> dict:
    """Held-out dE of true recipes: feature-only KM vs the KM the final method selected per fold (nested)."""
    res = {}
    for proto in ("strict", "grouped"):
        seen, cl = seen_mask(proto), D.clusters(proto)
        old, new, names = np.full(len(D.Y), np.nan), np.full(len(D.Y), np.nan), []
        for f, (tr, va) in enumerate(D.folds[proto]):
            name, kw = km_for_fold(proto, f)
            names.append(name)
            old[va] = pie.km_true_recipe_de(D, pie.make_km(D, tr, dict(weighting="none")), va)
            new[va] = pie.km_true_recipe_de(D, pie.make_km(D, tr, kw), va)
        d = np.where(seen, new - old, np.nan)
        diff, p = signflip(d, cl)
        res[proto] = {"n": int(seen.sum()), "selected": names,
                      "old_median": float(np.median(old[seen])), "old_mean": float(old[seen].mean()),
                      "new_median": float(np.median(new[seen])), "new_mean": float(new[seen].mean()),
                      "mean_diff": diff, "p": p}
    return res


def a3_bootstrap_spectra(B: int = 200) -> dict:
    """Recipe-cluster bootstrap of the unit K/S spectra (all rows, KM_DEFAULT): spread of each dye's spectrum."""
    rec = D.df["recipe_no"].astype(str).to_numpy()
    groups = np.unique(rec)
    idx = {g: np.flatnonzero(rec == g) for g in groups}
    usage = A.sum(0)
    common = [j for j in np.argsort(-usage) if usage[j] >= 12 and D.dye_hue[j] in ("yellow", "red", "blue")]
    full = pie.make_km(D, np.arange(len(D.Y)), pie.KM_DEFAULT).k
    boots = []
    for _ in range(B):
        take = np.concatenate([idx[g] for g in RNG.choice(groups, len(groups))])
        boots.append(pie.make_km(D, take, pie.KM_DEFAULT).k[common])
    K = np.stack(boots)  # (B, n_common, n_wl)
    integ = K.sum(-1)  # integrated unit K/S (strength)
    cv = integ.std(0) / integ.mean(0)
    peak_wl = 400 + 10 * K.argmax(-1)
    return {"dyes": int(len(common)), "B": B,
            "cv_integrated_median": float(np.median(cv)), "cv_integrated_range": [float(cv.min()), float(cv.max())],
            "peak_wavelength_iqr_median_nm": float(np.median(np.subtract(*np.percentile(peak_wl, [75, 25], axis=0)))),
            "full_vs_boot_corr_median": float(np.median([np.corrcoef(full[j], K[:, n].mean(0))[0, 1] for n, j in enumerate(common)]))}


# ----------------------------------------------------------------------------------------------- part B
def _rows(exp: str, method: str, proto: str = "strict") -> pd.DataFrame:
    return pie.load_rows(exp, method, proto)


REF = {"strict": ("E00b", "ml_et_spectral_substrate_feasible"), "grouped": ("E00b", "ml_et_spectral_feasible")}
METRICS = ["f1", "exact_set", "active_mae", "union_mae", "family_mae", "total_abs_err"]


def b1_shade_weighted() -> dict:
    res = {}
    for proto in ("strict", "grouped"):
        a, b, cl = _rows("E16", "avg_base", proto), _rows(*REF[proto], proto), D.clusters(proto)
        out = {}
        for m in METRICS:
            sa = pd.Series(a[m].to_numpy()).groupby(cl).mean()
            sb = pd.Series(b[m].to_numpy()).groupby(cl).mean()
            diff, p = signflip(a[m].to_numpy() - b[m].to_numpy(), cl, weights="shades")
            out[m] = {"ours": float(sa.mean()), "ml": float(sb.mean()), "diff": diff, "p": p}
        ps = [out[m]["p"] for m in METRICS]
        for m, ph in zip(METRICS, holm(ps)):
            out[m]["p_holm"] = float(ph)
        res[proto] = out
    return res


def holm(p) -> np.ndarray:
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj, run = np.empty(len(p)), 0.0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def b2_strata() -> dict:
    st, cl = D.strata("strict"), D.clusters("strict")
    a, b = _rows("E16", "avg_base"), _rows(*REF["strict"])
    out = {}
    for s in (1, 2, 3):
        m = st == s
        d = np.where(m, a.f1.to_numpy() - b.f1.to_numpy(), np.nan)
        diff, p = signflip(d, cl)
        out[str(s)] = {"n": int(m.sum()), "ours": float(a.f1[m].mean()), "ml": float(b.f1[m].mean()), "diff": diff, "p": p}
    for s, ph in zip(("1", "2", "3"), holm([out[s]["p"] for s in ("1", "2", "3")])):
        out[s]["p_holm"] = float(ph)
    return out


def b3_physics() -> dict:
    """No-physics ranker vs final method; blend vs KM amounts on the same chosen sets."""
    res = {}
    for proto in ("strict", "grouped"):
        cl = D.clusters(proto)
        a, n = _rows("E16", "avg_base", proto), _rows("E16", "avg_base_no_physics", proto)
        out = {"no_physics": {}}
        ps = []
        for m in METRICS:
            diff, p = signflip(a[m].to_numpy() - n[m].to_numpy(), cl)
            out["no_physics"][m] = {"with": float(a[m].mean()), "without": float(n[m].mean()), "diff": diff, "p": p}
            ps.append(p)
        for m, ph in zip(METRICS, holm(ps)):
            out["no_physics"][m]["p_holm"] = float(ph)
        # blend (alpha = 0.5, stored rows) vs KM amounts (alpha = 1) on the same chosen sets
        km = {f: pie.make_km(D, tr, pie.KM_DEFAULT) for f, (tr, _) in enumerate(D.folds[proto])}
        P_km, _, _ = pie.run_averaged_ranker(D, proto, pie.BASE_ALL, soft=False, alpha=1.0)
        r_km = pie.row_metrics(D.Y, P_km, D, km, pie.row_fold_index(D, proto))
        amt = ["active_mae", "family_mae", "union_mae", "total_abs_err"]
        out["blend_vs_km"], ps = {}, []
        for m in amt:
            diff, p = signflip(a[m].to_numpy() - r_km[m].to_numpy(), cl)
            out["blend_vs_km"][m] = {"blend": float(a[m].mean()), "km": float(r_km[m].mean()), "diff": diff, "p": p}
            ps.append(p)
        for m, ph in zip(amt, holm(ps)):
            out["blend_vs_km"][m]["p_holm"] = float(ph)
        res[proto] = out
    return res


def b4_conformal() -> dict:
    """Coverage over ALL predicted dyes (wrongly named dyes count as misses) and a shade-level split check."""
    P = np.load(pie.ROWS_DIR / "E16__avg_base__strict__pred.npy")
    fold = pie.row_fold_index(D, "strict")
    cl = D.clusters("strict")
    rows_, cols_ = np.nonzero((P > pie.ACTIVE) & A)
    lr = np.abs(np.log(P[rows_, cols_] / D.Y[rows_, cols_]))
    fam = D.dye_hue[cols_]
    out = {}
    for alpha in (0.1,):
        cover, q_used = np.zeros(len(lr), bool), np.zeros(len(lr))
        for f in range(5):
            for g in np.unique(fam):
                cal, te = (fold[rows_] != f) & (fam == g), (fold[rows_] == f) & (fam == g)
                s = np.sort(lr[cal]); m = len(s); kq = int(np.ceil((m + 1) * (1 - alpha)))
                q = np.inf if kq > m else s[kq - 1]
                cover[te], q_used[te] = lr[te] <= q, q
        n_correct = len(lr)
        pr_rows, pr_cols = np.nonzero(P > pie.ACTIVE)
        n_pred = len(pr_rows)
        shade_cov = pd.Series(cover).groupby(cl[rows_]).mean()
        out["correct_named"] = {"n_dyes": n_correct, "coverage": float(cover.mean()), "shades": int(shade_cov.size),
                                "shade_mean_coverage": float(shade_cov.mean())}
        out["all_predicted"] = {"n_dyes": int(n_pred), "covered": int(cover.sum()),
                                "coverage": float(cover.sum() / n_pred),
                                "share_correctly_named": float(n_correct / n_pred)}
        # shade-level split: calibrate on half of the shades, test on the other half (repeated)
        sh = np.unique(cl[rows_])
        covs = []
        for rep in range(200):
            rr = np.random.default_rng(rep)
            cal_sh = set(rr.choice(sh, len(sh) // 2, replace=False))
            calm = np.array([c in cal_sh for c in cl[rows_]])
            cv = []
            for g in np.unique(fam):
                s = np.sort(lr[calm & (fam == g)]); m = len(s); kq = int(np.ceil((m + 1) * (1 - alpha)))
                q = np.inf if kq > m else s[kq - 1]
                te = ~calm & (fam == g)
                cv.append(lr[te] <= q)
            covs.append(np.concatenate(cv).mean())
        out["shade_split_200"] = {"mean": float(np.mean(covs)), "p05": float(np.percentile(covs, 5)),
                                  "p95": float(np.percentile(covs, 95))}
    return out


def b5_strength_weighted() -> dict:
    """Hue-family MAE with each dye weighted by its integrated unit K/S (calibrated on the training fold)."""
    res = {}
    for proto in ("strict", "grouped"):
        cl = D.clusters(proto)
        Pa = np.load(pie.ROWS_DIR / f"E16__avg_base__{proto}__pred.npy")
        Pb = np.load(pie.ROWS_DIR / f"{REF[proto][0]}__{REF[proto][1]}__{proto}__pred.npy")
        ea, eb = np.zeros(len(D.Y)), np.zeros(len(D.Y))
        for f, (tr, va) in enumerate(D.folds[proto]):
            k = pie.make_km(D, tr, pie.KM_DEFAULT).k
            s = k.sum(1)
            s = s / np.median(s[A[tr].any(0)])  # strength relative to the median seen dye
            for E, P in ((ea, Pa), (eb, Pb)):
                err = 0.0
                for h in ("yellow", "red", "blue", "black"):
                    m = D.dye_hue == h
                    err = err + np.abs((P[va][:, m] * s[m]).sum(1) - (D.Y[va][:, m] * s[m]).sum(1))
                E[va] = err / 4
        diff, p = signflip(ea - eb, cl)
        res[proto] = {"ours": float(ea.mean()), "ml": float(eb.mean()), "diff": diff, "p": p}
    return res


def main() -> None:
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    if part in ("A", "all"):
        res["A1_colour_matching"] = a1_colour_matching(); print("A1", json.dumps(res["A1_colour_matching"], indent=1))
        res["A2_nested_calibration"] = a2_nested_calibration(); print("A2", json.dumps(res["A2_nested_calibration"], indent=1))
        res["A3_bootstrap_spectra"] = a3_bootstrap_spectra(); print("A3", json.dumps(res["A3_bootstrap_spectra"], indent=1))
    if part in ("B", "all"):
        for name, fn in (("B1_shade_weighted", b1_shade_weighted), ("B2_strata", b2_strata), ("B3_physics", b3_physics),
                         ("B4_conformal", b4_conformal), ("B5_strength_weighted", b5_strength_weighted)):
            res[name] = fn(); print(name, json.dumps(res[name], indent=1))
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
