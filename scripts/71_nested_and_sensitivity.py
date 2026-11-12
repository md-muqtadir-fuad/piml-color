"""Second-round review analyses (2026-11-12): nested selection, identifiability variants, colour and coverage checks.

N  Fully nested pipeline. Inside every outer training fold, chosen on inner folds only:
     - Extra Trees settings (max_features x min_samples_leaf grid) for the ML amounts and ML features of the method,
       by inner-CV union MAE of the top-3 sparsified prediction;
     - the ML reference's settings and input set (spectral or spectral + substrate), by the same criterion;
     - the ranker variant (probabilities averaged over the L2 grid, or one L2 value), by held-out NLL with each
       inner fold left out in turn (as in the nested E12 variant of script 60);
     - the blend weight alpha, by inner union MAE at the selected variant (as in E12).
   KM constants are already chosen on inner folds (script 60). The random-forest dye prior keeps its fixed settings.
   For a non-default Extra Trees setting, the cached candidate blocks are reused: the ML predictions are replaced and
   each block is extended with that model's top-3 sets, the only candidates that depend on the ML model.
   Run on the main strict split and the ten repeated shade-to-fold assignments of script 70 (experiment R).
I  Identifiability variants of exhaustive KM matching on measured targets (strict CV): baseline (beta 0.3,
   eta 0.01), eta = 0, Saunderson correction (k1 0.015, k2 0.6) and both.
F  KM-forward colour error (CMC(2:1), max over D65 and F11) of the recommended recipes, from stored predictions.
C  Coverage of the 90% per-family amount intervals in the ten repeated fold assignments.
T  Main comparison restricted to the 209 cotton-dominant lab dips (stored predictions).

Outputs: outputs/review_experiments/ (N_*, I_*, F_*, C_*, T_*); new bundle caches in outputs/review_experiments/cache_nested.
Run: python scripts/71_nested_and_sensitivity.py --only F T C I N
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear, nnls
from scipy.special import softmax
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rev = _load("rev70", "70_review_experiments.py")   # signflip, holm, random_shade_folds
pie = rev.pie                                        # the same script-60 module instance

from piml_color_recipe.inverse.km_physics import (  # noqa: E402
    delta_e_cmc, enumerate_subsets, ks_to_reflectance, reflectance_to_ks, reflectance_to_lab,
)

OUT = ROOT / "outputs" / "review_experiments"
MAIN_CACHE = pie.CACHE_DIR
REP_CACHE = OUT / "cache"
NEST_CACHE = OUT / "cache_nested"
METRICS = rev.METRICS
DEFAULT_ET = dict(max_features=1.0, min_samples_leaf=3)
ET_GRID = [DEFAULT_ET] + [dict(max_features=mf, min_samples_leaf=ml) for mf in (1.0, 0.5, 0.2) for ml in (1, 3, 5)
                          if dict(max_features=mf, min_samples_leaf=ml) != DEFAULT_ET]
VARIANTS = {"avg": list(pie.L2_AVG), **{f"l2_{g:g}": [g] for g in pie.L2_AVG}}
ML_REF_STORED = {"strict": "ml_et_spectral_substrate_feasible", "grouped": "ml_et_spectral_feasible"}


# ------------------------------------------------------------------ shared helpers
def split(name: str, data) -> tuple[str, Path]:
    """Register a split; returns (protocol name, default-bundle cache dir) and sets the matching tree seed."""
    if name == "main":
        pie.SEED = 42
        return "strict", MAIN_CACHE
    r = int(name.removeprefix("r"))
    proto = f"strict_r{r}"
    pie.PROTOCOLS[proto] = ("random_shade_folds", "target_id")
    data.folds[proto] = rev.random_shade_folds(data, 1000 + r)
    pie.SEED = 42 + r
    return proto, REP_CACHE


def et_predict(data, tr, va, cfg: dict, group: str) -> np.ndarray:
    est = make_pipeline(SimpleImputer(strategy="median"),
                        ExtraTreesRegressor(n_estimators=120, random_state=pie.SEED, n_jobs=-1, **cfg))
    return est.fit(data.X(group)[tr], data.Y[tr]).predict(data.X(group)[va])


def union_mae(data, P: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return pie.row_metrics(data.Y[rows], P)["union_mae"].to_numpy()


def tests_vs(a: pd.DataFrame, b: pd.DataFrame, cl: np.ndarray) -> dict:
    res = {m: rev.signflip(a[m].to_numpy(), b[m].to_numpy(), cl) for m in METRICS}
    ph = rev.holm([p for _, p in res.values()])
    return {m: {"a": float(np.nanmean(a[m])), "b": float(np.nanmean(b[m])), "delta": d, "p": p, "p_holm": h}
            for (m, (d, p)), h in zip(res.items(), ph)}


# ------------------------------------------------------------------ N: fully nested pipeline
def select_et(data, proto: str, tr: np.ndarray, group: str) -> tuple[dict, float]:
    splits = pie.inner_splits(data, proto, tr)
    best, best_s = None, np.inf
    for cfg in ET_GRID:                         # default first: ties keep the default
        errs = [union_mae(data, pie.feasible(et_predict(data, itr, iva, cfg, group)), iva) for itr, iva in splits]
        s = float(np.mean(np.concatenate(errs)))
        if s < best_s - 1e-12:
            best, best_s = cfg, s
    return best, best_s


def extend_block(data, b: dict, tr: np.ndarray, km_kw: dict, ml: np.ndarray) -> dict:
    """Add the rows' ML top-3 sets that the cached block lacks, with their bounded KM fits and per-candidate fields
    computed exactly as in script 60's build_block. Only these sets depend on the ML model; everything else in the
    block (KM fits of the training-pool candidates, dye prior) is unchanged."""
    A = data.Y > pie.ACTIVE
    keys = {tuple(int(d) for d in s[s >= 0]) for s in b["subsets"]}
    new = sorted({tuple(sorted(int(d) for d in s)) for s in pie.ml_top_sets(ml) if 0 < len(s) <= 3} - keys)
    if not new:
        return b
    arr = -np.ones((len(new), b["subsets"].shape[1]), dtype=int)
    for i, s in enumerate(new):
        arr[i, : len(s)] = s
    model = pie.make_km(data, tr, {k: v for k, v in km_kw.items() if not k.startswith("_")})
    c_max = float(data.Y[tr].max())
    va = b["va"]
    de = np.zeros((len(va), len(new)))
    conc = np.zeros((len(va), len(new), arr.shape[1]))
    for i, r in enumerate(va):
        sol = pie.solve_subsets(model, data.KS[r], arr, c_max=c_max)
        de[i], conc[i] = np.maximum(sol.de_d65, sol.de_f11), sol.conc
    out = dict(b)
    out["subsets"] = np.vstack([b["subsets"], arr])
    out["de"] = np.hstack([b["de"], de])
    out["conc"] = np.concatenate([b["conc"], conc], axis=1)
    out["logit_sum"] = np.hstack([b["logit_sum"], np.stack([pie.subset_logit_sum(arr, b["prob"][i]) for i in range(len(va))])])
    out["counts"] = np.concatenate([b["counts"], np.zeros(len(new))])
    n_c = len(out["subsets"])
    out["set_logp"] = np.log((out["counts"] + 0.5) / (out["counts"].sum() + 0.5 * n_c))
    sizes = (out["subsets"] >= 0).sum(1)
    size_counts = np.array([(A[tr].sum(1) == k).sum() for k in range(out["subsets"].shape[1] + 1)], dtype=float)
    out["card_logp"] = np.log((size_counts[sizes] + 1) / (size_counts.sum() + len(size_counts)))
    cand_mask = np.zeros((len(new), data.Y.shape[1]), bool)
    for c, s in enumerate(arr):
        cand_mask[c, s[s >= 0]] = True
    true_sets = A[va]
    tp = (true_sets[:, None, :] & cand_mask[None]).sum(-1)
    npred, ntrue = cand_mask.sum(1)[None, :], true_sets.sum(1)[:, None]
    prec = np.divide(tp, npred, out=np.zeros(tp.shape), where=npred > 0)
    rec = np.divide(tp, ntrue, out=np.zeros(tp.shape), where=ntrue > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(tp.shape), where=(prec + rec) > 0)
    full = np.zeros((len(va), len(new), data.Y.shape[1]))
    for slot in range(arr.shape[1]):
        ok = np.flatnonzero(arr[:, slot] >= 0)
        full[:, ok, arr[ok, slot]] = conc[:, ok, slot]
    union = true_sets[:, None, :] | cand_mask[None]
    umae = (np.abs(full - data.Y[va][:, None, :]) * union).sum(-1) / union.sum(-1)
    out["cand_mask"] = np.vstack([b["cand_mask"], cand_mask])
    out["f1"] = np.hstack([b["f1"], f1])
    out["umae"] = np.hstack([b["umae"], umae])
    return out


def nested_bundle(data, proto: str, f: int, cfg: dict, default_cache: Path) -> tuple[dict, str]:
    """Bundle of the final method with the ML model of setting cfg: the cached default bundle, with the inner and
    outer ML predictions replaced and each block extended by the new ML top-3 sets."""
    pie.CACHE_DIR = default_cache
    bd = pie.fold_bundle(data, proto, f)
    if cfg == DEFAULT_ET:
        return bd, proto
    km_kw = (pie.BUNDLE_VERSIONS[pie.BUNDLE_VERSION] or pie.KM_GRID)[bd["km"]]
    tr, va = data.folds[proto][f]
    splits = pie.inner_splits(data, proto, tr)
    inner_ml = [et_predict(data, itr, iva, cfg, "spectral_substrate") for itr, iva in splits]
    outer_ml = et_predict(data, tr, va, cfg, "spectral_substrate")
    inner = [extend_block(data, b, itr, km_kw, ml) for b, ml, (itr, _) in zip(bd["inner"], inner_ml, splits)]
    outer = extend_block(data, bd["outer"], tr, km_kw, outer_ml)
    return {"km": bd["km"], "inner": inner, "inner_ml": inner_ml, "outer": outer, "outer_ml": outer_ml}, proto


def _inner_items(data, bd: dict, splits, use) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    items = []
    for b, ml, (itr, _) in zip(bd["inner"], bd["inner_ml"], splits):
        al = pie.allowed_candidates(data, b, ml, itr)
        F = pie._fix_set_logp(pie.block_features(data, b, ml, itr, use), use, b, al)
        items.append((F, pie._positive_masked(b, al), al))
    return items


def _standardize(items, idx):
    flat = np.concatenate([items[k][0][items[k][2]] for k in idx])
    return flat.mean(0), flat.std(0) + 1e-9


def _probs(train, F_te, al_te, l2s) -> np.ndarray:
    return np.mean([softmax(np.where(al_te, F_te @ pie.fit_choice_model_masked(train, l2), -np.inf), axis=1)
                    for l2 in l2s], axis=0)


def nested_fold(data, proto: str, f: int, default_cache: Path, use) -> tuple[np.ndarray, np.ndarray, dict]:
    tr, va = data.folds[proto][f]
    cfg_m, _ = select_et(data, proto, tr, "spectral_substrate")
    ref = {g: select_et(data, proto, tr, g) for g in ("spectral", "spectral_substrate")}
    g_ref = min(ref, key=lambda g: ref[g][1])
    P_ref = pie.feasible(et_predict(data, tr, va, ref[g_ref][0], g_ref))

    bd, tag = nested_bundle(data, proto, f, cfg_m, default_cache)
    splits = pie.inner_splits(data, tag, tr)
    items = _inner_items(data, bd, splits, use)
    k_all = list(range(len(items)))
    # ranker variant by held-out NLL, each inner fold left out in turn; keep the held-out choices per variant
    nll = {v: [] for v in VARIANTS}
    choices = {v: [] for v in VARIANTS}
    for h in k_all:
        rest = [k for k in k_all if k != h]
        mu, sd = _standardize(items, rest)
        train = [((items[k][0] - mu) / sd, items[k][1], items[k][2]) for k in rest]
        F_h, pos_h, al_h = (items[h][0] - mu) / sd, items[h][1], items[h][2]
        per_l2 = {g: _probs(train, F_h, al_h, [g]) for g in pie.L2_AVG}
        for v, l2s in VARIANTS.items():
            prob = np.mean([per_l2[g] for g in l2s], axis=0)
            has = pos_h >= 0                                   # pos: index of the positive candidate, -1 if none
            p_pos = prob[np.flatnonzero(has), pos_h[has]]
            nll[v].append(-np.log(np.clip(p_pos, 1e-12, None)))
            choices[v].append(prob.argmax(1))
    nll_mean = {v: float(np.mean(np.concatenate(x))) for v, x in nll.items()}
    variant = min(nll_mean, key=lambda v: (round(nll_mean[v], 9), v != "avg"))
    alpha_err = {a: float(np.mean(np.concatenate([
        union_mae(data, pie.chosen_vectors(data, b, c, ml, a), b["va"])
        for b, c, ml in zip(bd["inner"], choices[variant], bd["inner_ml"])]))) for a in pie.ALPHAS}
    alpha = min(alpha_err, key=lambda a: (round(alpha_err[a], 9), abs(a - 0.5)))
    # final fit on all inner folds, applied to the outer validation fold
    mu, sd = _standardize(items, k_all)
    train = [((F - mu) / sd, pos, al) for F, pos, al in items]
    al_out = pie.allowed_candidates(data, bd["outer"], bd["outer_ml"], tr)
    F_out = (pie._fix_set_logp(pie.block_features(data, bd["outer"], bd["outer_ml"], tr, use), use, bd["outer"], al_out)
             - mu) / sd
    prob = _probs(train, F_out, al_out, VARIANTS[variant])
    P_ours = pie.chosen_vectors(data, bd["outer"], prob.argmax(1), bd["outer_ml"], alpha)
    log = {"fold": f, "et_method": cfg_m, "et_ref": ref[g_ref][0], "ref_inputs": g_ref, "variant": variant,
           "alpha": alpha, "nll": nll_mean, "alpha_union_mae": alpha_err}
    return P_ours, P_ref, log


def exp_n(data, splits=None) -> pd.DataFrame:
    splits = splits or ["main"] + [f"r{r}" for r in range(1, 11)]
    path = OUT / "N_nested.csv"
    done = pd.read_csv(path) if path.exists() else pd.DataFrame()
    rows = done.to_dict("records")
    logs_path = OUT / "N_nested_selections.json"
    logs = json.loads(logs_path.read_text()) if logs_path.exists() else {}
    use = pie.BASE_ALL
    for name in splits:
        if not done.empty and name in set(done["split"]):
            continue
        t0 = time.time()
        proto, cache = split(name, data)
        P_ours, P_ref = np.zeros_like(data.Y), np.zeros_like(data.Y)
        logs[name] = []
        for f, (_, va) in enumerate(data.folds[proto]):
            po, pr, lg = nested_fold(data, proto, f, cache, use)
            P_ours[va], P_ref[va] = po, pr
            logs[name].append(lg)
            print(f"  [N/{name}] fold {f}: ET {lg['et_method']}, ref {lg['ref_inputs']} {lg['et_ref']}, "
                  f"ranker {lg['variant']}, alpha {lg['alpha']}", flush=True)
        np.save(OUT / f"N_{name}_ours_pred.npy", P_ours)
        np.save(OUT / f"N_{name}_ref_pred.npy", P_ref)
        m_o, m_r = pie.row_metrics(data.Y, P_ours, data), pie.row_metrics(data.Y, P_ref, data)
        t = tests_vs(m_o, m_r, data.clusters(proto))
        rec = {"split": name, "seconds": round(time.time() - t0)}
        for m, v in t.items():
            rec |= {f"{m}_ours": v["a"], f"{m}_ml": v["b"], f"{m}_delta": v["delta"], f"{m}_p": v["p"],
                    f"{m}_p_holm": v["p_holm"]}
        rows.append(rec)
        pd.DataFrame(rows).to_csv(path, index=False)
        logs_path.write_text(json.dumps(logs, indent=1, default=str))
        print(f"[N] {name}: famMAE {rec['family_mae_ours']:.3f} vs {rec['family_mae_ml']:.3f} "
              f"(d {rec['family_mae_delta']:+.3f}, p_Holm {rec['family_mae_p_holm']:.3f}); F1 {rec['f1_ours']:.3f} vs "
              f"{rec['f1_ml']:.3f}; {rec['seconds']}s", flush=True)
    df = pd.DataFrame(rows)
    rep = df[df["split"] != "main"]
    summ = {"n_repeats": int(len(rep))}
    for m in METRICS:
        summ[m] = {"main": {k: float(df.loc[df.split == "main", f"{m}_{k}"].iloc[0])
                            for k in ("ours", "ml", "delta", "p_holm")} if (df.split == "main").any() else None,
                   "repeats_delta_mean": float(rep[f"{m}_delta"].mean()), "repeats_delta_min": float(rep[f"{m}_delta"].min()),
                   "repeats_delta_max": float(rep[f"{m}_delta"].max()), "repeats_ours_mean": float(rep[f"{m}_ours"].mean()),
                   "repeats_ml_mean": float(rep[f"{m}_ml"].mean()),
                   "n_favor_ours": int(((rep[f"{m}_delta"] > 0) if m in ("f1", "exact_set") else (rep[f"{m}_delta"] < 0)).sum()),
                   "n_sig_holm": int((rep[f"{m}_p_holm"] < 0.05).sum()),
                   "n_sig_holm_favor_ours": int(((rep[f"{m}_p_holm"] < 0.05) & ((rep[f"{m}_delta"] > 0) if m in ("f1", "exact_set")
                                                                                else (rep[f"{m}_delta"] < 0))).sum())}
    sel = [lg for v in logs.values() for lg in v]
    summ["selections"] = {k: pd.Series([json.dumps(lg[k], default=str) for lg in sel]).value_counts().to_dict()
                          for k in ("et_method", "et_ref", "ref_inputs", "variant", "alpha")}
    (OUT / "N_nested_summary.json").write_text(json.dumps(summ, indent=1))
    print("[N] summary:", json.dumps(summ, indent=1), flush=True)
    return df


# ------------------------------------------------------------------ I: identifiability variants
K1, K2 = 0.015, 0.6   # k1 just below the lowest measured target reflectance (0.020), so the inverse is defined


def saund_inv(rm: np.ndarray) -> np.ndarray:
    return np.clip((rm - K1) / (1 - K1 - K2 + K2 * rm), 1e-4, 1.0)


def saund_fwd(ri: np.ndarray) -> np.ndarray:
    return K1 + (1 - K1) * (1 - K2) * ri / (1 - K2 * ri)


def fit_sets(model, target_ks, subsets, c_max, saunderson: bool, lab65, lab11):
    """Bounded KM fit of every subset (as km_physics.solve_subsets); colour scored in measured-reflectance space."""
    from piml_color_recipe.inverse.km_physics import _row_weights
    w = _row_weights(target_ks, model.weighting)
    rhs = (target_ks - model.substrate) * w
    k_ext = np.vstack([model.k, np.zeros((1, model.k.shape[1]))])
    wk = k_ext * w[None, :]
    u_max = float(model.transform(np.array([c_max]))[0])
    u = np.zeros(subsets.shape)
    for i, s in enumerate(subsets):
        idx = s[s >= 0]
        if len(idx) == 1:
            a = wk[idx[0]]
            den = float(a @ a)
            u[i, 0] = min(max(float(a @ rhs) / den, 0.0), u_max) if den > 0 else 0.0
        else:
            sol, _ = nnls(wk[idx].T, rhs)
            if sol.max() > u_max:
                sol = lsq_linear(wk[idx].T, rhs, bounds=(0.0, u_max), method="bvls").x
            u[i, : len(idx)] = sol
    ks_pred = model.substrate[None, :] + np.einsum("ns,nsw->nw", u, k_ext[subsets])
    refl = ks_to_reflectance(ks_pred)
    if saunderson:
        refl = saund_fwd(refl)
    return np.maximum(delta_e_cmc(lab65[None, :], reflectance_to_lab(refl, "D65")),
                      delta_e_cmc(lab11[None, :], reflectance_to_lab(refl, "FL11")))


def exp_i(data) -> dict:
    A = data.Y > pie.ACTIVE
    ks_s = reflectance_to_ks(saund_inv(data.R))
    variants = {"baseline": (0.01, False), "eta0": (0.0, False), "saunderson": (0.01, True), "eta0_saunderson": (0.0, True)}
    out = {}
    for name, (eta, saund) in variants.items():
        t0 = time.time()
        d2 = copy.copy(data)
        d2.KS = ks_s if saund else data.KS
        kw = dict(pie.KM_DEFAULT, shrink=eta)
        ranks, best_true, f1s, same_fam, true_err, de_true_fit, de_best = [], [], [], [], [], [], []
        for f, (tr, va) in enumerate(data.folds["strict"]):
            model = pie.make_km(d2, tr, kw)
            usable = np.flatnonzero(A[tr].any(0))
            rows = [r for r in va if A[r].any() and set(np.flatnonzero(A[r])) <= set(usable)]
            subsets = enumerate_subsets(usable, 3)
            keys = [tuple(sorted(s[s >= 0])) for s in subsets]
            c_max = float(data.Y[tr].max())
            for r in rows:
                de = fit_sets(model, d2.KS[r], subsets, c_max, saund, data.lab_d65[r], data.lab_f11[r])
                ti = keys.index(tuple(sorted(np.flatnonzero(A[r]))))
                rank = 1 + int((de < de[ti] - 1e-9).sum())
                ranks.append(rank)
                best_true.append(rank == 1)
                order = np.argsort(de)
                bw = next(c for c in order if c != ti)                      # best-fitting wrong set
                b, t = set(keys[int(order[0])]), set(keys[ti])
                f1s.append(2 * len(b & t) / (len(b) + len(t)))
                same_fam.append(sorted(data.dye_hue[list(keys[bw])]) == sorted(data.dye_hue[list(keys[ti])]))
                de_true_fit.append(float(de[ti]))
                de_best.append(float(de[bw]))
                refl = ks_to_reflectance(model.predict_ks(data.Y[[r]]))
                if saund:
                    refl = saund_fwd(refl)
                true_err.append(float(max(delta_e_cmc(data.lab_d65[[r]], reflectance_to_lab(refl, "D65"))[0],
                                          delta_e_cmc(data.lab_f11[[r]], reflectance_to_lab(refl, "FL11"))[0])))
        ranks = np.array(ranks)
        out[name] = {"n_rows": int(len(ranks)), "true_best_share": float(np.mean(best_true)),
                     "median_rank": float(np.median(ranks)), "rank_iqr": [float(np.percentile(ranks, 25)),
                                                                          float(np.percentile(ranks, 75))],
                     "top10_share": float(np.mean(ranks <= 10)), "f1_of_best": float(np.mean(f1s)),
                     "best_wrong_same_family_share": float(np.mean(same_fam)),
                     "median_de_true_set_fitted": float(np.median(de_true_fit)),
                     "median_de_best_wrong": float(np.median(de_best)),
                     "median_km_error_true_amounts": float(np.median(true_err))}
        print(f"[I] {name}: {out[name]} ({time.time() - t0:.0f}s)", flush=True)
        (OUT / "I_identifiability_variants.json").write_text(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------ F: forward colour error of recommended recipes
def exp_f(data) -> dict:
    out = {}
    for proto in ("strict", "grouped"):
        rf_idx = pie.row_fold_index(data, proto)
        km = {f: pie.make_km(data, tr, pie.KM_DEFAULT) for f, (tr, _) in enumerate(data.folds[proto])}
        true_de = pie.row_metrics(data.Y, data.Y, data, km, rf_idx)["km_de_max"].to_numpy()
        meth = {"ours": ("E16", "avg_base"), "ml_reference": ("E00b", ML_REF_STORED[proto]),
                "retrieval": ("E00b", "nearest_historical_spectral")}
        de = {k: pie.load_rows(e, m, proto)["km_de_max"].to_numpy() for k, (e, m) in meth.items()}
        de["true_recipe"] = true_de
        cl = data.clusters(proto)
        seen = data.strata(proto) < 3          # every dye of the true recipe occurs in the training fold
        res = {}
        for subset, mask in (("all", np.ones(len(cl), bool)), ("seen_dyes", seen)):
            r = {k: {"n": int(mask.sum()), "median": float(np.nanmedian(v[mask])),
                     "share_le_1": float(np.nanmean(v[mask] <= 1.0)), "share_le_2": float(np.nanmean(v[mask] <= 2.0))}
                 for k, v in de.items()}
            tests = {f"ours_vs_{b}": rev.signflip(de["ours"][mask], de[b][mask], cl[mask])
                     for b in ("ml_reference", "retrieval")}
            ph = rev.holm([p for _, p in tests.values()])
            r["tests"] = {k: {"delta_mean": d, "p": p, "p_holm": h} for (k, (d, p)), h in zip(tests.items(), ph)}
            res[subset] = r
        out[proto] = res
        print(f"[F/{proto}]", json.dumps(res, indent=1), flush=True)
    (OUT / "F_forward_de.json").write_text(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------ C: interval coverage across repeats
def interval_coverage(data, P: np.ndarray, fold_of: np.ndarray, level: float = 0.9) -> dict:
    A = data.Y > pie.ACTIVE
    pred = P > pie.ACTIVE
    rows, cols = np.nonzero(pred & A)
    lr = np.abs(np.log(P[rows, cols] / data.Y[rows, cols]))
    fam = data.dye_hue[cols]
    fr = fold_of[rows]
    q_of = {}
    cover = np.zeros(len(lr), bool)
    for f in np.unique(fold_of):
        for g in np.unique(fam):
            cal, te = (fr != f) & (fam == g), (fr == f) & (fam == g)
            s = np.sort(lr[cal]); m = len(s); kq = int(np.ceil((m + 1) * level))
            q = np.inf if kq > m else s[kq - 1]
            cover[te] = lr[te] <= q
            q_of[(f, g)] = q
    n_pred = int(pred.sum())
    return {"n_correct": int(len(lr)), "coverage_correct": float(cover.mean()), "n_predicted": n_pred,
            "coverage_all_predicted": float(cover.sum() / n_pred),
            "factor_by_family": {g: float(np.exp(np.mean([q for (f, gg), q in q_of.items() if gg == g and np.isfinite(q)])))
                                 for g in np.unique(fam)}}


def exp_c(data) -> dict:
    out = {}
    P = np.load(pie.ROWS_DIR / "E16__avg_base__strict__pred.npy")
    out["main"] = interval_coverage(data, P, pie.row_fold_index(data, "strict"))
    for r in range(1, 11):
        proto, cache = split(f"r{r}", data)
        pie.CACHE_DIR = cache
        P, _, _ = pie.run_averaged_ranker(data, proto, pie.BASE_ALL, soft=False, own_ml_only=True)
        np.save(OUT / f"C_r{r}_ours_pred.npy", P)
        out[f"r{r}"] = interval_coverage(data, P, pie.row_fold_index(data, proto))
        print(f"[C] r{r}: {out[f'r{r}']}", flush=True)
    rep = [out[f"r{r}"] for r in range(1, 11)]
    out["summary"] = {k: {"mean": float(np.mean([x[k] for x in rep])), "min": float(np.min([x[k] for x in rep])),
                          "max": float(np.max([x[k] for x in rep]))} for k in ("coverage_correct", "coverage_all_predicted")}
    print("[C] main:", out["main"], "\n[C] repeats:", out["summary"], flush=True)
    (OUT / "C_coverage_repeats.json").write_text(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------ T: cotton-dominant subset
def exp_t(data) -> dict:
    cotton = data.df["production__cotton"].to_numpy(float) >= 50
    out = {"n_cotton": int(cotton.sum())}
    for proto in ("strict", "grouped"):
        a = pie.load_rows("E16", "avg_base", proto)[cotton].reset_index(drop=True)
        b = pie.load_rows("E00b", ML_REF_STORED[proto], proto)[cotton].reset_index(drop=True)
        out[proto] = tests_vs(a, b, data.clusters(proto)[cotton])
        print(f"[T/{proto}] " + "; ".join(f"{m} {v['a']:.3f} vs {v['b']:.3f} (d {v['delta']:+.3f}, p_Holm {v['p_holm']:.3f})"
                                          for m, v in out[proto].items()), flush=True)
    (OUT / "T_cotton_only.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=["F", "T", "C", "I", "N"])
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    data = pie.Data()
    for k in args.only:
        t0 = time.time()
        {"N": exp_n, "I": exp_i, "F": exp_f, "C": exp_c, "T": exp_t}[k](data)
        print(f"== {k} finished in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
