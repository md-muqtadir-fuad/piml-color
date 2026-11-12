"""NSGA-II feature-group selection for the inverse recipe model, nested inside each outer fold.

Decision vector: 25 pre-defined feature groups (reflectance / K/S wavelength bands, spectral descriptor
families, substrate blocks). Objectives (all minimised, inner CV only):
    1. -mean active-dye F1 of the Extra Trees inverse model (top-3 post-processing)
    2. mean union MAE
    3. number of selected input columns
The final subset per outer fold is the Pareto-front member with the best inner F1 (ties: lower union
MAE, then fewer columns). The outer fold is evaluated once with the reference Extra Trees settings.

Usage:
    python scripts/62_nsga2_feature_selection.py --proto strict grouped
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

from pymoo.algorithms.moo.nsga2 import NSGA2  # noqa: E402
from pymoo.core.problem import ElementwiseProblem  # noqa: E402
from pymoo.operators.crossover.pntx import TwoPointCrossover  # noqa: E402
from pymoo.operators.mutation.bitflip import BitflipMutation  # noqa: E402
from pymoo.operators.sampling.rnd import BinaryRandomSampling  # noqa: E402
from pymoo.optimize import minimize  # noqa: E402
from sklearn.ensemble import ExtraTreesRegressor  # noqa: E402

POP, GENS, SEARCH_TREES, SEED = 24, 15, 60, 42
BANDS = [(400, 440), (450, 490), (500, 540), (550, 590), (600, 640), (650, 690), (700, 750)]


def feature_groups(cols: list[str]) -> dict[str, list[int]]:
    idx = {c: i for i, c in enumerate(cols)}
    g: dict[str, list[int]] = {}
    for kind, prefix, tag in (("R", "target_refl__reflectance_", "refl"), ("KS", "target_ks__ks_", "ks")):
        for lo, hi in BANDS:
            g[f"{kind}_{lo}-{hi}nm"] = [idx[f"{prefix}{w}nm"] for w in range(lo, hi + 1, 10) if f"{prefix}{w}nm" in idx]
        g[f"{tag}_summary_stats"] = [idx[c] for c in cols if c.startswith(f"feat_{tag}_") and
                                     any(s in c for s in ("_mean", "_std", "_min", "_max", "_range", "_area", "_slope"))
                                     and "deriv" not in c and "band" not in c]
        g[f"{tag}_derivatives"] = [idx[c] for c in cols if c.startswith(f"feat_{tag}_") and "deriv" in c]
        g[f"{tag}_band_means"] = [idx[c] for c in cols if c.startswith(f"feat_{tag}_band_mean")]
    g["substrate_weight"] = [idx[c] for c in ("production__gsm", "production__sample_weight") if c in idx]
    g["liquor_ratio"] = [idx[c] for c in ("production__l2r_numeric",) if c in idx]
    g["fibre_composition"] = [idx[c] for c in cols if c in {
        "production__fiber_total_pct", "production__modal", "production__viscose", "production__cotton",
        "production__polyester", "production__lycra", "production__is_lycra_blend", "production__is_polyester_blend",
        "feat_process_cotton_polyester_ratio"} or c.startswith("feat_cat_production_dominant_fiber")]
    g["fabric_type"] = [idx[c] for c in cols if c.startswith("feat_cat_production_fabric_")]
    g["knit_structure"] = [idx[c] for c in cols if c.startswith("feat_cat_production_kniting_")]
    assert sorted(i for v in g.values() for i in v) == list(range(len(cols))), "groups must partition the columns"
    return g


def et_predict(Xtr, Ytr, Xva, trees: int) -> np.ndarray:
    est = ExtraTreesRegressor(n_estimators=trees, max_features=1.0, min_samples_leaf=3, random_state=pie.SEED, n_jobs=1)
    return est.fit(Xtr, Ytr).predict(Xva)


def impute(X: np.ndarray, tr: np.ndarray) -> np.ndarray:
    med = np.nanmedian(X[tr], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return np.where(np.isfinite(X), X, med)


class FeatureProblem(ElementwiseProblem):
    def __init__(self, data, proto: str, tr: np.ndarray, groups: dict[str, list[int]]) -> None:
        super().__init__(n_var=len(groups), n_obj=3, xl=0, xu=1, vtype=bool)
        self.ds, self.groups = data, list(groups.values())  # pymoo's Problem owns `self.data`
        self.X = impute(data.X("spectral_substrate"), tr)
        self.splits = pie.inner_splits(data, proto, tr)
        self.cache: dict[tuple, tuple[float, float]] = {}

    def inner_score(self, mask: np.ndarray) -> tuple[float, float]:
        key = tuple(bool(m) for m in mask)
        if key not in self.cache:
            cols = [i for m, g in zip(mask, self.groups) if m for i in g]
            f1, umae, n = 0.0, 0.0, 0
            for itr, iva in self.splits:
                P = pie.feasible(et_predict(self.X[itr][:, cols], self.ds.Y[itr], self.X[iva][:, cols], SEARCH_TREES))
                m = pie.row_metrics(self.ds.Y[iva], P)
                f1 += m["f1"].sum()
                umae += m["union_mae"].sum()
                n += len(iva)
            self.cache[key] = (f1 / n, umae / n)
        return self.cache[key]

    def _evaluate(self, x, out, *args, **kwargs):
        mask = np.asarray(x, bool)
        if not mask.any():
            out["F"] = [0.0, 10.0, 0.0]
            return
        f1, umae = self.inner_score(mask)
        out["F"] = [-f1, umae, float(sum(len(g) for m, g in zip(mask, self.groups) if m))]


def run(proto: str, data) -> dict:
    cols = data.features["spectral_substrate"]
    groups = feature_groups(cols)
    names = list(groups)
    rf_idx = pie.row_fold_index(data, proto)
    km = {f: pie.make_km(data, tr, pie.KM_DEFAULT) for f, (tr, _) in enumerate(data.folds[proto])}
    X_all = data.X("spectral_substrate")
    P = np.zeros_like(data.Y)
    log, fronts = [], []
    for f, (tr, va) in enumerate(data.folds[proto]):
        t0 = time.time()
        prob = FeatureProblem(data, proto, tr, groups)
        full = prob.inner_score(np.ones(len(names), bool))
        algo = NSGA2(pop_size=POP, sampling=BinaryRandomSampling(), crossover=TwoPointCrossover(),
                     mutation=BitflipMutation(), eliminate_duplicates=True)
        res = minimize(prob, algo, ("n_gen", GENS), seed=SEED + f, verbose=False)
        front = [(np.asarray(x, bool), -F[0], F[1], F[2]) for x, F in zip(np.atleast_2d(res.X), np.atleast_2d(res.F))]
        mask, f1_in, umae_in, n_cols = sorted(front, key=lambda t: (-t[1], t[2], t[3]))[0]
        sel = [i for m, g in zip(mask, groups.values()) if m for i in g]
        X = impute(X_all, tr)
        P[va] = pie.feasible(et_predict(X[tr][:, sel], data.Y[tr], X[va][:, sel], 120))
        chosen = [n for m, n in zip(mask, names) if m]
        log.append({"protocol": proto, "fold": f, "n_groups": int(mask.sum()), "n_columns": int(n_cols),
                    "inner_f1_selected": f1_in, "inner_umae_selected": umae_in, "inner_f1_all_features": full[0],
                    "inner_umae_all_features": full[1], "front_size": len(front), "evaluations": len(prob.cache),
                    "groups": json.dumps(chosen)})
        fronts += [{"protocol": proto, "fold": f, "inner_f1": t[1], "inner_umae": t[2], "n_columns": t[3],
                    "groups": json.dumps([n for m, n in zip(t[0], names) if m])} for t in front]
        print(f"  [NSGA2/{proto}] fold {f}: {int(mask.sum())} groups / {int(n_cols)} cols, inner F1 {f1_in:.3f} "
              f"(all features {full[0]:.3f}), front {len(front)}, {len(prob.cache)} evals, {time.time() - t0:.0f}s", flush=True)
    rows = pie.row_metrics(data.Y, P, data, km, rf_idx)
    res = pie.record("N01", "nsga2_et_selected_features", proto, rows, P, data)
    pd.DataFrame(log).to_csv(pie.OUT_DIR / f"N01_{proto}_selection.csv", index=False)
    pd.DataFrame(fronts).to_csv(pie.OUT_DIR / f"N01_{proto}_pareto_fronts.csv", index=False)
    return res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proto", nargs="+", default=["strict"])
    args = parser.parse_args()
    pie.RESULTS_CSV = pie.OUT_DIR / "results_n01.csv"
    pie.STRATA_CSV_NAME = "results_by_stratum_n01.csv"
    data = pie.Data()
    out = [run(p, data) for p in args.proto]
    pie.print_table(out)


if __name__ == "__main__":
    main()
