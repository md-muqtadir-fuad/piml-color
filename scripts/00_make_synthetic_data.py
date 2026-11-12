"""Generate a fully synthetic dataset with the structure the analysis pipeline expects.

No industrial data are read or used. Every dye, shade, recipe and lab dip below is invented:

* a palette of 36 fictitious reactive dyes (11 yellow, 12 red, 12 blue, 1 black that is never used), whose unit
  K/S spectra are smooth Gaussian absorption bands, not measured or calibrated spectra;
* 51 fictitious target shades, each made from one to three dyes (mostly one per hue family), with Langmuir-type
  uptake u(c) = c / (1 + 0.3 c) and smooth spectral noise standing in for model mismatch;
* 85 recipes (some shades get a second or third recipe with a within-hue dye swap or other amounts) and 212 lab
  dips (small amount variations of their recipe);
* substrate variables, process auxiliaries (salt and soda ash dosed from the total dye, as in practice) and
  lab-dip color differences drawn from simple distributions.

Results obtained on these data do not reproduce the numbers of the paper; they only show that the code runs.

Outputs (paths read by the pipeline):
    data/processed/features_supervised.csv
    data/global_naming/global_dye_mapping_active_36.csv
    data/interim/recipe_dye_long.csv
    data/splits/groupkfold_target_id.json, data/splits/groupkfold_recipe_no.json

Run: python scripts/00_make_synthetic_data.py [--seed 2026]
"""
from __future__ import annotations

import argparse
import json
import string
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from piml_color_recipe.features.spectral_features import add_spectral_descriptors  # noqa: E402
from piml_color_recipe.inverse.km_physics import WAVELENGTHS, ks_to_reflectance, reflectance_to_ks  # noqa: E402

N_SHADES, N_RECIPES, N_LABDIPS = 51, 85, 212
HUES = ["yellow"] * 11 + ["red"] * 12 + ["blue"] * 12 + ["black"]
BAND = {"yellow": ((420, 455), (28, 40), (0.6, 2.0)), "red": ((505, 545), (25, 38), (0.6, 2.0)),
        "blue": ((590, 640), (35, 50), (0.5, 1.6))}
BETA = 0.3
ACTIVE = 0.01
FABRICS = ["single jersey", "rib", "interlock", "fleece", "pique"]


def palette(rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    wl = WAVELENGTHS.astype(float)
    rows, spectra, count = [], [], {}
    for hue in HUES:
        count[hue] = count.get(hue, 0) + 1
        name = f"synthetic {hue} {count[hue]:02d}"
        if hue == "black":
            k = np.full(len(wl), 1.5)
        else:
            (c_lo, c_hi), (s_lo, s_hi), (a_lo, a_hi) = BAND[hue]
            c, s, a = rng.uniform(c_lo, c_hi), rng.uniform(s_lo, s_hi), rng.uniform(a_lo, a_hi)
            k = a * np.exp(-0.5 * ((wl - c) / s) ** 2) + 0.01 * a
            if hue in ("red", "blue"):  # weaker secondary band on the short-wavelength side
                k += 0.2 * a * np.exp(-0.5 * ((wl - (c - 70)) / 30) ** 2)
        spectra.append(k)
        rows.append({"global_dye_name": f"Dye {len(rows) + 1:02d}", "dye_name_clean": name,
                     "dye_column": f"dye_{name.replace(' ', '_')}_pct", "original_brand_inferred": "synthetic",
                     "original_hue_inferred": hue})
    return pd.DataFrame(rows), np.array(spectra)


def km_ks(c: np.ndarray, k: np.ndarray, s: np.ndarray) -> np.ndarray:
    u = c / (1.0 + BETA * c)
    return s + u @ k


def split_counts(rng: np.random.Generator, n_groups: int, total: int, cap: int | None = None) -> np.ndarray:
    counts = np.ones(n_groups, dtype=int)
    weights = rng.gamma(0.8, 1.0, n_groups)
    while counts.sum() < total:
        j = rng.choice(n_groups, p=weights / weights.sum())
        if cap is None or counts[j] < cap:
            counts[j] += 1
    return counts


def make(seed: int) -> None:
    rng = np.random.default_rng(seed)
    mapping, K = palette(rng)
    hue = mapping["original_hue_inferred"].to_numpy()
    D = len(hue)
    wl = WAVELENGTHS
    substrate = 0.015 + 0.01 * np.exp(-(wl - 400) / 60.0)
    popularity = np.zeros(D)
    for h in ("yellow", "red", "blue"):
        idx = np.flatnonzero(hue == h)
        popularity[idx] = 1.0 / np.arange(1, len(idx) + 1) ** 1.1

    def pick(h: str, exclude: int | None = None) -> int:
        idx = np.flatnonzero(hue == h)
        p = popularity[idx].copy()
        if exclude is not None:
            p[idx == exclude] = 0.0
        return int(rng.choice(idx, p=p / p.sum()))

    # ---- shades: dye set, amounts and target spectrum
    shades = []
    for s in range(N_SHADES):
        n = rng.choice([3, 2, 1], p=[0.85, 0.10, 0.05])
        hues = ["yellow", "red", "blue"] if n == 3 else list(rng.choice(["yellow", "red", "blue"], size=n, replace=False))
        dyes = [pick(h) for h in hues]
        total = float(np.clip(rng.lognormal(np.log(0.9), 0.6) if rng.random() < 0.5 else rng.lognormal(np.log(3.5), 0.35), 0.02, 7.0))
        conc = np.zeros(D)
        conc[dyes] = np.maximum(total * rng.dirichlet(np.full(len(dyes), 1.2)), 0.012)
        noise = gaussian_filter1d(rng.standard_normal(len(wl)), 3.0)
        ks = km_ks(conc, K, substrate) * np.exp(0.06 * noise / noise.std())
        shades.append({"target_id": f"SYNTHETIC-{s + 1:03d}", "conc": conc, "ks": ks})

    # ---- recipes per shade and lab dips per recipe
    rec_per_shade = split_counts(rng, N_SHADES, N_RECIPES, cap=3)
    recipes = []
    for s, sh in enumerate(shades):
        for r in range(rec_per_shade[s]):
            conc = sh["conc"].copy()
            if r > 0:
                act = np.flatnonzero(conc > ACTIVE)
                if rng.random() < 0.5 and len(act):
                    j = int(rng.choice(act))
                    j2 = pick(hue[j], exclude=j)
                    conc[j2], conc[j] = conc[j], 0.0
                else:
                    conc = conc * rng.lognormal(0.0, 0.15, D)
            recipes.append({"shade": s, "recipe_no": f"SYN{len(recipes) + 1:04d}", "conc": conc,
                            "gsm": int(np.clip(rng.normal(180, 30), 120, 280)),
                            "sample_weight": rng.choice([10.0, 9.5, 15.0], p=[0.8, 0.1, 0.1]),
                            "l2r": rng.choice([6.0, 10.0], p=[0.6, 0.4]),
                            "fiber": rng.choice(["cotton", "cotton_lycra", "cotton_polyester"], p=[0.96, 0.02, 0.02]),
                            "fabric": rng.choice(FABRICS, p=[0.5, 0.2, 0.15, 0.1, 0.05])})
    dips_per_recipe = split_counts(rng, N_RECIPES, N_LABDIPS, cap=12)

    rows, long_rows = [], []
    for ri, rec in enumerate(recipes):
        sh = shades[rec["shade"]]
        refl = ks_to_reflectance(sh["ks"])
        for d in range(dips_per_recipe[ri]):
            sample_id = f"{rec['recipe_no']}{string.ascii_lowercase[d]}"
            conc = rec["conc"] * np.where(rec["conc"] > 0, rng.lognormal(0.0, 0.07, D), 0.0)
            conc[(conc > 0) & (conc <= ACTIVE)] = 0.012
            total = float(conc.sum())
            cotton, polyester, lycra = {"cotton": (100, 0, 0), "cotton_lycra": (92, 0, 8),
                                        "cotton_polyester": (60, 40, 0)}[rec["fiber"]]
            salt = float(max(5.0, round(15 + 14 * total + rng.normal(0, 4))))
            soda = float(max(5.0, round(5 + 3.5 * total + rng.normal(0, 1.5))))
            de65 = float(np.round(rng.uniform(0.1, 0.95), 2))
            de11 = float(np.round(np.clip(de65 + rng.normal(0, 0.1), 0.05, 1.0), 2))
            mi = float(np.round(rng.uniform(0.1, 1.4), 2))
            row = {"recipe_row_uid": f"rr_{len(rows) + 1:04d}_{sample_id.lower()}", "sl_no": len(rows) + 1,
                   "sample_id": sample_id, "recipe_no": rec["recipe_no"], "target_id": sh["target_id"],
                   "is_full_supervised": True, "is_full_supervised_row": True, "has_cmc_quality": True,
                   "has_production": True, "has_target_reflectance": True, "has_target_ks": True,
                   "has_dye_vector": True,
                   "dE_d65": de65, "dE_f11": de11, "dE_mean": round((de65 + de11) / 2, 3), "dE_max": max(de65, de11),
                   "dE_min": min(de65, de11), "metamerism_gap": round(de11 - de65, 2),
                   "metamerism_gap_abs": round(abs(de11 - de65), 2), "met_index_d65": mi, "met_index_f11": mi,
                   "dye_count": int((conc > ACTIVE).sum()), "total_dye_pct": round(total, 4),
                   "production__fabric": rec["fabric"], "production__gsm": rec["gsm"],
                   "production__sample_weight": rec["sample_weight"], "production__salt": salt,
                   "production__soda_ash": soda, "production__fiber_total_pct": 100.0,
                   "production__chemical_load": salt + soda, "production__salt_soda_ratio": round(salt / soda, 4),
                   "production__chemical_load_per_gsm": round((salt + soda) / rec["gsm"], 5),
                   "production__l2r_numeric": rec["l2r"], "production__modal": 0.0, "production__viscose": 0.0,
                   "production__cotton": float(cotton), "production__polyester": float(polyester),
                   "production__lycra": float(lycra), "production__is_lycra_blend": int(lycra > 0),
                   "production__is_polyester_blend": int(polyester > 0),
                   "feat_process_salt_minus_soda": salt - soda, "feat_process_salt_plus_soda": salt + soda,
                   "feat_process_salt_per_gsm": round(salt / rec["gsm"], 5),
                   "feat_process_soda_per_gsm": round(soda / rec["gsm"], 5),
                   "feat_process_cotton_polyester_ratio": round(cotton / max(polyester, 1.0), 4)}
            for f in FABRICS:
                row[f"feat_cat_production_fabric_{f}"] = int(rec["fabric"] == f)
            dom = max((("cotton", cotton), ("polyester", polyester), ("lycra", lycra)), key=lambda t: t[1])[0]
            for f in ("cotton", "polyester", "lycra"):
                row[f"feat_cat_production_dominant_fiber_{f}"] = int(dom == f)
            for j, col in enumerate(mapping["dye_column"]):
                row[f"vec__{col}"] = round(float(conc[j]), 4)
            row["vec__dye_count_vectorized"] = row["dye_count"]
            row["vec__total_dye_pct_vectorized"] = row["total_dye_pct"]
            for w, rv, kv in zip(wl, refl, reflectance_to_ks(refl)):
                row[f"target_refl__reflectance_{w}nm"] = round(float(100 * rv), 4)
                row[f"target_ks__ks_{w}nm"] = round(float(kv), 6)
            rows.append(row)
            order = {"yellow": 1, "red": 2, "blue": 3, "black": 3}
            placed = {order[hue[j]]: j for j in np.flatnonzero(conc > ACTIVE)}
            for pos in (1, 2, 3):
                j = placed.get(pos)
                long_rows.append({"sample_id": sample_id, "recipe_no": rec["recipe_no"], "target_id": sh["target_id"],
                                  "dye_position": pos, "dye_name": mapping["dye_name_clean"][j] if j is not None else np.nan,
                                  "dye_pct": round(float(conc[j]), 4) if j is not None else 0.0, "dye_present": j is not None})

    df = pd.DataFrame(rows)
    refl_cols = [c for c in df.columns if c.startswith("target_refl__")]
    ks_cols = [c for c in df.columns if c.startswith("target_ks__")]
    df, _ = add_spectral_descriptors(df, refl_cols, ks_cols)

    for sub in ("processed", "global_naming", "interim", "splits"):
        (ROOT / "data" / sub).mkdir(parents=True, exist_ok=True)
    df.to_csv(ROOT / "data/processed/features_supervised.csv", index=False)
    mapping.to_csv(ROOT / "data/global_naming/global_dye_mapping_active_36.csv", index=False)
    pd.DataFrame(long_rows).to_csv(ROOT / "data/interim/recipe_dye_long.csv", index=False)
    for split, col in (("groupkfold_target_id", "target_id"), ("groupkfold_recipe_no", "recipe_no")):
        folds = []
        for f, (tr, va) in enumerate(GroupKFold(n_splits=5).split(df, groups=df[col])):
            folds.append({"split_name": split, "fold_id": f + 1, "group_col": col, "n_train": len(tr), "n_val": len(va),
                          "train_row_uids": df["recipe_row_uid"].iloc[tr].tolist(),
                          "val_row_uids": df["recipe_row_uid"].iloc[va].tolist()})
        (ROOT / f"data/splits/{split}.json").write_text(json.dumps(
            {"split_name": split, "group_col": col, "n_rows": len(df), "n_folds": 5, "synthetic": True, "folds": folds},
            indent=2), encoding="utf-8")
    print(f"synthetic data: {len(df)} lab dips, {df.recipe_no.nunique()} recipes, {df.target_id.nunique()} shades, "
          f"{D} dyes ({(df[[f'vec__{c}' for c in mapping.dye_column]] > ACTIVE).any().sum()} used)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    make(ap.parse_args().seed)


if __name__ == "__main__":
    main()
