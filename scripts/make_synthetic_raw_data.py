"""Generate fully synthetic RAW files in the layout of the dyehouse exports, so the whole pipeline can run.

No industrial data are read or used. Everything below is invented:

* 36 fictitious reactive dyes ("Synth Yellow Y01", ...; 11 yellow, 12 red, 12 blue, 1 black) whose unit K/S spectra
  are smooth Gaussian absorption bands, not measured or calibrated spectra;
* 63 fictitious target shades with codes "PANTONE 00-0001 TCX" ... (no real PANTONE code starts with 00-);
* 101 recipes and 259 lab dips with recipe numbers "00k0001" and sample ids "00k0001a" ...;
* a library of target spectra computed with Kubelka-Munk mixing, Langmuir-type uptake u(c) = c / (1 + 0.3 c) and
  smooth spectral noise; 12 of the 63 targets are deliberately missing from the library, so the cleaning step has
  incomplete records to exclude (expected after cleaning: 212 lab dips, 85 recipes, 51 shades);
* production variables (fabric, knit, fiber, GSM, liquor ratio; salt and soda ash dosed from the depth of shade)
  and lab-dip color differences drawn from simple distributions.

Results obtained on these data do not reproduce the numbers of the paper; they only show that the code runs.

Outputs: data/raw/recipe.csv, data/raw/cmc.csv, data/raw/production.csv, data/raw/pantones.csv
Run: python scripts/make_synthetic_raw_data.py [--seed 2026]
Then: scripts 00-04, 06, and the analysis scripts (see README).
"""
from __future__ import annotations

import argparse
import string
import sys
from pathlib import Path

import colour
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from piml_color_recipe.inverse.km_physics import WAVELENGTHS, ks_to_reflectance  # noqa: E402

BETA = 0.3
N_COMPLETE_SHADES, N_COMPLETE_RECIPES, N_COMPLETE_DIPS = 51, 85, 212
N_EXTRA_SHADES, N_EXTRA_RECIPES, N_EXTRA_DIPS = 12, 16, 47      # targets without a library spectrum
N_LIBRARY_ONLY = 40                                             # library shades never ordered
HUE_CODES = {"yellow": "Y", "red": "R", "blue": "B", "black": "K"}
POSITION = {"yellow": 1, "red": 2, "blue": 3, "black": 3}
FABRICS = ["single jersey", "rib", "interlock", "fleece", "lycra single jersey", "lycra rib", "mesh"]
KNIT2 = {"single jersey": "single_jersey", "rib": "rib", "interlock": "interlock", "fleece": "fleece",
         "lycra single jersey": "single_jersey", "lycra rib": "rib", "mesh": "mesh"}
LIBRARY_COLUMNS = [
    "source_file", "record_type", "record_index", "standard_name", "shade_name", "batch_name", "datetime_unix",
    "datetime_utc", "pass_fail_judge", "refl_points", "refl_interval_nm", "refl_low_nm", "viewing", "cord_illum",
    "x", "y", "z", "measdll_params_raw", "contrast_ratio", "input_method", "r0", "bi_tappi525_uvexclude",
    "wi_cie_uvexcluded", "wi_cie_tint_uvexclude", "cie_de", "chroma_qc_hdesc", "chroma_qc_cdesc", "chroma_qc_ldesc",
    "std_frm_file_name", "cmc_1", "cmc_c", "plot_color_space", "plot_color_space_cord", "avg_method", "avg_fixed_num",
    "avg_auto_method", "avg_auto_min_reads", "avg_auto_max_reads", "avg_auto_tolerance", "std_tol_values",
    "std_tol_params", "meas_bandwidth", "meas_firmwareversion", "meas_geometry", "meas_instrument_name",
    "meas_manufacture", "meas_measarea", "meas_measspot", "meas_measurementsource", "meas_measurementtype",
    "meas_model", "meas_number_of_bands", "meas_serial_number", "meas_specular_component", "meas_start_band",
    "meas_uv_cutoff", "meas_uv_filterpercent",
] + [f"reflectance_{w}nm" for w in range(360, 751, 10)]


def palette(rng: np.random.Generator) -> tuple[list[str], np.ndarray, np.ndarray]:
    hues = ["yellow"] * 11 + ["red"] * 12 + ["blue"] * 12 + ["black"]
    bands = {"yellow": ((420, 455), (28, 40), (0.6, 2.0)), "red": ((505, 545), (25, 38), (0.6, 2.0)),
             "blue": ((590, 640), (35, 50), (0.5, 1.6))}
    wl = np.arange(360, 751, 10, dtype=float)  # spectra over the full library range
    names, spectra, count = [], [], {}
    for h in hues:
        count[h] = count.get(h, 0) + 1
        names.append(f"Synth {h.capitalize()} {HUE_CODES[h]}{count[h]:02d}")
        if h == "black":
            spectra.append(np.full(len(wl), 1.5))
            continue
        (c_lo, c_hi), (s_lo, s_hi), (a_lo, a_hi) = bands[h]
        c, s, a = rng.uniform(c_lo, c_hi), rng.uniform(s_lo, s_hi), rng.uniform(a_lo, a_hi)
        k = a * np.exp(-0.5 * ((wl - c) / s) ** 2) + 0.01 * a
        if h in ("red", "blue"):
            k += 0.2 * a * np.exp(-0.5 * ((wl - (c - 70)) / 30) ** 2)
        spectra.append(k)
    return names, np.array(hues), np.array(spectra)


def mix_reflectance(conc: np.ndarray, k: np.ndarray, rng: np.random.Generator, noise: float = 0.06) -> np.ndarray:
    wl = np.arange(360, 751, 10, dtype=float)
    substrate = 0.015 + 0.01 * np.exp(-(wl - 360) / 60.0)
    ks = substrate + (conc / (1.0 + BETA * conc)) @ k
    z = gaussian_filter1d(rng.standard_normal(len(wl)), 3.0)
    return 100.0 * ks_to_reflectance(ks * np.exp(noise * z / z.std()))


def xyz_d65_10(refl_pct: np.ndarray) -> np.ndarray:
    wl = np.arange(360, 751, 10)
    sd = colour.SpectralDistribution(dict(zip(wl, refl_pct / 100.0)))
    cmfs = colour.MSDS_CMFS["CIE 1964 10 Degree Standard Observer"]
    return colour.sd_to_XYZ(sd, cmfs, colour.SDS_ILLUMINANTS["D65"], shape=colour.SpectralShape(360, 750, 10))


def split_counts(rng: np.random.Generator, n_groups: int, total: int, cap: int) -> np.ndarray:
    counts = np.ones(n_groups, dtype=int)
    w = rng.gamma(0.8, 1.0, n_groups)
    while counts.sum() < total:
        j = rng.choice(n_groups, p=w / w.sum())
        if counts[j] < cap:
            counts[j] += 1
    return counts


def make(seed: int) -> None:
    rng = np.random.default_rng(seed)
    names, hue, K = palette(rng)
    D = len(names)
    pop = np.zeros(D)
    for h in ("yellow", "red", "blue"):
        idx = np.flatnonzero(hue == h)
        pop[idx] = 1.0 / np.arange(1, len(idx) + 1) ** 1.1

    def pick(h: str, exclude: int | None = None, prefer: list[int] | None = None) -> int:
        idx = np.flatnonzero(hue == h)
        if prefer:
            cand = [j for j in prefer if hue[j] == h and j != exclude]
            if cand:
                return cand[0]
        p = pop[idx].copy()
        if exclude is not None:
            p[idx == exclude] = 0.0
        return int(rng.choice(idx, p=p / p.sum()))

    def shade_recipe(prefer: list[int] | None = None) -> np.ndarray:
        n = rng.choice([3, 2, 1], p=[0.85, 0.10, 0.05])
        hs = ["yellow", "red", "blue"] if n == 3 else list(rng.choice(["yellow", "red", "blue"], size=n, replace=False))
        dyes = [pick(h, prefer=prefer) for h in hs]
        total = float(np.clip(rng.lognormal(np.log(0.9), 0.6) if rng.random() < 0.5 else rng.lognormal(np.log(3.5), 0.35), 0.02, 7.0))
        conc = np.zeros(D)
        conc[dyes] = np.maximum(total * rng.dirichlet(np.full(len(dyes), 1.2)), 0.012)
        return conc

    # ---- targets: 51 complete + 12 without library spectrum (+ 40 library-only shades)
    n_targets = N_COMPLETE_SHADES + N_EXTRA_SHADES
    targets = [f"PANTONE 00-{k + 1:04d} TCX" for k in range(n_targets)]
    shade_conc = [shade_recipe() for _ in range(N_COMPLETE_SHADES)]
    used = set(np.flatnonzero(np.vstack(shade_conc).sum(0) > 0))
    unused = [j for j in range(D) if j not in used and hue[j] != "black"]
    for _ in range(N_EXTRA_SHADES):  # the excluded shades use the dyes not used elsewhere, so every dye occurs
        c = shade_recipe(prefer=unused)
        unused = [j for j in unused if c[j] == 0]
        shade_conc.append(c)

    # ---- recipes and lab dips
    rec_complete = split_counts(rng, N_COMPLETE_SHADES, N_COMPLETE_RECIPES, cap=3)
    rec_extra = split_counts(rng, N_EXTRA_SHADES, N_EXTRA_RECIPES, cap=3)
    rec_per_shade = np.concatenate([rec_complete, rec_extra])
    recipes = []
    for s in range(n_targets):
        for r in range(rec_per_shade[s]):
            conc = shade_conc[s].copy()
            if r > 0:
                act = np.flatnonzero(conc > 0)
                if rng.random() < 0.5 and len(act):
                    j = int(rng.choice(act))
                    j2 = pick(hue[j], exclude=j)
                    conc[j2], conc[j] = conc[j], 0.0
                else:
                    conc = conc * rng.lognormal(0.0, 0.15, D)
            recipes.append({"shade": s, "complete": s < N_COMPLETE_SHADES, "conc": conc})
    # the black dye occurs once, in an excluded recipe (as in the real records)
    black = int(np.flatnonzero(hue == "black")[0])
    last = recipes[-1]["conc"]
    blue = np.flatnonzero((hue == "blue") & (last > 0))
    if len(blue):
        last[black], last[blue[0]] = last[blue[0]], 0.0
    else:
        last[black] = 0.5
    dips = np.concatenate([split_counts(rng, N_COMPLETE_RECIPES, N_COMPLETE_DIPS, cap=12),
                           split_counts(rng, N_EXTRA_RECIPES, N_EXTRA_DIPS, cap=12)])

    recipe_rows, cmc_rows, prod_rows = [], [], []
    failed_one = False
    for ri, rec in enumerate(recipes):
        recipe_no = f"00k{ri + 1:04d}"
        target = targets[rec["shade"]]
        for d in range(dips[ri]):
            sample_id = f"{recipe_no}{string.ascii_lowercase[d]}"
            conc = rec["conc"] * np.where(rec["conc"] > 0, rng.lognormal(0.0, 0.07, D), 0.0)
            conc[(conc > 0) & (conc < 0.012)] = 0.012
            slots = {1: ("Dummy", 0.0), 2: ("Dummy", 0.0), 3: ("Dummy", 0.0)}
            for j in np.flatnonzero(conc > 0):
                slots[POSITION[hue[j]]] = (names[j], round(float(conc[j]), 4))
            row = {"sl_no": len(recipe_rows) + 1, "sample_id": sample_id, "target_id": target, "recipe_no": recipe_no}
            for p in (1, 2, 3):
                row[f"dye{p}"], row[f"dye{p}(%)"] = slots[p]
            recipe_rows.append(row)
            # lab-dip color difference against the target, under D65 and F11 (CMC 2:1); one recorded failure
            met = round(float(rng.uniform(0.1, 1.4)), 2)
            base = 1.65 if (not failed_one and rec["complete"] and rng.random() < 0.02) else float(rng.uniform(0.1, 0.95))
            failed_one = failed_one or base > 1.0
            for illum in ("d65", "f11"):
                de = round(float(np.clip(base + (rng.normal(0, 0.08) if illum == "f11" else 0.0), 0.05, 1.8)), 2)
                v = rng.normal(size=3)
                v = de * v / np.linalg.norm(v)
                h = rng.uniform(0, 2 * np.pi)
                cmc_rows.append({"sl_no": len(cmc_rows) + 1, "sample_id": sample_id, "target_id": target,
                                 "recipe_no": recipe_no, "illum": illum, "dE": de,
                                 "pf_decision": "pass" if de <= 1.0 else "fail",
                                 "dL": round(v[0], 2), "da": round(v[1] * np.cos(h) - v[2] * np.sin(h), 2),
                                 "db": round(v[1] * np.sin(h) + v[2] * np.cos(h), 2), "dc_csc": round(v[1], 2),
                                 "dh_sh": round(v[2], 2), "met_index": met})
        total = float(rec["conc"].sum())
        fabric = rng.choice(FABRICS, p=[0.45, 0.15, 0.1, 0.1, 0.1, 0.05, 0.05])
        fiber = rng.choice(["cotton", "cotton_polyester", "cotton_lycra"], p=[0.94, 0.03, 0.03])
        cotton, polyester, lycra = {"cotton": (100, 0, 0), "cotton_polyester": (60, 40, 0),
                                    "cotton_lycra": (95, 0, 5)}[fiber]
        if "lycra" in fabric:
            cotton, lycra = cotton - 5 if lycra == 0 else cotton, 5
        prod_rows.append({"sl_no": len(prod_rows) + 1, "target_id": target, "recipe_no": recipe_no,
                          "kniting_struc1": rng.choice(["sj", "11", "22", "il", ""], p=[0.3, 0.15, 0.1, 0.05, 0.4]),
                          "kniting_struc2": KNIT2[fabric], "fabric": fabric,
                          "l2r": rng.choice(["1:6", "1:10", "1:8"], p=[0.6, 0.35, 0.05]),
                          "modal": 0, "viscose": 0, "cotton": cotton, "polyester": polyester, "lycra": lycra,
                          "shade": f"synthetic shade {rec['shade'] + 1:03d}", "target_id.1": "",
                          "gsm": int(np.clip(round(rng.normal(190, 35)), 150, 550)),
                          "sample_weight": rng.choice([10, 9.5, 15], p=[0.9, 0.05, 0.05]),
                          "salt": int(np.clip(5 * round((15 + 14 * total + rng.normal(0, 4)) / 5), 5, 100)),
                          "soda_ash": int(np.clip(round(5 + 3 * total + rng.normal(0, 1.5)), 5, 20)),
                          "total_trial": int(np.clip(dips[ri] + rng.integers(4, 12), 5, 24))})
    for extra in range(2):  # production records of recipes that never reached the recipe table
        row = dict(prod_rows[-1])
        row.update({"sl_no": len(prod_rows) + 1, "recipe_no": f"00k{len(recipes) + extra + 1:04d}"})
        prod_rows.append(row)

    # ---- library of target spectra (one STANDARD_DATA record, the rest BATCH_DATA)
    lib = []
    shades_in_library = list(range(N_COMPLETE_SHADES)) + [None] * N_LIBRARY_ONLY
    for i, s in enumerate(shades_in_library):
        conc = shade_conc[s] if s is not None else shade_recipe()
        code = targets[s] if s is not None else f"PANTONE 00-{5001 + i:04d} TCX"
        refl = mix_reflectance(conc, K, rng)
        start = 360 if rng.random() < 0.1 else 400
        bands = 40 if start == 360 else 36
        inst = rng.choice(["SYN-SPECTRO-1", "SYN-SPECTRO-2", "SYN-SPECTRO-3", "SYN-SPECTRO-4"])
        area = "LAV" if rng.random() < 0.9 else "SAV"
        ts = int(rng.integers(1_600_000_000, 1_710_000_000))
        x, y, z = xyz_d65_10(refl)
        rec = {c: "" for c in LIBRARY_COLUMNS}
        rec.update({"source_file": "SYNTHETIC_LIBRARY.QTX", "record_type": "BATCH_DATA", "record_index": i + 1,
                    "standard_name": "PANTONE 00-0000 TCX", "shade_name": code, "batch_name": code,
                    "datetime_unix": ts, "datetime_utc": pd.Timestamp(ts, unit="s").strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "pass_fail_judge": 1, "refl_points": bands, "refl_interval_nm": 10, "refl_low_nm": start,
                    "viewing": "SCI", "cord_illum": "D65/10 deg", "x": round(x, 6), "y": round(y, 6), "z": round(z, 6),
                    "measdll_params_raw": (f"Instrument Name:{inst},Manufacture:synthetic,Model:{inst},Serial number:000000,"
                                           f"FirmwareVersion:unknown,Geometry:d/8,Specular component:I,MeasSpot:N/A,"
                                           f"MeasArea:{area} ,UV-Filter%:UVEXC ,UV-Cutoff:None,MeasurementType:R,"
                                           f"MeasurementSource:Instrument,Number of bands:{bands},Start band:{start},Bandwidth:10"),
                    "input_method": 130, "meas_bandwidth": 10, "meas_firmwareversion": "unknown", "meas_geometry": "d/8",
                    "meas_instrument_name": inst, "meas_manufacture": "synthetic", "meas_measarea": area,
                    "meas_measurementsource": "Instrument", "meas_measurementtype": "R", "meas_model": inst,
                    "meas_number_of_bands": bands, "meas_serial_number": "000000", "meas_specular_component": "I",
                    "meas_start_band": start, "meas_uv_filterpercent": "UVEXC"})
        for w, v in zip(range(360, 751, 10), refl):
            if w >= start:
                rec[f"reflectance_{w}nm"] = round(float(v), 6)
        lib.append(rec)
    std = dict(lib[0])
    std.update({"record_type": "STANDARD_DATA", "record_index": 0, "batch_name": "", "shade_name": "PANTONE 00-0000 TCX"})
    lib.insert(0, std)

    out = ROOT / "data" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(recipe_rows).to_csv(out / "recipe.csv", index=False)
    pd.DataFrame(cmc_rows).to_csv(out / "cmc.csv", index=False)
    pd.DataFrame(prod_rows).to_csv(out / "production.csv", index=False)
    pd.DataFrame(lib, columns=LIBRARY_COLUMNS).to_csv(out / "pantones.csv", index=False)
    dyes_used = int((np.vstack([r["conc"] for r in recipes]) > 0).any(0).sum())
    print(f"synthetic raw data: {len(recipe_rows)} lab dips, {len(recipes)} recipes, {n_targets} targets "
          f"({N_COMPLETE_SHADES} in the library), {len(lib)} library records, {dyes_used} of {D} dyes used")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    make(ap.parse_args().seed)


if __name__ == "__main__":
    main()
