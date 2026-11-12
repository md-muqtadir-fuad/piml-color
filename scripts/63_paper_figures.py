"""Publication figures for the revised Results and Analysis section.

Static figures for a two-column journal layout (single column 3.35 in, full width 7.0 in), saved as vector
PDF + 300 dpi PNG to outputs/figures/, each with a CSV of the plotted values (table-view twin).

Colour follows the entity in every figure (validated with the dataviz palette validator):
    our method = blue #2a78d6, ML reference = orange #eb6834, nearest-recipe retrieval = aqua #1baf7a,
    FT-Transformer = yellow #eda100, classical CCM = magenta #e87ba4.
Hue families use their semantic hues: yellow #eda100, red #e34948, blue #2a78d6 (black family unused).

Statistics match the paper: paired sign-flip permutation test over shade clusters (strict) or recipe
clusters (grouped), Holm correction within each declared family; error bars are 95 % cluster-bootstrap CIs.

Usage:
    python scripts/63_paper_figures.py            # all figures
    python scripts/63_paper_figures.py --only R4  # one figure
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location("pie", ROOT / "scripts" / "60_physics_inverse_experiments.py")
pie = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pie)

OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- style (reference palette, print surface)
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C_OURS, C_ML, C_RET, C_FT, C_CCM = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
FAMILY = {"yellow": "#eda100", "red": "#e34948", "blue": "#2a78d6"}
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
W1, W2 = 3.35, 7.0

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 7.5,
    "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-", "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42, "lines.linewidth": 1.6, "lines.markersize": 4,
})

D = pie.Data()
RNG_SEED = 42


_SHADE_IDS = {s: f"shade_{k + 1:02d}" for k, s in enumerate(sorted(set(D.df["target_id"].astype(str))))}


def save(fig, name: str, table: pd.DataFrame | None = None) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    if table is not None:
        table = table.copy()
        for c in table.columns:  # NDA: never write target-shade codes; use anonymous shade ids
            if table[c].dtype == object or pd.api.types.is_string_dtype(table[c]):
                table[c] = table[c].map(lambda v: _SHADE_IDS.get(str(v), v))
        table.to_csv(OUT / f"{name}_data.csv", index=False)
    plt.close(fig)
    print(f"  saved {name}")


def panel_label(ax, letter: str, title: str = "") -> None:
    """Panel letter + short title, left-aligned above the axes."""
    ax.set_title(f"({letter})  {title}".rstrip(), loc="left", fontsize=7.5, color=INK, pad=6)


def rows(exp: str, method: str, proto: str = "strict") -> pd.DataFrame:
    return pie.load_rows(exp, method, proto)


def cluster_ci(v: np.ndarray, cl: np.ndarray, n: int = 4000) -> tuple[float, float, float]:
    ok = np.isfinite(v)
    g = pd.Series(v[ok]).groupby(cl[ok]).agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = np.random.default_rng(RNG_SEED).integers(0, len(s), size=(n, len(s)))
    b = s[idx].sum(1) / c[idx].sum(1)
    return float(s.sum() / c.sum()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def signflip(a: np.ndarray, b: np.ndarray, cl: np.ndarray, n: int = 20000) -> tuple[float, float]:
    ok = np.isfinite(a) & np.isfinite(b)
    g = pd.Series(a[ok] - b[ok]).groupby(cl[ok]).agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    obs = s.sum() / c.sum()
    null = (np.random.default_rng(RNG_SEED).choice([-1, 1], size=(n, len(s))) * s).sum(1) / c.sum()
    b = int((np.abs(null) >= abs(obs) - 1e-12).sum())
    return float(obs), (b + 1) / (n + 1)  # the observed sign vector counts as one draw, so p > 0


def holm(p) -> np.ndarray:
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj, run = np.empty(len(p)), 0.0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def p_text(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


# ============================================================ M1 framework schematic (methods figure)

def fig_m1() -> None:
    fig, ax = plt.subplots(figsize=(W2, 2.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(-1, 41)
    ax.axis("off")

    def box(x, y, w, h, title, body, edge):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", linewidth=0.9,
                                    edgecolor=edge, facecolor="#ffffff"))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=7.5, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 5.6, body, ha="center", va="top", fontsize=6.1, color=INK2, linespacing=1.3)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, linewidth=0.9, color=MUTED))

    box(1, 21, 17, 16, "Production logs", "212 lab dips, 85 recipes,\n51 target shades\nR(λ), K/S(λ), substrate\n(salt / soda ash excluded)", AXIS)
    box(23, 21, 19, 16, "KM calibration", "unit K/S spectra k_j(λ)\nfrom historical mixtures:\nLangmuir uptake, relative\nweights, family shrinkage", C_OURS)
    box(23, 2.5, 19, 15.5, "Candidate dye sets", "historical sets, subsets,\none-dye within-hue swaps,\nML top-3 set\n(trichromatic slots)", AXIS)
    box(47, 21, 21, 16, "Physics per candidate", "bounded KM fit of amounts\nc^KM(S) to target K/S;\ncolor error ΔE_max(S)\n(CMC 2:1, D65 / F11)", C_OURS)
    box(47, 2.5, 21, 15.5, "Learned dye-set choice", "conditional logit over\ncandidates, 14 features\n(physics, history, ML);\nL2-averaged, no tuning", C_ML)
    box(73, 11, 26, 16, "Recipe and shortlist", "chosen set Ŝ = argmax P(S | x)\namounts = 0.5·KM + 0.5·ML\nranked shortlist for lab dips", INK2)
    arrow(18.4, 29, 22.6, 29)
    arrow(9.5, 20.6, 22.6, 10.5)
    arrow(42.4, 29, 46.6, 29)
    arrow(42.4, 14.5, 46.6, 22.5)
    arrow(42.4, 8.5, 46.6, 8.5)
    arrow(57.5, 20.6, 57.5, 18.4)
    arrow(68.4, 27, 72.6, 22.5)
    arrow(68.4, 9.5, 72.6, 15)
    ax.set_ylim(1.4, 38.2)
    save(fig, "fig_M1_framework")


# ================================================================== R1 data audit (leak, slots, strata)

def fig_r1() -> None:
    df = D.df
    tot = D.Y.sum(1)
    salt = pd.to_numeric(df["production__salt"], errors="coerce").to_numpy()
    soda = pd.to_numeric(df["production__soda_ash"], errors="coerce").to_numpy()
    long = pd.read_csv(ROOT / "data/interim/recipe_dye_long.csv")
    long = long[long.dye_present == True]  # noqa: E712
    mapping = pd.read_csv(ROOT / "data/global_naming/global_dye_mapping_active_36.csv")
    long["hue"] = long.dye_name.map(dict(zip(mapping.dye_name_clean, mapping.original_hue_inferred)))
    long = long[long.hue.isin(FAMILY)]
    ct = pd.crosstab(long.dye_position, long.hue)[["yellow", "red", "blue"]]
    share = ct.div(ct.sum(1), axis=0)
    strata = D.strata("strict")

    fig, axes = plt.subplots(1, 4, figsize=(W2, 2.0), gridspec_kw={"width_ratios": [1, 1, 1.05, 1.15], "wspace": 0.6})
    for ax, y, name, letter in ((axes[0], salt, "Salt (g/L)", "a"), (axes[1], soda, "Soda ash (g/L)", "b")):
        ok = np.isfinite(y)
        rho = pd.Series(y[ok]).corr(pd.Series(tot[ok]), method="spearman")
        ax.scatter(tot[ok], y[ok], s=9, color=C_OURS, edgecolor="#ffffff", linewidth=0.4, alpha=0.9)
        ax.set_xlabel("Total dye (% OWF)")
        ax.set_ylabel(name)
        panel_label(ax, letter, f"Spearman ρ = {rho:.2f}")
    ax = axes[2]
    im = ax.imshow(share.to_numpy(), cmap=matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ), vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(3), ["Yellow", "Red", "Blue"])
    ax.set_yticks(range(3), ["Position 1", "Position 2", "Position 3"])
    ax.set_xlabel("Dye hue family")
    for i in range(3):
        for j in range(3):
            v = share.to_numpy()[i, j]
            ax.text(j, i, f"{100 * v:.0f}%", ha="center", va="center", fontsize=6.8, color="#ffffff" if v > 0.5 else INK)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    panel_label(ax, "c", "Trichromatic positions")
    ax = axes[3]
    n = [int((strata == k).sum()) for k in (1, 2, 3)]
    labels = ["Dye set seen in training", "Novel combination of seen dyes", "Contains an unseen dye"]
    ax.barh(range(3), n, color=C_OURS, height=0.42)
    for i, (v, lab) in enumerate(zip(n, labels)):
        ax.text(0, i - 0.34, lab, va="bottom", ha="left", fontsize=6.4, color=INK)
        ax.text(v + 2, i, f"{v} ({100 * v / len(strata):.0f}%)", va="center", fontsize=6.6, color=INK)
    ax.set_yticks([])
    ax.set_ylim(2.55, -0.85)
    ax.set_xlim(0, 125)
    ax.set_xlabel("Rows (unseen-shade CV)")
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    panel_label(ax, "d", "Generalisation strata")
    table = pd.concat([
        pd.DataFrame({"panel": "a/b", "total_dye": tot, "salt": salt, "soda_ash": soda}),
        share.reset_index().melt(id_vars="dye_position", var_name="hue", value_name="share").assign(panel="c"),
        pd.DataFrame({"panel": "d", "stratum": [1, 2, 3], "rows": n}),
    ], ignore_index=True)
    save(fig, "fig_R1_data_audit", table)


# ============================================================= R2 KM calibration and recovered spectra

def fig_r2() -> None:
    A = D.Y > pie.ACTIVE
    import pickle

    # Calibrated KM = the (beta, eta) the final method selected by inner CV in each outer fold (nested).
    def selected(proto: str, f: int) -> dict:
        name = pickle.loads((pie.CACHE_DIR / f"bundle_v1__{proto}__fold{f}.pkl").read_bytes())["km"]
        return pie.KM_GRID[name]

    variants = {"Feature-only KM": lambda p, f: dict(weighting="none"), "Calibrated KM": selected,
                "Calibrated KM +\nabsorption bands": lambda p, f: {**selected(p, f), "band_mask": True}}
    colours = [MUTED, C_OURS, "#184f95"]
    res = {}
    for proto in ("strict", "grouped"):
        de, seen = {}, np.zeros(len(D.Y), bool)
        for name, kw_of in variants.items():
            v = np.zeros(len(D.Y))
            for f, (tr, va) in enumerate(D.folds[proto]):
                v[va] = pie.km_true_recipe_de(D, pie.make_km(D, tr, kw_of(proto, f)), va)
                s = A[tr].any(0)
                seen[va] = [(s | ~A[r]).all() for r in va]
            de[name] = v
        res[proto] = (de, seen)

    fig = plt.figure(figsize=(W2, 2.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.32)
    table = []
    for k, proto in enumerate(("strict", "grouped")):
        ax = fig.add_subplot(gs[0, k])
        de, seen = res[proto]
        cl = D.clusters(proto)
        names = list(variants)
        for name, col in zip(names, colours):
            x = np.sort(de[name][seen])
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=col, linewidth=1.6, label=name)
            table.append(pd.DataFrame({"protocol": proto, "variant": name.replace("\n", " "), "dE_max_true_recipe": de[name][seen]}))
        diff, p = signflip(np.where(seen, de[names[1]], np.nan), np.where(seen, de[names[0]], np.nan), cl)
        print(f"  R2 {proto}: calibrated vs feature-only, Δ mean = {diff:+.2f}, {p_text(p)}; "
              f"means {de[names[0]][seen].mean():.2f} -> {de[names[1]][seen].mean():.2f}, medians "
              f"{np.median(de[names[0]][seen]):.2f} -> {np.median(de[names[1]][seen]):.2f} (n = {seen.sum()})")
        d2, p2 = signflip(np.where(seen, de[names[2]], np.nan), np.where(seen, de[names[1]], np.nan), cl)
        print(f"  R2 {proto}: + absorption bands vs calibrated, Δ mean = {d2:+.2f}, {p_text(p2)}")
        ax.set_xscale("log")
        ax.set_xlim(0.15, 40)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("True-recipe ΔE$_{\\max}$ (CMC, log)")
        ax.set_ylabel("Cumulative share of rows" if k == 0 else "")
        panel_label(ax, "ab"[k], "Unseen shades" if proto == "strict" else "Seen shades")
        if k == 0:
            handles, labels = ax.get_legend_handles_labels()
    # (c) unit K/S spectra recovered from production logs (all rows, no band constraint)
    ax = fig.add_subplot(gs[0, 2])
    model = pie.make_km(D, np.arange(len(D.Y)), pie.KM_DEFAULT)
    usage = (D.Y > pie.ACTIVE).sum(0)
    wl = np.arange(400, 751, 10)
    common = [j for j in np.argsort(-usage) if usage[j] >= 12 and D.dye_hue[j] in FAMILY]
    for fam, col in FAMILY.items():
        first = True
        for j in common:
            if D.dye_hue[j] != fam:
                continue
            kj = model.k[j]
            ax.plot(wl, kj / kj.max(), color=col, linewidth=1.1, alpha=0.9, label=f"{fam.capitalize()} dyes" if first else None)
            table.append(pd.DataFrame({"protocol": "all rows", "variant": f"unit K/S {D.dye_names[j]} ({fam})", "wavelength_nm": wl, "k_normalised": kj / kj.max()}))
            first = False
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Unit K/S (normalised)")
    ax.set_xlim(400, 750)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=6.2, handlelength=1.4)
    panel_label(ax, "c", "Calibrated unit K/S spectra")
    fig.legend(handles, [l.replace("\n", " ") for l in labels], loc="lower center", ncol=3,
               bbox_to_anchor=(0.36, -0.2), fontsize=6.4, handlelength=1.8)
    save(fig, "fig_R2_km_calibration", pd.concat(table, ignore_index=True))


# ================================================================== R3 colour alone does not identify dyes

def fig_r3() -> None:
    A = D.Y > pie.ACTIVE
    best = np.full(len(D.Y), np.nan)
    true_de = np.full(len(D.Y), np.nan)
    # Both axes use KM-fitted amounts (Eq. 8): x = the true dye set, y = the best of all sets of up to three dyes.
    for f, (tr, va) in enumerate(D.folds["strict"]):
        z = np.load(pie.CACHE_DIR / f"E02_lang0.3_shrink0.01_cmax__strict__fold{f}.npz")
        de_all = np.maximum(z["de_d65"], z["de_f11"])
        best[z["val"]] = de_all.min(1)
        keys = {tuple(s[s >= 0]): c for c, s in enumerate(z["subsets"])}
        for i, r in enumerate(z["val"]):
            c = keys.get(tuple(np.flatnonzero(A[r])))
            if c is not None:
                true_de[r] = de_all[i, c]
    seen = np.zeros(len(D.Y), bool)
    for tr, va in D.folds["strict"]:
        s = A[tr].any(0)
        seen[va] = [(s | ~A[r]).all() for r in va]
    cl = D.clusters("strict")
    methods = [("KM matching", "E02", "ccm_min_de_max", C_CCM),
               ("ML reference", "E00b", "ml_et_spectral_substrate_feasible", C_ML),
               ("Our method", "E16", "avg_base", C_OURS)]

    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.3), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.35})
    ax = axes[0]
    m = seen & np.isfinite(best) & np.isfinite(true_de)
    ax.scatter(true_de[m], best[m], s=10, color=C_CCM, edgecolor="#ffffff", linewidth=0.4)
    lim = (0.05, 40)
    ax.plot(lim, lim, color=MUTED, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("True dye set, fitted ΔE$_{\\max}$")
    ax.set_ylabel("Best of all sets, ΔE$_{\\max}$")
    below = float((best[m] < true_de[m] - 1e-9).mean())
    print(f"  R3: {100 * below:.1f}% of {m.sum()} lab dips below y = x (another set fits better than the true set); "
          f"{len(np.unique(cl[m]))} shades")
    ax.text(0.5, 0.5, "y = x", transform=ax.transAxes, rotation=45, fontsize=6.3, color=MUTED, ha="center", va="bottom")
    panel_label(ax, "a", "Unseen shades, seen dyes")
    ax = axes[1]
    ref = rows("E00b", "ml_et_spectral_substrate_feasible")
    tab = []
    for i, (name, ex, mth, col) in enumerate(methods):
        r = rows(ex, mth)
        mean, lo, hi = cluster_ci(r.f1.to_numpy(), cl)
        ax.bar(i, mean, color=col, width=0.62)
        ax.errorbar(i, mean, yerr=[[mean - lo], [hi - mean]], color=INK, capsize=2.5, linewidth=0.8)
        ax.text(i, hi + 0.015, f"{mean:.2f}", ha="center", fontsize=6.8, color=INK)
        tab.append({"method": name.replace("\n", " "), "f1": mean, "ci_lo": lo, "ci_hi": hi})
    d, p = signflip(rows("E02", "ccm_min_de_max").f1.to_numpy(), ref.f1.to_numpy(), cl)
    print(f"  R3: CCM vs ML F1, Δ = {d:+.3f}, {p_text(p)}")
    ax.set_xticks(range(3), [m_[0] for m_ in methods], fontsize=6.6)
    ax.set_ylabel("Dye-set F1")
    ax.set_ylim(0, 0.6)
    ax.grid(axis="x", visible=False)
    panel_label(ax, "b", "Dye identification")
    table = pd.concat([pd.DataFrame({"panel": "a", "true_recipe_dE": true_de, "best_ccm_dE": best, "dyes_seen": seen}),
                       pd.DataFrame(tab).assign(panel="b")], ignore_index=True)
    save(fig, "fig_R3_colour_does_not_identify_dyes", table)


# ============================================================ R4 main comparison: effects and strata

METRICS = [("f1", "Dye-set F1", +1), ("exact_set", "Exact recipe", +1), ("active_mae", "MAE on true dyes", -1),
           ("union_mae", "Union MAE", -1), ("family_mae", "Hue-family MAE", -1), ("total_abs_err", "Total-dye error", -1)]


def fig_r4() -> None:
    fig = plt.figure(figsize=(W2, 3.1))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1.35, 1, 1], hspace=0.75, wspace=0.45)
    table = []
    # (a) forest plot: our method minus ML, strict (blue) and grouped (aqua-free: use hollow blue)
    ax = fig.add_subplot(gs[:, 0])
    refs = {"strict": ("E00b", "ml_et_spectral_substrate_feasible"), "grouped": ("E00b", "ml_et_spectral_feasible")}
    y0 = np.arange(len(METRICS))[::-1]
    for k, proto in enumerate(("strict", "grouped")):
        cl = D.clusters(proto)
        a, b = rows("E16", "avg_base", proto), rows(*refs[proto], proto)
        diffs, ps, cis = [], [], []
        for key, _, _ in METRICS:
            x, y = a[key].to_numpy(), b[key].to_numpy()
            d, p = signflip(x, y, cl)
            m, lo, hi = cluster_ci(x - y, cl)
            diffs.append(d)
            ps.append(p)
            cis.append((lo, hi))
        ph = holm(ps)
        off = 0.17 if proto == "strict" else -0.17
        for i, ((key, label, better), d, (lo, hi), p_) in enumerate(zip(METRICS, diffs, cis, ph)):
            sig = p_ < 0.05
            face = C_OURS if proto == "strict" else "#ffffff"
            ax.plot([lo, hi], [y0[i] + off] * 2, color=C_OURS, linewidth=1.1)
            ax.plot(d, y0[i] + off, marker="o", markersize=4.5, markerfacecolor=face, markeredgecolor=C_OURS, markeredgewidth=1.0, linestyle="none")
            if sig:
                ax.text(hi + 0.006, y0[i] + off, "*", va="center", fontsize=8, color=INK)
            table.append({"panel": "a", "protocol": proto, "metric": label, "diff_vs_ML": d, "ci_lo": lo, "ci_hi": hi,
                          "p_signflip": ps[i], "p_holm": p_, "better_if": "higher" if better > 0 else "lower"})
    ax.axvline(0, color=AXIS, linewidth=0.8)
    ax.set_yticks(y0, [f"{lbl}  ({'↑' if b > 0 else '↓'})" for _, lbl, b in METRICS])
    ax.set_xlabel("Our method − ML reference")
    ax.grid(axis="y", visible=False)
    ax.plot([], [], marker="o", color=C_OURS, linestyle="none", label="Unseen shades")
    ax.plot([], [], marker="o", markerfacecolor="#ffffff", markeredgecolor=C_OURS, linestyle="none", label="Seen shades")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.11), ncol=2, fontsize=6.2, handletextpad=0.3, columnspacing=1.0)
    panel_label(ax, "a", "Our method vs ML")
    # (b-e) by stratum, unseen shades: F1 and family MAE for ML, ours, retrieval
    methods = [("ML reference", "E00b", "ml_et_spectral_substrate_feasible", C_ML), ("Our method", "E16", "avg_base", C_OURS),
               ("Nearest-recipe retrieval", "E00b", "nearest_historical_spectral", C_RET)]
    strata = D.strata("strict")
    cl = D.clusters("strict")
    for col_i, (key, label) in enumerate((("f1", "Dye-set F1"), ("family_mae", "Hue-family MAE"))):
        ax = fig.add_subplot(gs[:, 1 + col_i])
        width = 0.26
        for m_i, (mname, ex, mth, colr) in enumerate(methods):
            r = rows(ex, mth)[key].to_numpy()
            for s_i, s in enumerate((1, 2, 3)):
                v = np.where(strata == s, r, np.nan)
                mean, lo, hi = cluster_ci(v, cl)
                x = s_i + (m_i - 1) * width
                ax.bar(x, mean, width=width * 0.92, color=colr, label=mname if s_i == 0 else None)
                ax.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]], color=INK, capsize=1.6, linewidth=0.7)
                table.append({"panel": "bc"[col_i], "metric": label, "method": mname, "stratum": s, "mean": mean, "ci_lo": lo, "ci_hi": hi})
        ax.set_xticks(range(3), ["Set seen\nbefore", "Novel\ncombination", "Unseen\ndye"], fontsize=6.6)
        ax.set_ylabel(label)
        ax.grid(axis="x", visible=False)
        panel_label(ax, "bc"[col_i], f"{label} by stratum")
        if col_i == 0:
            h_b, l_b = ax.get_legend_handles_labels()
            fig.legend(h_b, l_b, loc="lower center", bbox_to_anchor=(0.71, -0.14), ncol=3, fontsize=6.4, handlelength=1.2)
    save(fig, "fig_R4_main_comparison", pd.DataFrame(table))


# ============================================================ R5 physics contribution to the amounts

def fig_r5() -> None:
    cache = OUT / "_r5_ml_amounts_rows.csv"
    blend = rows("E16", "avg_base")
    if cache.exists():
        ml_only = pd.read_csv(cache)
    else:  # same dye choice as the final method, amounts from the ML regressor only (alpha = 0)
        P, _, _ = pie.run_averaged_ranker(D, "strict", pie.BASE_ALL, soft=False, alpha=0.0)
        ml_only = pie.row_metrics(D.Y, P, D)
        ml_only.to_csv(cache, index=False)
    cl = D.clusters("strict")
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.55), gridspec_kw={"wspace": 0.35})
    table = []
    for ax, (key, label), letter in zip(axes, (("family_mae", "Hue-family MAE"), ("active_mae", "MAE on the true dyes")), "ab"):
        a, b = blend[key].to_numpy(), ml_only[key].to_numpy()
        d, p = signflip(a, b, cl)
        g = pd.DataFrame({"cl": cl, "a": a, "b": b}).groupby("cl").mean()
        ax.scatter(g.b, g.a, s=14, color=C_OURS, edgecolor="#ffffff", linewidth=0.5, zorder=3)
        hi = float(np.nanmax(g[["a", "b"]].to_numpy())) * 1.08
        ax.plot([0, hi], [0, hi], color=MUTED, linewidth=0.8)
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("ML amounts")
        ax.set_ylabel("0.5·KM + 0.5·ML")
        tie = np.isclose(g.a, g.b, atol=1e-9)
        print(f"  R5 {key}: Δ = {d:+.3f}, {p_text(p)}; shades below / on / above the line: "
              f"{100 * float(((g.a < g.b) & ~tie).mean()):.0f}% / {100 * float(tie.mean()):.0f}% / {100 * float(((g.a > g.b) & ~tie).mean()):.0f}%")
        panel_label(ax, letter, label)
        table.append(g.reset_index().rename(columns={"a": f"{key}_blend", "b": f"{key}_ml_only"}).assign(metric=key, diff=d, p=p))
    save(fig, "fig_R5_physics_amounts", pd.concat(table, ignore_index=True))


# ============================================================ R6 which inputs matter (spectral importance)

def fig_r6() -> None:
    cache = OUT / "_r6_permutation_rows.npz"
    spec = importlib.util.spec_from_file_location("nsga", ROOT / "scripts" / "62_nsga2_feature_selection.py")
    nsga = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nsga)
    groups = nsga.feature_groups(D.features["spectral_substrate"])
    names = list(groups)
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        base_fam, perm_fam = z["base_fam"], z["perm_fam"]
    else:
        from sklearn.ensemble import ExtraTreesRegressor

        X_all = D.X("spectral_substrate")
        rng = np.random.default_rng(0)
        R = 10
        Pb = np.zeros_like(D.Y)
        Pp = np.zeros((len(names), R, *D.Y.shape))
        for tr, va in D.folds["strict"]:
            X = nsga.impute(X_all, tr)
            est = ExtraTreesRegressor(n_estimators=120, max_features=1.0, min_samples_leaf=3, random_state=pie.SEED, n_jobs=-1).fit(X[tr], D.Y[tr])
            Pb[va] = pie.feasible(est.predict(X[va]))
            for gi, cols in enumerate(groups.values()):
                for r in range(R):
                    Xp = X[va].copy()
                    Xp[:, cols] = Xp[rng.permutation(len(va))][:, cols]
                    Pp[gi, r][va] = pie.feasible(est.predict(Xp))
        base_fam = pie.row_metrics(D.Y, Pb, D)["family_mae"].to_numpy()
        perm_fam = np.array([np.mean([pie.row_metrics(D.Y, Pp[gi, r], D)["family_mae"].to_numpy() for r in range(R)], 0)
                             for gi in range(len(names))])
        np.savez(cache, base_fam=base_fam, perm_fam=perm_fam, names=np.array(names))
    cl = D.clusters("strict")
    stats = []
    for gi, g in enumerate(names):
        d, p = signflip(perm_fam[gi], base_fam, cl)
        m, lo, hi = cluster_ci(perm_fam[gi] - base_fam, cl)
        stats.append({"group": g, "rise": d, "ci_lo": lo, "ci_hi": hi, "p": p})
    st = pd.DataFrame(stats)
    st["p_holm"] = holm(st.p)

    fig = plt.figure(figsize=(W2, 2.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    bands = nsga.BANDS
    centres = np.array([(lo + hi) / 2 for lo, hi in bands])
    for lo, hi, fam in ((400, 490, "yellow"), (500, 590, "red"), (600, 690, "blue")):
        ax.axvspan(lo, hi, color=FAMILY[fam], alpha=0.10, linewidth=0)
        ax.text((lo + hi) / 2, 0.97, f"{fam}\nabsorption", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.0, color=INK2)
    for prefix, label, colr in (("KS_", "K/S (absorption)", C_OURS), ("R_", "Reflectance", MUTED)):
        sub = st[st.group.str.startswith(prefix)].reset_index(drop=True)
        ax.plot(centres, sub.rise, color=colr, linewidth=1.4, label=label, zorder=2)
        ax.errorbar(centres, sub.rise, yerr=[sub.rise - sub.ci_lo, sub.ci_hi - sub.rise], color=colr, linewidth=0.7, capsize=1.5, linestyle="none", zorder=2)
        sig = sub.p_holm < 0.05
        ax.scatter(centres[sig], sub.rise[sig], s=22, color=colr, edgecolor="#ffffff", linewidth=0.6, zorder=3)
        ax.scatter(centres[~sig], sub.rise[~sig], s=22, facecolor="#ffffff", edgecolor=colr, linewidth=0.9, zorder=3)
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_xlim(400, 750)
    ax.set_ylim(-0.012, 0.118)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Δ hue-family MAE")
    ax.legend(loc="upper right", fontsize=6.3, handlelength=1.4, bbox_to_anchor=(1.0, 0.80))
    panel_label(ax, "a", "Spectral bands")
    ax = fig.add_subplot(gs[0, 1])
    nonspec = st[~st.group.str.match(r"^(KS_|R_)")].copy()
    pretty = {"refl_summary_stats": "R stats", "refl_derivatives": "R deriv.",
              "refl_band_means": "R bands", "ks_summary_stats": "K/S stats", "ks_derivatives": "K/S deriv.",
              "ks_band_means": "K/S bands", "substrate_weight": "GSM", "liquor_ratio": "L:R",
              "fibre_composition": "Fibre", "fabric_type": "Fabric", "knit_structure": "Knit"}
    nonspec["label"] = nonspec.group.map(pretty)
    nonspec = nonspec.sort_values("rise")
    yy = np.arange(len(nonspec))
    is_sub = nonspec.group.isin(["substrate_weight", "liquor_ratio", "fibre_composition", "fabric_type", "knit_structure"])
    colours = np.where(is_sub, MUTED, C_OURS)
    ax.barh(yy, nonspec.rise, color=colours, height=0.62)
    ax.errorbar(nonspec.rise, yy, xerr=[nonspec.rise - nonspec.ci_lo, nonspec.ci_hi - nonspec.rise], color=INK, linewidth=0.6, capsize=1.3, linestyle="none")
    for y_, (_, r) in zip(yy, nonspec.iterrows()):
        if r.p_holm < 0.05:
            ax.text(r.ci_hi + 0.001, y_, "*", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(yy, nonspec.label, fontsize=6.4)
    ax.axvline(0, color=AXIS, linewidth=0.8)
    ax.set_xlabel("Δ hue-family MAE")
    ax.grid(axis="y", visible=False)
    panel_label(ax, "b", "Other feature groups")
    save(fig, "fig_R6_input_importance", st)


# ===================================================================== R7 ranked shortlist (Hit@k)

def fig_r7() -> None:
    cache = OUT / "_r7_ranks.csv"
    A = D.Y > pie.ACTIVE
    K = 10
    if cache.exists():
        rk = pd.read_csv(cache)
    else:
        _, rank, _ = pie.run_averaged_ranker(D, "strict", pie.BASE_ALL, soft=False)
        Xs = np.nan_to_num(D.X("spectral"))
        knn = np.full(len(D.Y), np.inf)
        from sklearn.preprocessing import StandardScaler

        for tr, va in D.folds["strict"]:
            sc = StandardScaler().fit(Xs[tr])
            dist = ((sc.transform(Xs[va])[:, None, :] - sc.transform(Xs[tr])[None]) ** 2).sum(-1)
            for i, r in enumerate(va):
                sets = []
                for j in tr[np.argsort(dist[i])]:
                    s = tuple(np.flatnonzero(A[j]))
                    if s not in sets:
                        sets.append(s)
                    if len(sets) == K:
                        break
                t = tuple(np.flatnonzero(A[r]))
                if t in sets:
                    knn[r] = sets.index(t) + 1
        rk = pd.DataFrame({"ours": rank, "retrieval": knn})
        rk.to_csv(cache, index=False)
    strata = D.strata("strict")
    cl = D.clusters("strict")
    ml_exact = rows("E00b", "ml_et_spectral_substrate_feasible").exact_set.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.3), gridspec_kw={"wspace": 0.3})
    table = []
    ks = np.arange(1, K + 1)
    for ax, (mask, title), letter in zip(axes, ((np.ones(len(strata), bool), "All unseen shades"), (strata == 1, "Unseen shades, seen dye set")), "ab"):
        ends = []
        for col, name, colr in (("ours", "Our method", C_OURS), ("retrieval", "Nearest-recipe retrieval", C_RET)):
            r = rk[col].to_numpy()
            vals = [cluster_ci(np.where(mask, (r <= k).astype(float), np.nan), cl) for k in ks]
            m = np.array([v[0] for v in vals])
            lo = np.array([v[1] for v in vals])
            hi = np.array([v[2] for v in vals])
            ax.fill_between(ks, lo, hi, color=colr, alpha=0.12, linewidth=0)
            ax.plot(ks, m, color=colr, marker="o", markersize=3.2, label=name)
            ends.append((m[-1], name))
            table.append(pd.DataFrame({"panel": letter, "method": name, "k": ks, "hit": m, "ci_lo": lo, "ci_hi": hi}))
        (v1, n1), (v2, n2) = ends
        if abs(v1 - v2) < 0.05:
            hi_v, lo_v = (v1, v2) if v1 >= v2 else (v2, v1)
            mid = (v1 + v2) / 2
            ax.text(K + 0.2, mid + 0.03, f"{hi_v:.2f}", va="center", fontsize=6.4, color=INK)
            ax.text(K + 0.2, mid - 0.03, f"{lo_v:.2f}", va="center", fontsize=6.4, color=INK)
        else:
            for v, _ in ends:
                ax.text(K + 0.2, v, f"{v:.2f}", va="center", fontsize=6.4, color=INK)
        e = cluster_ci(np.where(mask, ml_exact, np.nan), cl)
        ax.plot(1, e[0], marker="D", markersize=4.5, color=C_ML, linestyle="none", label="ML reference")
        table.append(pd.DataFrame({"panel": [letter], "method": ["ML reference"], "k": [1], "hit": [e[0]], "ci_lo": [e[1]], "ci_hi": [e[2]]}))
        ax.set_xticks(ks)
        ax.set_xlim(0.6, K + 1.4)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Shortlist length k")
        ax.set_ylabel("Share of lab dips\n(recipe in the top k)")
        panel_label(ax, letter, title)
        if letter == "a":
            ax.legend(loc="upper left", fontsize=6.2, handlelength=1.4)
    d1, p1 = signflip((rk.ours.to_numpy() <= 1).astype(float), (rk.retrieval.to_numpy() <= 1).astype(float), cl)
    print(f"  R7: k = 1 vs retrieval, Δ = {d1:+.3f}, {p_text(p1)}")
    save(fig, "fig_R7_shortlist", pd.concat(table, ignore_index=True))


FIGS = {"M1": fig_m1, "R1": fig_r1, "R2": fig_r2, "R3": fig_r3, "R4": fig_r4, "R5": fig_r5, "R6": fig_r6, "R7": fig_r7}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", default=list(FIGS))
    args = parser.parse_args()
    for k in args.only:
        print(f"[{k}]")
        FIGS[k]()


if __name__ == "__main__":
    main()
