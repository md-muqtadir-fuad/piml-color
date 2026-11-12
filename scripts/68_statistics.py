"""Appendix B/C: variable statistics and exploratory data analysis of the 212 analysis rows.

Uses the same `Data` object as the experiments (scripts/60), so every count matches the paper.
Dye names are replaced by the anonymised labels (Dye A ... Dye AJ); no shade codes or sample ids
are written.

Outputs (outputs/appendix/):
    tab_B_numeric_body.tex, tab_B_categorical_body.tex, tab_B_dyes_body.tex
    figures/fig_C1_spectra.pdf, figures/fig_C2_recipes.pdf, figures/fig_C3_correlation.pdf
    appendix_stats.json (numbers quoted in the appendix text)

Run: python scripts/68_appendix_statistics.py
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
_spec = importlib.util.spec_from_file_location("figs", ROOT / "scripts" / "63_paper_figures.py")
figs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(figs)  # paper style (rcParams, colours) and the shared Data object

import colour  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

pie, D = figs.pie, figs.D
OUT = ROOT / "outputs" / "appendix"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
TAU = pie.ACTIVE
WIDTH = 6.5  # appendix \textwidth in inches, so fonts print at their set size
df = D.df
num = lambda c: pd.to_numeric(df[c], errors="coerce")  # noqa: E731


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print(f"  saved {name}")


def fmt(x: float, d: int) -> str:
    return "--" if pd.isna(x) else f"{x:.{d}f}"


# --------------------------------------------------------------------------- shared quantities
A = D.Y > TAU
total = D.Y.sum(1)
n_active = A.sum(1)
recipe = df["recipe_no"].astype(str).to_numpy()
target = df["target_id"].astype(str).to_numpy()
fam_amount = {h: D.Y[:, D.dye_hue == h].sum(1) for h in ("yellow", "red", "blue")}
other_mask = ~np.isin(D.dye_hue, ["yellow", "red", "blue"])
fam_amount["black/other"] = D.Y[:, other_mask].sum(1)
mapping = pd.read_csv(ROOT / "data/global_naming/global_dye_mapping_active_36.csv")
label = dict(zip(mapping["dye_column"], mapping["global_dye_name"]))
dye_labels = [label[c.removeprefix("vec__")] for c in D.dye_cols]


# --------------------------------------------------------------------------- Table B1: numeric variables
def numeric_rows() -> list[tuple[str, str, pd.Series, int]]:
    """(group, name, values, decimals)."""
    r = []
    sub = [("GSM (g\\,m$^{-2}$)", "production__gsm", 0), ("Sample weight (g)", "production__sample_weight", 2),
           ("Liquor ratio (1:$x$)", "production__l2r_numeric", 1), ("Cotton (\\%)", "production__cotton", 1),
           ("Viscose (\\%)", "production__viscose", 1), ("Modal (\\%)", "production__modal", 1),
           ("Polyester (\\%)", "production__polyester", 1), ("Elastane (\\%)", "production__lycra", 1)]
    for name, c, d in sub:
        r.append(("Substrate inputs", name, num(c), d))
    for name, c, d in [("Salt (as recorded)", "production__salt", 1), ("Soda ash (as recorded)", "production__soda_ash", 1)]:
        r.append(("Process auxiliaries (excluded)", name, num(c), d))
    r.append(("Recipe (outputs)", "Total dye (\\% OWF)", pd.Series(total), 3))
    r.append(("Recipe (outputs)", "Active dyes per lab dip", pd.Series(n_active.astype(float)), 2))
    for h in ("yellow", "red", "blue"):  # the one black dye column is never active in the 212 rows
        r.append(("Recipe (outputs)", f"{h.capitalize()} dyes (\\% OWF)", pd.Series(fam_amount[h]), 3))
    for name, c in [("Reflectance, 400--490 nm (\\%)", "feat_refl_band_mean_blue_400_490"),
                    ("Reflectance, 500--570 nm (\\%)", "feat_refl_band_mean_green_500_570"),
                    ("Reflectance, 580--620 nm (\\%)", "feat_refl_band_mean_yellow_orange_580_620"),
                    ("Reflectance, 630--750 nm (\\%)", "feat_refl_band_mean_red_630_750"),
                    ("$K/S$, mean over 400--750 nm", "feat_ks_mean")]:
        r.append(("Target spectrum (inputs)", name, num(c), 2))
    for name, c in [("$\\Delta E_{\\mathrm{CMC}}$, D65", "dE_d65"), ("$\\Delta E_{\\mathrm{CMC}}$, F11", "dE_f11"),
                    ("$\\Delta E_{\\max}$", "dE_max")]:
        r.append(("Lab-dip color difference (not used)", name, num(c), 2))
    return r


def table_numeric() -> pd.DataFrame:
    lines, recs, last = [], [], None
    for group, name, s, d in numeric_rows():
        if group != last:
            if last is not None:
                lines.append("\\addlinespace[2pt]")
            lines.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{group}}}}} \\\\")
            last = group
        q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
        vals = [s.notna().sum(), s.isna().sum(), s.mean(), s.std(), med, q1, q3, s.min(), s.max()]
        recs.append(dict(group=group, variable=name, n=vals[0], missing=vals[1], mean=vals[2], sd=vals[3],
                         median=med, q1=q1, q3=q3, min=vals[7], max=vals[8]))
        lines.append(f"\\quad {name} & {vals[0]} & {vals[1]} & {fmt(vals[2], d)} & {fmt(vals[3], d)} & "
                     f"{fmt(med, d)} & {fmt(q1, d)}--{fmt(q3, d)} & {fmt(vals[7], d)}--{fmt(vals[8], d)} \\\\")
    (OUT / "tab_B_numeric_body.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------- Table B2: categorical variables
def table_categorical() -> dict:
    fab = df["production__fabric"] if "production__fabric" in df.columns else None
    if fab is None:  # one-hot columns only
        cols = [c for c in df.columns if c.startswith("feat_cat_production_fabric_")]
        fab = df[cols].idxmax(axis=1).str.removeprefix("feat_cat_production_fabric_")
    dom_cols = [c for c in df.columns if c.startswith("feat_cat_production_dominant_fiber_")]
    dom = df[dom_cols].idxmax(axis=1).str.removeprefix("feat_cat_production_dominant_fiber_")
    blend = np.where(num("production__is_lycra_blend") > 0, "with elastane", "without elastane")
    lines, out = [], {}
    for title, s in [("Fabric construction", fab), ("Dominant fiber", dom), ("Elastane", pd.Series(blend))]:
        t = pd.DataFrame({"v": s.astype(str).str.replace("_", " ").to_numpy(), "recipe": recipe})
        g = t.groupby("v").agg(dips=("v", "size"), recipes=("recipe", "nunique")).sort_values("dips", ascending=False)
        lines.append(f"\\multicolumn{{3}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\")
        for v, row in g.iterrows():
            lines.append(f"\\quad {v} & {row.dips} & {row.recipes} \\\\")
        lines.append("\\addlinespace[2pt]")
        out[title] = g.to_dict(orient="index")
    (OUT / "tab_B_categorical_body.tex").write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    return out


# --------------------------------------------------------------------------- Table B3: dye usage
def table_dyes() -> pd.DataFrame:
    recs = []
    for j, lab in enumerate(dye_labels):
        used = A[:, j]
        amounts = D.Y[used, j]
        recs.append(dict(dye=lab, family=D.dye_hue[j], dips=int(used.sum()), recipes=len(set(recipe[used])),
                         median=float(np.median(amounts)) if used.any() else np.nan,
                         min=float(amounts.min()) if used.any() else np.nan,
                         max=float(amounts.max()) if used.any() else np.nan))
    t = pd.DataFrame(recs)
    t["order"] = t["dye"].str.removeprefix("Dye ").map(lambda s: (len(s), s))
    t = t.sort_values(["recipes", "order"], ascending=[False, True]).drop(columns="order")
    lines = [f"{r.dye} & {r.family} & {r.recipes} & {r.dips} & {fmt(r.median, 3)} & "
             f"{fmt(r.min, 3)}--{fmt(r.max, 3)} \\\\" if r.dips else
             f"{r.dye} & {r.family} & 0 & 0 & -- & -- \\\\" for r in t.itertuples()]
    (OUT / "tab_B_dyes_body.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return t


# --------------------------------------------------------------------------- Figure C1: target spectra
def srgb_of_targets(rows: np.ndarray) -> np.ndarray:
    wl = np.arange(400, 760, 10)
    out = []
    cmfs = colour.MSDS_CMFS["CIE 1964 10 Degree Standard Observer"]
    ill = colour.SDS_ILLUMINANTS["D65"]
    for r in rows:
        sd = colour.SpectralDistribution(dict(zip(wl, D.R[r])))
        xyz = colour.sd_to_XYZ(sd, cmfs, ill) / 100.0
        out.append(np.clip(colour.XYZ_to_sRGB(xyz), 0, 1))
    return np.array(out)


def fig_spectra() -> dict:
    first = pd.Series(np.arange(len(df))).groupby(target).first().to_numpy()
    rgb = srgb_of_targets(first)
    wl = np.arange(400, 760, 10)
    fig, ax = plt.subplots(1, 2, figsize=(WIDTH, 2.4))
    for k, r in enumerate(first):
        ax[0].plot(wl, 100 * D.R[r], color=rgb[k], lw=1.0, alpha=0.9)
        ax[1].plot(wl, D.KS[r], color=rgb[k], lw=1.0, alpha=0.9)
    ax[0].set(xlabel="Wavelength (nm)", ylabel="Reflectance (%)", xlim=(400, 750), ylim=(0, None))
    ax[1].set(xlabel="Wavelength (nm)", ylabel="$K/S$ (log scale)", xlim=(400, 750), yscale="log")
    figs.panel_label(ax[0], "a", f"Target reflectance ({len(first)} shades)")
    figs.panel_label(ax[1], "b", "Kubelka–Munk $K/S$")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig_C1_spectra")
    return {"shades": int(len(first))}


# --------------------------------------------------------------------------- Figure C2: recipes and approval quality
def fig_recipes() -> dict:
    fig, ax = plt.subplots(1, 3, figsize=(WIDTH, 2.3))
    # (a) total dye per lab dip
    bins = np.logspace(np.log10(max(total[total > 0].min(), 1e-3)), np.log10(total.max()), 18)
    ax[0].hist(total[total > 0], bins=bins, color=figs.SEQ[3], edgecolor="white", linewidth=0.6)
    ax[0].set(xscale="log", xlabel="Total dye (% OWF)", ylabel="Lab dips")
    figs.panel_label(ax[0], "a", "Depth of shade")
    # (b) amount of each active dye by hue family
    fams = ["yellow", "red", "blue"]
    data = [D.Y[:, D.dye_hue == h][A[:, D.dye_hue == h]] for h in fams]
    bp = ax[1].boxplot(data, widths=0.5, patch_artist=True, showfliers=True,
                       medianprops=dict(color=figs.INK, lw=1.2),
                       flierprops=dict(marker="o", ms=2.5, mfc=figs.MUTED, mec="none", alpha=0.7))
    for patch, h in zip(bp["boxes"], fams):
        patch.set(facecolor=figs.FAMILY[h], alpha=0.55, edgecolor=figs.FAMILY[h])
    ax[1].set_xticks([1, 2, 3], [f"{h}\n(n={len(x)})" for h, x in zip(fams, data)])
    ax[1].set(yscale="log", ylabel="Amount per active dye (% OWF)")
    figs.panel_label(ax[1], "b", "Amount by hue family")
    # (c) approval colour difference
    edges = np.arange(0, 1.85, 0.1)
    ax[2].hist(num("dE_d65"), bins=edges, histtype="step", lw=1.4, color=figs.C_OURS, label="D65")
    ax[2].hist(num("dE_f11"), bins=edges, histtype="step", lw=1.4, color=figs.C_ML, label="F11")
    ax[2].set(xlabel="$\\Delta E_{\\mathrm{CMC(2:1)}}$ of lab dip", ylabel="Lab dips")
    ax[2].legend(loc="upper right")
    figs.panel_label(ax[2], "c", "Lab-dip color difference")
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig_C2_recipes")
    return {"per_family_active_n": {h: int(len(x)) for h, x in zip(fams, data)}}


# --------------------------------------------------------------------------- Figure C3: Spearman correlations
def fig_correlation() -> dict:
    cols = {
        "GSM": num("production__gsm"), "Sample wt.": num("production__sample_weight"),
        "L:R": num("production__l2r_numeric"), "Cotton": num("production__cotton"),
        "Polyester": num("production__polyester"), "Elastane": num("production__lycra"),
        "Viscose": num("production__viscose"), "Modal": num("production__modal"),
        "Salt*": num("production__salt"), "Soda ash*": num("production__soda_ash"),
        "R mean": num("feat_refl_mean"), "K/S mean": num("feat_ks_mean"),
        "K/S 400–490": num("feat_ks_band_mean_blue_400_490"), "K/S 500–570": num("feat_ks_band_mean_green_500_570"),
        "K/S 630–750": num("feat_ks_band_mean_red_630_750"),
        "Total dye": pd.Series(total), "Yellow dye": pd.Series(fam_amount["yellow"]),
        "Red dye": pd.Series(fam_amount["red"]), "Blue dye": pd.Series(fam_amount["blue"]),
    }
    M = pd.DataFrame({k: v.to_numpy() for k, v in cols.items()})
    rho = M.corr(method="spearman").to_numpy()
    names = list(cols)
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f4f3ee", "#e34948"])
    fig, ax = plt.subplots(figsize=(WIDTH, 5.0))
    im = ax.imshow(rho, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=60, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.grid(False)
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j and abs(rho[i, j]) >= 0.5 and np.isfinite(rho[i, j]):
                ax.text(j, i, f"{rho[i, j]:.2f}".replace("0.", "."), ha="center", va="center", fontsize=5.5,
                        color="white" if abs(rho[i, j]) > 0.8 else figs.INK)
    for k in (7.5, 9.5, 14.5):
        ax.axhline(k, color=figs.INK2, lw=0.6)
        ax.axvline(k, color=figs.INK2, lw=0.6)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Spearman $\\rho$")
    cb.outline.set_visible(False)
    save(fig, "fig_C3_correlation")

    # spectral collinearity: adjacent wavelengths and band means
    R, KS = D.R, D.KS
    adj_r = [spearmanr(R[:, k], R[:, k + 1])[0] for k in range(R.shape[1] - 1)]
    adj_ks = [spearmanr(KS[:, k], KS[:, k + 1])[0] for k in range(KS.shape[1] - 1)]
    rk = lambda a, b: float(spearmanr(cols[a], cols[b], nan_policy="omit")[0])  # noqa: E731
    fiber_sum = (num("production__cotton") + num("production__polyester") + num("production__lycra")
                 + num("production__viscose") + num("production__modal"))
    return {
        "rho_salt_total": rk("Salt*", "Total dye"), "rho_soda_total": rk("Soda ash*", "Total dye"),
        "rho_cotton_polyester": rk("Cotton", "Polyester"), "rho_gsm_total": rk("GSM", "Total dye"),
        "rho_rmean_total": rk("R mean", "Total dye"), "rho_ksmean_total": rk("K/S mean", "Total dye"),
        "adjacent_refl_rho_min": float(np.min(adj_r)), "adjacent_refl_rho_median": float(np.median(adj_r)),
        "adjacent_ks_rho_min": float(np.min(adj_ks)), "adjacent_ks_rho_median": float(np.median(adj_ks)),
        "fiber_sum_min": float(fiber_sum.min()), "fiber_sum_max": float(fiber_sum.max()),
        "max_abs_rho_substrate_vs_total": float(max(abs(rk(a, "Total dye")) for a in names[:8])),
    }


def main() -> None:
    stats = {
        "rows": int(len(df)), "recipes": int(len(set(recipe))), "shades": int(len(set(target))),
        "dyes_active": int(A.any(0).sum()), "dye_columns": int(A.shape[1]),
        "recipes_by_n_dyes": pd.Series([A[recipe == r][0].sum() for r in sorted(set(recipe))]).value_counts().sort_index().to_dict(),
        "dips_per_recipe": pd.Series(recipe).value_counts().describe()[["min", "50%", "max"]].to_dict(),
        "dips_per_shade": pd.Series(target).value_counts().describe()[["min", "50%", "max"]].to_dict(),
    }
    numeric = table_numeric()
    stats["categorical"] = table_categorical()
    dyes = table_dyes()
    stats["dyes_single_recipe"] = int((dyes["recipes"] == 1).sum())
    stats.update(fig_spectra())
    stats.update(fig_recipes())
    stats.update(fig_correlation())
    numeric.to_csv(OUT / "tab_B_numeric.csv", index=False)
    dyes.to_csv(OUT / "tab_B_dyes.csv", index=False)
    (OUT / "appendix_stats.json").write_text(json.dumps(stats, indent=2, default=float), encoding="utf-8")
    print(json.dumps(stats, indent=2, default=float))


if __name__ == "__main__":
    main()
