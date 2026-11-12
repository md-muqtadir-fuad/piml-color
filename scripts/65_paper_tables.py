"""Paper tables for the revised manuscript, recomputed from per-row outputs.

Writes LaTeX table bodies and CSVs to outputs/tables/ and prints every number.
Tests: paired sign-flip permutation test over clusters (target shade for strict CV, recipe for grouped CV),
20,000 flips, Holm correction within each declared family (the same helpers as scripts/63_paper_figures.py).

Run: python scripts/65_paper_tables.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location("figs", ROOT / "scripts" / "63_paper_figures.py")
figs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(figs)
pie, D = figs.pie, figs.D

OUT = ROOT / "outputs" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
PROTOS = ("strict", "grouped")
ML_REF = {"strict": ("E00b", "ml_et_spectral_substrate_feasible"), "grouped": ("E00b", "ml_et_spectral_feasible")}
METRICS = figs.METRICS  # (key, label, +1 if higher is better)


def fmt(v: float, nd: int = 3) -> str:
    return f"{v:.{nd}f}"


def fmt_d(v: float) -> str:
    """Signed difference with a typeset minus/plus sign."""
    return ("$-$" if v < 0 else "$+$") + f"{abs(v):.3f}"


def fmt_p(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


# ------------------------------------------------------------------ T-data: dataset and strata

def dye_facts(threshold: float) -> dict:
    """Recipe-level dye counts; rows of one recipe share its dye set, so the first row is used."""
    A = D.Y > threshold
    rec_sets = pd.Series([frozenset(np.flatnonzero(r)) for r in A]).groupby(D.clusters("grouped")).first()
    dye_recipes = pd.Series([j for s in rec_sets for j in s]).value_counts()
    sizes = rec_sets.map(len).value_counts()
    return {"dyes_used": len(dye_recipes), "dye_sets": len({s for s in rec_sets if s}),
            "dyes_one_recipe": int((dye_recipes == 1).sum()),
            **{f"recipes_{k}_dyes": int(sizes.get(k, 0)) for k in (3, 2, 1, 0)}}


def table_data() -> None:
    recorded, active = dye_facts(0.0), dye_facts(pie.ACTIVE)
    strata = {p: np.bincount(D.strata(p), minlength=4)[1:] for p in PROTOS}
    n = {"rows": len(D.Y), "recipes": len(np.unique(D.clusters("grouped"))), "shades": len(np.unique(D.clusters("strict")))}
    print("[data]", n, "| recorded", recorded, "| active", active, "| strata", {p: s.tolist() for p, s in strata.items()})
    pd.DataFrame([n | {f"recorded_{k}": v for k, v in recorded.items()} | {f"active_{k}": v for k, v in active.items()}
                  | {f"strata_{p}": s.tolist() for p, s in strata.items()}]).to_csv(OUT / "tab_data.csv", index=False)
    pct = {p: [f"{v} ({100 * v / n['rows']:.0f}\\,\\%)" for v in s] for p, s in strata.items()}
    lines = [
        rf"Lab-dip observations (rows) & \multicolumn{{2}}{{c}}{{{n['rows']}}} \\",
        rf"Recipes / target shades & \multicolumn{{2}}{{c}}{{{n['recipes']} / {n['shades']}}} \\",
        r"\addlinespace[2pt]",
        r"& As recorded & Active ($c>\tau$) \\",
        r"\cmidrule(lr){2-2}\cmidrule(lr){3-3}",
        rf"Dyes used (of {D.Y.shape[1]}) & {recorded['dyes_used']} & {active['dyes_used']} \\",
        rf"Distinct dye sets & {recorded['dye_sets']} & {active['dye_sets']} \\",
        rf"Dyes used in one recipe only & {recorded['dyes_one_recipe']} & {active['dyes_one_recipe']} \\",
        rf"Recipes with 3 / 2 / 1 / 0 dyes & {recorded['recipes_3_dyes']} / {recorded['recipes_2_dyes']} / "
        rf"{recorded['recipes_1_dyes']} / {recorded['recipes_0_dyes']} & {active['recipes_3_dyes']} / "
        rf"{active['recipes_2_dyes']} / {active['recipes_1_dyes']} / {active['recipes_0_dyes']} \\",
        r"\addlinespace[2pt]",
        r"Rows by stratum & Strict CV & Grouped CV \\",
        r"\cmidrule(lr){2-2}\cmidrule(lr){3-3}",
        rf"\quad Dye set seen in training & {pct['strict'][0]} & {pct['grouped'][0]} \\",
        rf"\quad Novel combination of seen dyes & {pct['strict'][1]} & {pct['grouped'][1]} \\",
        rf"\quad Contains an unseen dye & {pct['strict'][2]} & {pct['grouped'][2]} \\",
    ]
    (OUT / "tab_data_body.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ T-main: main comparison, both protocols

MAIN_METHODS = [
    ("Exhaustive KM matching", "E02", "ccm_min_de_max"),
    ("Nearest-recipe retrieval", "E00b", "nearest_historical_spectral"),
    ("FT-Transformer", "A04", "ft_transformer"),
    ("FT-Transformer + slot head", "A07d", "ft_logic_slot_ensemble"),
    ("ML reference (Extra Trees)", None, None),
    ("Our method", "E16", "avg_base"),
    ("\quad without physics features", "E16", "avg_base_no_physics"),
]


def table_main() -> None:
    out, csv = [], []
    for proto in PROTOS:
        cl = D.clusters(proto)
        data = {}
        for name, exp, meth in MAIN_METHODS:
            exp, meth = (exp, meth) if exp else ML_REF[proto]
            data[name] = pie.load_rows(exp, meth, proto)
        for name, df in data.items():
            vals = [df[k].mean() for k, _, _ in METRICS]
            csv.append({"protocol": proto, "method": name} | {k: v for (k, _, _), v in zip(METRICS, vals)})
        # ML variants: confirm the reference is the stronger ET variant under each protocol
        for meth in ("ml_et_spectral_feasible", "ml_et_spectral_substrate_feasible"):
            print(f"  [{proto}] {meth}: F1 {pie.load_rows('E00b', meth, proto).f1.mean():.3f}")

        def tests(a: pd.DataFrame, b: pd.DataFrame) -> tuple[list[float], list[float]]:
            d_p = [figs.signflip(a[k].to_numpy(), b[k].to_numpy(), cl) for k, _, _ in METRICS]
            return [d for d, _ in d_p], list(figs.holm([p for _, p in d_p]))

        d_ml, p_ml = tests(data["Our method"], data["ML reference (Extra Trees)"])
        d_ft, p_ft = tests(data["FT-Transformer + slot head"], data["FT-Transformer"])
        d_np, p_np = tests(data["Our method"], data["\quad without physics features"])
        print(f"[main {proto}] ours-no_physics diffs {np.round(d_np, 3)}, p_holm {np.round(p_np, 4)}")
        out.append((proto, data, (d_ml, p_ml), (d_ft, p_ft), (d_np, p_np)))
        print(f"[main {proto}] ours-ML diffs {np.round(d_ml, 3)}, p_holm {np.round(p_ml, 4)}")
        print(f"[main {proto}] slot-FT diffs {np.round(d_ft, 3)}, p_holm {np.round(p_ft, 4)}")
    pd.DataFrame(csv).to_csv(OUT / "tab_main.csv", index=False)

    lines = []
    head = {"strict": "Unseen shades (strict CV, 51 shades)", "grouped": "Seen shades (grouped CV, 85 recipes)"}
    for proto, data, (d_ml, p_ml), (d_ft, p_ft), (d_np, p_np) in out:
        lines.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\textit{{{head[proto]}}}}} \\")
        for name, df in data.items():
            cells = [fmt(df[k].mean()) for k, _, _ in METRICS]
            lines.append(f"{name} & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace[1pt]")
        lines.append(r"\quad Ours vs ML, $p_{\mathrm{Holm}}$ & " + " & ".join(fmt_p(p) for p in p_ml) + r" \\")
        lines.append(r"\quad Ours vs no physics, $p_{\mathrm{Holm}}$ & " + " & ".join(fmt_p(p) for p in p_np) + r" \\")
        lines.append(r"\quad Slot head vs FT, $p_{\mathrm{Holm}}$ & " + " & ".join(fmt_p(p) for p in p_ft) + r" \\")
        if proto == "strict":
            lines.append(r"\midrule")
    (OUT / "tab_main_body.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ T-amounts: physics amount ablation

AMOUNT_METRICS = [("active_mae", "MAE on true dyes"), ("family_mae", "Hue-family MAE"),
                  ("union_mae", "Union MAE"), ("total_abs_err", "Total-dye error")]


def table_amounts() -> None:
    res, csv = {}, []
    for proto in PROTOS:
        cl = D.clusters(proto)
        per_alpha = {}
        for alpha in (0.0, 0.5, 1.0):
            P, _, _ = pie.run_averaged_ranker(D, proto, pie.BASE_ALL, soft=False, alpha=alpha)
            per_alpha[alpha] = pie.row_metrics(D.Y, P, D)
        stored = pie.load_rows("E16", "avg_base", proto)
        for k, _ in AMOUNT_METRICS:  # the alpha = 0.5 run must reproduce the stored final method
            assert np.allclose(per_alpha[0.5][k].to_numpy(), stored[k].to_numpy(), atol=1e-9, equal_nan=True), (proto, k)
        tests = {}  # one Holm family per comparison (alpha vs ML amounts) over the four amount metrics, as in Fig. R4
        for alpha, base in ((0.5, 0.0), (1.0, 0.0), ("blend_vs_km", None)):
            if base is None:  # blend (0.5) vs KM amounts (1.0)
                d_p = [figs.signflip(per_alpha[0.5][k].to_numpy(), per_alpha[1.0][k].to_numpy(), cl) for k, _ in AMOUNT_METRICS]
            else:
                d_p = [figs.signflip(per_alpha[alpha][k].to_numpy(), per_alpha[base][k].to_numpy(), cl) for k, _ in AMOUNT_METRICS]
            for (k, _), (d, p), ph in zip(AMOUNT_METRICS, d_p, figs.holm([p for _, p in d_p])):
                tests[(alpha, k)] = (d, p, ph)
                print(f"[amounts {proto}] {alpha} (vs ML amounts unless blend_vs_km), {k}: diff {d:+.3f}, p {p:.4f}, p_holm {ph:.4f}")
                csv.append({"protocol": proto, "alpha": alpha, "metric": k, "diff_vs_ml_amounts": d, "p": p, "p_holm": ph})
        res[proto] = (per_alpha, tests)
        for a, m in per_alpha.items():
            csv.append({"protocol": proto, "alpha": a} | {k: m[k].mean() for k, _ in AMOUNT_METRICS})
    pd.DataFrame(csv).to_csv(OUT / "tab_amounts.csv", index=False)

    names = {0.0: r"ML amounts ($\alpha=0$)", 0.5: r"Blend ($\alpha=0.5$)", 1.0: r"KM amounts ($\alpha=1$)"}
    head = {"strict": "Unseen shades (strict CV)", "grouped": "Seen shades (grouped CV)"}
    lines = []
    for proto in PROTOS:
        per_alpha, tests = res[proto]
        lines.append(rf"\multicolumn{{5}}{{@{{}}l}}{{\textit{{{head[proto]}}}}} \\")
        for a in (0.0, 0.5, 1.0):
            lines.append(f"{names[a]} & " + " & ".join(fmt(per_alpha[a][k].mean()) for k, _ in AMOUNT_METRICS) + r" \\")
        lines.append(r"\addlinespace[1pt]")
        for a, label in ((0.5, "Blend vs ML"), (1.0, "KM vs ML"), ("blend_vs_km", "Blend vs KM")):
            lines.append(rf"\quad {label}, $p_{{\mathrm{{Holm}}}}$ & "
                         + " & ".join(fmt_p(tests[(a, k)][2]) for k, _ in AMOUNT_METRICS) + r" \\")
        if proto == "strict":
            lines.append(r"\midrule")
    (OUT / "tab_amounts_body.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ T-negative: methods without a significant gain (supplement)

NEGATIVE = [
    ("Gradient boosting", "XGBoost", "A01", "xgb_plain"),
    ("Gradient boosting", "LightGBM", "A01", "lgbm_plain"),
    ("Attention + boosting", "Attention $\\rightarrow$ XGBoost", "A01", "attn_to_xgb"),
    ("Attention + boosting", "Attention $\\rightarrow$ LightGBM", "A01", "attn_to_lgbm"),
    ("Attention + boosting", "GBDT leaves $\\rightarrow$ attention", "A02", "gbdt_leaves_to_attention"),
    ("Attention + boosting", "Soft attention trees", "A03", "soft_attention_trees"),
    ("Neural", "KAN, 1 layer", "A05", "kan_1_layer"),
    ("Neural", "KAN, 2 layers", "A05", "kan_2_layer_h8"),
    ("Search", "Composite cost, nested selection", "A08", "proxy_nested_top1"),
    ("Search", "NSGA-II feature selection, nested", "N01", "nsga2_et_selected_features"),
    ("Search", "Ranker with nested L2 tuning", "E12", "nested_nll_all"),
    ("Other", "Physics-based augmentation, nested", "E08", "aug_et_inner_cv"),
    ("Other", "Confidence gate (ranker or ML), nested", "E09", "gate_all__gate"),
    ("Other", "Two-stage slot classifier", "T01", "thesis_two_stage"),
]
FT_LOSSES = ["ft_mse", "ft_mae", "ft_huber", "ft_log_mse", "ft_active_weighted_mse", "ft_family_mse", "ft_hurdle",
             "ft_composition", "ft_soft_f1", "ft_km_physics"]


def table_negative() -> None:
    proto = "strict"
    cl = D.clusters(proto)
    ref = pie.load_rows(*ML_REF[proto], proto)
    entries = [(g, label, exp, meth) for g, label, exp, meth in NEGATIVE] + [("Loss", m, "A06", m) for m in FT_LOSSES]
    stats = []
    for group, label, exp, meth in entries:
        df = pie.load_rows(exp, meth, proto)
        d_f1, p_f1 = figs.signflip(df.f1.to_numpy(), ref.f1.to_numpy(), cl)
        d_fam, p_fam = figs.signflip(df.family_mae.to_numpy(), ref.family_mae.to_numpy(), cl)
        stats.append({"group": group, "method": label, "f1": df.f1.mean(), "d_f1": d_f1, "p_f1": p_f1,
                      "family_mae": df.family_mae.mean(), "d_family_mae": d_fam, "p_family_mae": p_fam})
    st = pd.DataFrame(stats)
    st["p_holm_f1"] = figs.holm(st.p_f1)  # Holm across all methods in the table, per metric
    st["p_holm_family_mae"] = figs.holm(st.p_family_mae)
    st.to_csv(OUT / "tab_negative.csv", index=False)
    for r in st.itertuples():
        print(f"[negative] {r.method:40s} F1 {r.f1:.3f} ({r.d_f1:+.3f}, p_holm {r.p_holm_f1:.4f}); "
              f"family MAE {r.family_mae:.3f} ({r.d_family_mae:+.3f}, p_holm {r.p_holm_family_mae:.4f})")

    lines, last = [], None
    for r in st[st.group != "Loss"].itertuples():
        if r.group != last and last is not None:
            lines.append(r"\addlinespace[2pt]")
        last = r.group
        lines.append(f"{r.method} & {fmt(r.f1)} & {fmt_d(r.d_f1)} & {fmt_p(r.p_holm_f1)} & {fmt(r.family_mae)} & "
                     f"{fmt_d(r.d_family_mae)} & {fmt_p(r.p_holm_family_mae)} \\\\")
    lo = st[st.group == "Loss"]

    def rng(col: str, p: bool = False) -> str:
        a, b = lo[col].min(), lo[col].max()
        return f"{fmt_p(a)}--{fmt_p(b)}" if p else f"{fmt_d(a)} to {fmt_d(b)}" if col.startswith("d_") else f"{a:.3f}--{b:.3f}"

    lines.append(r"\addlinespace[2pt]")
    lines.append(f"FT-Transformer, 10 loss functions & {rng('f1')} & {rng('d_f1')} & {rng('p_holm_f1', True)} & "
                 f"{rng('family_mae')} & {rng('d_family_mae')} & {rng('p_holm_family_mae', True)} \\\\")
    (OUT / "tab_negative_body.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[negative] ML reference: F1 {ref.f1.mean():.3f}, family MAE {ref.family_mae.mean():.3f}")


if __name__ == "__main__":
    table_data()
    table_main()
    table_amounts()
    table_negative()
