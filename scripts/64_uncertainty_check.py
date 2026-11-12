"""U01 feasibility check: does the ranker's own probability carry usable uncertainty? (strict CV, cached bundles)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.special import softmax
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("pie", ROOT / "scripts" / "60_physics_inverse_experiments.py")
pie = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pie)

D = pie.Data()
proto = "strict"
use = pie.BASE_ALL
A = D.Y > pie.ACTIVE
n = len(D.Y)
rec = {k: np.full(n, np.nan) for k in ("p_top", "margin", "entropy", "de_best", "de_chosen", "f1", "exact", "p_true", "fold")}
probs, f1s = {}, {}
for f, (tr, va) in enumerate(D.folds[proto]):
    bd, prob, allowed = pie.fold_choice_probs(D, proto, f, use)  # final method, each row's own candidates (Eq. 7)
    o = bd["outer"]
    srt = np.sort(prob, 1)[:, ::-1]
    ch = prob.argmax(1)
    exact_c = (o["f1"] >= 1 - 1e-9) & allowed  # candidate equals the true set
    for i, r in enumerate(va):
        rec["p_top"][r] = srt[i, 0]
        rec["margin"][r] = srt[i, 0] - srt[i, 1]
        rec["entropy"][r] = -(prob[i] * np.log(prob[i] + 1e-12)).sum()
        rec["de_best"][r] = o["de"][i].min()
        rec["de_chosen"][r] = o["de"][i, ch[i]]
        rec["f1"][r] = o["f1"][i, ch[i]]
        rec["exact"][r] = float(exact_c[i, ch[i]])
        rec["p_true"][r] = prob[i][exact_c[i]].sum()  # 0 if the true set is not a candidate
        rec["fold"][r] = f
        probs[r], f1s[r] = prob[i], np.where(allowed[i], o["f1"][i], 0.0)  # other rows' candidates excluded

st = D.strata(proto)
cl = D.clusters(proto)
shades = np.unique(cl)
rng = np.random.default_rng(0)


def shade_mean(x):
    return np.array([x[cl == s].mean() for s in shades])


def cluster_boot_auc(y, s, B=2000):
    idx = {sh: np.flatnonzero(cl == sh) for sh in shades}
    vals = []
    for _ in range(B):
        take = np.concatenate([idx[sh] for sh in rng.choice(shades, len(shades))])
        if len(np.unique(y[take])) == 2:
            vals.append(roc_auc_score(y[take], s[take]))
    return np.percentile(vals, [2.5, 97.5])


def shade_perm_spearman(x, y, B=20000):
    xs, ys = shade_mean(x), shade_mean(y)
    r0 = spearmanr(xs, ys).statistic
    null = np.array([spearmanr(rng.permutation(xs), ys).statistic for _ in range(B)])
    return r0, (np.sum(np.abs(null) >= abs(r0)) + 1) / (B + 1)


print(f"rows {n}, shades {len(shades)}; exact rate {rec['exact'].mean():.3f}, mean F1 {rec['f1'].mean():.3f}")
print("\n1) Does confidence predict correctness? (row AUROC with shade-bootstrap CI; shade-level Spearman, permutation p)")
for name in ("p_top", "margin"):
    auc = roc_auc_score(rec["exact"], rec[name])
    lo, hi = cluster_boot_auc(rec["exact"], rec[name])
    r, p = shade_perm_spearman(rec[name], rec["f1"])
    print(f"  {name:7s}: AUROC(exact) {auc:.3f} [{lo:.3f}, {hi:.3f}];  shade Spearman with F1 {r:+.3f}, p {p:.4f}")
for name in ("de_best", "de_chosen"):
    auc = roc_auc_score(rec["exact"], -rec[name])
    lo, hi = cluster_boot_auc(rec["exact"], -rec[name])
    print(f"  -{name:9s}: AUROC(exact) {auc:.3f} [{lo:.3f}, {hi:.3f}]")

print("\n2) Calibration of p_top (reliability; ECE)")
bins = np.array([0, .2, .4, .6, .8, 1.0001])
ece = 0
for a, b in zip(bins[:-1], bins[1:]):
    m = (rec["p_top"] >= a) & (rec["p_top"] < b)
    if m.any():
        ece += m.mean() * abs(rec["p_top"][m].mean() - rec["exact"][m].mean())
        print(f"  p_top [{a:.1f},{min(b,1):.1f}): n {m.sum():3d}, mean conf {rec['p_top'][m].mean():.3f}, exact rate {rec['exact'][m].mean():.3f}")
print(f"  ECE {ece:.3f}")

print("\n3) Selective prediction: F1 / exact on the most confident fraction (by p_top)")
order = np.argsort(-rec["p_top"])
for cov in (1.0, 0.75, 0.5, 0.25):
    k = int(round(cov * n))
    sel = order[:k]
    print(f"  coverage {cov:.2f}: F1 {rec['f1'][sel].mean():.3f}, exact {rec['exact'][sel].mean():.3f}, stratum-3 share {np.mean(st[sel]==3):.2f}")

print("\n4) Flagging rows that need an unseen dye (stratum 3)")
y3 = (st == 3).astype(int)
for name, s in (("1 - p_top", 1 - rec["p_top"]), ("entropy", rec["entropy"]), ("de_best", rec["de_best"])):
    auc = roc_auc_score(y3, s)
    lo, hi = cluster_boot_auc(y3, s)
    print(f"  {name:9s}: AUROC {auc:.3f} [{lo:.3f}, {hi:.3f}]")

print("\n5) Cross-conformal dye-set prediction sets (score = 1 - p(true set); calibrate on the other 4 folds)")
score = 1 - rec["p_true"]
for alpha in (0.1, 0.2, 0.3, 0.4):
    cover, size = np.zeros(n), np.zeros(n)
    for f in range(5):
        cal, te = rec["fold"] != f, rec["fold"] == f
        s = np.sort(score[cal])
        m = len(s)
        kq = int(np.ceil((m + 1) * (1 - alpha)))
        q = np.inf if kq > m else s[kq - 1]
        for r in np.flatnonzero(te):
            inset = ((1 - probs[r]) <= q + 1e-12) & (f1s[r] >= 0) & (probs[r] > 0)
            size[r] = inset.sum()
            cover[r] = float(((f1s[r] >= 1 - 1e-9) & inset).any())
    sc = shade_mean(cover)
    print(f"  target {1-alpha:.0%}: coverage {cover.mean():.3f} (shade-level {sc.mean():.3f}), median set size {np.median(size):.0f}, "
          f"mean {size.mean():.1f}; by stratum cov {[round(cover[st==k].mean(),2) for k in (1,2,3)]}")
print(f"  ceiling: true set among candidates for {np.mean(rec['p_true']>0):.3f} of rows "
      f"(by stratum {[round(np.mean(rec['p_true'][st==k]>0),2) for k in (1,2,3)]})")

print("\n6) Amount intervals: cross-conformal on log(pred/true) for correctly named dyes (final method amounts)")
P = np.load(ROOT / "outputs/physics_inverse_experiments/rows/E16__avg_base__strict__pred.npy")
rows, cols = np.nonzero((P > pie.ACTIVE) & A)
lr = np.abs(np.log(P[rows, cols] / D.Y[rows, cols]))
fr = rec["fold"][rows]
for alpha in (0.1, 0.2):
    cov, width = [], []
    for f in range(5):
        cal, te = fr != f, fr == f
        s = np.sort(lr[cal]); m = len(s); kq = int(np.ceil((m + 1) * (1 - alpha)))
        q = s[min(kq, m) - 1]
        cov.append(lr[te] <= q); width.append(np.full(te.sum(), q))
    cov, width = np.concatenate(cov), np.concatenate(width)
    print(f"  target {1-alpha:.0%}: coverage {cov.mean():.3f}, interval pred x/÷ {np.exp(width.mean()):.2f}  (n dyes {len(lr)})")

print("\n7) Amount intervals: robustness (pooled vs per-hue-family Mondrian), by family and stratum, shade-bootstrap CI")
fam = D.dye_hue[cols]
shade_of = cl[rows]


def conformal(alpha, groups=None):
    """Cross-conformal over outer folds; with groups, a separate quantile per group (Mondrian)."""
    cover, q_used = np.zeros(len(lr), bool), np.zeros(len(lr))
    for f in range(5):
        for g in (np.unique(groups) if groups is not None else [None]):
            sel_g = np.ones(len(lr), bool) if g is None else groups == g
            cal, te = (fr != f) & sel_g, (fr == f) & sel_g
            s = np.sort(lr[cal]); m = len(s); kq = int(np.ceil((m + 1) * (1 - alpha)))
            q = np.inf if kq > m else s[kq - 1]
            cover[te], q_used[te] = lr[te] <= q, q
    return cover, q_used


def boot_ci(cover, B=5000):
    sh = np.unique(shade_of); idx = {s: np.flatnonzero(shade_of == s) for s in sh}
    vals = [cover[np.concatenate([idx[s] for s in rng.choice(sh, len(sh))])].mean() for _ in range(B)]
    return np.percentile(vals, [2.5, 97.5])


for alpha in (0.1, 0.2):
    for label, groups in (("pooled", None), ("per family", fam)):
        cov, q = conformal(alpha, groups)
        lo, hi = boot_ci(cov)
        fin = np.isfinite(q)
        by_f = {g: (round(cov[fam == g].mean(), 3), round(float(np.exp(q[fam == g].mean())), 2), int((fam == g).sum())) for g in np.unique(fam)}
        by_s = {k: (round(cov[st[rows] == k].mean(), 3), int((st[rows] == k).sum())) for k in (1, 2, 3)}
        print(f"  {1-alpha:.0%} {label:10s}: coverage {cov.mean():.3f} [{lo:.3f}, {hi:.3f}], factor x/÷ {np.exp(q[fin].mean()):.2f}, "
              f"shades {len(np.unique(shade_of))}\n      by family (cov, factor, n) {by_f}\n      by stratum (cov, n) {by_s}")
