"""Attention + gradient-boosting hybrids for the inverse recipe task (exploratory).

Four combinations, all predicting the 36-dye recipe vector from spectral + substrate inputs and then
applying the published top-3 post-processing (so they compare directly with the ET reference):

  A1  attention -> trees : wavelength-token attention network; its embedding and wavelength attention
                           weights (out-of-fold inside each training fold) are added to LightGBM / XGBoost inputs
  A2  trees -> attention : LightGBM leaf indices as tokens, embedded and attention-pooled, MLP head
  A3  differentiable trees with attention : NODE-style soft oblivious trees whose split features are chosen
                           by softmax attention over the inputs
  A4  attention only     : FT-Transformer (per-feature tokens, CLS token, transformer encoder)

Controls: plain LightGBM and XGBoost on the same inputs. Hyperparameters are fixed in advance (no tuning);
neural models average 3 seeds. Results go to the same tables as scripts/60 (exp ids A00-A04).

Usage:
    python scripts/61_attention_hybrid_experiments.py --exp A01
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location("pie", ROOT / "scripts" / "60_physics_inverse_experiments.py")
pie = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pie)

import lightgbm as lgb  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import xgboost as xgb  # noqa: E402

import warnings  # noqa: E402

warnings.filterwarnings("ignore", message="X does not have valid feature names")
torch.set_num_threads(2)
SEEDS = (0, 1, 2)
EPOCHS = 400
LR = 3e-3
WD = 1e-3
D = 16


# ----------------------------------------------------------------------------- inputs


class Inputs:
    """Median-imputed, standardised spectral + substrate inputs with wavelength-token indices."""

    def __init__(self, data) -> None:
        self.cols = data.features["spectral_substrate"]
        self.X = data.X("spectral_substrate")
        self.refl = [i for i, c in enumerate(self.cols) if c.startswith("target_refl__")]
        self.ks = [i for i, c in enumerate(self.cols) if c.startswith("target_ks__")]
        self.rest = [i for i in range(len(self.cols)) if i not in set(self.refl) | set(self.ks)]

    def fit_transform(self, tr: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
        Xtr = self.X[tr]
        med = np.nanmedian(Xtr, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        fill = lambda A: np.where(np.isfinite(A), A, med)  # noqa: E731
        mu, sd = fill(Xtr).mean(0), fill(Xtr).std(0)
        sd = np.where(sd > 1e-9, sd, 1.0)
        return [((fill(self.X[idx]) - mu) / sd).astype(np.float32) for idx in (tr, *others)]


def train_net(make, Xtr: np.ndarray, Ytr: np.ndarray, seed: int, epochs: int = EPOCHS) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    X, Y = torch.tensor(Xtr), torch.tensor(Ytr, dtype=torch.float32)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = ((model(X) - Y) ** 2).mean()
        loss.backward()
        opt.step()
    model.eval()
    return model


def predict_seeds(make, Xtr, Ytr, Xva, epochs: int = EPOCHS) -> np.ndarray:
    preds = []
    for s in SEEDS:
        m = train_net(make, Xtr, Ytr, s, epochs)
        with torch.no_grad():
            preds.append(m(torch.tensor(Xva)).numpy())
    return np.mean(preds, axis=0)


# ------------------------------------------------------------------------ tree models


def lgbm_multi(Xtr, Ytr, Xva, seed: int = 0) -> np.ndarray:
    out = np.zeros((len(Xva), Ytr.shape[1]))
    for j in range(Ytr.shape[1]):
        if (Ytr[:, j] > 0).sum() < 2:
            out[:, j] = Ytr[:, j].mean()
            continue
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=7, min_child_samples=5,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                              random_state=seed, verbose=-1)
        m.fit(Xtr, Ytr[:, j])
        out[:, j] = m.predict(Xva)
    return out


def xgb_multi(Xtr, Ytr, Xva, seed: int = 0) -> np.ndarray:
    m = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8,
                         reg_lambda=1.0, random_state=seed, n_jobs=2, tree_method="hist")
    m.fit(Xtr, Ytr)
    return m.predict(Xva)


# ------------------------------------------------------------------ A1: attention -> trees


class WavelengthAttention(nn.Module):
    """Tokens = 36 wavelengths (reflectance, K/S) + one substrate token + CLS; one attention layer."""

    def __init__(self, refl, ks, rest, n_out: int, d: int = D, heads: int = 2) -> None:
        super().__init__()
        self.refl, self.ks, self.rest = refl, ks, rest
        self.tok = nn.Linear(2, d)
        self.pos = nn.Parameter(torch.randn(len(refl), d) * 0.02)
        self.sub = nn.Linear(len(rest), d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_out)

    def encode(self, X: torch.Tensor):
        wl = self.tok(torch.stack([X[:, self.refl], X[:, self.ks]], -1)) + self.pos
        seq = torch.cat([self.cls.expand(len(X), -1, -1), self.sub(X[:, self.rest]).unsqueeze(1), wl], 1)
        out, w = self.attn(seq, seq, seq, need_weights=True, average_attn_weights=True)
        h = self.norm(seq + out)
        return h[:, 0], w[:, 0, 2:]  # CLS embedding, CLS attention over the wavelengths

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(X)[0])


def attention_features(inp: Inputs, Xtr, Ytr, Xva, seed: int = 0) -> np.ndarray:
    make = lambda: WavelengthAttention(inp.refl, inp.ks, inp.rest, Ytr.shape[1])  # noqa: E731
    m = train_net(make, Xtr, Ytr, seed)
    with torch.no_grad():
        z, w = m.encode(torch.tensor(Xva))
    return np.hstack([z.numpy(), w.numpy()])


def exp_a01(data) -> list[dict]:
    """A1: attention network embeddings + wavelength attention as extra inputs to LightGBM / XGBoost (stacked OOF)."""
    inp = Inputs(data)
    results = []
    for proto in pie.PROTOCOLS:
        rf_idx = pie.row_fold_index(data, proto)
        km = {f: pie.make_km(data, tr, pie.KM_DEFAULT) for f, (tr, _) in enumerate(data.folds[proto])}
        preds = {k: np.zeros_like(data.Y) for k in ("lgbm_plain", "xgb_plain", "attn_to_lgbm", "attn_to_xgb", "attn_net_alone")}
        for f, (tr, va) in enumerate(data.folds[proto]):
            t0 = time.time()
            Xtr, Xva = inp.fit_transform(tr, va)
            Ytr = data.Y[tr]
            # Out-of-fold attention features for the training rows (inner CV), full-fold net for validation rows.
            pos = {r: k for k, r in enumerate(tr)}
            Ztr = np.zeros((len(tr), D + len(inp.refl)), dtype=np.float32)
            for itr, iva in pie.inner_splits(data, proto, tr):
                a, b = [pos[r] for r in itr], [pos[r] for r in iva]
                Ztr[b] = attention_features(inp, Xtr[a], Ytr[a], Xtr[b])
            Zva = attention_features(inp, Xtr, Ytr, Xva)
            preds["lgbm_plain"][va] = lgbm_multi(Xtr, Ytr, Xva)
            preds["xgb_plain"][va] = xgb_multi(Xtr, Ytr, Xva)
            preds["attn_to_lgbm"][va] = lgbm_multi(np.hstack([Xtr, Ztr]), Ytr, np.hstack([Xva, Zva]))
            preds["attn_to_xgb"][va] = xgb_multi(np.hstack([Xtr, Ztr]), Ytr, np.hstack([Xva, Zva]))
            make = lambda: WavelengthAttention(inp.refl, inp.ks, inp.rest, Ytr.shape[1])  # noqa: E731
            preds["attn_net_alone"][va] = predict_seeds(make, Xtr, Ytr, Xva)
            print(f"  [A01/{proto}] fold {f}: {time.time() - t0:.0f}s", flush=True)
        for name, P in preds.items():
            results.append(pie.record("A01", name, proto, pie.row_metrics(data.Y, pie.feasible(P), data, km, rf_idx),
                                      pie.feasible(P), data))
    return results


# ------------------------------------------------------------------ A2: trees -> attention


class LeafAttention(nn.Module):
    """Each tree's leaf is a token; a learned query attention-pools them; MLP head."""

    def __init__(self, n_tokens: int, n_out: int, d: int = D) -> None:
        super().__init__()
        self.emb = nn.Embedding(n_tokens, d)
        self.query = nn.Parameter(torch.randn(d) * 0.1)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, n_out))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        e = self.emb(ids)  # (n, T, d)
        a = torch.softmax(e @ self.query / np.sqrt(e.shape[-1]), dim=1)
        return self.head((a.unsqueeze(-1) * e).sum(1))


def leaf_tokens(data, Xtr, tr, Xva, n_trees: int = 50, leaves: int = 8) -> tuple[np.ndarray, np.ndarray, int]:
    """LightGBM on dense targets (4 hue-family totals + total dye); leaf index per tree -> global token id."""
    targets = np.column_stack([data.F[tr], data.Y[tr].sum(1)])
    tok_tr, tok_va, offset = [], [], 0
    for t in range(targets.shape[1]):
        m = lgb.LGBMRegressor(n_estimators=n_trees, learning_rate=0.1, num_leaves=leaves, min_child_samples=5,
                              random_state=0, verbose=-1)
        m.fit(Xtr, targets[:, t])
        a, b = m.predict(Xtr, pred_leaf=True), m.predict(Xva, pred_leaf=True)
        n_t = a.shape[1]
        base = offset + np.arange(n_t) * leaves
        tok_tr.append(a + base)
        tok_va.append(b + base)
        offset += n_t * leaves
    return np.hstack(tok_tr).astype(np.int64), np.hstack(tok_va).astype(np.int64), offset


def exp_a02(data) -> list[dict]:
    """A2: GBDT leaf-index tokens -> attention pooling -> MLP (He et al. 2014-style GBDT features)."""
    inp = Inputs(data)
    results = []
    for proto in pie.PROTOCOLS:
        rf_idx = pie.row_fold_index(data, proto)
        km = {f: pie.make_km(data, tr, pie.KM_DEFAULT) for f, (tr, _) in enumerate(data.folds[proto])}
        P = np.zeros_like(data.Y)
        for f, (tr, va) in enumerate(data.folds[proto]):
            t0 = time.time()
            Xtr, Xva = inp.fit_transform(tr, va)
            ttr, tva, n_tok = leaf_tokens(data, Xtr, tr, Xva)
            preds = []
            for s in SEEDS:
                torch.manual_seed(s)
                m = LeafAttention(n_tok, data.Y.shape[1])
                opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
                ids, Y = torch.tensor(ttr), torch.tensor(data.Y[tr], dtype=torch.float32)
                for _ in range(EPOCHS):
                    opt.zero_grad()
                    loss = ((m(ids) - Y) ** 2).mean()
                    loss.backward()
                    opt.step()
                m.eval()
                with torch.no_grad():
                    preds.append(m(torch.tensor(tva)).numpy())
            P[va] = np.mean(preds, 0)
            print(f"  [A02/{proto}] fold {f}: {n_tok} leaf tokens, {time.time() - t0:.0f}s", flush=True)
        results.append(pie.record("A02", "gbdt_leaves_to_attention", proto,
                                  pie.row_metrics(data.Y, pie.feasible(P), data, km, rf_idx), pie.feasible(P), data))
    return results


# ------------------------------------------- A3: differentiable trees with attention


class SoftAttentionTrees(nn.Module):
    """NODE-style ensemble of soft oblivious trees: each split's feature is a softmax attention over inputs."""

    def __init__(self, n_in: int, n_out: int, n_trees: int = 16, depth: int = 3) -> None:
        super().__init__()
        self.sel = nn.Parameter(torch.randn(n_trees, depth, n_in) * 0.1)
        self.thr = nn.Parameter(torch.randn(n_trees, depth) * 0.5)
        self.log_temp = nn.Parameter(torch.zeros(n_trees, depth))
        self.leaf = nn.Parameter(torch.randn(n_trees, 2 ** depth, n_out) * 0.01)
        bits = torch.tensor([[(code >> d) & 1 for d in range(depth)] for code in range(2 ** depth)], dtype=torch.float32)
        self.register_buffer("bits", bits)  # (L, depth)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        att = torch.softmax(self.sel, -1)
        f = torch.einsum("nf,tdf->ntd", X, att)
        right = torch.sigmoid((f - self.thr) / torch.exp(self.log_temp))  # (n, T, depth)
        p = right.unsqueeze(2) * self.bits + (1 - right.unsqueeze(2)) * (1 - self.bits)  # (n, T, L, depth)
        return torch.einsum("ntl,tlo->no", p.prod(-1), self.leaf) / self.sel.shape[0]


def exp_a03(data) -> list[dict]:
    """A3: soft oblivious trees with attention-based feature selection (NODE-like), 3 seeds."""
    inp = Inputs(data)
    results = []
    for proto in pie.PROTOCOLS:
        rf_idx = pie.row_fold_index(data, proto)
        km = {f: pie.make_km(data, tr, pie.KM_DEFAULT) for f, (tr, _) in enumerate(data.folds[proto])}
        P = np.zeros_like(data.Y)
        for f, (tr, va) in enumerate(data.folds[proto]):
            t0 = time.time()
            Xtr, Xva = inp.fit_transform(tr, va)
            make = lambda: SoftAttentionTrees(Xtr.shape[1], data.Y.shape[1])  # noqa: E731
            P[va] = predict_seeds(make, Xtr, data.Y[tr], Xva)
            print(f"  [A03/{proto}] fold {f}: {time.time() - t0:.0f}s", flush=True)
        results.append(pie.record("A03", "soft_attention_trees", proto,
                                  pie.row_metrics(data.Y, pie.feasible(P), data, km, rf_idx), pie.feasible(P), data))
    return results


# ------------------------------------------------------------------- A4: FT-Transformer


class FTTransformer(nn.Module):
    def __init__(self, n_in: int, n_out: int, d: int = D, layers: int = 2, heads: int = 2) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_in, d) * 0.1)
        self.b = nn.Parameter(torch.zeros(n_in, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, n_out))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        tok = X.unsqueeze(-1) * self.W + self.b
        h = self.enc(torch.cat([self.cls.expand(len(X), -1, -1), tok], 1))
        return self.head(h[:, 0])


def exp_a04(data) -> list[dict]:
    """A4: FT-Transformer on all inputs (one token per feature), 3 seeds."""
    inp = Inputs(data)
    results = []
    for proto in pie.PROTOCOLS:
        rf_idx = pie.row_fold_index(data, proto)
        km = {f: pie.make_km(data, tr, pie.KM_DEFAULT) for f, (tr, _) in enumerate(data.folds[proto])}
        P = np.zeros_like(data.Y)
        for f, (tr, va) in enumerate(data.folds[proto]):
            t0 = time.time()
            Xtr, Xva = inp.fit_transform(tr, va)
            make = lambda: FTTransformer(Xtr.shape[1], data.Y.shape[1])  # noqa: E731
            P[va] = predict_seeds(make, Xtr, data.Y[tr], Xva, epochs=200)
            print(f"  [A04/{proto}] fold {f}: {time.time() - t0:.0f}s", flush=True)
        results.append(pie.record("A04", "ft_transformer", proto,
                                  pie.row_metrics(data.Y, pie.feasible(P), data, km, rf_idx), pie.feasible(P), data))
    return results


# ------------------------------------------------- A5: Kolmogorov-Arnold network (1-2 layers)


class KANLayer(nn.Module):
    """KAN layer: every input->output edge carries a learnable univariate function
    phi(x) = w_base * silu(x) + sum_k c_k B_k(x)  (cubic B-splines on a uniform grid over [-3, 3])."""

    def __init__(self, n_in: int, n_out: int, grid: int = 5, order: int = 3, lo: float = -3.0, hi: float = 3.0) -> None:
        super().__init__()
        h = (hi - lo) / grid
        self.register_buffer("knots", torch.arange(-order, grid + order + 1, dtype=torch.float32) * h + lo)
        self.order = order
        self.clip = (lo - order * h + 1e-4, hi + order * h - 1e-4)
        self.base = nn.Linear(n_in, n_out)
        self.coef = nn.Parameter(torch.randn(n_out, n_in, grid + order) * (0.1 / np.sqrt(n_in)))

    def basis(self, x: torch.Tensor) -> torch.Tensor:
        g, x = self.knots, x.clamp(*self.clip).unsqueeze(-1)
        B = ((x >= g[:-1]) & (x < g[1:])).float()
        for p in range(1, self.order + 1):  # Cox-de Boor recursion
            B = ((x - g[: -(p + 1)]) / (g[p:-1] - g[: -(p + 1)]) * B[..., :-1]
                 + (g[p + 1:] - x) / (g[p + 1:] - g[1:-p]) * B[..., 1:])
        return B  # (n, n_in, grid + order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(nn.functional.silu(x)) + torch.einsum("nik,oik->no", self.basis(x), self.coef)


class KAN(nn.Module):
    def __init__(self, n_in: int, n_out: int, hidden: int | None = None) -> None:
        super().__init__()
        if hidden is None:
            self.net = KANLayer(n_in, n_out)
        else:
            self.net = nn.Sequential(KANLayer(n_in, hidden), nn.LayerNorm(hidden), KANLayer(hidden, n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def exp_a05(data) -> list[dict]:
    """A5: KAN with one layer (additive spline model) and two layers (hidden width 8), 3 seeds."""
    inp = Inputs(data)
    results = []
    for proto in pie.PROTOCOLS:
        rf_idx = pie.row_fold_index(data, proto)
        km = {f: pie.make_km(data, tr, pie.KM_DEFAULT) for f, (tr, _) in enumerate(data.folds[proto])}
        preds = {"kan_1_layer": np.zeros_like(data.Y), "kan_2_layer_h8": np.zeros_like(data.Y)}
        for f, (tr, va) in enumerate(data.folds[proto]):
            t0 = time.time()
            Xtr, Xva = inp.fit_transform(tr, va)
            n_in, n_out = Xtr.shape[1], data.Y.shape[1]
            preds["kan_1_layer"][va] = predict_seeds(lambda: KAN(n_in, n_out), Xtr, data.Y[tr], Xva)
            preds["kan_2_layer_h8"][va] = predict_seeds(lambda: KAN(n_in, n_out, hidden=8), Xtr, data.Y[tr], Xva)
            print(f"  [A05/{proto}] fold {f}: {time.time() - t0:.0f}s", flush=True)
        for name, P in preds.items():
            results.append(pie.record("A05", name, proto, pie.row_metrics(data.Y, pie.feasible(P), data, km, rf_idx),
                                      pie.feasible(P), data))
    return results


# ------------------------------------------- A06: FT-Transformer with alternative cost functions

KM_LAMBDA = 1.0  # weight of the Kubelka-Munk consistency term, fixed before running


def _outputs(loss: str, n_d: int) -> int:
    return {"hurdle": 2 * n_d, "soft_f1": 2 * n_d, "composition": n_d + 1}.get(loss, n_d)


def decode(loss: str, out: torch.Tensor) -> torch.Tensor:
    """Map raw network outputs to a non-negative recipe vector."""
    n = out.shape[1]
    if loss == "log_mse":
        return torch.expm1(out).clamp(min=0)
    if loss in ("hurdle", "soft_f1"):
        h = n // 2
        return torch.sigmoid(out[:, :h]) * nn.functional.softplus(out[:, h:])
    if loss == "composition":
        return nn.functional.softplus(out[:, -1:]) * torch.softmax(out[:, :-1], 1)
    return out


def loss_value(loss: str, out: torch.Tensor, b: dict) -> torch.Tensor:
    Y, A = b["Y"], b["A"]
    if loss == "mse":
        return ((out - Y) ** 2).mean()
    if loss == "mae":
        return (out - Y).abs().mean()
    if loss == "huber":
        return nn.functional.smooth_l1_loss(out, Y, beta=0.1)
    if loss == "log_mse":
        return ((out - torch.log1p(Y)) ** 2).mean()
    if loss == "active_weighted_mse":
        w = 1.0 + 10.0 * A
        return (w * (out - Y) ** 2).sum() / w.sum()
    if loss in ("hurdle", "soft_f1"):
        h = out.shape[1] // 2
        logits, amt = out[:, :h], nn.functional.softplus(out[:, h:])
        amount = ((amt - Y) ** 2 * A).sum() / A.sum().clamp(min=1)
        if loss == "hurdle":
            return nn.functional.binary_cross_entropy_with_logits(logits, A) + amount
        p = torch.sigmoid(logits)
        soft_f1 = 2 * (p * A).sum(1) / (p.sum(1) + A.sum(1) + 1e-6)
        return (1 - soft_f1).mean() + amount
    if loss == "composition":
        T = Y.sum(1)
        T_hat = nn.functional.softplus(out[:, -1])
        ok = T > 0
        logpi = torch.log_softmax(out[:, :-1], 1)
        ce = -(Y[ok] / T[ok, None] * logpi[ok]).sum(1).mean()
        return ((torch.log1p(T_hat) - torch.log1p(T)) ** 2).mean() + ce
    if loss == "family_mse":
        H = b["H"]
        return ((out - Y) ** 2).mean() + ((out @ H - Y @ H) ** 2).mean()
    if loss == "km_physics":
        c = out.clamp(min=0)
        u = c / (1 + b["beta"] * c)
        ks = b["s"] + u @ b["K"]
        phys = (((ks - b["KS"]) / b["KS"].clamp(min=1e-3)) ** 2).mean()
        return ((out - Y) ** 2).mean() + KM_LAMBDA * phys
    raise ValueError(loss)


A06_LOSSES = ["mse", "mae", "huber", "log_mse", "active_weighted_mse", "hurdle", "soft_f1", "composition",
              "family_mse", "km_physics"]


def ft_with_loss(data, inp, tr, va, loss: str, seeds=(0,), epochs: int = 200) -> np.ndarray:
    Xtr, Xva = inp.fit_transform(tr, va)
    n_d = data.Y.shape[1]
    b = {"Y": torch.tensor(data.Y[tr], dtype=torch.float32)}
    b["A"] = (b["Y"] > pie.ACTIVE).float()
    b["H"] = torch.tensor(np.column_stack([(data.dye_hue == g).astype(np.float32) for g in data.families]))
    if loss == "km_physics":
        km = pie.make_km(data, tr, pie.KM_DEFAULT)
        b.update(K=torch.tensor(km.k, dtype=torch.float32), s=torch.tensor(km.substrate, dtype=torch.float32),
                 KS=torch.tensor(data.KS[tr], dtype=torch.float32), beta=float(km.beta))
    preds = []
    X, Xv = torch.tensor(Xtr), torch.tensor(Xva)
    for seed in seeds:
        torch.manual_seed(seed)
        m = FTTransformer(Xtr.shape[1], _outputs(loss, n_d))
        opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
        for _ in range(epochs):
            m.train()
            opt.zero_grad()
            loss_value(loss, m(X), b).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            preds.append(decode(loss, m(Xv)).numpy())
    return np.mean(preds, 0)


def exp_a06(data, losses: list[str] | None = None, protos=("strict",), seeds=(0,), tag: str = "A06") -> list[dict]:
    """FT-Transformer trained with each cost function (screening: 1 seed; same folds and fixed settings)."""
    inp = Inputs(data)
    results = []
    for proto in protos:
        rf_idx = pie.row_fold_index(data, proto)
        km = {f: pie.make_km(data, tr, pie.KM_DEFAULT) for f, (tr, _) in enumerate(data.folds[proto])}
        for loss in losses or A06_LOSSES:
            t0 = time.time()
            P = np.zeros_like(data.Y)
            for tr, va in data.folds[proto]:
                P[va] = ft_with_loss(data, inp, tr, va, loss, seeds)
            P = pie.feasible(P)
            results.append(pie.record(tag, f"ft_{loss}", proto, pie.row_metrics(data.Y, P, data, km, rf_idx), P, data))
            print(f"  [{tag}/{proto}] {loss}: F1 {results[-1]['f1']:.3f}  exact {results[-1]['exact_set']:.3f}  "
                  f"union_mae {results[-1]['union_mae']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    return results


# ----------------------------- A07: logic-structured cost function (designed before A06 results)

COLOUR_WEIGHT = 0.05  # on mean CIELAB dE76 (typically 2-10), fixed before running


def _lab_torch(R: torch.Tensor, illuminant: str) -> torch.Tensor:
    """Differentiable CIELAB (10 deg observer) from reflectance factors, same weights as km_physics."""
    from piml_color_recipe.inverse.km_physics import _WEIGHTS

    W = torch.tensor(_WEIGHTS[illuminant][0], dtype=torch.float32)
    xyz = R @ W
    white = W.sum(0)
    t = xyz / white
    d = 6.0 / 29.0
    f = torch.where(t > d ** 3, t.clamp(min=1e-6) ** (1.0 / 3.0), t / (3 * d * d) + 4.0 / 29.0)
    return torch.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


def km_colour_error(c: torch.Tensor, b: dict) -> torch.Tensor:
    """Mean dE76 between the target and the KM-predicted colour of recipe c, averaged over D65 and F11."""
    u = c / (1 + b["beta"] * c)
    ks = (b["s"] + u @ b["K"]).clamp(min=1e-6)
    R = 1 + ks - torch.sqrt(ks * ks + 2 * ks)
    err = 0
    for ill in ("D65", "FL11"):
        err = err + torch.sqrt(((_lab_torch(R, ill) - b[f"lab_{ill}"]) ** 2).sum(1) + 1e-8).mean()
    return err / 2


class SlotHead(nn.Module):
    """FT-Transformer body with a slot-structured head: per hue slot, a masked softmax over the dyes
    allowed in that slot (+ 'none') and a slot amount."""

    def __init__(self, n_in: int, n_d: int, allowed: np.ndarray) -> None:
        super().__init__()
        self.body = FTTransformer(n_in, 3 * (n_d + 1) + 3)
        self.n_d = n_d
        mask = np.full((3, n_d + 1), -1e9, dtype=np.float32)
        mask[allowed] = 0.0
        self.register_buffer("mask", torch.tensor(mask))

    def forward(self, X: torch.Tensor):
        out = self.body(X)
        logits = out[:, : 3 * (self.n_d + 1)].view(-1, 3, self.n_d + 1) + self.mask
        amount = nn.functional.softplus(out[:, 3 * (self.n_d + 1):])
        return logits, amount

    def expected_recipe(self, logits, amount) -> torch.Tensor:
        p = torch.softmax(logits, -1)[..., : self.n_d]  # drop 'none'
        return (p * amount.unsqueeze(-1)).sum(1)

    def hard_recipe(self, logits, amount) -> np.ndarray:
        cls = logits.argmax(-1).numpy()
        amt = amount.numpy()
        P = np.zeros((len(cls), self.n_d))
        for i in range(len(cls)):
            for slot in range(3):
                j = cls[i, slot]
                if j < self.n_d:
                    P[i, j] += amt[i, slot]
        return P


def ft_logic(data, inp, tr, va, variant: str, lab_pos: np.ndarray, seeds=(0,), epochs: int = 200, project: bool = False):
    Xtr, Xva = inp.fit_transform(tr, va)
    n_d = data.Y.shape[1]
    Y = torch.tensor(data.Y[tr], dtype=torch.float32)
    km = pie.make_km(data, tr, pie.KM_DEFAULT)
    b = {"K": torch.tensor(km.k, dtype=torch.float32), "s": torch.tensor(km.substrate, dtype=torch.float32),
         "beta": float(km.beta),
         "lab_D65": torch.tensor(data.lab_d65[tr], dtype=torch.float32),
         "lab_FL11": torch.tensor(data.lab_f11[tr], dtype=torch.float32)}
    slot_cls = np.where(lab_pos[tr] >= 0, lab_pos[tr], n_d)  # n_d = 'none'
    slot_amt = np.where(lab_pos[tr] >= 0, data.Y[tr][np.arange(len(tr))[:, None], np.maximum(lab_pos[tr], 0)], 0.0)
    allowed = np.zeros((3, n_d + 1), bool)
    for slot in range(3):
        allowed[slot, np.unique(slot_cls[:, slot])] = True
    cls_t, amt_t = torch.tensor(slot_cls), torch.tensor(slot_amt, dtype=torch.float32)
    filled = (cls_t < n_d).float()
    T = Y.sum(1)
    X, Xv = torch.tensor(Xtr), torch.tensor(Xva)
    preds = []
    for seed in seeds:
        torch.manual_seed(seed)
        if variant == "mse_colour":
            m = FTTransformer(Xtr.shape[1], n_d)
        else:
            m = SlotHead(Xtr.shape[1], n_d, allowed)
        opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
        for _ in range(epochs):
            m.train()
            opt.zero_grad()
            if variant == "mse_colour":
                out = m(X)
                loss = ((out - Y) ** 2).mean() + COLOUR_WEIGHT * km_colour_error(out.clamp(min=0), b)
            else:
                logits, amount = m(X)
                ce = nn.functional.cross_entropy(logits.reshape(-1, n_d + 1), cls_t.reshape(-1))
                amt_loss = (((amount - amt_t) ** 2) * filled).sum() / filled.sum()
                loss = ce + amt_loss
                if variant == "logic_full":
                    c = m.expected_recipe(logits, amount)
                    loss = loss + COLOUR_WEIGHT * km_colour_error(c, b) + ((torch.log1p(c.sum(1)) - torch.log1p(T)) ** 2).mean()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            preds.append(m(Xv).clamp(min=0).numpy() if variant == "mse_colour" else m.hard_recipe(*m(Xv)))
    P = np.mean(preds, 0)
    if project:
        # Hard colour consistency: keep the network's dye choice, re-solve the amounts with bounded KM.
        from piml_color_recipe.inverse.km_physics import solve_subsets

        c_max = float(data.Y[tr].max())
        Q = np.zeros_like(P)
        for i, r in enumerate(va):
            dyes = np.flatnonzero(P[i] > pie.ACTIVE)
            dyes = dyes[km.usable[dyes]]
            if len(dyes) == 0:
                continue
            sub = -np.ones((1, 3), dtype=int)
            sub[0, : len(dyes[:3])] = dyes[:3]
            Q[i, dyes[:3]] = solve_subsets(km, data.KS[r], sub, c_max=c_max).conc[0, : len(dyes[:3])]
        return P, Q
    return P


A07_VARIANTS = ["logic_slot", "logic_full", "mse_colour"]


def ft_logic_slot_seeds(data, inp, tr, va, lab_pos: np.ndarray, seeds=(0, 1, 2), epochs: int = 200) -> dict[str, np.ndarray]:
    """logic_slot per seed, plus a proper ensemble: average slot probabilities and amounts across seeds,
    then decode one dye per slot (averaging hard recipes would mix dyes from different seeds)."""
    Xtr, Xva = inp.fit_transform(tr, va)
    n_d = data.Y.shape[1]
    slot_cls = np.where(lab_pos[tr] >= 0, lab_pos[tr], n_d)
    slot_amt = np.where(lab_pos[tr] >= 0, data.Y[tr][np.arange(len(tr))[:, None], np.maximum(lab_pos[tr], 0)], 0.0)
    allowed = np.zeros((3, n_d + 1), bool)
    for slot in range(3):
        allowed[slot, np.unique(slot_cls[:, slot])] = True
    cls_t, amt_t = torch.tensor(slot_cls), torch.tensor(slot_amt, dtype=torch.float32)
    filled = (cls_t < n_d).float()
    X, Xv = torch.tensor(Xtr), torch.tensor(Xva)
    out, probs, amounts = {}, [], []
    for seed in seeds:
        torch.manual_seed(seed)
        m = SlotHead(Xtr.shape[1], n_d, allowed)
        opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
        for _ in range(epochs):
            m.train()
            opt.zero_grad()
            logits, amount = m(X)
            loss = (nn.functional.cross_entropy(logits.reshape(-1, n_d + 1), cls_t.reshape(-1))
                    + (((amount - amt_t) ** 2) * filled).sum() / filled.sum())
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            logits, amount = m(Xv)
            out[f"seed{seed}"] = m.hard_recipe(logits, amount)
            probs.append(torch.softmax(logits, -1))
            amounts.append(amount)
    p_mean, a_mean = torch.stack(probs).mean(0), torch.stack(amounts).mean(0)
    out["ensemble"] = m.hard_recipe(torch.log(p_mean + 1e-12), a_mean)
    return out


def _exp_a07d(data, proto: str) -> list[dict]:
    inp = Inputs(data)
    lab_pos = pie.position_labels(data)
    rf_idx = pie.row_fold_index(data, proto)
    km = {f: pie.make_km(data, tr, pie.KM_DEFAULT) for f, (tr, _) in enumerate(data.folds[proto])}
    preds: dict[str, np.ndarray] = {}
    t0 = time.time()
    for tr, va in data.folds[proto]:
        for k, P in ft_logic_slot_seeds(data, inp, tr, va, lab_pos).items():
            preds.setdefault(k, np.zeros_like(data.Y))[va] = P
    results = []
    for k, P in preds.items():
        P = pie.feasible(P)
        results.append(pie.record("A07d", f"ft_logic_slot_{k}", proto, pie.row_metrics(data.Y, P, data, km, rf_idx), P, data))
        print(f"  [A07d/{proto}] {k}: F1 {results[-1]['f1']:.3f}  exact {results[-1]['exact_set']:.3f}  "
              f"union_mae {results[-1]['union_mae']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    return results


def exp_a07d_strict(data) -> list[dict]:
    """logic_slot: per-seed results and a probability-averaged 3-seed ensemble (strict CV)."""
    return _exp_a07d(data, "strict")


def exp_a07d_grouped(data) -> list[dict]:
    """logic_slot: per-seed results and a probability-averaged 3-seed ensemble (grouped CV)."""
    return _exp_a07d(data, "grouped")


def exp_a07c_strict(data) -> list[dict]:
    """3-seed confirmation of the slot-structured loss (strict CV)."""
    return exp_a07(data, ["logic_slot"], ("strict",), seeds=(0, 1, 2), exp_id="A07c")


def exp_a07c_grouped(data) -> list[dict]:
    """3-seed confirmation of the slot-structured loss (grouped CV)."""
    return exp_a07(data, ["logic_slot"], ("grouped",), seeds=(0, 1, 2), exp_id="A07c")


def exp_a07(data, variants: list[str] | None = None, protos=("strict",), seeds=(0,), exp_id: str = "A07") -> list[dict]:
    """Logic-structured cost functions on the FT-Transformer (1 seed screen, same folds as A06)."""
    inp = Inputs(data)
    lab_pos = pie.position_labels(data)
    results = []
    for proto in protos:
        rf_idx = pie.row_fold_index(data, proto)
        km = {f: pie.make_km(data, tr, pie.KM_DEFAULT) for f, (tr, _) in enumerate(data.folds[proto])}
        for variant in variants or A07_VARIANTS:
            t0 = time.time()
            project = variant == "logic_full"
            out = {f"ft_{variant}": np.zeros_like(data.Y)}
            if project:
                out["ft_logic_full_projected"] = np.zeros_like(data.Y)
            for tr, va in data.folds[proto]:
                r = ft_logic(data, inp, tr, va, variant, lab_pos, seeds, project=project)
                if project:
                    out[f"ft_{variant}"][va], out["ft_logic_full_projected"][va] = r
                else:
                    out[f"ft_{variant}"][va] = r
            for name, P in out.items():
                P = pie.feasible(P)
                results.append(pie.record(exp_id, name, proto, pie.row_metrics(data.Y, P, data, km, rf_idx), P, data))
                print(f"  [{exp_id}/{proto}] {name}: F1 {results[-1]['f1']:.3f}  exact {results[-1]['exact_set']:.3f}  "
                      f"union_mae {results[-1]['union_mae']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    return results


# ------------------- A08: brute-force composite cost  L = sum_k w_k * g(L_k),  nested selection

TERMS = ["mse", "mae", "wmse", "bce", "soft_f1", "colour", "total", "family"]
FORMS = ["linear", "exp", "log"]


class ProxyMLP(nn.Module):
    """Fast proxy body for the cost-function search: presence logits + amounts (36 + 36)."""

    def __init__(self, n_in: int, n_d: int) -> None:
        super().__init__()
        self.n_d = n_d
        self.net = nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 2 * n_d))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)


def composite_terms(out: torch.Tensor, b: dict) -> list[torch.Tensor]:
    n = out.shape[1] // 2
    logits, amt = out[:, :n], nn.functional.softplus(out[:, n:])
    p = torch.sigmoid(logits)
    c = p * amt
    Y, A = b["Y"], b["A"]
    w = 1.0 + 10.0 * A
    soft_f1 = 2 * (p * A).sum(1) / (p.sum(1) + A.sum(1) + 1e-6)
    return [
        ((c - Y) ** 2).mean(),
        (c - Y).abs().mean(),
        (w * (c - Y) ** 2).sum() / w.sum(),
        nn.functional.binary_cross_entropy_with_logits(logits, A),
        (1 - soft_f1).mean(),
        km_colour_error(c, b) / 10.0,  # dE76 / 10 so that the term is O(1)
        ((torch.log1p(c.sum(1)) - torch.log1p(Y.sum(1))) ** 2).mean(),
        ((c @ b["H"] - Y @ b["H"]) ** 2).mean(),
    ]


def combine(terms: list[torch.Tensor], w: np.ndarray, form: str) -> torch.Tensor:
    total = 0.0
    for t, wk in zip(terms, w):
        if wk == 0:
            continue
        if form == "linear":
            total = total + wk * t
        elif form == "exp":
            total = total + wk * (torch.exp(t.clamp(max=5.0)) - 1.0)
        else:
            total = total + wk * torch.log1p(t)
    return total


def sample_costs(n_per_form: int = 40, seed: int = 7) -> list[tuple[str, np.ndarray]]:
    """Random sparse weight vectors per form (log-uniform 0.01-10), always with an amount or presence anchor."""
    rng = np.random.default_rng(seed)
    costs = []
    for form in FORMS:
        costs.append((form, np.array([1.0] + [0.0] * (len(TERMS) - 1))))  # MSE-only reference
        while sum(1 for f, _ in costs if f == form) < n_per_form + 1:
            on = rng.random(len(TERMS)) < 0.5
            if not on[:4].any():
                continue
            costs.append((form, np.where(on, 10 ** rng.uniform(-2, 1, len(TERMS)), 0.0)))
    return costs


def _batch(data, tr, km) -> dict:
    Y = torch.tensor(data.Y[tr], dtype=torch.float32)
    return {"Y": Y, "A": (Y > pie.ACTIVE).float(),
            "H": torch.tensor(np.column_stack([(data.dye_hue == g).astype(np.float32) for g in data.families])),
            "K": torch.tensor(km.k, dtype=torch.float32), "s": torch.tensor(km.substrate, dtype=torch.float32),
            "beta": float(km.beta), "lab_D65": torch.tensor(data.lab_d65[tr], dtype=torch.float32),
            "lab_FL11": torch.tensor(data.lab_f11[tr], dtype=torch.float32)}


def train_cost(make, X: np.ndarray, b: dict, form: str, w: np.ndarray, Xv: np.ndarray, seed: int = 0, epochs: int = 300) -> np.ndarray:
    torch.manual_seed(seed)
    m = make()
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    Xt, Xvt = torch.tensor(X), torch.tensor(Xv)
    for _ in range(epochs):
        m.train()
        opt.zero_grad()
        combine(composite_terms(m(Xt), b), w, form).backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        out = m(Xvt)
        n = out.shape[1] // 2
        return (torch.sigmoid(out[:, :n]) * nn.functional.softplus(out[:, n:])).numpy()


def exp_a08(data, protos=("strict",), top_k: int = 5) -> list[dict]:
    """Brute-force composite cost search on a fast proxy network, selected inside inner folds."""
    import pandas as pd

    inp = Inputs(data)
    costs = sample_costs()
    results, log = [], []
    for proto in protos:
        rf_idx = pie.row_fold_index(data, proto)
        km_eval = {f: pie.make_km(data, tr, pie.KM_DEFAULT) for f, (tr, _) in enumerate(data.folds[proto])}
        preds = {k: np.zeros_like(data.Y) for k in ("proxy_mse_only", "proxy_nested_top5", "proxy_nested_top1", "proxy_outer_best_OPTIMISTIC")}
        outer_f1 = np.zeros((len(costs), len(data.Y)))
        outer_P = np.zeros((len(costs), *data.Y.shape))
        for f, (tr, va) in enumerate(data.folds[proto]):
            t0 = time.time()
            inner_f1 = np.zeros(len(costs))
            for itr, iva in pie.inner_splits(data, proto, tr):
                Xa, Xb = inp.fit_transform(itr, iva)
                b = _batch(data, itr, pie.make_km(data, itr, pie.KM_DEFAULT))
                make = lambda: ProxyMLP(Xa.shape[1], data.Y.shape[1])  # noqa: E731
                for k, (form, w) in enumerate(costs):
                    P = pie.feasible(train_cost(make, Xa, b, form, w, Xb))
                    inner_f1[k] += pie.row_metrics(data.Y[iva], P)["f1"].sum()
            inner_f1 /= len(tr)
            Xtr, Xva = inp.fit_transform(tr, va)
            b = _batch(data, tr, pie.make_km(data, tr, pie.KM_DEFAULT))
            make = lambda: ProxyMLP(Xtr.shape[1], data.Y.shape[1])  # noqa: E731
            for k, (form, w) in enumerate(costs):
                outer_P[k][va] = train_cost(make, Xtr, b, form, w, Xva)
                outer_f1[k, va] = pie.row_metrics(data.Y[va], pie.feasible(outer_P[k][va]))["f1"].to_numpy()
            order = np.argsort(-inner_f1)
            preds["proxy_mse_only"][va] = outer_P[0][va]
            preds["proxy_nested_top1"][va] = outer_P[order[0]][va]
            preds["proxy_nested_top5"][va] = outer_P[order[:top_k]][:, va].mean(0)
            for rank, k in enumerate(order[:top_k]):
                log.append({"protocol": proto, "fold": f, "rank": rank + 1, "form": costs[k][0], "inner_f1": inner_f1[k],
                            **dict(zip(TERMS, costs[k][1]))})
            print(f"  [A08/{proto}] fold {f}: best inner F1 {inner_f1[order[0]]:.3f} ({costs[order[0]][0]}), "
                  f"MSE-only inner F1 {inner_f1[0]:.3f}  {time.time() - t0:.0f}s", flush=True)
        best = int(np.argmax(outer_f1.mean(1)))  # chosen on the outer folds themselves: optimistic by construction
        preds["proxy_outer_best_OPTIMISTIC"] = outer_P[best]
        dist = outer_f1.mean(1)
        print(f"  [A08/{proto}] outer F1 over {len(costs)} costs: min {dist.min():.3f} median {np.median(dist):.3f} "
              f"max {dist.max():.3f} (MSE-only {dist[0]:.3f}); best = {costs[best][0]} {np.round(costs[best][1], 3)}")
        np.save(pie.OUT_DIR / f"A08_{proto}_outer_f1_by_cost.npy", dist)
        for name, P in preds.items():
            P = pie.feasible(P)
            results.append(pie.record("A08", name, proto, pie.row_metrics(data.Y, P, data, km_eval, rf_idx), P, data))
    pd.DataFrame(log).to_csv(pie.OUT_DIR / "A08_selected_costs.csv", index=False)
    pd.DataFrame([{"form": f, **dict(zip(TERMS, w))} for f, w in costs]).to_csv(pie.OUT_DIR / "A08_cost_grid.csv", index=False)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True, nargs="+")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--losses", nargs="+", default=None)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.losses:
        globals()["A06_LOSSES"] = args.losses
    # Separate summary files per process so parallel runs cannot overwrite each other; merged afterwards.
    tag = args.tag or "_".join(e.lower() for e in args.exp)
    pie.RESULTS_CSV = pie.OUT_DIR / f"results_{tag}.csv"
    pie.STRATA_CSV_NAME = f"results_by_stratum_{tag}.csv"
    data = pie.Data()
    for exp in args.exp:
        t0 = time.time()
        res = globals()[f"exp_{exp.lower()}"](data)
        pie.print_table(res)
        print(f"{exp} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
