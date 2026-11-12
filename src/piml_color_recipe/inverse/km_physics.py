"""Kubelka-Munk colour-mixing model and sparse colour-matching solver.

Single-constant Kubelka-Munk theory treats a dyed textile as additive in K/S:

    K/S_mix(lambda) = K/S_substrate(lambda) + sum_i u(c_i) * k_i(lambda)

where c_i is the concentration (% OWF) of dye i, k_i(lambda) is its unit K/S
spectrum and u(.) is a concentration-uptake transform (identity for ideal
linear build-up). The per-dye spectra and the substrate term are estimated
from historical recipes with non-negative least squares, because no
single-dye calibration dyeings are available.

The inverse solver enumerates every dye subset of size <= ``max_size``, fits
non-negative concentrations for each subset against the target K/S spectrum,
and reports the CMC colour difference of every candidate. Selection rules
(physics-only or combined with a learned dye prior) are applied downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import colour
import numpy as np
from scipy.optimize import nnls

WAVELENGTHS = np.arange(400, 751, 10)
_SHAPE = colour.SpectralShape(400, 750, 10)
_KS_FLOOR = 1e-3


def _tristimulus_weights(illuminant: str) -> tuple[np.ndarray, np.ndarray]:
    cmfs = colour.MSDS_CMFS["CIE 1964 10 Degree Standard Observer"].copy().align(_SHAPE)
    sd = colour.SDS_ILLUMINANTS[illuminant].copy().align(_SHAPE)
    weights = sd.values[:, None] * cmfs.values
    weights = weights * (100.0 / weights[:, 1].sum())
    white_xy = colour.XYZ_to_xy(weights.sum(axis=0) / 100.0)
    return weights, white_xy


_WEIGHTS = {name: _tristimulus_weights(name) for name in ("D65", "FL11")}


def reflectance_to_lab(reflectance: np.ndarray, illuminant: str = "D65") -> np.ndarray:
    """CIELAB (10 degree observer) from reflectance factors in [0, 1] sampled at WAVELENGTHS."""
    weights, white_xy = _WEIGHTS[illuminant]
    xyz = np.asarray(reflectance, dtype=float) @ weights
    return colour.XYZ_to_Lab(xyz / 100.0, white_xy)


def delta_e_cmc(lab_reference: np.ndarray, lab_sample: np.ndarray) -> np.ndarray:
    """CMC(2:1) colour difference with the first argument as the standard."""
    return np.asarray(colour.delta_E(lab_reference, lab_sample, method="CMC"), dtype=float)


def ks_to_reflectance(ks: np.ndarray) -> np.ndarray:
    ks = np.maximum(np.asarray(ks, dtype=float), 0.0)
    return 1.0 + ks - np.sqrt(ks ** 2 + 2.0 * ks)


def reflectance_to_ks(reflectance: np.ndarray) -> np.ndarray:
    r = np.clip(np.asarray(reflectance, dtype=float), 1e-4, 1.0)
    return (1.0 - r) ** 2 / (2.0 * r)


def _row_weights(ks: np.ndarray, weighting: str) -> np.ndarray:
    ks = np.maximum(ks, _KS_FLOOR)
    if weighting == "relative":
        return 1.0 / ks
    if weighting == "sqrt":
        return 1.0 / np.sqrt(ks)
    if weighting == "none":
        return np.ones_like(ks)
    raise ValueError(f"Unknown weighting: {weighting}")


@dataclass
class KMModel:
    k: np.ndarray  # (n_dyes, n_wl) unit K/S per transformed concentration
    substrate: np.ndarray  # (n_wl,)
    usable: np.ndarray  # (n_dyes,) dyes with an estimated spectrum
    uptake: str = "linear"
    beta: float = 0.0
    weighting: str = "relative"

    def transform(self, conc: np.ndarray) -> np.ndarray:
        c = np.maximum(np.asarray(conc, dtype=float), 0.0)
        if self.uptake == "linear":
            return c
        if self.uptake == "power":
            return np.power(c, self.beta)
        if self.uptake == "langmuir":
            return c / (1.0 + self.beta * c)
        raise ValueError(f"Unknown uptake: {self.uptake}")

    def inverse_transform(self, u: np.ndarray) -> np.ndarray:
        u = np.maximum(np.asarray(u, dtype=float), 0.0)
        if self.uptake == "linear":
            return u
        if self.uptake == "power":
            return np.power(u, 1.0 / self.beta)
        if self.uptake == "langmuir":
            # Saturating uptake cannot exceed 1/beta; cap at a large finite concentration.
            denom = np.maximum(1.0 - self.beta * u, 1e-3)
            return u / denom
        raise ValueError(f"Unknown uptake: {self.uptake}")

    def predict_ks(self, conc: np.ndarray) -> np.ndarray:
        return self.substrate + self.transform(conc) @ self.k

    def predict_reflectance(self, conc: np.ndarray) -> np.ndarray:
        return ks_to_reflectance(self.predict_ks(conc))


def fit_km(
    conc: np.ndarray,
    ks: np.ndarray,
    *,
    uptake: str = "linear",
    beta: float = 0.0,
    weighting: str = "relative",
    ridge: float = 0.0,
    fit_substrate: bool = True,
    min_rows: int = 1,
    active_threshold: float = 0.01,
    shrink: float = 0.0,
    groups: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> KMModel:
    """Estimate unit K/S spectra per dye (and a substrate term) wavelength by wavelength.

    ``ridge`` shrinks every spectrum towards zero; ``shrink`` pulls each dye towards
    the mean spectrum of its ``groups`` label. Both act within a wavelength, so the
    fit stays separable. ``upper`` (n_dyes, n_wl) bounds each unit spectrum from above, e.g. to
    impose near-zero absorption outside a dye family's absorption band.
    """
    from scipy.optimize import lsq_linear

    conc = np.maximum(np.asarray(conc, dtype=float), 0.0)
    ks = np.asarray(ks, dtype=float)
    usable = (conc > active_threshold).sum(axis=0) >= min_rows
    model = KMModel(np.zeros((conc.shape[1], ks.shape[1])), np.zeros(ks.shape[1]), usable, uptake, beta, weighting)
    u = model.transform(conc[:, usable])
    n_rows, n_dyes = u.shape
    design = np.column_stack([np.ones(n_rows), u]) if fit_substrate else u
    n_coef = design.shape[1]
    offset = 1 if fit_substrate else 0
    pens = []
    if ridge > 0:
        p = np.zeros((n_dyes, n_coef))
        p[:, offset:] = np.sqrt(ridge) * np.eye(n_dyes)
        pens.append(p)
    if shrink > 0 and groups is not None:
        g = np.asarray(groups)[usable]
        p = np.zeros((n_dyes, n_coef))
        for d in range(n_dyes):
            members = np.flatnonzero(g == g[d])
            p[d, offset + members] -= 1.0 / len(members)
            p[d, offset + d] += 1.0
        pens.append(np.sqrt(shrink) * p)
    penalty = np.vstack(pens) if pens else None
    for j in range(ks.shape[1]):
        w = _row_weights(ks[:, j], weighting)
        a = design * w[:, None]
        b = ks[:, j] * w
        if penalty is not None:
            a = np.vstack([a, penalty])
            b = np.concatenate([b, np.zeros(len(penalty))])
        if upper is None:
            coef, _ = nnls(a, b)
        else:
            hi = np.concatenate([[np.inf] if fit_substrate else [], np.asarray(upper)[usable, j]])
            coef = lsq_linear(a, b, bounds=(np.zeros(n_coef), hi), method="bvls").x
        if fit_substrate:
            model.substrate[j] = coef[0]
            model.k[usable, j] = coef[1:]
        else:
            model.k[usable, j] = coef
    return model


def fit_km_joint(
    conc: np.ndarray,
    ks: np.ndarray,
    *,
    uptake: str = "linear",
    beta: float = 0.0,
    weighting: str = "relative",
    smooth: float = 0.0,
    shrink: float = 0.0,
    groups: np.ndarray | None = None,
    active_threshold: float = 0.01,
) -> KMModel:
    """Joint fit over all wavelengths with optional regularisation of the dye spectra.

    ``smooth`` penalises second differences of each unit K/S spectrum along
    wavelength. ``shrink`` pulls each dye towards the mean spectrum of its
    ``groups`` label (e.g. hue family), which stabilises rarely used dyes.
    Both penalties are applied in the weighted residual scale.
    """
    from scipy import sparse
    from scipy.linalg import cholesky, solve_triangular

    conc = np.maximum(np.asarray(conc, dtype=float), 0.0)
    ks = np.asarray(ks, dtype=float)
    n_rows, n_wl = ks.shape
    usable = (conc > active_threshold).sum(axis=0) >= 1
    model = KMModel(np.zeros((conc.shape[1], n_wl)), np.zeros(n_wl), usable, uptake, beta, weighting)
    u = model.transform(conc[:, usable])
    n_dyes = u.shape[1]
    w = _row_weights(ks, weighting)  # (n_rows, n_wl)

    # Unknowns: substrate (n_wl) then k[d, wl] flattened dye-major.
    rows, cols, vals = [], [], []
    r_idx = np.arange(n_rows * n_wl).reshape(n_rows, n_wl)
    rows.append(r_idx.ravel())
    cols.append(np.tile(np.arange(n_wl), n_rows))
    vals.append(w.ravel())
    for d in range(n_dyes):
        nz = np.flatnonzero(u[:, d] > 0)
        if len(nz) == 0:
            continue
        rows.append(r_idx[nz].ravel())
        cols.append((n_wl + d * n_wl + np.arange(n_wl))[None, :].repeat(len(nz), 0).ravel())
        vals.append((u[nz, d][:, None] * w[nz]).ravel())
    a_fit = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n_rows * n_wl, n_wl + n_dyes * n_wl)
    )
    blocks = [a_fit]
    rhs = [(ks * w).ravel()]
    scale = float(np.median(w * np.maximum(ks, _KS_FLOOR)))  # penalties in the same units as weighted residuals
    if smooth > 0:
        d2 = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n_wl - 2, n_wl))
        pen = sparse.block_diag([d2] * n_dyes)
        blocks.append(sparse.hstack([sparse.csr_matrix((pen.shape[0], n_wl)), pen]) * (np.sqrt(smooth) * scale))
        rhs.append(np.zeros(pen.shape[0]))
    if shrink > 0 and groups is not None:
        g = np.asarray(groups)[usable]
        m = np.zeros((n_dyes, n_dyes))
        for d in range(n_dyes):
            members = np.flatnonzero(g == g[d])
            m[d, d] += 1.0
            m[d, members] -= 1.0 / len(members)
        pen = sparse.kron(sparse.csr_matrix(m), sparse.eye(n_wl))
        blocks.append(sparse.hstack([sparse.csr_matrix((pen.shape[0], n_wl)), pen]) * (np.sqrt(shrink) * scale))
        rhs.append(np.zeros(pen.shape[0]))
    a = sparse.vstack(blocks).tocsr()
    b = np.concatenate(rhs)
    # Exact NNLS on the Cholesky-reduced normal equations: ||Ax - b|| and ||L^T x - L^-1 A^T b||
    # differ by a constant, and the reduced system is square (n_unknowns x n_unknowns).
    gram = (a.T @ a).toarray()
    gram[np.diag_indices_from(gram)] += 1e-10 * max(np.trace(gram) / len(gram), 1.0)
    chol = cholesky(gram, lower=True)
    x, _ = nnls(chol.T, solve_triangular(chol, a.T @ b, lower=True), maxiter=50 * len(gram))
    model.substrate = x[:n_wl]
    model.k[usable] = x[n_wl:].reshape(n_dyes, n_wl)
    return model


def enumerate_subsets(dyes: np.ndarray, max_size: int = 3) -> np.ndarray:
    """All dye subsets of size 1..max_size, padded with -1."""
    combos = [c for r in range(1, max_size + 1) for c in combinations(np.asarray(dyes).tolist(), r)]
    out = -np.ones((len(combos), max_size), dtype=int)
    for i, combo in enumerate(combos):
        out[i, : len(combo)] = combo
    return out


@dataclass
class SubsetSolutions:
    subsets: np.ndarray  # (n_sub, max_size) dye indices, -1 padded
    conc: np.ndarray  # (n_sub, max_size) fitted concentrations
    de_d65: np.ndarray  # (n_sub,)
    de_f11: np.ndarray  # (n_sub,)

    def recipe_vector(self, row: int, n_dyes: int) -> np.ndarray:
        vec = np.zeros(n_dyes)
        for slot, dye in enumerate(self.subsets[row]):
            if dye >= 0:
                vec[dye] = self.conc[row, slot]
        return vec


def solve_subsets(model: KMModel, target_ks: np.ndarray, subsets: np.ndarray, c_max: float | None = None) -> SubsetSolutions:
    """Fit concentrations in [0, c_max] for every subset and score the colour match."""
    from scipy.optimize import lsq_linear

    target_ks = np.asarray(target_ks, dtype=float)
    w = _row_weights(target_ks, model.weighting)
    rhs = (target_ks - model.substrate) * w
    k_ext = np.vstack([model.k, np.zeros((1, model.k.shape[1]))])  # index -1 -> zero spectrum
    weighted_k = k_ext * w[None, :]
    u_max = float(model.transform(np.array([c_max]))[0]) if c_max is not None else np.inf
    u = np.zeros(subsets.shape, dtype=float)
    for i, subset in enumerate(subsets):
        idx = subset[subset >= 0]
        if len(idx) == 1:
            a = weighted_k[idx[0]]
            denom = float(a @ a)
            u[i, 0] = min(max(float(a @ rhs) / denom, 0.0), u_max) if denom > 0 else 0.0
        else:
            sol, _ = nnls(weighted_k[idx].T, rhs)
            if sol.max() > u_max:
                sol = lsq_linear(weighted_k[idx].T, rhs, bounds=(0.0, u_max), method="bvls").x
            u[i, : len(idx)] = sol
    ks_pred = model.substrate[None, :] + np.einsum("ns,nsw->nw", u, k_ext[subsets])
    refl_pred = ks_to_reflectance(ks_pred)
    refl_target = ks_to_reflectance(target_ks)[None, :]
    de = {}
    for ill in ("D65", "FL11"):
        de[ill] = delta_e_cmc(reflectance_to_lab(refl_target, ill), reflectance_to_lab(refl_pred, ill))
    conc = model.inverse_transform(u)
    conc[subsets < 0] = 0.0
    return SubsetSolutions(subsets, conc, de["D65"], de["FL11"])
