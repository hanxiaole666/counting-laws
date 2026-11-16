"""Period-Sliding Window Attention (PSWA): a REAL sparse attention implementation.

Unlike the original 03_evaluate.py (which only wrote ``config.sliding_window``),
this module materialises an explicit boolean mask, runs a block-sparse softmax,
and measures retained attention mass. The mask is:

    S(i) = {j : i-w < j <= i}                    local sliding window
         U {j : j <= i, j % g == 0}              periodic anchors (data-driven g)
         U {j : j < s}                           attention sinks

all under a causal constraint j <= i. This is a genuine algorithmic object with
a falsifiable quality bound, not a config flag.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MODES = ("full", "sliding", "longformer_global", "nsa_like", "pswa")


def build_mask(n: int, w: int = 256, g: int = 96, s: int = 4,
               mode: str = "pswa") -> np.ndarray:
    """Return a causal boolean attention mask of shape (n, n)."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode}; choose from {MODES}")
    idx = np.arange(n)
    causal = idx[None, :] <= idx[:, None]
    if mode == "full":
        return causal
    local = (idx[None, :] > idx[:, None] - w) & causal
    if mode == "sliding":
        return local
    sinks = (idx[None, :] < s) & causal
    if mode == "longformer_global":
        # sliding + a small set of task-global tokens visible to every query
        return local | sinks
    if mode == "nsa_like":
        # sliding + strided compressed branch (one key per g positions)
        compressed = (idx[None, :] % g == 0) & causal
        return local | compressed | sinks
    anchors = (idx[None, :] % g == 0) & causal
    return local | anchors | sinks


def softmax_rows(scores: np.ndarray, allowed: np.ndarray | None = None) -> np.ndarray:
    """Numerically stable row-wise softmax, optionally masked (-inf outside)."""
    s = scores.astype(np.float64, copy=True)
    if allowed is not None:
        s = np.where(allowed, s, -np.inf)
    m = np.max(s, axis=1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    e = np.exp(s - m)
    z = e.sum(axis=1, keepdims=True)
    return e / np.where(z > 0, z, 1.0)


def full_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dense causal attention. Returns (output, softmax probs)."""
    d = q.shape[1]
    scores = (q @ k.T) / math.sqrt(d)
    n = q.shape[0]
    causal = np.tril(np.ones((n, n), dtype=bool))
    probs = softmax_rows(scores, causal)
    return probs @ v, probs


def sparse_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                     mask: np.ndarray, block: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Block-sparse attention: never materialises the n x n score matrix.

    For each query block we gather only the key indices that ANY row of the
    block attends to, compute a small (B x |idx|) score tile, and apply the
    corresponding sub-mask. Returns (output, measured_peak_tensor_bytes).
    """
    n, d = q.shape
    out = np.zeros_like(q, dtype=np.float64)
    peak = 0
    for b in range(0, n, block):
        hi = min(n, b + block)
        sub = mask[b:hi]                       # (B, n) bool
        col = np.flatnonzero(sub.any(axis=0))
        if col.size == 0:
            continue
        kt = k[col]                            # (|idx|, d)
        vt = v[col]
        tile = (q[b:hi] @ kt.T) / math.sqrt(d)
        probs = softmax_rows(tile, sub[:, col])
        out[b:hi] = probs @ vt
        live = (sub.nbytes + col.nbytes + kt.nbytes + vt.nbytes
                + tile.nbytes + probs.nbytes + out.nbytes)
        peak = max(peak, int(live))
    return out, np.array([peak], dtype=np.int64)


def retained_mass(probs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """rho_i = sum of attention probability kept by the sparse pattern."""
    return (probs * mask).sum(axis=1)


def quality_bound(probs: np.ndarray, v: np.ndarray, mask: np.ndarray,
                  out_dense: np.ndarray) -> dict:
    """Falsifiable bound: ||o_full - o_sparse||_inf <= 2 * eps * V_max.

    Proof sketch: with rho_i = sum_{j in S(i)} a_ij and eps_i = 1 - rho_i,
    the renormalised sparse distribution a_S has total-variation distance
    exactly eps_i from a; since |E_a[V] - E_{a_S}[V]| <= TV * (max V - min V)
    <= 2 eps_i V_max, the bound follows per coordinate.
    """
    rho = retained_mass(probs, mask)
    eps = 1.0 - rho
    vmax = float(np.max(np.abs(v)))
    out_sparse = np.zeros_like(out_dense)
    for i in range(probs.shape[0]):
        a_s = np.where(mask[i], probs[i], 0.0)
        z = a_s.sum()
        if z > 0:
            out_sparse[i] = (a_s / z) @ v
    diff = np.abs(out_dense - out_sparse)
    bound = 2.0 * eps * vmax
    return {
        "rho_mean": float(rho.mean()), "rho_min": float(rho.min()),
        "eps_mean": float(eps.mean()), "eps_max": float(eps.max()),
        "v_max": vmax,
        "empirical_max_abs_diff": float(diff.max()),
        "theoretical_bound_max": float(bound.max()),
        "bound_holds": bool(np.all(diff <= bound[:, None] + 1e-9)),
        "bound_tightness_ratio": float(diff.max() / bound.max()) if bound.max() > 0 else None,
    }


def complexity_curve(ns, w: int, g: int, s: int, mode: str = "pswa") -> list[dict]:
    rows = []
    for n in ns:
        m = build_mask(n, w, g, s, mode)
        nnz = int(m.sum())
        rows.append({"n": int(n), "nnz": nnz, "dense_nnz": int(n * (n + 1) / 2),
                     "sparsity": float(1 - nnz / (n * (n + 1) / 2)),
                     "score_matrix_mib": float(nnz * 4 / 2**20)})
    return rows


def fit_loglog_slope(rows: list[dict]) -> float:
    x = np.log(np.array([r["n"] for r in rows], dtype=float))
    y = np.log(np.array([r["nnz"] for r in rows], dtype=float))
    return float(np.polyfit(x, y, 1)[0])


def sequence_from_events(features: np.ndarray, d: int = 32,
                         seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic Q/K/V projection of a REAL entity event sequence.

    ``features`` is an (n, f) matrix of numeric behavioural features. A fixed
    random projection (seeded, no fitting) maps it to d dimensions; Q/K/V are
    three different projections. No labels are used here.
    """
    rng = np.random.default_rng(seed)
    f = features.shape[1]
    p_q = rng.normal(0, 1 / math.sqrt(f), (f, d))
    p_k = rng.normal(0, 1 / math.sqrt(f), (f, d))
    p_v = rng.normal(0, 1 / math.sqrt(f), (f, d))
    x = np.nan_to_num(features.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)
    return x @ p_q, x @ p_k, x @ p_v


def assert_ablation_nontrivial(out_full: np.ndarray, out_pswa: np.ndarray,
                               tol: float = 1e-6) -> dict:
    """The original pipeline's fatal flaw: full and pswa produced identical output."""
    delta = float(np.max(np.abs(out_full - out_pswa)))
    if delta <= tol:
        raise AssertionError(
            f"AB LATION IS A NO-OP: max|full-pswa|={delta} <= {tol}; "
            "refusing to report a spurious attention ablation")
    return {"max_abs_diff": delta, "nontrivial": True, "tol": tol}
