"""Inference the original design was missing.

The reviewer-facing question is not "is the point estimate bigger" but "is the
difference larger than resampling noise, and does it survive multiplicity and
cost sensitivity". This module adds:

  * paired permutation test on the SAME test rows (valid for AUC-PR, where
    DeLong's variance formula does not apply);
  * paired bootstrap confidence intervals for the difference in AUC-PR and in
    Recall@FPR=1%;
  * cluster-aware bootstrap (entities, not rows) because transactions from one
    card are not independent;
  * expected-cost curves over a range of cost ratios, with the optimal
    threshold chosen on VALIDATION and evaluated on TEST (no test leakage).
"""
from __future__ import annotations

import numpy as np


def _aucpr(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def _recall_at_fpr(y: np.ndarray, s: np.ndarray, target: float = 0.01) -> float:
    from sklearn.metrics import roc_curve
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    j = int(np.argmin(np.abs(fpr - target)))
    return float(tpr[j])


def paired_permutation(y, s_a, s_b, n_perm: int = 2000, seed: int = 0,
                       metric: str = "auc_pr") -> dict:
    """Randomisation test: swap the two systems' scores *within* each row.

    Under H0 the two systems are exchangeable per row, so swapping is the exact
    null. Applicable to any paired metric, including AUC-PR.
    """
    y = np.asarray(y).astype(int)
    s_a = np.asarray(s_a, float)
    s_b = np.asarray(s_b, float)
    fn = _aucpr if metric == "auc_pr" else _recall_at_fpr
    observed = fn(y, s_a) - fn(y, s_b)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        flip = rng.random(len(y)) < 0.5
        aa = np.where(flip, s_b, s_a)
        bb = np.where(flip, s_a, s_b)
        null[i] = fn(y, aa) - fn(y, bb)
    p = float((np.abs(null) >= abs(observed) - 1e-12).mean())
    return {"metric": metric, "observed_delta": float(observed),
            "p_two_sided": p, "n_perm": n_perm,
            "null_q": {"q025": float(np.quantile(null, 0.025)),
                       "q500": float(np.quantile(null, 0.5)),
                       "q975": float(np.quantile(null, 0.975))},
            "note": "row-wise exchangeability under H0; no distributional "
                    "assumption on the metric"}


def paired_bootstrap(y, s_a, s_b, B: int = 1000, seed: int = 0,
                     groups=None) -> dict:
    """Paired bootstrap CI for delta AUC-PR / delta Recall@1%FPR.

    With ``groups`` the resampling unit is the entity (cluster), which is the
    statistically correct unit when several transactions share a card.
    """
    y = np.asarray(y).astype(int)
    s_a = np.asarray(s_a, float)
    s_b = np.asarray(s_b, float)
    rng = np.random.default_rng(seed)
    n = len(y)
    if groups is not None:
        groups = np.asarray(groups)
        uniq, inv = np.unique(groups, return_inverse=True)
        members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    d_pr, d_rc, a_pr, b_pr = [], [], [], []
    for _ in range(B):
        if groups is None:
            idx = rng.integers(0, n, n)
        else:
            pick = rng.integers(0, len(members), len(members))
            idx = np.concatenate([members[p] for p in pick])
        a_pr.append(_aucpr(y[idx], s_a[idx]))
        b_pr.append(_aucpr(y[idx], s_b[idx]))
        d_pr.append(a_pr[-1] - b_pr[-1])
        d_rc.append(_recall_at_fpr(y[idx], s_a[idx]) - _recall_at_fpr(y[idx], s_b[idx]))
    q = lambda v, lo, hi: [float(np.nanquantile(v, lo)), float(np.nanquantile(v, hi))]
    return {"B": B, "resample_unit": "entity" if groups is not None else "row",
            "delta_auc_pr_mean": float(np.nanmean(d_pr)),
            "delta_auc_pr_ci95": q(d_pr, 0.025, 0.975),
            "delta_recall_at_1pct_fpr_ci95": q(d_rc, 0.025, 0.975),
            "a_auc_pr_ci95": q(a_pr, 0.025, 0.975),
            "b_auc_pr_ci95": q(b_pr, 0.025, 0.975),
            "p_delta_positive": float(np.nanmean(np.array(d_pr) > 0))}


def cost_curve(y_val, s_val, y_test, s_test,
               ratios=(1, 5, 10, 20, 50, 100)) -> list[dict]:
    """Expected cost per transaction for a grid of cost ratios c_fn : c_fp.

    Threshold is chosen on validation for each ratio, then applied to test.
    """
    y_val = np.asarray(y_val).astype(int)
    y_test = np.asarray(y_test).astype(int)
    s_val = np.asarray(s_val, float)
    s_test = np.asarray(s_test, float)
    rows = []
    grid = np.unique(np.quantile(s_val, np.linspace(0.001, 0.999, 200)))
    for r in ratios:
        best = None
        for t in grid:
            pv = s_val >= t
            cost_v = (r * ((y_val == 1) & ~pv).sum() + ((y_val == 0) & pv).sum()
                      ) / max(1, len(y_val))
            if best is None or cost_v < best[1]:
                best = (t, cost_v)
        t = best[0]
        pt = s_test >= t
        fn = int(((y_test == 1) & ~pt).sum())
        fp = int(((y_test == 0) & pt).sum())
        rows.append({"cost_ratio_fn_fp": r, "threshold_from_validation": float(t),
                     "test_cost_per_tx": float((r * fn + fp) / max(1, len(y_test))),
                     "test_recall": float(((y_test == 1) & pt).sum() / max(1, (y_test == 1).sum())),
                     "test_fpr": float(((y_test == 0) & pt).sum() / max(1, (y_test == 0).sum())),
                     "validation_cost_per_tx": float(best[1])})
    return rows
