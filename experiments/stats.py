"""Rebuilt statistical protocol (fixes the original B-6 / B-11 defects).

Replaces:
  * n=5 seed-level Wilcoxon (minimum achievable p = 1/32, so after Bonferroni
    the test could NEVER be significant) -> DeLong on paired continuous scores,
    McNemar on paired predictions, Holm correction across comparisons;
  * IID bootstrap of a single test set -> entity-cluster and time-block
    bootstrap that reflect the actual dependence structure;
  * silently-empty paired files -> explicit KeyError.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm
from sklearn.metrics import average_precision_score, roc_auc_score


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    xs = np.asarray(x)[order]
    n = len(xs)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = t
    return out


def delong_auc(y: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Return (AUC-ROC, DeLong variance) for one score vector."""
    y = np.asarray(y).astype(int)
    s = np.asarray(scores, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return float("nan"), float("nan")
    tz = _midrank(s)
    tx = _midrank(pos)
    ty = _midrank(neg)
    auc = (tz[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    v01 = (tz[y == 1] - tx) / n0
    v10 = 1.0 - (tz[y == 0] - ty) / n1
    var = np.var(v01, ddof=1) / n1 + np.var(v10, ddof=1) / n0
    return float(auc), float(var)


def delong_paired_test(y: np.ndarray, s_a: np.ndarray, s_b: np.ndarray) -> dict:
    """Paired DeLong test for AUC(a) - AUC(b)."""
    y = np.asarray(y).astype(int)
    s_a = np.asarray(s_a, float)
    s_b = np.asarray(s_b, float)
    pos_a, neg_a = s_a[y == 1], s_a[y == 0]
    pos_b, neg_b = s_b[y == 1], s_b[y == 0]
    n1, n0 = len(pos_a), len(neg_a)
    auc_a = roc_auc_score(y, s_a)
    auc_b = roc_auc_score(y, s_b)
    tza, tzb = _midrank(s_a), _midrank(s_b)
    v01_a = (tza[y == 1] - _midrank(pos_a)) / n0
    v10_a = 1 - (tza[y == 0] - _midrank(neg_a)) / n1
    v01_b = (tzb[y == 1] - _midrank(pos_b)) / n0
    v10_b = 1 - (tzb[y == 0] - _midrank(neg_b)) / n1
    cov = (np.cov(v01_a, v01_b, ddof=1)[0, 1] / n1
           + np.cov(v10_a, v10_b, ddof=1)[0, 1] / n0)
    _, va = delong_auc(y, s_a)
    _, vb = delong_auc(y, s_b)
    var_diff = va + vb - 2 * cov
    if var_diff <= 0:
        return {"auc_a": auc_a, "auc_b": auc_b, "delta": auc_a - auc_b,
                "se": 0.0, "z": 0.0, "p_two_sided": 1.0,
                "ci95": [auc_a - auc_b, auc_a - auc_b],
                "note": "non-positive variance of the difference; treated as null"}
    se = math.sqrt(var_diff)
    z = (auc_a - auc_b) / se
    p = 2 * (1 - norm.cdf(abs(z)))
    return {"auc_a": float(auc_a), "auc_b": float(auc_b),
            "delta": float(auc_a - auc_b), "se": float(se), "z": float(z),
            "p_two_sided": float(p),
            "ci95": [float(auc_a - auc_b - 1.96 * se),
                     float(auc_a - auc_b + 1.96 * se)]}


def _aucpr(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def cluster_bootstrap(y, scores, groups, B: int = 200, seed: int = 0,
                      alpha: float = 0.05) -> dict:
    """Resample ENTITIES (cards/accounts), not rows."""
    y = np.asarray(y)
    scores = np.asarray(scores, float)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_group[g] for g in pick])
        vals.append(_aucpr(y[rows], scores[rows]))
    vals = np.array([v for v in vals if np.isfinite(v)])
    return {"estimator": "entity-cluster bootstrap of AUC-PR",
            "point": _aucpr(y, scores), "B": B,
            "ci95": [float(np.quantile(vals, alpha / 2)),
                     float(np.quantile(vals, 1 - alpha / 2))],
            "sd": float(vals.std(ddof=1)), "seed": seed,
            "n_entities": int(len(uniq))}


def block_bootstrap(y, scores, order, block_len: int = 512, B: int = 200,
                    seed: int = 0, alpha: float = 0.05) -> dict:
    """Resample contiguous TIME blocks, preserving temporal dependence."""
    y = np.asarray(y)
    scores = np.asarray(scores, float)
    order = np.argsort(np.asarray(order))
    y, scores = y[order], scores[order]
    n = len(y)
    n_blocks = max(1, math.ceil(n / block_len))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        starts = rng.integers(0, max(1, n - block_len), size=n_blocks)
        rows = np.concatenate([np.arange(s, min(n, s + block_len)) for s in starts])[:n]
        vals.append(_aucpr(y[rows], scores[rows]))
    vals = np.array([v for v in vals if np.isfinite(v)])
    return {"estimator": f"time-block bootstrap (block_len={block_len}) of AUC-PR",
            "point": _aucpr(y, scores), "B": B,
            "ci95": [float(np.quantile(vals, alpha / 2)),
                     float(np.quantile(vals, 1 - alpha / 2))],
            "sd": float(vals.std(ddof=1)), "seed": seed, "n_blocks": n_blocks}


def mcnemar_exact(y, pred_a, pred_b) -> dict:
    """Exact binomial McNemar on discordant pairs (paired predictions)."""
    a = np.asarray(pred_a).astype(int)
    b = np.asarray(pred_b).astype(int)
    yt = np.asarray(y).astype(int)
    ca_wb = int(((a == yt) & (b != yt)).sum())   # a correct, b wrong
    cb_wa = int(((b == yt) & (a != yt)).sum())
    n = ca_wb + cb_wa
    if n == 0:
        return {"b": ca_wb, "c": cb_wa, "p_exact": 1.0, "statistic": 0.0}
    from scipy.stats import binomtest
    p = float(binomtest(min(ca_wb, cb_wa), n, 0.5).pvalue)
    chi2 = (abs(ca_wb - cb_wa) - 1) ** 2 / n if n > 0 else 0.0
    return {"b": ca_wb, "c": cb_wa, "n_discordant": n,
            "p_exact": p, "chi2_cc": float(chi2)}


def holm(pvals: list[float], alpha: float = 0.05) -> dict:
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, float)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adjusted[i] = min(1.0, running)
    return {"alpha": alpha, "raw": list(map(float, pvals)),
            "adjusted": adjusted.tolist(),
            "reject": [bool(a < alpha) for a in adjusted],
            "method": "Holm-Bonferroni step-down"}


def seed_power(delta: float, sd: float, alpha: float = 0.05,
               target_power: float = 0.8, max_seeds: int = 200) -> dict:
    """Number of independent training seeds needed for a paired test."""
    if sd <= 0 or delta == 0:
        return {"required_seeds": None, "note": "zero effect or zero variance"}
    z_a = norm.ppf(1 - alpha / 2)
    z_b = norm.ppf(target_power)
    n = math.ceil(((z_a + z_b) * sd / abs(delta)) ** 2)
    return {"required_seeds": min(n, max_seeds), "delta": delta, "sd": sd,
            "alpha": alpha, "target_power": target_power,
            "formula": "n = ((z_{1-a/2}+z_power)*sd/delta)^2"}


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    sp = math.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1))
                   / max(1, n1 + n2 - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def cliffs_delta(a: np.ndarray, b: np.ndarray, max_n: int = 2000) -> float:
    """Non-parametric effect size; subsamples deterministically for tractability."""
    rng = np.random.default_rng(0)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) > max_n:
        a = rng.choice(a, max_n, replace=False)
    if len(b) > max_n:
        b = rng.choice(b, max_n, replace=False)
    gt = int((a[:, None] > b[None, :]).sum())
    lt = int((a[:, None] < b[None, :]).sum())
    return float((gt - lt) / (len(a) * len(b)))


def platt_calibrate(s_val, y_val, s_test):
    """Continuous logit -> calibrated probability, fitted on VALIDATION only."""
    from sklearn.linear_model import LogisticRegression
    s_val = np.asarray(s_val, float).reshape(-1, 1)
    s_test = np.asarray(s_test, float).reshape(-1, 1)
    eps = 1e-9
    logit_v = np.log(np.clip(s_val, eps, 1 - eps) / (1 - np.clip(s_val, eps, 1 - eps)))
    logit_t = np.log(np.clip(s_test, eps, 1 - eps) / (1 - np.clip(s_test, eps, 1 - eps)))
    lr = LogisticRegression(max_iter=1000)
    lr.fit(logit_v, np.asarray(y_val).astype(int))
    return lr.predict_proba(logit_t)[:, 1]


def brier_ece(y, p, bins: int = 10) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    brier = float(np.mean((p - y) ** 2))
    edges = np.linspace(0, 1, bins + 1)
    ece, rows = 0.0, []
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if not m.any():
            continue
        acc, conf = float(y[m].mean()), float(p[m].mean())
        ece += m.sum() / len(y) * abs(acc - conf)
        rows.append({"bin": i, "n": int(m.sum()), "accuracy": acc,
                     "mean_confidence": conf})
    return {"brier": brier, "ece": float(ece), "calibration_bins": rows}


def stratified_errors(y, scores, threshold, strata: np.ndarray,
                      min_n: int = 200) -> list[dict]:
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, float)
    pred = (scores >= threshold).astype(int)
    out = []
    for name in np.unique(strata):
        m = np.asarray(strata) == name
        if m.sum() < min_n:
            continue
        tp = int(((pred == 1) & (y == 1) & m).sum())
        fp = int(((pred == 1) & (y == 0) & m).sum())
        fn = int(((pred == 0) & (y == 1) & m).sum())
        tn = int(((pred == 0) & (y == 0) & m).sum())
        ap = _aucpr(y[m], scores[m]) if len(np.unique(y[m])) > 1 else None
        out.append({"stratum": str(name), "n": int(m.sum()),
                    "fraud_rate": float(y[m].mean()),
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "precision": tp / (tp + fp) if tp + fp else 0.0,
                    "recall": tp / (tp + fn) if tp + fn else 0.0,
                    "auc_pr": float(ap) if ap is not None and np.isfinite(ap) else None})
    return out


def metrics(y, scores, threshold: float | None = None,
            target_fpr: float = 0.01, c_fn: float = 10.0,
            c_fp: float = 1.0) -> dict:
    """Primary metrics: AUC-PR, Recall@FPR=1%, expected business loss."""
    from sklearn.metrics import roc_curve, confusion_matrix
    y = np.asarray(y).astype(int)
    s = np.asarray(scores, float)
    fpr, tpr, thr = roc_curve(y, s)
    j = int(np.argmin(np.abs(fpr - target_fpr)))
    recall_at_fpr = float(tpr[j])
    if threshold is None:
        threshold = float(thr[j])
    pred = (s >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc_pr": _aucpr(y, s),
        "auc_roc": float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else None,
        "recall_at_fpr_1pct": recall_at_fpr,
        "achieved_fpr": float(fpr[j]),
        "threshold": threshold,
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "expected_loss": float(c_fn * fn + c_fp * fp),
        "c_fn": c_fn, "c_fp": c_fp,
    }
