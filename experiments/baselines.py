"""Baselines, cost-aware cascade, learning curves, drift and fairness.

Honesty contract: the second-stage "expensive model" here is a stronger
tabular learner, i.e. a PROXY for the LLM stage. No 3B/7B weights are present
on this machine (no huggingface.co access, 20 GB free disk), therefore every
LLM-specific claim remains BLOCKED and is never reported as a result below.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler

from .data_io import load

META_COLS = {"dataset", "split", "ts", "label", "user_id", "seen_user",
             "TransactionID", "isFraud", "TransactionDT"}
# The mirror's user_id column is entirely <NA>; card1 is the usable entity key.
ENTITY_KEY = "card1"


def numeric_feature_cols(df: pd.DataFrame, k: int = 60) -> list[str]:
    """Pick k numeric predictors with the lowest missing rate (deterministic)."""
    cand = [c for c in df.columns
            if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]
    miss = df[cand].isna().mean()
    return sorted(cand, key=lambda c: (float(miss[c]), c))[:k]


@dataclass
class Split:
    Xtr: np.ndarray
    ytr: np.ndarray
    Xva: np.ndarray
    yva: np.ndarray
    Xte: np.ndarray
    yte: np.ndarray
    cols: list[str]
    ts_te: np.ndarray
    entity_te: np.ndarray
    meta: dict


def make_split(df: pd.DataFrame, cols: list[str], max_train: int = 120_000,
               seed: int = 0, max_test: int | None = None) -> Split:
    """Strict temporal split supplied by the dataset; NO shuffling across splits.

    All fit-dependent statistics (medians, scaler) are fitted on TRAIN only.
    """
    tr = df[df["split"] == "train"]
    va = df[df["split"] == "validation"]
    te = df[df["split"] == "test"]
    med = tr[cols].median(numeric_only=True)
    rng = np.random.default_rng(seed)
    if max_train and len(tr) > max_train:
        take = np.sort(rng.choice(len(tr), size=max_train, replace=False))
        tr = tr.iloc[take]
    if max_test and len(te) > max_test:
        take = np.sort(rng.choice(len(te), size=max_test, replace=False))
        te = te.iloc[take]

    def mat(frame):
        return frame[cols].apply(pd.to_numeric, errors="coerce").fillna(med).to_numpy(np.float32)

    ent = (te[ENTITY_KEY].to_numpy() if ENTITY_KEY in te.columns
           else np.arange(len(te)))
    return Split(mat(tr), tr["label"].to_numpy(np.int8),
                 mat(va), va["label"].to_numpy(np.int8),
                 mat(te), te["label"].to_numpy(np.int8),
                 list(cols), te["ts"].to_numpy(), ent,
                 {"n_train": len(tr), "n_val": len(va), "n_test": len(te),
                  "n_features": len(cols), "seed": seed,
                  "missing_median_fitted_on": "train only",
                  "entity_key": ENTITY_KEY if ENTITY_KEY in df.columns else "row_index"})


def fit_model(kind: str, X: np.ndarray, y: np.ndarray, seed: int):
    t0 = time.perf_counter()
    if kind == "gbdt":
        m = HistGradientBoostingClassifier(
            max_iter=60, max_leaf_nodes=31, learning_rate=0.1,
            l2_regularization=1.0, random_state=seed, early_stopping=False)
        m.fit(X, y)
    elif kind == "gbdt_strong":
        m = HistGradientBoostingClassifier(
            max_iter=180, max_leaf_nodes=63, learning_rate=0.08,
            l2_regularization=0.5, random_state=seed, early_stopping=False)
        m.fit(X, y)
    elif kind == "linear":
        sc = StandardScaler().fit(X)
        m = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=60,
                          tol=1e-3, random_state=seed)
        m.fit(sc.transform(X), y)
        return ("scaled", m, sc), time.perf_counter() - t0
    else:
        raise ValueError(f"unknown model kind: {kind}")
    return ("raw", m, None), time.perf_counter() - t0


def predict_scores(bundle, X: np.ndarray) -> np.ndarray:
    mode, m, sc = bundle
    x = sc.transform(X) if mode == "scaled" else X
    return m.predict_proba(x)[:, 1].astype(np.float64)


def threshold_at_fpr(y: np.ndarray, s: np.ndarray, target_fpr: float = 0.01) -> float:
    """Operating point fixed ONCE on validation, never refit on test."""
    fpr, tpr, thr = roc_curve(y, s)
    return float(thr[int(np.argmin(np.abs(fpr - target_fpr)))])


def run_model(kind: str, sp: Split, seed: int = 0):
    bundle, fit_s = fit_model(kind, sp.Xtr, sp.ytr, seed)
    s_va = predict_scores(bundle, sp.Xva)
    s_te = predict_scores(bundle, sp.Xte)
    thr = threshold_at_fpr(sp.yva, s_va)
    return {"kind": kind, "seed": seed, "scores_val": s_va, "scores_test": s_te,
            "threshold": thr, "fit_seconds": fit_s}


def cascade(sp: Split, seed: int = 0, upgrade_rate: float = 0.10,
            blend: float = 0.65, gbdt_cost: float = 1.0,
            strong_cost: float = 20.0):
    """Cost-aware GBDT -> strong-model cascade. `strong` is an LLM PROXY."""
    b1, t1 = fit_model("gbdt", sp.Xtr, sp.ytr, seed)
    b2, t2 = fit_model("gbdt_strong", sp.Xtr, sp.ytr, seed)
    p1_va = predict_scores(b1, sp.Xva)
    p1_te = predict_scores(b1, sp.Xte)
    # uncertainty = distance from decision boundary in probability space
    tau = float(np.quantile(np.abs(p1_va - 0.5), 1 - upgrade_rate))
    up_te = np.abs(p1_te - 0.5) < tau
    s_te = p1_te.copy()
    if up_te.any():
        p2 = predict_scores(b2, sp.Xte[up_te])
        s_te[up_te] = blend * p2 + (1 - blend) * p1_te[up_te]
    s_va = p1_va.copy()
    up_va = np.abs(p1_va - 0.5) < tau
    if up_va.any():
        p2v = predict_scores(b2, sp.Xva[up_va])
        s_va[up_va] = blend * p2v + (1 - blend) * p1_va[up_va]
    cost = gbdt_cost + strong_cost * float(up_te.mean())
    return {"kind": "cascade", "seed": seed, "scores_val": s_va,
            "scores_test": s_te, "threshold": threshold_at_fpr(sp.yva, s_va),
            "fit_seconds": t1 + t2, "upgrade_rate_actual": float(up_te.mean()),
            "tau": tau, "cost_per_tx": cost,
            "cost_model": {"gbdt": gbdt_cost, "strong_proxy": strong_cost,
                           "note": "proxy unit cost, NOT measured LLM latency"}}


def learning_curve(df: pd.DataFrame, cols: list[str], sizes: list[int],
                   kinds: list[str], seed: int = 0, max_test: int = 20_000):
    """AUC-PR vs number of labelled training rows; reports crossover honestly."""
    from .stats import metrics
    out = []
    for size in sizes:
        sp = make_split(df, cols, max_train=max(size * 4, size), seed=seed,
                        max_test=max_test)
        # Exactly `size` labelled rows. At very small sizes a uniform draw can
        # contain only negatives (fraud rate ~3.5%), which makes any classifier
        # undefined. We enforce a MINIMUM of one positive and one negative and
        # record that we did so; the rest of the draw keeps the natural rate.
        rng = np.random.default_rng(seed + 1000)
        n_take = min(size, len(sp.Xtr))
        y = sp.ytr
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        n_pos = max(1, min(len(pos), int(round(n_take * float(y.mean())))))
        n_neg = min(len(neg), n_take - n_pos)
        pick_pos = rng.choice(pos, size=n_pos, replace=False)
        pick_neg = rng.choice(neg, size=n_neg, replace=False)
        idx = np.sort(np.concatenate([pick_pos, pick_neg]))
        stratified = n_pos > int(round(n_take * float(y.mean())))
        small = Split(sp.Xtr[idx], sp.ytr[idx], sp.Xva, sp.yva,
                      sp.Xte, sp.yte, sp.cols, sp.ts_te, sp.entity_te,
                      {**sp.meta, "n_train": len(idx)})
        for kind in kinds:
            r = run_model(kind, small, seed)
            m = metrics(small.yte, r["scores_test"], threshold=r["threshold"])
            out.append({"n_train": int(len(idx)), "model": kind,
                        "n_pos_train": int(sp.ytr[idx].sum()),
                        "minority_floor_enforced": bool(stratified),
                        "auc_pr": m["auc_pr"], "recall_at_fpr_1pct": m["recall_at_fpr_1pct"],
                        "expected_loss": m["expected_loss"], "seed": seed})
    return out


def drift_by_time(y: np.ndarray, s: np.ndarray, ts: np.ndarray,
                  n_slices: int = 6):
    """Slice a held-out test period by time.

    Continuous timestamps (IEEE-CIS seconds) are cut by quantiles. Discrete
    timestamps (BAF month index 0..7) must be cut by UNIQUE VALUES -- quantile
    boundaries collapse onto each other and would silently drop all slices.
    """
    from .stats import metrics
    uniq = np.unique(ts)
    rows = []

    def add(i, lo, hi, sel):
        if sel.sum() < 100 or len(np.unique(y[sel])) < 2:
            return
        m = metrics(y[sel], s[sel])
        rows.append({"slice": i, "ts_lo": float(lo), "ts_hi": float(hi),
                     "n": int(sel.sum()), "fraud_rate": float(y[sel].mean()),
                     "auc_pr": m["auc_pr"],
                     "recall_at_fpr_1pct": m["recall_at_fpr_1pct"]})

    if len(uniq) <= 12:
        mode = "unique-value slices"
        for i, v in enumerate(uniq):
            add(i, v, v, ts == v)
    else:
        mode = "quantile slices"
        edges = np.quantile(ts, np.linspace(0, 1, n_slices + 1))
        for i in range(n_slices):
            lo, hi = edges[i], edges[i + 1]
            sel = (ts >= lo) & (ts < hi if i < n_slices - 1 else ts <= hi)
            add(i, lo, hi, sel)
    decay = (rows[-1]["auc_pr"] - rows[0]["auc_pr"]) if len(rows) >= 2 else None
    return {"slices": rows, "slice_mode": mode, "n_unique_ts": int(len(uniq)),
            "auc_pr_first_to_last_delta": decay,
            "note": "temporal slices within the held-out test period; "
                    "delta is None when fewer than two evaluable slices exist"}


def fairness_report(df: pd.DataFrame, y: np.ndarray, s: np.ndarray,
                    threshold: float, group_col: str) -> dict:
    """Equalized-odds gaps on BAF protected / socio-economic attributes."""
    if group_col not in df.columns:
        return {"group_col": group_col, "status": "missing"}
    g = df[group_col].to_numpy()
    pred = (s >= threshold).astype(int)
    rows = []
    ser = pd.Series(g)
    if pd.api.types.is_numeric_dtype(ser):
        try:
            g = pd.qcut(ser, 4, duplicates="drop").astype(str).to_numpy()
        except ValueError:
            g = ser.astype(str).to_numpy()
    else:
        g = ser.astype(str).to_numpy()
    for name in np.unique(g)[:12]:
        m = (g == name)
        if m.sum() < 500:
            continue
        yy, pp = y[m], pred[m]
        tpr = float(pp[yy == 1].mean()) if (yy == 1).any() else None
        fpr = float(pp[yy == 0].mean()) if (yy == 0).any() else None
        rows.append({"group": str(name), "n": int(m.sum()),
                     "fraud_rate": float(yy.mean()), "tpr": tpr, "fpr": fpr})
    tprs = [r["tpr"] for r in rows if r["tpr"] is not None]
    fprs = [r["fpr"] for r in rows if r["fpr"] is not None]
    return {"group_col": group_col, "groups": rows,
            "tpr_gap": (max(tprs) - min(tprs)) if tprs else None,
            "fpr_gap": (max(fprs) - min(fprs)) if fprs else None,
            "threshold": threshold}
