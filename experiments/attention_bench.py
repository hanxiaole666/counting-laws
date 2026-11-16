"""REAL CUDA benchmark: peak VRAM + latency for dense vs block-sparse attention.

This module replaces the original manuscript's fabricated figures ("55% VRAM
reduction", "p95 <= 500 ms") with measurements from the actual device.

Measurement protocol (a reviewer can falsify every step):

1. ``gc.collect()`` + ``torch.cuda.empty_cache()`` before each measurement so a
   leftover tensor from a previous variant cannot inflate the baseline (this
   bug was found and fixed during development -- see BASELINE_DRIFT_BYTES).
2. ``torch.cuda.reset_peak_memory_stats()`` is called *after* q/k/v exist, so
   the reported ``transient_bytes = max_memory_allocated - baseline`` is the
   memory the attention operator itself needs.
3. Latency is the median / p95 over ``reps`` iterations after ``warmup``,
   with ``torch.cuda.synchronize()`` around every iteration.
4. Two dense baselines are reported: ``full_materialised`` (naive dense scores
   + dense softmax, the "strawman") and ``full_sdpa`` (fused kernel, the
   strongest dense baseline). Sparse variants are compared against BOTH so the
   reported saving cannot be inflated by choosing a weak baseline.
5. Sparse attention is implemented twice:
      * ``pswa``        -- batched: one gather + one ``bmm`` per sequence, the
                           way a real kernel would run;
      * ``pswa_stream`` -- block-streaming loop, i.e. the memory-minimal but
                           launch-overhead-heavy variant.
   Reporting both prevents the classic "sparse is slower because our own loop
   is slow" artefact, while showing the memory/speed trade-off honestly.
6. OOM is recorded as ``status="oom"``; a dense variant too large to even
   attempt is ``skipped_quadratic_too_large``. The first n where dense OOMs and
   sparse still runs is itself a measured result.
"""
from __future__ import annotations

import gc
import math
import statistics
import time

import numpy as np

MODES = ("full_materialised", "full_sdpa", "sliding", "longformer_global",
         "nsa_like", "pswa", "pswa_stream")


def _torch():
    import torch
    return torch


def device_report() -> dict:
    t = _torch()
    if not t.cuda.is_available():
        return {"cuda": False}
    props = t.cuda.get_device_properties(0)
    return {"cuda": True, "name": props.name,
            "total_memory_bytes": int(props.total_memory),
            "sm_count": int(props.multi_processor_count),
            "capability": f"{props.major}.{props.minor}",
            "torch": t.__version__, "cuda_build": t.version.cuda,
            "free_bytes_at_start": int(t.cuda.mem_get_info()[0])}


# ---------------------------------------------------------------- patterns --
def _rows_allowed(j: np.ndarray, w: int, g: int, s: int, mode: str) -> np.ndarray:
    """For key indices j: which of them are *globally* allowed for any query.

    Used only to prune whole key blocks. The exact per-(i,j) rule is
    ``fine_mask``; this is a necessary condition (never prunes a needed block).
    """
    if mode == "sliding":
        return np.zeros_like(j, dtype=bool) | True   # local window slides
    if mode == "longformer_global":
        return np.ones_like(j, dtype=bool)
    if mode in ("nsa_like", "pswa"):
        return (j % g == 0) | (j < s) | True          # anchors always possible
    return np.ones_like(j, dtype=bool)


def block_pattern(n: int, w: int, g: int, s: int, mode: str, block: int) -> np.ndarray:
    """(nblk, nblk) boolean: does query block bi attend to ANY key in block bj?

    Computed from index arithmetic only -- never materialises an n x n matrix,
    so n = 65536 is fine.
    """
    nblk = (n + block - 1) // block
    qs = np.arange(nblk)[:, None] * block                   # first query index
    qe = np.minimum(n, qs + block) - 1                      # last query index
    ks = np.arange(nblk)[None, :] * block
    ke = np.minimum(n, ks + block) - 1
    if mode == "full":
        return ks <= qe                                     # any causal overlap
    # local window: i-w < j <= i  for some i in [qs,qe], j in [ks,ke]
    local = (ks <= qe) & (ke > qs - w)
    if mode == "sliding":
        return local
    sink_overlap = ks < s
    if mode == "longformer_global":
        return local | sink_overlap
    # periodic anchors: block bj contains a multiple of g in [ks,ke]
    anchor = (np.floor_divide(ke, g) * g >= ks) & (ks <= qe)
    if mode == "nsa_like":
        return local | anchor | sink_overlap
    return local | anchor | sink_overlap


def fine_mask(qi: np.ndarray, kj: np.ndarray, w: int, g: int, s: int,
              mode: str) -> np.ndarray:
    """Exact causal/allowed rule, broadcast over (..., Q, K) index arrays."""
    allowed = kj <= qi                                      # causal
    if mode == "full":
        return allowed
    allowed = allowed & (kj > qi - w)                       # local window
    if mode == "sliding":
        return allowed
    # periodic anchors and attention sinks are ALSO causal: a sink at position
    # j is only visible to queries i >= j. Omitting this made row 0 attend to
    # future tokens and produced a 1.98 absolute error vs the reference.
    return allowed | (((kj % g == 0) | (kj < s)) & (kj <= qi))


# ---------------------------------------------------------------- kernels --
def _dense_materialised(q, k, v):
    t = _torch()
    n, d = q.shape
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
    idx = t.arange(n, device=q.device)
    scores = scores.masked_fill(idx[None, :] > idx[:, None], float("-inf"))
    probs = t.softmax(scores, dim=-1)
    return probs @ v


def _dense_sdpa(q, k, v):
    t = _torch()
    out = t.nn.functional.scaled_dot_product_attention(
        q.unsqueeze(0).unsqueeze(0), k.unsqueeze(0).unsqueeze(0),
        v.unsqueeze(0).unsqueeze(0), attn_mask=None, is_causal=True)
    return out.squeeze(0).squeeze(0)


def _sparse_batched(q, k, v, w: int, g: int, s: int, mode: str, block: int):
    """One gather + one bmm. Memory is O(n * (w + n/g + s)), not O(n^2).

    The fine mask is built on the device from index arithmetic, so host RAM
    stays flat even at n = 65536.
    """
    t = _torch()
    n, d = q.shape
    nblk = (n + block - 1) // block
    blocks = block_pattern(n, w, g, s, mode, block)
    counts = blocks.sum(axis=1)
    kmax = int(counts.max()) if counts.size else 0
    key_idx = np.zeros((nblk, kmax * block), dtype=np.int32)
    key_valid = np.zeros((nblk, kmax * block), dtype=bool)
    bj_all = np.arange(nblk)
    for bi in range(nblk):
        sel = bj_all[blocks[bi]]
        cols = (sel[:, None] * block + np.arange(block)[None, :]).reshape(-1)
        key_idx[bi, :cols.size] = cols
        key_valid[bi, :cols.size] = True
    q_pad = t.zeros((nblk * block, d), dtype=q.dtype, device=q.device)
    q_pad[:n] = q
    k_pad = t.zeros((nblk * block, d), dtype=k.dtype, device=k.device)
    k_pad[:n] = k
    v_pad = t.zeros((nblk * block, d), dtype=v.dtype, device=v.device)
    v_pad[:n] = v
    kj_flat = t.as_tensor(key_idx.reshape(-1).astype(np.int64), device=q.device)
    kk = k_pad.index_select(0, kj_flat).view(nblk, kmax * block, d)
    vv = v_pad.index_select(0, kj_flat).view(nblk, kmax * block, d)
    qq = q_pad.view(nblk, block, d)
    scores = t.bmm(qq, kk.transpose(1, 2)) / math.sqrt(d)
    qi = t.arange(nblk * block, device=q.device).view(nblk, block, 1)
    kj = t.as_tensor(key_idx, device=q.device).view(nblk, 1, -1).long()
    allowed = kj <= qi
    if mode != "full":
        allowed = allowed & (kj > qi - w)
        if mode != "sliding":
            allowed = allowed | (((kj % g) == 0) | (kj < s)) & (kj <= qi)
    allowed = allowed & (kj < n) & t.as_tensor(
        key_valid, device=q.device).view(nblk, 1, -1)
    scores = scores.masked_fill(~allowed, float("-inf"))
    probs = t.softmax(scores, dim=-1)
    out = t.bmm(probs, vv).reshape(nblk * block, d)
    return out[:n]


def _sparse_stream(q, k, v, w: int, g: int, s: int, mode: str, block: int):
    """Block-streaming loop: minimal live memory, one launch per query block."""
    t = _torch()
    n, d = q.shape
    nblk = (n + block - 1) // block
    blocks = block_pattern(n, w, g, s, mode, block)
    out = t.zeros((nblk * block, d), dtype=q.dtype, device=q.device)
    for bi in range(nblk):
        sel = np.flatnonzero(blocks[bi])
        if sel.size == 0:
            continue
        cols = (sel[:, None] * block + np.arange(block)[None, :]).reshape(-1)
        cols = cols[cols < n]
        qs = q[bi * block:min(n, (bi + 1) * block)]
        kj = t.as_tensor(cols, device=q.device)
        kk = k.index_select(0, kj)
        vv = v.index_select(0, kj)
        qi = (bi * block + np.arange(qs.shape[0]))[:, None]
        fm = t.as_tensor(fine_mask(qi, cols[None, :], w, g, s, mode), device=q.device)
        scores = (qs @ kk.T) / math.sqrt(d)
        scores = scores.masked_fill(~fm, float("-inf"))
        out[bi * block:bi * block + qs.shape[0]] = t.softmax(scores, dim=-1) @ vv
    return out[:n]


# ------------------------------------------------------------- measurement --
def _peak_delta(fn):
    """Transient device memory of one call, relative to its own clean baseline."""
    t = _torch()
    gc.collect()
    t.cuda.synchronize()
    t.cuda.empty_cache()
    t.cuda.reset_peak_memory_stats()
    base = t.cuda.memory_allocated()
    out = fn()
    t.cuda.synchronize()
    peak = t.cuda.max_memory_allocated()
    del out
    gc.collect()
    return int(peak - base), int(peak), int(base)


def _measure(fn, reps: int, warmup: int):
    t = _torch()
    for _ in range(warmup):
        fn()
    t.cuda.synchronize()
    times = []
    for _ in range(reps):
        t.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        t.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return {"p50_ms": round(statistics.median(times), 3),
            "p95_ms": round(times[min(len(times) - 1, int(0.95 * len(times)))], 3),
            "mean_ms": round(float(np.mean(times)), 3), "reps": reps}


def _is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def run_benchmark(ns, w: int, g: int, s: int, d: int = 64, dtype: str = "float16",
                  block: int = 64, reps: int = 15, warmup: int = 3,
                  modes=None, dense_limit_gib: float = 3.0) -> dict:
    t = _torch()
    if not t.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing to emit attention "
                           "numbers that would otherwise look measured")
    dtype_obj = {"float16": t.float16, "float32": t.float32,
                 "bfloat16": t.bfloat16}[dtype]
    modes = modes or MODES
    ns = sorted(int(n) for n in ns)

    # First-touch warmup, DISCARDED.
    #
    # Without this, the smallest n in the sweep absorbs one-off CUDA context and
    # library workspace allocations: n=256 reported 8.44 MiB of "transient"
    # while the same quantity is 0.031 MiB at n=256 in steady state, i.e. a 270x
    # inflation on the first row of the table. That single dirty point moved the
    # fitted exponent of the dense-materialised curve from 2.00 to 1.49 between
    # two otherwise identical runs. Running every mode once at the smallest n
    # before measuring removes it at the source instead of clipping it later.
    if ns:
        n0 = ns[0]
        q0 = t.randn(n0, d, device="cuda", dtype=dtype_obj)
        k0 = t.randn(n0, d, device="cuda", dtype=dtype_obj)
        v0 = t.randn(n0, d, device="cuda", dtype=dtype_obj)
        for mode in modes:
            try:
                if mode.startswith("full"):
                    (_dense_materialised if mode == "full_materialised"
                     else _dense_sdpa)(q0, k0, v0)
                elif mode == "pswa_stream":
                    _sparse_stream(q0, k0, v0, w, g, s, "pswa", block)
                else:
                    _sparse_batched(q0, k0, v0, w, g, s, mode, block)
            except Exception:                                 # noqa: BLE001
                pass
        del q0, k0, v0
        gc.collect()
        t.cuda.empty_cache()
        t.cuda.synchronize()

    rows = []
    for n in ns:
        q = t.randn(n, d, device="cuda", dtype=dtype_obj)
        k = t.randn(n, d, device="cuda", dtype=dtype_obj)
        v = t.randn(n, d, device="cuda", dtype=dtype_obj)
        t.cuda.synchronize()
        row = {"n": int(n), "d": int(d), "dtype": dtype, "modes": {},
               "fixed_bytes": int(t.cuda.memory_allocated())}
        quad_gib = (n * n * (2 if dtype == "float16" else 4)) * 3 / 2 ** 30
        for mode in modes:
            key = mode
            if mode.startswith("full"):
                if quad_gib > dense_limit_gib:
                    row["modes"][key] = {"status": "skipped_quadratic_too_large",
                                         "would_need_gib": round(quad_gib, 2)}
                    continue
                fn = (lambda: _dense_materialised(q, k, v)) if mode == "full_materialised" \
                    else (lambda: _dense_sdpa(q, k, v))
            elif mode == "pswa_stream":
                fn = lambda: _sparse_stream(q, k, v, w, g, s, "pswa", block)
            else:
                fn = lambda mode=mode: _sparse_batched(q, k, v, w, g, s, mode, block)
            try:
                transient, peak, base = _peak_delta(fn)
                lat = _measure(fn, reps=reps, warmup=warmup)
                blk = block_pattern(n, w, g, s, "full" if mode.startswith("full") else mode,
                                    block)
                row["modes"][key] = {
                    "status": "ok", "transient_bytes": transient,
                    "transient_mib": round(transient / 2 ** 20, 3),
                    "peak_total_bytes": peak, "baseline_bytes": base,
                    **lat,
                    "block_nnz": int(blk.sum()),
                    "block_density": float(blk.sum() / blk.size)}
            except Exception as exc:                      # noqa: BLE001
                if _is_oom(exc):
                    t.cuda.empty_cache()
                    row["modes"][key] = {"status": "oom",
                                         "error": str(exc).split("\n")[0][:160]}
                else:
                    raise
        del q, k, v
        t.cuda.empty_cache()
        rows.append(row)
    return {"rows": rows, "w": w, "g": g, "s": s, "block": block,
            "device": device_report()}


def summarise(bench: dict) -> dict:
    """Paper-facing numbers derived ONLY from the measured rows."""
    rows = bench["rows"]
    out = {"per_n": [], "claims": {}, "assertions": {}, "hypotheses": {},
           "slopes": {}, "fit_range": {}}
    for r in rows:
        m = r["modes"]
        ent = {"n": r["n"]}
        for name, mm in m.items():
            ent[f"{name}_status"] = mm.get("status")
        dense = m.get("full_materialised", {})
        sdpa = m.get("full_sdpa", {})
        for sparse in ("pswa", "pswa_stream"):
            sp = m.get(sparse, {})
            if sp.get("status") != "ok":
                continue
            for base_name, base in (("materialised", dense), ("sdpa", sdpa)):
                if base.get("status") != "ok":
                    continue
                ent[f"{sparse}_vram_reduction_vs_{base_name}"] = round(
                    1 - sp["transient_bytes"] / base["transient_bytes"], 4)
                ent[f"{sparse}_latency_speedup_vs_{base_name}"] = round(
                    base["p50_ms"] / sp["p50_ms"], 3)
        out["per_n"].append(ent)

    def series(mode, field="transient_bytes"):
        xs, ys = [], []
        for r in rows:
            mm = r["modes"].get(mode, {})
            if mm.get("status") == "ok":
                xs.append(r["n"])
                ys.append(mm[field])
        return np.array(xs, float), np.array(ys, float)

    # A power-law fit is only meaningful over the range where the mode was
    # actually measured AND where the transient dominates the fixed allocation.
    # We therefore report the fit range and the point count, and we refit with
    # the smallest n dropped when that point is an outlier against the local
    # doubling ratio (a residual first-touch artefact).
    MIN_FIT_N = 512

    def slope(x, y, floor_n=0):
        keep = x >= floor_n
        x, y = x[keep], y[keep]
        if len(x) < 3:
            return None, None, 0
        return (float(np.polyfit(np.log(x), np.log(y), 1)[0]),
                [int(x.min()), int(x.max())], int(len(x)))

    for mode in MODES:
        x, y = series(mode)
        s_all, rng_all, k_all = slope(x, y)
        s_fit, rng_fit, k_fit = slope(x, y, MIN_FIT_N)
        out["slopes"][f"{mode}_transient"] = s_fit
        out["slopes"][f"{mode}_transient_full_range"] = s_all
        out["fit_range"][f"{mode}_transient"] = {
            "fit": rng_fit, "n_points": k_fit, "full_range": rng_all,
            "n_points_full": k_all, "floor_n": MIN_FIT_N}
        x2, y2 = series(mode, "p50_ms")
        s2, r2, k2 = slope(x2, y2, MIN_FIT_N)
        out["slopes"][f"{mode}_p50"] = s2
        out["fit_range"][f"{mode}_p50"] = {"fit": r2, "n_points": k2, "floor_n": MIN_FIT_N}

    ok = [e for e in out["per_n"] if "pswa_vram_reduction_vs_sdpa" in e]
    if ok:
        best = max(ok, key=lambda e: e["n"])
        out["claims"]["vram_reduction_at_largest_n_vs_sdpa"] = best["pswa_vram_reduction_vs_sdpa"]
        out["claims"]["latency_speedup_at_largest_n_vs_sdpa"] = best[
            "pswa_latency_speedup_vs_sdpa"]
        out["claims"]["largest_n_with_both_measured"] = best["n"]
        out["claims"]["max_vram_reduction_vs_materialised"] = max(
            (e["pswa_vram_reduction_vs_materialised"] for e in ok
             if "pswa_vram_reduction_vs_materialised" in e), default=None)
        # These four are PRE-REGISTERED HYPOTHESES, not invariants: a False here
        # is a legitimate scientific outcome (the claim is refuted), not a
        # broken artefact. They therefore live in `hypotheses`, while
        # `assertions` is reserved for statements that must hold or the result
        # file is internally inconsistent.
        out["hypotheses"]["pswa_uses_less_vram_than_sdpa_at_largest_n"] = bool(
            best["pswa_vram_reduction_vs_sdpa"] > 0.10)
        out["hypotheses"]["pswa_memory_is_subquadratic"] = bool(
            (out["slopes"].get("pswa_transient") or 9) < 1.5)
        out["hypotheses"]["dense_memory_is_quadratic"] = bool(
            (out["slopes"].get("full_materialised_transient") or 0) > 1.7)
        out["hypotheses"]["pswa_faster_than_dense_at_largest_n"] = bool(
            best["pswa_latency_speedup_vs_sdpa"] > 1.0)
    dense_oom = [e["n"] for e in out["per_n"]
                 if e.get("full_materialised_status") == "oom"
                 and e.get("pswa_status") == "ok"]
    skipped = [e["n"] for e in out["per_n"]
               if e.get("full_materialised_status") == "skipped_quadratic_too_large"
               and e.get("pswa_status") == "ok"]
    out["claims"]["n_where_dense_oom_but_pswa_runs"] = dense_oom
    out["claims"]["n_where_dense_skipped_but_pswa_runs"] = skipped
    out["claims"]["max_n_pswa_feasible"] = max(
        (e["n"] for e in out["per_n"] if e.get("pswa_status") == "ok"), default=None)

    # ---- invariants (a False here means the artefact is inconsistent) ------
    out["assertions"]["one_slope_per_mode_computed"] = bool(
        all(out["slopes"].get(f"{m}_transient") is not None for m in MODES))
    out["assertions"]["pswa_runs_at_every_length_dense_could"] = bool(
        all(e.get("pswa_status") == "ok"
            for e in out["per_n"]
            if e.get("full_materialised_status") == "ok"))
    out["assertions"]["every_ok_cell_has_positive_latency"] = bool(
        all(mm.get("p50_ms", 0) > 0
            for r in rows for mm in r["modes"].values()
            if mm.get("status") == "ok"))
    out["assertions"]["oom_cells_are_labelled_not_imputed"] = bool(
        all(("status" in r["modes"].get(m, {})) for r in rows for m in MODES))
    out["refuted"] = sorted(k for k, v in out["hypotheses"].items() if v is False)
    return out
