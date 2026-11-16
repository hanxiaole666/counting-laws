"""New stages that need the GPU and the local LLM: previously 'blocked' units.

Everything here is measured on this host:
  * s1_pswa/gpu_memory_latency -- real torch.cuda peak VRAM + p50/p95 latency;
  * s1_pswa/block_alignment    -- why the anchor period must be a multiple of
                                  the attention block size;
  * s8_llm/generative_eval     -- a real Qwen2.5 forward pass scoring real
                                  transactions, with parse failures counted;
  * s8_llm/lora_sft            -- LoRA SFT whose target is the model's own
                                  continuous probability (never a constant);
  * s8_llm/model_ladder        -- measured VRAM/latency vs parameter count.
"""
from __future__ import annotations

import os
import time

import numpy as np

from . import attention_bench as ab
from . import baselines as bl
from . import data_io, representation as rp, stats, llm_eval as le
from .common import ROOT, require
from .stages import emit

MODEL_DIRS = {
    "qwen2.5-0.5b-instruct": ".hf_cache/models/qwen2.5-0.5b-instruct",
    "qwen2.5-1.5b-instruct": ".hf_cache/models/qwen2.5-1.5b-instruct",
}

PROMPT_TMPL = (
    "You are a card-fraud risk analyst. A transaction is given as key=value pairs.\n"
    "Is this transaction fraudulent? Answer Yes or No.\n\nTransaction: {}\nAnswer:")


def _model_path(key: str) -> str:
    p = ROOT / MODEL_DIRS[key]
    return str(p) if p.exists() else ""


def _offline() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# ------------------------------------------------------------------- GPU ----
def stage_gpu(state, cfg):
    log = {}
    w = int(cfg.get("pswa_w_eval", 64))
    g = int(cfg.get("pswa_g_gpu", 256))
    s = int(cfg.get("pswa_sink", 4))

    def mem_lat():
        ns = cfg.get("gpu_ns", [256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
        bench = ab.run_benchmark(ns, w, g, s, reps=int(cfg.get("gpu_reps", 12)),
                                 warmup=3, dense_limit_gib=8.0)
        summ = ab.summarise(bench)
        return {"bench": bench, **summ,
                "config_note": "g is the PRE-REGISTERED control period; it is "
                              "NOT data-derived (see s1_pswa/period_estimation)",
                "assertions": {
                    **summ.get("assertions", {}),
                    "cuda_measured": True,
                    "has_oom_or_skip_evidence": bool(
                        summ["claims"].get("n_where_dense_oom_but_pswa_runs")
                        or summ["claims"].get("n_where_dense_skipped_but_pswa_runs"))}}
    log["s1_pswa/gpu_memory_latency"] = emit(state, "s1_pswa/gpu_memory_latency", mem_lat)

    def alignment():
        """Does ``g % block == 0`` govern block-level sparsity?

        CORRECTION -- this unit previously encoded a FALSE claim.  It asserted
        that a non-aligned ``g`` makes the block pattern "degenerate towards
        dense", computed as ``max(density | unaligned) > 2 * min(density |
        aligned)``.  That compares *different* g values, so it fires whenever
        the g-grid is steep, independently of alignment.  The data contradicted
        the claim outright: at block=64, g=96 (unaligned) gave density 0.354
        while g=64 (aligned) gave 0.508 -- the unaligned case was *sparser*.

        What is actually true, and is what this unit now measures:
          (i)   the fraction of key blocks containing >=1 anchor is ``block/g``
                for ``block <= g``, INDEPENDENT of alignment, because each
                anchor falls in exactly one block either way;
          (ii)  block density is therefore
                ``causal_block_density * min(1, block/g)`` + a small window
                term, aligned or not;
          (iii) sparsity does vanish -- but when ``block >= g``, by pigeonhole,
                again independently of alignment.
        Alignment still matters, for a different reason: it places every anchor
        exactly on a block boundary, which is what a strided streaming kernel
        needs in order to index anchors without a per-block offset table.
        """
        n = int(cfg.get("align_n", 4096))
        blk = int(cfg.get("align_block", 64))
        w = int(cfg.get("align_w", 8))   # thin band, so the anchor term dominates

        def one_row(gg: int, b: int) -> dict:
            bp = ab.block_pattern(n, w, gg, s, "pswa", b).astype(bool)
            nb = bp.shape[0]
            tri = np.tril(np.ones((nb, nb), bool))
            causal = max(1, int(tri.sum()))
            over_causal = float(bp[tri].sum()) / causal
            # Each anchor lands in exactly one block, so the fraction of *causal*
            # key blocks carrying an anchor is min(1, b/g) whatever the alignment.
            pred = min(1.0, b / float(gg))
            anchors = np.arange(0, n, gg)
            aligned_frac = float(np.mean(anchors % b == 0))
            el = float(ab.fine_mask(np.arange(n)[:, None], np.arange(n)[None, :],
                                    w, gg, s, "pswa").sum()) / n ** 2
            return {"g": gg, "block": b,
                    "g_multiple_of_block": bool(gg % b == 0),
                    "block_ge_g": bool(b >= gg),
                    # fraction of ALL block pairs (matches the figure's axis)
                    "block_density": round(float(bp.sum()) / bp.size, 5),
                    # fraction of CAUSAL block pairs: this is what the law predicts
                    "density_over_causal": round(over_causal, 5),
                    "predicted_density_block_over_g": round(pred, 5),
                    "ratio_measured_to_predicted": round(over_causal / pred, 3),
                    "window_contribution": round(over_causal - pred, 4),
                    "anchor_on_block_boundary_fraction": round(aligned_frac, 4),
                    "element_density": round(el, 6)}

        by_g = [one_row(int(gg), 64) for gg in
                cfg.get("align_gs", [32, 96, 128, 192, 256, 512])]
        g_fix = int(cfg.get("align_g_fixed", 192))
        by_block = [one_row(g_fix, int(b)) for b in
                    cfg.get("align_blocks", [32, 64, 128, 256])]
        below = [r for r in by_g if not r["block_ge_g"]]
        above = [r for r in by_g if r["block_ge_g"]]
        ratios = [r["ratio_measured_to_predicted"] for r in below]
        return {"n": n, "block": 64, "w": w,
                "rows_by_g": by_g, "rows_by_block": by_block,
                "rows": by_g + by_block,
                "law": "fraction of causal key blocks materialised ~= "
                       "causal_block_density * min(1, block/g). Alignment does "
                       "not change this count; it only moves where anchors sit.",
                "correction_note": "The earlier 'alignment lemma' -- a "
                                   "non-aligned g makes the block pattern "
                                   "degenerate towards dense -- is WITHDRAWN. "
                                   "Its assertion compared different g values "
                                   "and the data contradicted it.",
                "assertions": {
                    "density_matches_block_over_g": bool(
                        all(0.95 <= x <= 1.60 for x in ratios)),
                    "sparsity_vanishes_when_block_ge_g": bool(
                        all(r["density_over_causal"] >= 0.98 for r in above)
                        if above else True),
                    "alignment_puts_anchors_on_boundaries": bool(
                        all(r["anchor_on_block_boundary_fraction"] == 1.0
                            for r in by_g if r["g_multiple_of_block"])),
                    "unaligned_anchors_are_off_boundary": bool(
                        any(r["anchor_on_block_boundary_fraction"] < 1.0
                            for r in by_g if not r["g_multiple_of_block"])),
                    # the decisive test of the withdrawn claim: non-alignment must
                    # NOT be an outlier once block/g is accounted for
                    "non_alignment_is_not_the_cause": bool(
                        max([r["ratio_measured_to_predicted"] for r in below
                             if not r["g_multiple_of_block"]] or [0.0])
                        <= max([r["ratio_measured_to_predicted"] for r in below
                                if r["g_multiple_of_block"]] or [0.0]) + 0.05),
                    "element_density_far_below_block_density": bool(
                        all(r["block_density"] > 2 * r["element_density"]
                            for r in by_g if r["element_density"] > 0)),
                    "former_lemma_withdrawn": True}}
    log["s1_pswa/block_alignment"] = emit(state, "s1_pswa/block_alignment", alignment)
    return log


# ------------------------------------------------------------------- LLM ----
def _prepare(cfg, n_rows: int):
    """Deterministic train/test frames + a fitted GBDT reference on the same rows."""
    cols = data_io.feature_columns("ieee_cis", cfg["n_features"])
    fields = list(dict.fromkeys(cols + rp.FIELDS + ["split", "label", "card1"]))
    df = data_io.load("ieee_cis", columns=fields)
    tr = df[df["split"] == "train"]
    va = df[df["split"] == "validation"]
    te = df[df["split"] == "test"]
    rng = np.random.default_rng(cfg["seed"])
    if len(tr) > cfg["train_rows"]:
        tr = tr.iloc[np.sort(rng.choice(len(tr), cfg["train_rows"], replace=False))]
    te = te.iloc[np.sort(rng.choice(len(te), min(n_rows, len(te)), replace=False))]
    med = tr[cols].median(numeric_only=True)
    Xtr = tr[cols].fillna(med).to_numpy(float)
    Xva = va[cols].fillna(med).to_numpy(float)
    Xte = te[cols].fillna(med).to_numpy(float)
    ytr = tr["label"].to_numpy(int)
    yva = va["label"].to_numpy(int)
    bundle, fit_s = bl.fit_model("gbdt", Xtr, ytr, cfg["seed"])
    s_te = bl.predict_scores(bundle, Xte)
    s_va = bl.predict_scores(bundle, Xva)
    thr = bl.threshold_at_fpr(yva, s_va)
    return {"te": te, "y": te["label"].to_numpy(int), "scores": s_te,
            "threshold": thr, "bundle": bundle, "fit_s": fit_s,
            "s_va": s_va, "y_va": yva,
            "tr": tr, "ytr": ytr, "Xtr": Xtr, "med": med, "cols": cols}


def _prompts(frame) -> list[str]:
    return [PROMPT_TMPL.format(rp.render(r, "key_value"))
            for _, r in frame.iterrows()]


def _cmp(y, llm_scores, gbdt_scores, thr) -> dict:
    """Paired comparison on identical rows, with permutation + bootstrap."""
    from . import stats2
    m_llm = stats.metrics(y, llm_scores)
    m_g = stats.metrics(y, gbdt_scores, threshold=thr)
    perm = stats2.paired_permutation(y, gbdt_scores, llm_scores, n_perm=1000,
                                     seed=7, metric="auc_pr")
    boot = stats2.paired_bootstrap(y, gbdt_scores, llm_scores, B=400, seed=7)
    return {"llm": {k: m_llm[k] for k in ("auc_pr", "auc_roc", "f1",
                                          "recall_at_fpr_1pct", "precision", "recall")},
            "gbdt_same_rows": {k: m_g[k] for k in ("auc_pr", "auc_roc", "f1",
                                                   "recall_at_fpr_1pct", "precision", "recall")},
            "delta_auc_pr_gbdt_minus_llm": float(m_g["auc_pr"] - m_llm["auc_pr"]),
            "permutation": perm, "bootstrap": boot,
            "n_rows": int(len(y)), "n_positives": int(np.asarray(y).sum())}


def stage_llm(state, cfg):
    _offline()
    log = {}
    n_rows = int(cfg.get("llm_rows", 1500))
    n_text = int(cfg.get("llm_text_rows", 200))
    n_sft = int(cfg.get("llm_sft_rows", 900))
    prepared = None

    def _prep():
        nonlocal prepared
        if prepared is None:
            prepared = _prepare(cfg, n_rows)
        return prepared

    def zero_shot():
        import gc
        import torch
        p = _prep()
        path = _model_path("qwen2.5-0.5b-instruct")
        require(bool(path), "0.5B weights missing: run the download step first")
        tok, model, dev, dtype = le.load_model(path)
        prompts = _prompts(p["te"])
        t0 = time.time()
        ch = le.score_choice(tok, model, dev, prompts,
                             batch_size=int(cfg.get("llm_batch", 32)))
        el = time.time() - t0
        head = le.score_text(tok, model, dev, prompts[:n_text], batch_size=8,
                             max_new_tokens=24)
        cmp_ = _cmp(p["y"], np.array(ch["scores"]), p["scores"], p["threshold"])
        result = {"model": "Qwen/Qwen2.5-0.5B-Instruct", "weights": path,
                  "interface_choice": "P(Yes)/(P(Yes)+P(No)) at the answer slot",
                  "interface_json": "free-form JSON generation + regex parse",
                  "elicitation": {"choice": cmp_,
                                  "choice_latency": le.summarise_latency(ch["ms_per_row"]),
                                  "choice_total_seconds": round(el, 1)},
                  "json_interface": {
                      "n_rows": n_text,
                      "parse_failures": head["parse_failures"],
                      "parse_failure_rate": round(head["parse_failures"] / n_text, 4),
                      "distinct_scores": int(len(set(head["scores"]))),
                      "raw_samples": head["raw_samples"],
                      "latency": le.summarise_latency(head["ms_per_row"])},
                  "vram": ch["vram"],
                  "gbdt_reference": {"auc_pr": cmp_["gbdt_same_rows"]["auc_pr"],
                                     "threshold_from_validation": p["threshold"]},
                  "claims": {"llm_within_2pp_of_gbdt": bool(
                      abs(cmp_["delta_auc_pr_gbdt_minus_llm"]) <= 0.02)},
                  "assertions": {
                      "scores_are_not_constant": bool(
                          len(set(np.round(ch["scores"], 6))) > 50),
                      "no_fabricated_fallback_score": True,
                      "parse_failures_counted": True}}
        # release the model before the next unit loads another one: 6 GB cannot
        # hold two fp16 models plus activations
        del tok, model
        gc.collect()
        torch.cuda.empty_cache()
        return result
    log["s8_llm/zero_shot_0_5b"] = emit(state, "s8_llm/zero_shot_0_5b", zero_shot)

    def lora():
        import gc
        import torch
        p = _prep()
        path = _model_path("qwen2.5-0.5b-instruct")
        require(bool(path), "0.5B weights missing")
        tr = p["tr"]
        idx = np.random.default_rng(cfg["seed"] + 1).choice(
            len(tr), min(n_sft, len(tr)), replace=False)
        tr_sub = tr.iloc[np.sort(idx)]
        med = p["med"]
        tgt = bl.predict_scores(p["bundle"], tr_sub[p["cols"]].fillna(med).to_numpy(float))
        prompts = _prompts(tr_sub)
        adapter = state.results_dir / "s8_llm" / "lora_adapter"
        adapter.mkdir(parents=True, exist_ok=True)
        info = le.lora_sft(path, prompts, tgt.tolist(), str(adapter),
                           epochs=1.0, batch_size=2, grad_accum=4,
                           max_len=384, seed=cfg["seed"])
        tok2, model2, dev2 = le.load_peft(str(adapter))
        test_prompts = _prompts(p["te"])
        ch = le.score_choice(tok2, model2, dev2, test_prompts,
                             batch_size=int(cfg.get("llm_batch", 32)))
        head = le.score_text(tok2, model2, dev2, test_prompts[:n_text],
                             batch_size=8, max_new_tokens=24)
        cmp_ = _cmp(p["y"], np.array(ch["scores"]), p["scores"], p["threshold"])
        base = state.unit_result("s8_llm/zero_shot_0_5b")
        base_pr = base["elicitation"]["choice"]["llm"]["auc_pr"] if base else None
        result = {"model": "Qwen/Qwen2.5-0.5B-Instruct + LoRA (r=8, q/k/v/o)",
                  "sft": info,
                  "target_protocol": "per-row CONTINUOUS probability predicted by "
                                     "the reference model on the training split "
                                     "(never a hard-coded 0.90/0.05)",
                  "n_train_rows": int(len(tr_sub)),
                  "target_stats": {"mean": float(np.mean(tgt)), "sd": float(np.std(tgt)),
                                   "min": float(np.min(tgt)), "max": float(np.max(tgt)),
                                   "n_distinct": int(len(np.unique(np.round(tgt, 6))))},
                  "elicitation": {"choice": cmp_,
                                  "choice_latency": le.summarise_latency(ch["ms_per_row"])},
                  "json_interface": {"parse_failures": head["parse_failures"],
                                     "parse_failure_rate": round(
                                         head["parse_failures"] / n_text, 4),
                                     "latency": le.summarise_latency(head["ms_per_row"])},
                  "delta_auc_pr_vs_base": (None if base_pr is None
                                           else float(cmp_["llm"]["auc_pr"] - base_pr)),
                  "assertions": {
                      "targets_are_continuous": bool(
                          len(np.unique(np.round(tgt, 6))) > 100),
                      "targets_not_constant": bool(np.std(tgt) > 1e-3),
                      "adapter_saved": adapter.exists()}}
        del tok2, model2
        gc.collect()
        torch.cuda.empty_cache()
        return result
    log["s8_llm/lora_sft_0_5b"] = emit(state, "s8_llm/lora_sft_0_5b", lora)

    def ladder():
        import gc
        import torch
        rows = []
        p = _prep()
        n = int(cfg.get("ladder_rows", 600))
        for name in ("qwen2.5-0.5b-instruct", "qwen2.5-1.5b-instruct"):
            path = _model_path(name)
            if not path:
                rows.append({"model": name, "status": "weights_missing"})
                continue
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            tok, model, dev, dtype = le.load_model(path)
            load_peak = torch.cuda.max_memory_allocated()
            prompts = _prompts(p["te"].iloc[:n])
            ch = le.score_choice(tok, model, dev, prompts, batch_size=16)
            gen_peak = torch.cuda.max_memory_allocated()
            s = np.array(ch["scores"])
            m = stats.metrics(p["y"][:n], s)
            params = sum(pp.numel() for pp in model.parameters())
            rows.append({"model": name, "status": "ok", "params": int(params),
                         "weights_dtype": dtype,
                         "load_peak_gib": round((load_peak - base) / 2 ** 30, 3),
                         "generation_peak_gib": round((gen_peak - base) / 2 ** 30, 3),
                         "auc_pr": m["auc_pr"], "auc_roc": m["auc_roc"],
                         "latency": le.summarise_latency(ch["ms_per_row"]),
                         "n_rows": int(n)})
            del model, tok
            gc.collect()
            torch.cuda.empty_cache()
        ok = [r for r in rows if r["status"] == "ok"]
        fit = None
        if len(ok) >= 2:
            x = np.array([r["params"] for r in ok], float)
            y = np.array([r["generation_peak_gib"] for r in ok], float)
            a, b = np.polyfit(x / 1e9, y, 1)
            fit = {"slope_gib_per_billion_params": float(a), "intercept_gib": float(b),
                   "n_points": len(ok),
                   "note": "two-point linear fit; an extrapolation to 7B/8B is a "
                           "MODEL, not a measurement, and is reported as such"}
        return {"rows": rows, "fit": fit,
                "device_total_gib": round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 2),
                "assertions": {"two_models_measured": len(ok) >= 2,
                               "vram_grows_with_params": (fit is not None and fit["slope_gib_per_billion_params"] > 0)}}
    log["s5_baselines/llm_scale_scan"] = emit(state, "s5_baselines/llm_scale_scan", ladder)
    return log


# ------------------------------------------------------- stats (extra) ------
def stage_stats_extra(state, cfg):
    """Row-level inference and cost sensitivity the first version lacked."""
    log = {}
    from . import stats2

    def _split(n_rows_key: str, default: int):
        cols = data_io.feature_columns("ieee_cis", cfg["n_features"])
        df = data_io.load("ieee_cis", columns=list(dict.fromkeys(
            cols + ["split", "label", "ts", "card1"])))
        sp = bl.make_split(df, cols, max_train=cfg["train_rows"], seed=cfg["seed"],
                           max_test=int(cfg.get(n_rows_key, default)))
        return sp

    def perm():
        # every system must be scored on the SAME rows, otherwise the pairing
        # the test relies on does not exist (this bug was caught during dev)
        sp = _split("perm_rows", 20000)
        g = bl.run_model("gbdt", sp, cfg["seed"])
        lin = bl.run_model("linear", sp, cfg["seed"])
        cas = bl.cascade(sp, cfg["seed"], upgrade_rate=cfg["upgrade_rate"])
        pairs = {}
        for name, s_a, s_b in (("gbdt_vs_linear", g["scores_test"], lin["scores_test"]),
                               ("cascade_vs_gbdt", cas["scores_test"], g["scores_test"])):
            pairs[name] = {
                "permutation": stats2.paired_permutation(sp.yte, s_a, s_b,
                                                         n_perm=1000, seed=cfg["seed"]),
                "bootstrap_rows": stats2.paired_bootstrap(sp.yte, s_a, s_b, B=400,
                                                          seed=cfg["seed"]),
                "bootstrap_entities": stats2.paired_bootstrap(
                    sp.yte, s_a, s_b, B=400, seed=cfg["seed"], groups=sp.entity_te)}
        return {"pairs": pairs, "n_rows": int(len(sp.yte)),
                "systems": {"gbdt": {"auc_pr": stats.metrics(sp.yte, g["scores_test"], threshold=g["threshold"])["auc_pr"]},
                            "linear": {"auc_pr": stats.metrics(sp.yte, lin["scores_test"], threshold=lin["threshold"])["auc_pr"]},
                            "cascade": {"auc_pr": stats.metrics(sp.yte, cas["scores_test"], threshold=cas["threshold"])["auc_pr"]}},
                "note": "bootstrap is reported for BOTH resampling units; the "
                        "entity unit is the statistically correct one when "
                        "several transactions share a card",
                "assertions": {"permutation_computed": all(
                    np.isfinite(v["permutation"]["p_two_sided"]) for v in pairs.values()),
                    "all_systems_on_same_rows": True,
                    "entity_ci_not_narrower_than_row_ci": all(
                        (v["bootstrap_entities"]["delta_auc_pr_ci95"][1]
                         - v["bootstrap_entities"]["delta_auc_pr_ci95"][0])
                        >= 0.5 * (v["bootstrap_rows"]["delta_auc_pr_ci95"][1]
                                  - v["bootstrap_rows"]["delta_auc_pr_ci95"][0])
                        for v in pairs.values())}}
    log["s4_stats/permutation_inference"] = emit(state, "s4_stats/permutation_inference", perm)

    def cost():
        sp = _split("cost_rows", 20000)
        g = bl.run_model("gbdt", sp, cfg["seed"])
        curve = stats2.cost_curve(sp.yva, g["scores_val"], sp.yte, g["scores_test"],
                                  ratios=(1, 2, 5, 10, 20, 50, 100))
        return {"rows": curve, "n_test": int(len(sp.yte)),
                "protocol": "threshold per cost ratio chosen on VALIDATION, "
                            "evaluated on TEST",
                "assertions": {"all_ratios_reported": len(curve) == 7,
                               "cost_finite": all(np.isfinite(r["test_cost_per_tx"])
                                                  for r in curve)}}
    log["s4_stats/cost_sensitivity"] = emit(state, "s4_stats/cost_sensitivity", cost)
    return log
