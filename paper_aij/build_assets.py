"""Generate every number and table that appears in the AIJ manuscript.

Design rule: the LaTeX source contains no hand-typed measurement.  Every
quantity is emitted here from ``results/<run_id>/*.json`` into
``numbers.tex`` (macros) or ``tables/*.tex`` (floats).  ``check_numbers.py``
re-derives the values a second time and diffs them against the compiled
numbers, so a stale number cannot survive an edit.

Usage:  python paper_aij/build_assets.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
RUNS = ROOT / "results"
sys.path.insert(0, str(ROOT))


# ------------------------------------------------------------------ helpers --
def latest_run() -> Path:
    runs = sorted([p for p in RUNS.iterdir()
                   if p.is_dir() and (p / "RUN_METADATA.json").exists()])
    return runs[-1]


def load(run: Path, rel: str) -> dict:
    p = run / f"{rel}.json"
    if not p.exists():
        raise SystemExit(f"missing result: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def nnz_exact(n: int, w: int, g: int, s: int) -> int:
    """Closed form for |S| with S(i)={i-w<j<=i} U {j<s} U {j<=i, j%g==0}.

    For i < w the window already contains every earlier position, so the row
    contributes i+1.  For i >= w the window contributes w, the sinks add
    min(s, i-w+1) positions that the window has just dropped, and the anchors
    add floor((i-w)/g) positions (multiples of g that are >= s).
    """
    total = w * (w + 1) // 2
    for i in range(w, n):
        m = i - w
        total += w + min(s, m + 1) + m // g
    return total


def nnz_asymptotic(n: int, w: int, g: int, s: int) -> float:
    """n(w+s) + (n-w)^2/(2g): the form used for the complexity statement."""
    return n * (w + s) + (n - w) ** 2 / (2.0 * g)


def fmt(x, nd=4) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return f"{x:,}".replace(",", "{,}")
    if abs(x) >= 1000:
        return f"{x:,.1f}".replace(",", "{,}")
    return f"{x:.{nd}f}"


def pct(x, nd=2) -> str:
    return f"{100 * x:.{nd}f}"


MACROS: dict[str, str] = {}


def M(name: str, value: str) -> None:
    if name in MACROS:
        raise SystemExit(f"duplicate macro {name}")
    if value is None or str(value).strip() == "":
        # an empty macro prints as nothing and reads as a missing unit count in
        # the appendix; fail here instead of shipping a hole in the sentence
        raise SystemExit(f"empty value for macro {name}")
    MACROS[name] = value


def MT(name: str, text: str) -> None:
    """Text macro (escaped LaTeX)."""
    M(name, text)


# --------------------------------------------------------------------- main --
def main() -> int:
    run = latest_run()
    rid = run.name
    meta = json.loads((run / "RUN_METADATA.json").read_text(encoding="utf-8"))

    ieee = load(run, "data/ieee_cis")
    baf = load(run, "data/baf")
    ent = load(run, "s2_gbl/entity_sequences")
    cplx = load(run, "s1_pswa/complexity_curve")
    qual = load(run, "s1_pswa/quality_bound")
    per = load(run, "s1_pswa/period_estimation")
    gpu = load(run, "s1_pswa/gpu_memory_latency")
    blk = load(run, "s1_pswa/block_alignment")
    main_t = load(run, "s5_baselines/main_table")
    infer = load(run, "s4_stats/inference")
    perm = load(run, "s4_stats/permutation_inference")
    cost = load(run, "s4_stats/cost_sensitivity")
    cal = load(run, "s4_stats/calibration")
    seeds = load(run, "s4_stats/seed_sweep")
    eff = load(run, "s4_stats/effect_sizes")
    llm0 = load(run, "s8_llm/zero_shot_0_5b")
    sft = load(run, "s8_llm/lora_sft_0_5b")
    scale = load(run, "s5_baselines/llm_scale_scan")
    gas = load(run, "s3_protocol/onchain_gas")
    drift_i = load(run, "s6_drift/ieee_temporal")
    drift_b = load(run, "s6_drift/baf_monthly")
    fair = load(run, "s6_drift/fairness")
    curve = load(run, "s5_baselines/learning_curve")
    ladder = load(run, "s5_baselines/field_ladder")
    transfer = load(run, "s5_baselines/baf_transfer")
    chal = load(run, "s3_protocol/challenge_grid")

    # ---------------------------------------------------------------- header
    MT("RunId", rid.replace("_", r"\_"))
    # RUN_METADATA nests the unit counts under summary.counts
    _summ = meta.get("summary", {})
    _done = _summ.get("counts", {}).get("done", _summ.get("done"))
    M("RunUnits", str(_done))
    M("RunBlocked", str(_summ.get("counts", {}).get("blocked", 0)))
    M("RunElapsed", f"{meta.get('elapsed_s', 0):.0f}")

    # ---------------------------------------------------------------- data
    si, sb = ieee["schema"], baf["schema"]
    M("IeeeRows", fmt(si["rows"]))
    M("IeeeCols", fmt(si["columns"]))
    M("IeeeFraudPct", pct(si["fraud_rate"]))
    M("IeeeFraudCount", fmt(si["fraud_count"]))
    M("BafRows", fmt(sb["rows"]))
    M("BafCols", fmt(sb["columns"]))
    M("BafFraudPct", pct(sb["fraud_rate"]))
    M("IeeeSha", si["sha256"][:16].replace("_", r"\_"))
    M("TrainRows", fmt(si["split_counts"]["train"]))
    M("ValRows", fmt(si["split_counts"]["validation"]))
    M("TestRows", fmt(si["split_counts"]["test"]))
    M("NEtities", fmt(ent["length_distribution"]["n_entities"]))
    M("MedianLen", fmt(ent["length_distribution"]["median_len"], 0))
    M("PninetyLen", fmt(ent["length_distribution"]["p90_len"], 0))
    M("PninetyNineLen", fmt(ent["length_distribution"]["p99_len"], 0))
    M("MaxLen", fmt(ent["length_distribution"]["max_len"]))
    M("EntGteThousand", fmt(ent["length_distribution"]["entities_ge_1024"]))

    # -------------------------------------------------------- attention law
    w, g, s = cplx["w"], cplx["g"], cplx["s"]
    M("WinW", str(w))
    M("AnchorG", str(g))
    M("Sinks", str(s))
    pts = cplx["modes"]["pswa"]["curve"]
    rows = []
    max_err = 0.0
    for p in pts:
        n = p["n"]
        closed = nnz_exact(n, w, g, s)
        asym = nnz_asymptotic(n, w, g, s)
        err = abs(closed - p["nnz"]) / p["nnz"]
        max_err = max(max_err, err)
        dense = p["dense_nnz"]
        rows.append((n, p["nnz"], closed, asym, p["nnz"] / dense, dense))
    M("NnzLawMaxErrPct", f"{100 * max_err:.3f}")
    # local log-log exponent between consecutive points
    exps = []
    for a, b in zip(pts, pts[1:]):
        exps.append(math.log(b["nnz"] / a["nnz"]) / math.log(b["n"] / a["n"]))
    M("NnzLocalExponentMin", f"{min(exps):.3f}")
    M("NnzLocalExponentMax", f"{max(exps):.3f}")
    M("NnzSlopeFitted", f"{cplx['modes']['pswa']['loglog_slope']:.3f}")
    M("SlidingSlopeFitted", f"{cplx['modes']['sliding']['loglog_slope']:.3f}")
    # crossover: n where the anchor term equals the window term
    M("CrossoverLen", fmt(int(2 * g * (w + s))))
    last = pts[-1]
    anchor_share = (nnz_exact(last["n"], w, g, s) - last["n"] * w) / last["nnz"]
    M("AnchorShareAtLargestN", pct(anchor_share, 1))
    M("DensityAtLargestN", f"{100 * last['nnz'] / last['dense_nnz']:.2f}")
    M("DenseAtLargestN", fmt(last["dense_nnz"]))

    # ------------------------------------------------------ retained mass
    q_ = qual["quality_bound"]
    M("QualityN", fmt(qual["n"]))
    M("RhoMean", f"{q_['rho_mean']:.4f}")
    M("RhoMin", f"{q_['rho_min']:.4f}")
    M("EpsMean", f"{q_['eps_mean']:.4f}")
    M("EpsMax", f"{q_['eps_max']:.4f}")
    M("Vmax", f"{q_['v_max']:.3f}")
    M("EmpDiff", f"{q_['empirical_max_abs_diff']:.3f}")
    M("BoundMax", f"{q_['theoretical_bound_max']:.2f}")
    M("Tightness", pct(q_["bound_tightness_ratio"], 1))
    M("PeakTensorReductionPct", pct(qual["peak_intermediate_tensor_bytes"]
                                    ["reduction_fraction"], 1))

    # ------------------------------------------------------- period / ACF
    pe = per["period_estimation"]
    M("AcfLagOne", f"{pe['short_lag_peaks'][0]['acf']:.4f}")
    M("AcfAtG", f"{pe['acf_at_anchor_g']:.4f}")
    M("AcfNoiseFloor", f"{pe['acf_noise_floor_sd']:.4f}")
    M("AcfWeakPeak", f"{pe['weak_local_peak_acf']:.4f}")
    M("AcfWeakLag", str(pe["weak_local_peak_lag"]))
    M("AcfPeakLag", str(pe["anchor_band_peak_lag"]))
    M("NSequences", str(pe["sequences_used"]))
    M("AnchorBandLo", str(pe["anchor_band"][0]))
    M("AnchorBandHi", str(pe["anchor_band"][1]))

    # ------------------------------------------------------------ block law
    M("AlignN", fmt(blk["n"]))
    M("AlignBlock", str(blk["block"]))
    M("AlignW", str(blk["w"]))
    bb = blk["rows_by_block"]
    M("BlockRatioMin", f"{min(r['ratio_measured_to_predicted'] for r in blk['rows_by_g']):.3f}")
    M("BlockRatioMax", f"{max(r['ratio_measured_to_predicted'] for r in blk['rows_by_g']):.3f}")
    _un = [r["ratio_measured_to_predicted"] for r in blk["rows_by_g"]
           if not r["g_multiple_of_block"]]
    _al = [r["ratio_measured_to_predicted"] for r in blk["rows_by_g"]
           if r["g_multiple_of_block"]]
    M("BlockRatioUnalignedMax", f"{max(_un):.3f}" if _un else "n/a")
    M("BlockRatioAlignedMin", f"{min(_al):.3f}" if _al else "n/a")
    inflation = [r["block_density"] / r["element_density"] for r in blk["rows_by_g"]]
    M("BlockInflationMin", f"{min(inflation):.1f}")
    M("BlockInflationMax", f"{max(inflation):.1f}")
    # Inflation at the largest block size (rows_by_block): this is the number the
    # text quotes for "the same mask under a coarser kernel".
    _gran = [r["block_density"] / r["element_density"] for r in blk["rows_by_block"]]
    M("BlockInflationGranMax", f"{max(_gran):.1f}")
    M("BlockInflationGranMin", f"{min(_gran):.1f}")
    # Executed density of the reference mask, on both denominators, in percent.
    _atb = [r for r in blk["rows_by_block"] if r["block"] == blk["block"]]
    M("BlockDensCausalAtB",
      f"{100 * _atb[0]['density_over_causal']:.1f}" if _atb else "n/a")
    M("BlockDensCausalSatPct",
      f"{100 * max(r['density_over_causal'] for r in blk['rows_by_block']):.0f}")
    M("BlockDensGridAtB",
      f"{100 * _atb[0]['block_density']:.1f}" if _atb else "n/a")
    # Inflation at the reference setting (B = block, g = align_g_fixed).
    M("BlockInflationAtRef",
      f"{_atb[0]['block_density'] / _atb[0]['element_density']:.1f}" if _atb else "n/a")
    M("BlockElemDensPct",
      f"{100 * _atb[0]['element_density']:.2f}" if _atb else "n/a")
    # The two rows the withdrawn-lemma paragraph quotes (block density of the
    # unaligned period g=96 against the aligned period g=32 at B=64).
    def _bd(gg):
        r = [x for x in blk["rows_by_g"] if x["g"] == gg]
        return f"{r[0]['block_density']:.3f}" if r else "n/a"
    M("WithdrawDensUnaligned", _bd(96))    # g % B != 0, mid period
    M("WithdrawDensAligned", _bd(192))     # g % B == 0, long period
    M("WithdrawDensDensest", _bd(32))      # g < B, densest row in the grid
    big = [r for r in blk["rows_by_g"] if r["block_ge_g"]]
    M("BlockGeGCount", str(len(big)))

    # --------------------------------------------------------------- device
    dev = gpu["bench"]["device"]
    M("GpuName", dev["name"].replace("NVIDIA ", ""))
    M("GpuMemGiB", f"{dev['total_memory_bytes'] / 2 ** 30:.1f}")
    M("GpuCapability", dev["capability"])
    M("TorchVer", dev["torch"])
    M("CudaVer", dev["cuda_build"])
    M("GpuReps", str(gpu["bench"]["rows"][0]["modes"]["pswa"]["reps"]))
    M("GpuW", str(gpu["bench"]["w"]))
    M("GpuG", str(gpu["bench"]["g"]))
    M("GpuBlock", str(gpu["bench"]["block"]))
    cl = gpu["claims"]
    M("VramVsSdpaPct", f"{abs(cl['vram_reduction_at_largest_n_vs_sdpa']) * 100:,.0f}"
                        .replace(",", "{,}"))
    M("VramRatioVsSdpa", f"{-cl['vram_reduction_at_largest_n_vs_sdpa'] + 1:.1f}")
    M("LatencyRatioVsSdpa", f"{cl['latency_speedup_at_largest_n_vs_sdpa']:.3f}")
    M("LatencySlowdownVsSdpa", f"{1 / cl['latency_speedup_at_largest_n_vs_sdpa']:.1f}")
    M("VramVsMatPct", pct(cl["max_vram_reduction_vs_materialised"], 1))
    M("LargestN", fmt(cl["largest_n_with_both_measured"]))
    sl = gpu["slopes"]
    M("SlopeDense", f"{sl['full_materialised_transient']:.2f}")
    M("SlopeSdpa", f"{sl['full_sdpa_transient']:.2f}")
    M("SlopeLocal", f"{sl['longformer_global_transient']:.2f}")
    M("SlopeSliding", f"{sl['sliding_transient']:.2f}")
    M("SlopePswa", f"{sl['pswa_transient']:.2f}")
    M("SlopeStream", f"{sl['pswa_stream_transient']:.2f}")
    M("SlopeDenseFull", f"{sl['full_materialised_transient_full_range']:.2f}")
    M("SlopePswaFull", f"{sl['pswa_transient_full_range']:.2f}")
    M("SlopeStreamFull", f"{sl['pswa_stream_transient_full_range']:.2f}")

    byn = {r["n"]: r["modes"] for r in gpu["bench"]["rows"]}
    big_n = max(byn)
    from experiments import attention_bench as _ab0
    _b0 = _ab0.block_pattern(big_n, gpu["bench"]["w"], gpu["bench"]["g"],
                             gpu["bench"]["s"], "pswa", gpu["bench"]["block"]).astype(bool)
    _pad = _b0.sum(axis=1).max() / (_b0.sum() / _b0.shape[0])
    mb = byn[big_n]

    def cell(mode, field):
        v = mb.get(mode, {})
        return v.get(field) if v.get("status") == "ok" else None

    M("VramPswaBig", fmt(cell("pswa", "transient_mib"), 1))
    M("VramSdpaBig", fmt(cell("full_sdpa", "transient_mib"), 2))
    M("VramLocalBig", fmt(cell("longformer_global", "transient_mib"), 1))
    M("VramSlidingBig", fmt(cell("sliding", "transient_mib"), 1))
    M("VramStreamBig", fmt(cell("pswa_stream", "transient_mib"), 1))
    M("LatPswaBig", fmt(cell("pswa", "p50_ms"), 1))
    M("LatSdpaBig", fmt(cell("full_sdpa", "p50_ms"), 1))
    M("LatLocalBig", fmt(cell("longformer_global", "p50_ms"), 1))
    M("LatSlidingBig", fmt(cell("sliding", "p50_ms"), 1))
    M("LatStreamBig", fmt(cell("pswa_stream", "p50_ms"), 1))
    M("VramAfterDepad", f"{cell('pswa', 'transient_mib') / _pad:.0f}")
    M("LatAfterDepad", f"{cell('pswa', 'p50_ms') / _pad:.0f}")
    M("VramTimesSdpa", f"{cell('pswa', 'transient_mib') / cell('full_sdpa', 'transient_mib'):.0f}")
    M("MemLocalRatio", f"{cell('pswa', 'transient_mib') / cell('longformer_global', 'transient_mib'):.1f}")
    M("LatLocalRatio", f"{cell('pswa', 'p50_ms') / cell('longformer_global', 'p50_ms'):.1f}")
    M("LatSlidingRatio", f"{cell('full_sdpa', 'p50_ms') / cell('sliding', 'p50_ms'):.1f}")
    M("LatStreamRatio", f"{cell('pswa_stream', 'p50_ms') / cell('full_sdpa', 'p50_ms'):.1f}")
    M("OomN", fmt(min(cl["n_where_dense_oom_but_pswa_runs"])))

    # padding factor of the batched sparse kernel at the largest length
    from experiments import attention_bench as _ab
    _bp = _ab.block_pattern(big_n, gpu["bench"]["w"], gpu["bench"]["g"],
                            gpu["bench"]["s"], "pswa", gpu["bench"]["block"]).astype(bool)
    _mean = _bp.sum() / _bp.shape[0]
    M("PadFactor", f"{_bp.sum(axis=1).max() / _mean:.1f}")
    M("MeanSelBlocks", f"{_mean:.0f}")
    M("MaxSelBlocks", str(int(_bp.sum(axis=1).max())))
    M("NBlocks", str(_bp.shape[0]))
    # Work inflation at the *device* setting (B, g), measured block_alignment row,
    # not the hand-derived B/2: at n = align_n the window term still inflates the
    # element density, so the measured factor sits between B/2 and B.
    _dev_row = [r for r in blk["rows_by_g"] if r["g"] == gpu["bench"]["g"]]
    M("BlockWorkInflation",
      f"{_dev_row[0]['block_density'] / _dev_row[0]['element_density']:.1f}"
      if _dev_row else f"{gpu['bench']['block'] / 2:.0f}")

    # ---------------------------------------------------------- tabular main
    systems = {r["system"]: r for r in main_t["rows"]}
    for key, tag in (("linear", "Linear"), ("gbdt", "Gbdt"),
                     ("gbdt_strong", "Strong"), ("cascade", "Cascade")):
        r = systems[key]
        M(f"AucPr{tag}", f"{r['auc_pr']:.4f}")
        M(f"AucRoc{tag}", f"{r['auc_roc']:.4f}")
        M(f"Recall{tag}", pct(r["recall_at_fpr_1pct"], 1))
        M(f"Loss{tag}", fmt(r["expected_loss"]))
        M(f"Cost{tag}", f"{r['cost_per_tx']:.2f}")
    M("MainTrainRows", fmt(main_t["n_train"]))
    M("MainTestRows", fmt(main_t["n_test"]))
    M("MainFeatures", str(main_t["n_features"]))
    M("UpgradeRate", pct(systems["cascade"]["upgrade_rate_actual"], 1))

    dl = infer["delong_gbdt_vs_linear"]
    M("DelongDelta", f"{dl['delta']:.4f}")
    M("DelongCiLo", f"{dl['ci95'][0]:.4f}")
    M("DelongCiHi", f"{dl['ci95'][1]:.4f}")
    M("DelongZ", f"{dl['z']:.2f}")
    mc = infer["mcnemar"]
    M("McNemarB", str(mc["b"]))
    M("McNemarC", str(mc["c"]))
    M("McNemarDiscordant", str(mc["n_discordant"]))
    M("McNemarP", f"{mc['p_exact']:.2e}".replace("e-", r"\times10^{-").replace("e+", r"\times10^{") + "}")
    M("ClusterCiLo", f"{infer['cluster_bootstrap']['ci95'][0]:.4f}")
    M("ClusterCiHi", f"{infer['cluster_bootstrap']['ci95'][1]:.4f}")
    M("BlockCiLo", f"{infer['block_bootstrap']['ci95'][0]:.4f}")
    M("BlockCiHi", f"{infer['block_bootstrap']['ci95'][1]:.4f}")
    M("RequiredSeeds", str(infer["power"]["required_seeds"]))
    M("SeedMean", f"{seeds['auc_pr_mean']:.4f}")
    M("SeedSd", f"{seeds['auc_pr_sd']:.4f}")
    M("SeedN", str(seeds["n_seeds"]))

    for name, tag in (("gbdt_vs_linear", "GvL"), ("cascade_vs_gbdt", "CvG")):
        p = perm["pairs"][name]
        M(f"PermDelta{tag}", f"{p['permutation']['observed_delta']:.4f}")
        M(f"PermN{tag}", str(p["permutation"]["n_perm"]))
        M(f"PermCiLo{tag}", f"{p['bootstrap_entities']['delta_auc_pr_ci95'][0]:.4f}")
        M(f"PermCiHi{tag}", f"{p['bootstrap_entities']['delta_auc_pr_ci95'][1]:.4f}")

    M("PermSamples", str(perm["pairs"]["gbdt_vs_linear"]["permutation"]["n_perm"]))
    cr = {r["cost_ratio_fn_fp"]: r for r in cost["rows"]}
    M("CostRatioMax", str(max(cr)))
    M("CostAtOne", f"{cr[1]['test_cost_per_tx']:.4f}")
    M("CostAtHundred", f"{cr[100]['test_cost_per_tx']:.4f}")
    M("RecallAtOne", pct(cr[1]["test_recall"], 1))
    M("RecallAtHundred", pct(cr[100]["test_recall"], 1))
    M("FprAtHundred", pct(cr[100]["test_fpr"], 1))

    M("BrierPre", f"{cal['brier_raw']:.4f}")
    M("BrierPost", f"{cal['brier_calibrated']:.4f}")
    M("EcePre", f"{cal['ece_raw']:.4f}")
    M("EcePost", f"{cal['ece_calibrated']:.4f}")
    M("PlattImproves", "true" if cal["claims"]["platt_reduces_brier"] else "false")
    M("CohensD", f"{eff['cohens_d_positive_class_scores_gbdt_vs_linear']:.4f}")
    M("CliffsDelta", f"{eff['cliffs_delta_positive_class_scores']:.4f}")

    # ------------------------------------------------------------------ LLM
    c0 = llm0["elicitation"]["choice"]
    M("LlmAucPr", f"{c0['llm']['auc_pr']:.4f}")
    M("LlmAucRoc", f"{c0['llm']['auc_roc']:.4f}")
    M("LlmFOne", f"{c0['llm']['f1']:.4f}")
    M("LlmRecall", pct(c0["llm"]["recall_at_fpr_1pct"], 1))
    M("GbdtSameAucPr", f"{c0['gbdt_same_rows']['auc_pr']:.4f}")
    M("GbdtSameAucRoc", f"{c0['gbdt_same_rows']['auc_roc']:.4f}")
    M("GbdtSameFOne", f"{c0['gbdt_same_rows']['f1']:.4f}")
    M("GbdtSameRecall", pct(c0["gbdt_same_rows"]["recall_at_fpr_1pct"], 1))
    M("LlmDelta", f"{c0['delta_auc_pr_gbdt_minus_llm']:.4f}")
    M("LlmCiLo", f"{c0['bootstrap']['delta_auc_pr_ci95'][0]:.4f}")
    M("LlmCiHi", f"{c0['bootstrap']['delta_auc_pr_ci95'][1]:.4f}")
    M("LlmRows", fmt(c0["n_rows"]))
    M("LlmPositives", str(c0["n_positives"]))
    lat = llm0["elicitation"]["choice_latency"]
    M("LlmMsRow", f"{lat['p50_ms_per_row']:.2f}")
    M("LlmRowsPerS", f"{lat['throughput_rows_per_s']:.2f}")
    jl = llm0["json_interface"]
    M("JsonMsRow", f"{jl['latency']['p50_ms_per_row']:.2f}")
    M("JsonRowsPerS", f"{jl['latency']['throughput_rows_per_s']:.2f}")
    M("InterfaceRatio", f"{jl['latency']['p50_ms_per_row'] / lat['p50_ms_per_row']:.1f}")
    M("ParseFailZero", pct(jl["parse_failure_rate"], 1))
    M("ParseFailRows", str(jl["parse_failures"]))
    M("ParseFailDenom", str(jl["n_rows"]))
    M("VramZeroShot", f"{llm0['vram']['peak_bytes'] / 2 ** 30:.2f}")

    c1 = sft["elicitation"]["choice"]
    M("SftAucPr", f"{c1['llm']['auc_pr']:.4f}")
    M("SftAucRoc", f"{c1['llm']['auc_roc']:.4f}")
    M("SftFOne", f"{c1['llm']['f1']:.4f}")
    M("SftRecall", pct(c1["llm"]["recall_at_fpr_1pct"], 1))
    M("SftDelta", f"{sft['delta_auc_pr_vs_base']:.4f}")
    M("SftParseFail", pct(sft["json_interface"]["parse_failure_rate"], 1))
    M("SftRows", fmt(sft["n_train_rows"]))
    M("SftSeconds", f"{sft['sft']['train_seconds']:.0f}")
    M("SftLoss", f"{sft['sft']['train_loss']:.3f}")
    M("SftTrainable", fmt(sft["sft"]["trainable_params"]))
    M("SftTrainablePct", pct(sft["sft"]["trainable_fraction"], 2))
    M("SftVram", f"{sft['sft']['peak_vram_bytes'] / 2 ** 30:.2f}")
    ts = sft["target_stats"]
    M("TargetMean", f"{sft['target_stats']['mean']:.4f}")
    M("TargetSd", f"{ts['sd']:.4f}")
    M("TargetDistinct", str(ts["n_distinct"]))
    M("TargetMin", f"{ts['min']:.5f}")
    M("TargetMax", f"{ts['max']:.4f}")
    sc = {r["model"]: r for r in scale["rows"]}
    small = sc["qwen2.5-0.5b-instruct"]
    large = sc["qwen2.5-1.5b-instruct"]
    M("SmallParams", f"{small['params'] / 1e9:.2f}")
    M("SmallVram", f"{small['generation_peak_gib']:.2f}")
    M("SmallAucPr", f"{small['auc_pr']:.4f}")
    M("SmallAucRoc", f"{small['auc_roc']:.4f}")
    M("SmallMsRow", f"{small['latency']['p50_ms_per_row']:.2f}")
    M("LargeParams", f"{large['params'] / 1e9:.2f}")
    M("LargeVram", f"{large['generation_peak_gib']:.2f}")
    M("LargeAucPr", f"{large['auc_pr']:.4f}")
    M("LargeAucRoc", f"{large['auc_roc']:.4f}")
    M("LargeMsRow", f"{large['latency']['p50_ms_per_row']:.2f}")
    M("ScaleRows", str(small['n_rows']))
    M("ScaleSlope", f"{scale['fit']['slope_gib_per_billion_params']:.2f}")
    M("ScalePoints", str(scale["fit"]["n_points"]))

    # ------------------------------------------------------------------ gas
    M("GasPerTx", fmt(gas["per_tx_log_gas_mean"]))
    M("GasPerTxSd", f"{gas['per_tx_log_gas_std']:.1f}")
    M("GasBatch", fmt(gas["batch_root_gas"]))
    M("GasVerify", fmt(gas["verify_inclusion_gas"]))
    M("GasDeploy", fmt(gas["deploy_gas"]))
    M("GasSavingPct", pct(gas["gas_saving_fraction"], 2))
    M("GasBatchSize", fmt(gas["batch_size"]))
    M("MerkleDepth", str(gas["merkle_depth"]))
    M("SolcVer", gas["solc"])

    # ---------------------------------------------------------------- drift
    M("IeeeDrift", f"{drift_i['auc_pr_first_to_last_delta']:+.4f}")
    M("BafDrift", f"{drift_b['auc_pr_first_to_last_delta']:+.4f}")
    M("BafPrevLo", pct(drift_b["slices"][0]["fraud_rate"], 2))
    M("BafPrevHi", pct(drift_b["slices"][-1]["fraud_rate"], 2))
    M("BafAucPr", f"{transfer['metrics']['auc_pr']:.4f}")
    M("BafAucRoc", f"{transfer['metrics']['auc_roc']:.4f}")
    M("BafRecall", pct(transfer["metrics"]["recall_at_fpr_1pct"], 1))
    M("BafFeatures", str(transfer["n_features"]))
    gaps = fair["reports"]
    worst = max(gaps, key=lambda r: r["tpr_gap"])
    M("FairAttr", worst["group_col"].replace("_", r"\_"))
    M("FairTprGap", f"{worst['tpr_gap']:.4f}")
    M("FairFprGap", f"{worst['fpr_gap']:.4f}")
    M("FairGroups", str(len(worst["groups"])))
    M("FairMinTprGap", f"{min(g['tpr_gap'] for g in gaps):.4f}")
    M("FairAttrs", str(len(gaps)))

    # ------------------------------------------------------- learning curve
    lc = {r["n_train"]: r["auc_pr"] for r in curve["rows"]}
    M("CurveSmallAucPr", f"{lc[min(lc)]:.4f}")
    M("CurveLargeAucPr", f"{lc[max(lc)]:.4f}")
    M("CurveSizes", str(len(lc)))
    lad = ladder["rows"]
    M("LadderSmallestAucPr", f"{lad[0]['auc_pr']:.4f}")
    M("LadderLargestAucPr", f"{lad[-1]['auc_pr']:.4f}")
    M("LadderSmall", str(lad[0]["n_fields"]))
    M("LadderLarge", str(lad[-1]["n_fields"]))
    M("CrossoverN", fmt(curve["crossover_n_train"]))

    # ------------------------------------------------------------- challenge
    M("ChallengeCells", str(len(chal["rows"])))
    rq = chal["required_q_for_delta_0.01"]
    M("ChallengeQ", f"{max(rq.values()):.4f}")
    _f02 = sorted([r for r in chal["rows"] if abs(r["f"] - 0.2) < 1e-9],
                  key=lambda r: r["q"])
    M("ChalDetLow", f"{_f02[0]['detection_rate']:.3f}")
    M("ChalDetHigh", f"{_f02[-1]['detection_rate']:.3f}")
    M("ChalTrials", str(_f02[0]["trials"]))
    M("ChalN", fmt(_f02[0]["N"]))
    M("ChalEscapeObs", f"{_f02[0]['empirical_escape']:.3f}")
    M("ChalEscapeTheory", f"{_f02[0]['theory_hypergeometric']:.3f}")
    M("BootstrapB", str(infer["cluster_bootstrap"]["B"]))
    M("BootstrapBlocks", str(infer["block_bootstrap"]["n_blocks"]))
    M("PowerDelta", f"{infer['power']['delta']:.4f}")
    M("ChallengeQSmall", f"{min(rq.values()):.4f}")
    M("DistinctScores", fmt(cal["n_distinct_test_scores"]))

    # ------------------------------------------------------------- emit tex
    lines = ["% AUTO-GENERATED by paper_aij/build_assets.py -- do not edit.",
             f"% source run: {rid}", ""]
    for k in sorted(MACROS):
        lines.append(f"\\newcommand{{\\{k}}}{{{MACROS[k]}}}")
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------------------------------------------------------------- tables
    write_tables(OUT / "tables", rows, blk, gpu, systems, cr, llm0, sft, cost)
    (OUT / "assets.json").write_text(json.dumps(
        {"run_id": rid, "macros": MACROS}, indent=1), encoding="utf-8")
    print(f"run {rid}: {len(MACROS)} macros, tables written")
    return 0


# ------------------------------------------------------------------- tables --
def write_tables(tdir: Path, nnz_rows, blk, gpu, systems, cr, llm0, sft, cost) -> None:
    tdir.mkdir(parents=True, exist_ok=True)

    # T1: counting law
    L = [r"\begin{tabular}{rrrrrr}", r"\toprule",
         r"$n$ & measured & Eq.~\eqref{eq:entrylaw} & rel.\ err & asymptotic & density \\",
         r"\midrule"]
    for n, meas, closed, asym, dens, dense in nnz_rows:
        L.append(f"{n:,}".replace(",", "{,}") + " & " +
                 f"{meas:,}".replace(",", "{,}") + " & " +
                 f"{closed:,}".replace(",", "{,}") + " & " +
                 f"{(closed - meas) / meas * 100:+.3f}\\% & " +
                 f"{asym:,.0f}".replace(",", "{,}") + " & " +
                 f"{100 * dens:.2f}\\% \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (tdir / "tab_counting.tex").write_text("\n".join(L) + "\n", encoding="utf-8")

    # T2: density law as a function of the anchor period (block size fixed)
    L = [r"\begin{tabular}{rrrrrr}", r"\toprule",
         r"$g$ & aligned & $\min(1,B/g)$ & window term & sum & measured \\",
         r"\midrule"]
    for r in blk["rows_by_g"]:
        anchor = r["predicted_density_block_over_g"]
        wterm = r["window_contribution"]
        L.append(f"{r['g']} & " + ("yes" if r["g_multiple_of_block"] else "no") + " & "
                 + f"{anchor:.3f} & {wterm:.3f} & {anchor + wterm:.3f} & "
                 + f"{r['density_over_causal']:.3f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (tdir / "tab_block.tex").write_text("\n".join(L) + "\n", encoding="utf-8")

    # T2b: granularity inflation (anchor period fixed, block size varies)
    # Both denominators are stated, because "executed density" is ambiguous:
    # block_density is a fraction of ALL block pairs (the whole grid, as the
    # kernel sees it), density_over_causal is a fraction of the causal block
    # pairs (what the law predicts). Inflation is block_density/element_density
    # and element_density is |S|/n^2, so numerator and denominator in the last
    # column are both over the whole grid and the ratio is the work multiplier.
    L = [r"\begin{tabular}{rrrrrr}", r"\toprule",
         r"$B$ & aligned & block pairs & causal blocks & elements & inflation \\",
         r"\midrule"]
    for r in blk["rows_by_block"]:
        L.append(f"{r['block']} & " + ("yes" if r["g_multiple_of_block"] else "no") + " & "
                 + f"{100 * r['block_density']:.2f}\\% & "
                 + f"{100 * r['density_over_causal']:.2f}\\% & "
                 + f"{100 * r['element_density']:.3f}\\% & "
                 + f"{r['block_density'] / r['element_density']:.1f}$\\times$ \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (tdir / "tab_granularity.tex").write_text("\n".join(L) + "\n", encoding="utf-8")

    # T3: device
    modes = [("full_materialised", "dense materialised"),
             ("full_sdpa", "fused SDPA"),
             ("sliding", "local window"),
             ("longformer_global", "local + sinks"),
             ("pswa", "window + anchors"),
             ("pswa_stream", "window + anchors, streamed")]
    L = [r"\begin{tabular}{r" + "rr" * len(modes) + "}", r"\toprule",
         r"$n$ & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{lab}}}" for _, lab in modes)
         + r" \\",
         " & " + " & ".join(["MiB & ms"] * len(modes)) + r" \\", r"\midrule"]
    for row in gpu["bench"]["rows"]:
        cells = []
        for key, _ in modes:
            m = row["modes"].get(key, {})
            if m.get("status") == "ok":
                mib = f"{m['transient_mib']:,.2f}".replace(",", "{,}")
                cells += [mib, f"{m['p50_ms']:.1f}"]
            else:
                cells += [m.get("status", "-"), "-"]
        L.append(f"{row['n']:,}".replace(",", "{,}") + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (tdir / "tab_device.tex").write_text("\n".join(L) + "\n", encoding="utf-8")

    # T4: main tabular
    L = [r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"system & AUC-PR & AUC-ROC & R@FPR$=1\%$ & loss & cost/tx \\", r"\midrule"]
    labels = {"linear": "linear (logistic)",
              "gbdt": "GBDT gate",
              "gbdt_strong": "GBDT strong",
              "cascade": "cascade (gate+strong)"}
    for k in ("linear", "gbdt", "gbdt_strong", "cascade"):
        r = systems[k]
        L.append(f"{labels[k]} & "
                 f"{r['auc_pr']:.4f} & {r['auc_roc']:.4f} & "
                 f"{100 * r['recall_at_fpr_1pct']:.1f}\\% & "
                 f"{r['expected_loss']:,.0f}".replace(",", "{,}") + " & "
                 f"{r['cost_per_tx']:.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (tdir / "tab_main.tex").write_text("\n".join(L) + "\n", encoding="utf-8")

    # T5: cost curve
    L = [r"\begin{tabular}{rrrrr}", r"\toprule",
         r"$c_{\mathrm{FN}}/c_{\mathrm{FP}}$ & threshold & cost/tx & recall & FPR \\",
         r"\midrule"]
    for r in cost["rows"]:
        L.append(f"{r['cost_ratio_fn_fp']} & {r['threshold_from_validation']:.4f} & "
                 f"{r['test_cost_per_tx']:.4f} & {100 * r['test_recall']:.1f}\\% & "
                 f"{100 * r['test_fpr']:.2f}\\% \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (tdir / "tab_cost.tex").write_text("\n".join(L) + "\n", encoding="utf-8")

    # T6: LLM
    c0 = llm0["elicitation"]["choice"]
    c1 = sft["elicitation"]["choice"]
    jl = llm0["json_interface"]
    L = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"system & AUC-PR & AUC-ROC & R@FPR$=1\%$ & F1 & parse fail & ms/row \\",
         r"\midrule"]
    L.append(r"GBDT, same rows & " +
             f"{c0['gbdt_same_rows']['auc_pr']:.4f} & {c0['gbdt_same_rows']['auc_roc']:.4f} & "
             f"{100 * c0['gbdt_same_rows']['recall_at_fpr_1pct']:.1f}\\% & "
             f"{c0['gbdt_same_rows']['f1']:.4f} & -- & -- \\\\")
    L.append(r"0.5B, logit interface & " +
             f"{c0['llm']['auc_pr']:.4f} & {c0['llm']['auc_roc']:.4f} & "
             f"{100 * c0['llm']['recall_at_fpr_1pct']:.1f}\\% & {c0['llm']['f1']:.4f} & "
             f"{100 * jl['parse_failure_rate']:.1f}\\% & "
             f"{llm0['elicitation']['choice_latency']['p50_ms_per_row']:.1f} \\\\")
    L.append(r"0.5B, JSON interface & -- & -- & -- & -- & "
             f"{100 * jl['parse_failure_rate']:.1f}\\% & "
             f"{jl['latency']['p50_ms_per_row']:.1f} \\\\")
    L.append(r"0.5B + LoRA (" + f"{sft['n_train_rows']}" + r" rows) & " +
             f"{c1['llm']['auc_pr']:.4f} & {c1['llm']['auc_roc']:.4f} & "
             f"{100 * c1['llm']['recall_at_fpr_1pct']:.1f}\\% & {c1['llm']['f1']:.4f} & "
             f"{100 * sft['json_interface']['parse_failure_rate']:.1f}\\% & "
             f"{sft['elicitation']['choice_latency']['p50_ms_per_row']:.1f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (tdir / "tab_llm.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
