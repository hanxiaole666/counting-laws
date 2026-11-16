"""Independent audit of the numbers that appear in the manuscript.

``build_assets.py`` derives every macro from a result file.  This script
re-derives a curated subset by a *different* code path (direct JSON traversal,
no shared helper) and diffs the result against ``numbers.tex``.  If the two
disagree, either the generator or the paper is wrong; either way the build
should stop.

Usage:  python paper_aij/check_numbers.py [-v]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def latest() -> Path:
    return sorted([p for p in (ROOT / "results").iterdir()
                   if p.is_dir() and (p / "RUN_METADATA.json").exists()])[-1]


def macros() -> dict[str, str]:
    txt = (HERE / "numbers.tex").read_text(encoding="utf-8")
    out = {}
    for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", txt, re.M):
        out[m.group(1)] = m.group(2)
    return out


def main(argv: list[str]) -> int:
    run = latest()
    M = macros()

    def j(rel: str) -> dict:
        return json.loads((run / f"{rel}.json").read_text(encoding="utf-8"))

    gpu = j("s1_pswa/gpu_memory_latency")
    blk = j("s1_pswa/block_alignment")
    llm = j("s8_llm/zero_shot_0_5b")
    sft = j("s8_llm/lora_sft_0_5b")
    gasc = j("s3_protocol/onchain_gas")
    main_t = {r["system"]: r for r in j("s5_baselines/main_table")["rows"]}
    qual = j("s1_pswa/quality_bound")["quality_bound"]
    cplx = j("s1_pswa/complexity_curve")
    infer = j("s4_stats/inference")

    # granularity terms, recomputed here (not imported from build_assets)
    def infl(r):
        return r["block_density"] / r["element_density"]

    gran = [infl(r) for r in blk["rows_by_block"]]
    at_b = next(r for r in blk["rows_by_block"] if r["block"] == blk["block"])
    caus = [r["density_over_causal"] for r in blk["rows_by_block"]]

    big = max(r["n"] for r in gpu["bench"]["rows"])
    bigrow = next(r for r in gpu["bench"]["rows"] if r["n"] == big)
    pswa = bigrow["modes"]["pswa"]
    sdpa = bigrow["modes"]["full_sdpa"]
    local = bigrow["modes"]["longformer_global"]
    sliding = bigrow["modes"]["sliding"]

    # (macro, independently recomputed value as it should be formatted)
    checks = [
        ("VramPswaBig", f"{pswa['transient_mib']:,.1f}".replace(",", "{,}")),
        ("VramSdpaBig", f"{sdpa['transient_mib']:.2f}"),
        ("VramLocalBig", f"{local['transient_mib']:,.1f}".replace(",", "{,}")),
        ("VramStreamBig", f"{stream['transient_mib']:,.1f}".replace(",", "{,}")
         if (stream := bigrow["modes"]["pswa_stream"]).get("status") == "ok" else ""),
        ("LatPswaBig", f"{pswa['p50_ms']:,.1f}".replace(",", "{,}")),
        ("LatSdpaBig", f"{sdpa['p50_ms']:,.1f}".replace(",", "{,}")),
        ("LatSlidingBig", f"{sliding['p50_ms']:,.1f}".replace(",", "{,}")),
        ("VramTimesSdpa", f"{pswa['transient_mib'] / sdpa['transient_mib']:.0f}"),
        ("LatencySlowdownVsSdpa",
         f"{pswa['p50_ms'] / sdpa['p50_ms']:.1f}"),
        ("LatSlidingRatio", f"{sdpa['p50_ms'] / sliding['p50_ms']:.1f}"),
        ("MemLocalRatio", f"{pswa['transient_mib'] / local['transient_mib']:.1f}"),
        ("LargestN", f"{big:,}".replace(",", "{,}")),
        ("SlopePswa", f"{gpu['slopes']['pswa_transient']:.2f}"),
        ("SlopeSdpa", f"{gpu['slopes']['full_sdpa_transient']:.2f}"),
        ("SlopeStream", f"{gpu['slopes']['pswa_stream_transient']:.2f}"),
        ("RhoMean", f"{qual['rho_mean']:.4f}"),
        ("EpsMean", f"{qual['eps_mean']:.4f}"),
        ("EmpDiff", f"{qual['empirical_max_abs_diff']:.3f}"),
        ("BoundMax", f"{qual['theoretical_bound_max']:.2f}"),
        ("GpuG", str(gpu["bench"]["g"])),
        ("GpuW", str(gpu["bench"]["w"])),
        ("GpuBlock", str(gpu["bench"]["block"])),
        ("AlignN", f"{blk['n']:,}".replace(",", "{,}")),
        ("LlmAucPr", f"{llm['elicitation']['choice']['llm']['auc_pr']:.4f}"),
        ("GbdtSameAucPr", f"{llm['elicitation']['choice']['gbdt_same_rows']['auc_pr']:.4f}"),
        ("LlmAucRoc", f"{llm['elicitation']['choice']['llm']['auc_roc']:.4f}"),
        ("SftAucRoc", f"{sft['elicitation']['choice']['llm']['auc_roc']:.4f}"),
        ("SftDelta", f"{sft['delta_auc_pr_vs_base']:.4f}"),
        ("LlmDelta", f"{llm['elicitation']['choice']['delta_auc_pr_gbdt_minus_llm']:.4f}"),
        ("ParseFailZero", f"{100 * llm['json_interface']['parse_failure_rate']:.1f}"),
        ("SftParseFail", f"{100 * sft['json_interface']['parse_failure_rate']:.1f}"),
        ("SftRows", f"{sft['n_train_rows']:,}".replace(",", "{,}")),
        ("AucPrCascade", f"{main_t['cascade']['auc_pr']:.4f}"),
        ("AucPrGbdt", f"{main_t['gbdt']['auc_pr']:.4f}"),
        ("GasPerTx", f"{gasc['per_tx_log_gas_mean']:,}".replace(",", "{,}")),
        ("GasSavingPct", f"{100 * gasc['gas_saving_fraction']:.2f}"),
        ("DelongDelta", f"{infer['delong_gbdt_vs_linear']['delta']:.4f}"),
        ("NnzSlopeFitted", f"{cplx['modes']['pswa']['loglog_slope']:.3f}"),
        # granularity table: three columns of Table 4 plus the two inflation
        # figures the prose quotes.
        ("BlockInflationGranMax", f"{max(gran):.1f}"),
        ("BlockInflationGranMin", f"{min(gran):.1f}"),
        ("BlockElemDensPct", f"{100 * at_b['element_density']:.2f}"),
        ("BlockDensCausalAtB", f"{100 * at_b['density_over_causal']:.1f}"),
        ("BlockDensCausalSatPct", f"{100 * max(caus):.0f}"),
        ("BlockInflationAtRef", f"{infl(at_b):.1f}"),
        ("BlockDensGridAtB", f"{100 * at_b['block_density']:.1f}"),
        ("BlockInflationMax", f"{max(infl(r) for r in blk['rows_by_g']):.1f}"),
        # the two rows quoted by the withdrawn-lemma paragraph, and the SFT
        # target mean, all recomputed from the per-row records
        ("WithdrawDensUnaligned",
         f"{next(r for r in blk['rows_by_g'] if r['g'] == 96)['block_density']:.3f}"),
        ("WithdrawDensAligned",
         f"{next(r for r in blk['rows_by_g'] if r['g'] == 192)['block_density']:.3f}"),
        ("WithdrawDensDensest",
         f"{next(r for r in blk['rows_by_g'] if r['g'] == 32)['block_density']:.3f}"),
        ("TargetMean", f"{sft['target_stats']['mean']:.4f}"),
        ("TargetDistinct", str(sft['target_stats']['n_distinct'])),
    ]

    bad = []
    for name, expected in checks:
        got = M.get(name)
        if got is None:
            bad.append((name, expected, "MISSING"))
        elif got != expected:
            bad.append((name, expected, got))

    # the closed form must reproduce the measured entry count exactly
    w, g, s = cplx["w"], cplx["g"], cplx["s"]
    for pt in cplx["modes"]["pswa"]["curve"]:
        n = pt["n"]
        closed = w * (w + 1) // 2 + sum(
            w + min(s, i - w + 1) + (i - w) // g for i in range(w, n))
        if closed != pt["nnz"]:
            bad.append((f"entry_law_n{n}", closed, pt["nnz"]))

    if bad:
        print(f"FAIL: {len(bad)} mismatch(es) against {run.name}")
        for name, exp, got in bad:
            print(f"  {name:22s} expected {exp!s:>12}  found {got!s:>12}")
        return 1

    print(f"PASS: {len(checks)} macros and "
          f"{len(cplx['modes']['pswa']['curve'])} entry-law points agree "
          f"with {run.name}")
    if "-v" in argv:
        for name, expected in checks[:8]:
            print(f"  {name:22s} {expected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
