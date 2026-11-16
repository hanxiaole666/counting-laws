"""S7: figures and tables with a strict figure<-data provenance chain.

This replaces scripts/06_make_figures.py, which silently emitted hardcoded
``EXAMPLE DATA`` plots when real results were missing. Here:
  * every required input must exist, otherwise SystemExit (fail-fast);
  * any input with ``mock: true`` is refused;
  * every figure is registered with the sha256 of its inputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import FIG_DIR, ROOT, atomic_write_json, atomic_write_text, sha256_file

REQUIRED = {
    "complexity": "s1_pswa/complexity_curve.json",
    "quality": "s1_pswa/quality_bound.json",
    "modes": "s1_pswa/mode_comparison.json",
    "pswa_scan": "s1_pswa/long_sequence_scan.json",
    "fidelity": "s2_gbl/fidelity_ablation.json",
    "challenge": "s3_protocol/challenge_grid.json",
    "predictable": "s3_protocol/predictable_attack.json",
    "seeds": "s4_stats/seed_sweep.json",
    "inference": "s4_stats/inference.json",
    "calibration": "s4_stats/calibration.json",
    "effect": "s4_stats/effect_sizes.json",
    "main": "s5_baselines/main_table.json",
    "curve": "s5_baselines/learning_curve.json",
    "ladder": "s5_baselines/field_ladder.json",
    "baf": "s5_baselines/baf_transfer.json",
    "drift_ieee": "s6_drift/ieee_temporal.json",
    "drift_baf": "s6_drift/baf_monthly.json",
    "fairness": "s6_drift/fairness.json",
    "errors": "s6_drift/error_stratification.json",
    # measured on this host with CUDA torch + local Qwen2.5 weights
    "gpu": "s1_pswa/gpu_memory_latency.json",
    "align": "s1_pswa/block_alignment.json",
    "llm0": "s8_llm/zero_shot_0_5b.json",
    "llmsft": "s8_llm/lora_sft_0_5b.json",
    "llmscale": "s5_baselines/llm_scale_scan.json",
    "gas": "s3_protocol/onchain_gas.json",
    "perm": "s4_stats/permutation_inference.json",
    "cost": "s4_stats/cost_sensitivity.json",
}

PLOT_SPECS = [
    ("fig1_complexity.png", ["complexity"], "Attention score entries vs sequence length"),
    ("fig2_challenge.png", ["challenge", "predictable"], "Challenge detection vs sampling rate"),
    ("fig3_learning_curve.png", ["curve"], "AUC-PR vs labelled training rows"),
    ("fig4_seed_sweep.png", ["seeds"], "AUC-PR across training seeds"),
    ("fig5_field_ladder.png", ["ladder"], "AUC-PR vs number of input fields"),
    ("fig6_drift.png", ["drift_ieee", "drift_baf"], "Temporal drift on two datasets"),
    ("fig7_representation.png", ["fidelity"], "Field-aligned surface forms"),
    ("fig8_attention_modes.png", ["modes", "quality"], "Sparse attention modes"),
    ("fig9_pswa_scan.png", ["pswa_scan"], "PSWA memory gain vs retained attention mass"),
    ("fig10_gpu_vram.png", ["gpu"], "MEASURED peak VRAM on the RTX 2060 (torch)"),
    ("fig11_block_alignment.png", ["align"], "Block density vs anchor period"),
    ("fig12_llm_vs_gbdt.png", ["llm0", "llmsft", "main"], "Local LLM vs tabular baseline"),
    ("fig13_onchain_gas.png", ["gas"], "EVM gas: per-tx log vs one Merkle root"),
    ("fig14_cost_curve.png", ["cost"], "Expected cost vs cost ratio"),
]


def _load_all(results_dir: Path) -> tuple[dict, dict]:
    from .common import load_json
    data, hashes = {}, {}
    missing = []
    for tag, rel in REQUIRED.items():
        p = results_dir / rel
        if not p.exists():
            missing.append(rel)
            continue
        d = load_json(p)
        if d.get("mock", False):
            raise SystemExit(f"FAIL-FAST: mock result fed into a paper figure: {rel}")
        data[tag] = d
        hashes[rel] = sha256_file(p)
    if missing:
        raise SystemExit(
            "FAIL-FAST: real results are missing; refusing to fabricate plots.\n"
            "Missing:\n  " + "\n  ".join(missing)
            + "\nRun: python -m experiments.run_all")
    return data, hashes


def _save(fig, name: str) -> str:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def build_plots(data: dict) -> dict:
    made = {}
    # Fig 1: complexity
    c = data["complexity"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for mode in ("full", "sliding", "longformer_global", "nsa_like", "pswa"):
        curve = c["modes"][mode]["curve"]
        ax.plot([r["n"] for r in curve], [r["nnz"] for r in curve],
                marker="o", label=mode)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("sequence length n"); ax.set_ylabel("attention score entries")
    ax.set_title(f"PSWA complexity (w={c['w']}, g={c['g']}, s={c['s']})")
    ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")
    made["fig1_complexity.png"] = _save(fig, "fig1_complexity.png")

    # Fig 2: challenge detection
    rows = data["challenge"]["rows"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for f in sorted({r["f"] for r in rows}):
        rs = sorted([r for r in rows if r["f"] == f], key=lambda r: r["q"])
        ax.plot([r["q"] for r in rs], [r["detection_rate"] for r in rs],
                marker="o", label=f"empirical f={f}")
        ax.plot([r["q"] for r in rs],
                [1 - r["theory_hypergeometric"] for r in rs],
                linestyle="--", alpha=0.6, label=f"theory f={f}")
    ax.set_xscale("log"); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("challenge sampling rate q"); ax.set_ylabel("detection probability")
    ax.set_title("Commitment + unpredictable challenge recomputation")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    made["fig2_challenge.png"] = _save(fig, "fig2_challenge.png")

    # Fig 3: learning curve
    rows = data["curve"]["rows"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for m in sorted({r["model"] for r in rows}):
        rs = sorted([r for r in rows if r["model"] == m], key=lambda r: r["n_train"])
        ax.plot([r["n_train"] for r in rs], [r["auc_pr"] for r in rs],
                marker="s", label=m)
    cross = data["curve"]["crossover_n_train"]
    if cross:
        ax.axvline(cross, color="grey", linestyle=":",
                   label=f"crossover n={cross}")
    ax.set_xscale("log"); ax.set_xlabel("labelled training rows")
    ax.set_ylabel("AUC-PR (held-out test)")
    ax.set_title("Learning curve (tabular models; LLM stage blocked)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    made["fig3_learning_curve.png"] = _save(fig, "fig3_learning_curve.png")

    # Fig 4: seed sweep
    rows = data["seeds"]["rows"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar([r["seed"] for r in rows], [r["auc_pr"] for r in rows], color="#4C72B0")
    ax.axhline(data["seeds"]["auc_pr_mean"], color="#C44E52", linestyle="--",
               label=f"mean={data['seeds']['auc_pr_mean']:.4f}")
    ax.set_xlabel("training seed"); ax.set_ylabel("AUC-PR")
    ax.set_title(f"{data['seeds']['n_seeds']} independent retraining seeds")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    made["fig4_seed_sweep.png"] = _save(fig, "fig4_seed_sweep.png")

    # Fig 5: field ladder
    rows = data["ladder"]["rows"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot([r["n_fields"] for r in rows], [r["auc_pr"] for r in rows], marker="o")
    ax.set_xlabel("number of numeric input fields"); ax.set_ylabel("AUC-PR")
    ax.set_title("Field ladder (same model, same split)"); ax.grid(alpha=0.3)
    made["fig5_field_ladder.png"] = _save(fig, "fig5_field_ladder.png")

    # Fig 6: drift
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for tag, label in (("drift_ieee", "IEEE-CIS"), ("drift_baf", "BAF")):
        s = data[tag]["slices"]
        ax.plot(range(len(s)), [x["auc_pr"] for x in s], marker="o", label=label)
    ax.set_xlabel("temporal slice (old -> new)"); ax.set_ylabel("AUC-PR")
    ax.set_title("Temporal drift on held-out test periods")
    ax.legend(); ax.grid(alpha=0.3)
    made["fig6_drift.png"] = _save(fig, "fig6_drift.png")

    # Fig 7: representation
    rows = data["fidelity"]["rows"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    names = [r["mode"] for r in rows]
    x = np.arange(len(names))
    ax.bar(x - 0.2, [r["tokens_per_event_mean"] for r in rows], width=0.4,
           label="tokens/event")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, [r["reverse_parse_recall"] for r in rows], width=0.4,
            color="#DD8452", label="reverse-parse recall")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("tokens / event"); ax2.set_ylabel("reverse-parse recall")
    ax.set_title("Field-aligned GBL-S surface forms (identical 10 fields)")
    ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)
    made["fig7_representation.png"] = _save(fig, "fig7_representation.png")

    # Fig 8: attention modes
    rows = data["modes"]["modes"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    names = [r["mode"] for r in rows]
    ax.bar(names, [r["nnz"] for r in rows], color="#55A868")
    ax.set_ylabel("retained attention entries"); ax.set_title(
        f"Sparse attention modes (n={data['modes']['n']})")
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    for i, r in enumerate(rows):
        ax.text(i, r["nnz"], f"rho={r['rho_mean']:.2f}", ha="center",
                va="bottom", fontsize=7, rotation=0)
    ax.grid(alpha=0.3, axis="y")
    made["fig8_attention_modes.png"] = _save(fig, "fig8_attention_modes.png")

    # Fig 9: PSWA n x w scan
    rows = data["pswa_scan"]["rows"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.0))
    for w in sorted({r["w"] for r in rows}):
        rs = sorted([r for r in rows if r["w"] == w], key=lambda r: r["n"])
        ax.plot([r["n"] for r in rs], [r["byte_reduction"] for r in rs],
                marker="o", label=f"w={w}")
        ax2.plot([r["n"] for r in rs], [r["eps_mean"] for r in rs],
                 marker="s", label=f"w={w}")
    ax.axhline(0.30, color="grey", linestyle=":", label="30% target")
    ax.set_xlabel("sequence length n"); ax.set_ylabel("peak tensor byte reduction")
    ax.set_title("Measured memory gain (host tensor bytes)")
    ax2.set_xlabel("sequence length n"); ax2.set_ylabel("discarded attention mass eps")
    ax2.set_title("Quality cost")
    for a in (ax, ax2):
        a.legend(fontsize=8); a.grid(alpha=0.3)
    made["fig9_pswa_scan.png"] = _save(fig, "fig9_pswa_scan.png")

    # Fig 10: MEASURED device VRAM and latency (this is the figure the original
    # manuscript faked with "55% VRAM reduction")
    g = data["gpu"]
    rows = g["bench"]["rows"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.2))
    for mode, style in (("full_materialised", "o-"), ("full_sdpa", "s-"),
                        ("pswa", "^-"), ("pswa_stream", "v-")):
        xs = [r["n"] for r in rows if r["modes"].get(mode, {}).get("status") == "ok"]
        ys = [max(r["modes"][mode]["transient_mib"], 1e-3) for r in rows
              if r["modes"].get(mode, {}).get("status") == "ok"]
        if xs:
            ax.plot(xs, ys, style, label=mode, ms=4)
            ax2.plot(xs, [r["modes"][mode]["p50_ms"] for r in rows
                          if r["modes"].get(mode, {}).get("status") == "ok"],
                     style, label=mode, ms=4)
    oom = [r["n"] for r in rows
           if r["modes"].get("full_materialised", {}).get("status") == "oom"]
    for n in oom:
        ax.axvline(n, color="#C44E52", linestyle=":", linewidth=1)
    if oom:
        ax.text(oom[0], ax.get_ylim()[1] * 0.5, " dense OOM", color="#C44E52", fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("sequence length n"); ax.set_ylabel("peak VRAM (MiB, measured)")
    ax.set_title(f"Peak device memory, {g['bench']['device']['name']}")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("sequence length n"); ax2.set_ylabel("p50 latency (ms, measured)")
    ax2.set_title("Latency per attention call")
    for a in (ax, ax2):
        a.legend(fontsize=7); a.grid(alpha=0.3, which="both")
    made["fig10_gpu_vram.png"] = _save(fig, "fig10_gpu_vram.png")

    # Fig 11: block density follows block/g, and dies when block >= g.
    # (This figure used to plot a claim that has since been withdrawn: an earlier
    # version said a non-multiple anchor period makes the pattern "dense".)
    rows = data["align"]["rows_by_g"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.0))
    x = np.array([r["block"] / r["g"] for r in rows])
    y = np.array([r["density_over_causal"] for r in rows])
    pred = np.minimum(1.0, x)
    al = np.array([r["g_multiple_of_block"] for r in rows])
    ax.plot([0, 1.05], [0, 1.05], "--", color="#888888", lw=1, label="y = block/g")
    ax.scatter(x[al], y[al], s=46, color="#4C72B0", label="g multiple of block", zorder=3)
    ax.scatter(x[~al], y[~al], s=70, facecolors="none", edgecolors="#C44E52",
               marker="s", linewidths=1.6, label="g NOT a multiple", zorder=4)
    ax.set_xlabel("block / g"); ax.set_ylabel("causal key blocks materialised")
    ax.set_title("Density follows block/g, aligned or not", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    lbl = [f"{r['g']}/{r['block']}" + ("" if r["g_multiple_of_block"] else "*") for r in rows]
    ax2.bar(np.arange(len(rows)), [r["ratio_measured_to_predicted"] for r in rows],
            color=["#4C72B0" if a else "#C44E52" for a in al])
    ax2.axhline(1.0, color="#333333", lw=1, ls="--")
    ax2.set_xticks(np.arange(len(rows))); ax2.set_xticklabels(lbl, fontsize=7, rotation=45)
    ax2.set_ylabel("measured / predicted"); ax2.set_ylim(0, 1.8)
    ax2.set_title("Residual = window term (all within 1.0–1.6×)", fontsize=10)
    ax2.grid(alpha=0.3, axis="y")
    fig.suptitle(f"Anchor period vs block granularity (n={data['align']['n']}, "
                 f"w={data['align']['w']})", fontsize=10)
    fig.tight_layout()
    made["fig11_block_alignment.png"] = _save(fig, "fig11_block_alignment.png")

    # Fig 12: local LLM vs tabular baseline on identical rows
    z = data["llm0"]["elicitation"]["choice"]
    s = data["llmsft"]["elicitation"]["choice"]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    names = ["GBDT\n(same rows)", "Qwen2.5-0.5B\nzero-shot", "Qwen2.5-0.5B\n+LoRA"]
    vals = [z["gbdt_same_rows"]["auc_pr"], z["llm"]["auc_pr"], s["llm"]["auc_pr"]]
    ax.bar(names, vals, color=["#4C72B0", "#DD8452", "#55A868"])
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("AUC-PR (held-out test)")
    ax.set_title(f"Local generative baseline, n={z['n_rows']} "
                 f"({z['n_positives']} positives)")
    ax.grid(alpha=0.3, axis="y")
    made["fig12_llm_vs_gbdt.png"] = _save(fig, "fig12_llm_vs_gbdt.png")

    # Fig 13: measured EVM gas
    gas = data["gas"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    labels = [f"per-tx LOG1\nx{gas['batch_size']}", "one Merkle root\nper batch"]
    vals = [gas["total_gas_per_tx_design"], gas["total_gas_batch_design"]]
    ax.bar(labels, vals, color=["#C44E52", "#55A868"])
    ax.set_yscale("log"); ax.set_ylabel("gas for one batch (measured on EVM)")
    ax.set_title(f"Gas: {gas['gas_saving_fraction']:.1%} saved "
                 f"(solc {gas['solc']}, py-evm)")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    made["fig13_onchain_gas.png"] = _save(fig, "fig13_onchain_gas.png")

    # Fig 14: cost sensitivity
    rows = data["cost"]["rows"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot([r["cost_ratio_fn_fp"] for r in rows],
            [r["test_cost_per_tx"] for r in rows], "o-")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("cost ratio c_fn : c_fp"); ax.set_ylabel("expected cost per transaction")
    ax.set_title("Cost sensitivity (threshold chosen on validation)")
    ax.grid(alpha=0.3, which="both")
    made["fig14_cost_curve.png"] = _save(fig, "fig14_cost_curve.png")
    return made


def build_tables(data: dict) -> str:
    L = ["# Tables (generated from results; no hand-typed numbers)", ""]
    L += ["## Table 1 · Main systems on IEEE-CIS held-out test", "",
          "| system | AUC-PR | AUC-ROC | Recall@FPR=1% | F1 | expected loss | cost/tx |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for r in data["main"]["rows"]:
        L.append(f"| {r['system']} | {r['auc_pr']:.4f} | {r['auc_roc']:.4f} | "
                 f"{r['recall_at_fpr_1pct']:.4f} | {r['f1']:.4f} | "
                 f"{r['expected_loss']:.0f} | {r['cost_per_tx']:.2f} |")
    L += ["", f"n_train={data['main']['n_train']}, n_test={data['main']['n_test']}, "
          f"fields={data['main']['n_features']}; {data['main']['threshold_policy']}.", ""]
    L += ["## Table 2 · Statistical protocol", "",
          "| quantity | value |", "|---|---:|"]
    inf = data["inference"]
    L += [
        f"| DeLong AUC(gbdt) - AUC(linear) | {inf['delong_gbdt_vs_linear']['delta']:+.4f} |",
        f"| DeLong 95% CI | [{inf['delong_gbdt_vs_linear']['ci95'][0]:+.4f}, {inf['delong_gbdt_vs_linear']['ci95'][1]:+.4f}] |",
        f"| DeLong p (two-sided) | {inf['delong_gbdt_vs_linear']['p_two_sided']:.3g} |",
        f"| entity-cluster bootstrap 95% CI | [{inf['cluster_bootstrap']['ci95'][0]:.4f}, {inf['cluster_bootstrap']['ci95'][1]:.4f}] |",
        f"| time-block bootstrap 95% CI | [{inf['block_bootstrap']['ci95'][0]:.4f}, {inf['block_bootstrap']['ci95'][1]:.4f}] |",
        f"| McNemar discordant pairs | {inf['mcnemar']['n_discordant']} |",
        f"| McNemar exact p | {inf['mcnemar']['p_exact']:.3g} |",
        f"| Holm-adjusted p | {', '.join(f'{p:.3g}' for p in inf['holm']['adjusted'])} |",
        f"| seeds required (80% power) | {inf['power']['required_seeds']} |",
        "", "## Table 3 · Verification protocol", "",
        "| f | q | empirical detection | theory (hypergeometric) | escape 95% CI |",
        "|---:|---:|---:|---:|---|"]
    for r in data["challenge"]["rows"]:
        L.append(f"| {r['f']:.2f} | {r['q']:.3f} | {r['detection_rate']:.3f} | "
                 f"{1-r['theory_hypergeometric']:.3f} | "
                 f"[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] |")
    L += ["", "## Table 4 · Field-aligned surface forms", "",
          "| mode | tokens/event | reverse-parse recall | numeric rel. err | probe acc |",
          "|---|---:|---:|---:|---:|"]
    for r in data["fidelity"]["rows"]:
        e = r["numeric_relative_error_mean"]
        a = r["probe_productcd_accuracy"]
        L.append(f"| {r['mode']} | {r['tokens_per_event_mean']:.1f} | "
                 f"{r['reverse_parse_recall']:.3f} | "
                 f"{'—' if e is None else f'{e:.2e}'} | "
                 f"{'—' if a is None else f'{a:.3f}'} |")
    L += ["", "## Table 5 · Cross-dataset and drift", "",
          "| dataset | AUC-PR first slice | AUC-PR last slice | delta |",
          "|---|---:|---:|---:|"]
    for tag, label in (("drift_ieee", "IEEE-CIS"), ("drift_baf", "BAF")):
        d = data[tag]
        s = d["slices"]
        if s:
            delta = d["auc_pr_first_to_last_delta"]
            L.append(f"| {label} | {s[0]['auc_pr']:.4f} | {s[-1]['auc_pr']:.4f} | "
                     f"{'— (only one evaluable slice)' if delta is None else f'{delta:+.4f}'} |")
    L += ["", f"BAF held-out AUC-PR = {data['baf']['metrics']['auc_pr']:.4f}.", ""]
    L += ["## Table 6 · PSWA n x w scan (real entity sequence)", "",
          "| n | w | nnz | dense nnz | rho | eps | peak-byte reduction | max abs output diff |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in data["pswa_scan"]["rows"]:
        L.append(f"| {r['n']} | {r['w']} | {r['nnz']} | {r['dense_nnz']} | "
                 f"{r['rho_mean']:.4f} | {r['eps_mean']:.4f} | "
                 f"{r['byte_reduction']:.3f} | {r['max_abs_output_diff']:.4g} |")
    qual = data["quality"]
    L += ["", f"Quality bound: empirical max |delta out| = "
          f"{qual['quality_bound']['empirical_max_abs_diff']:.4g}, theoretical bound = "
          f"{qual['quality_bound']['theoretical_bound_max']:.4g}, holds = "
          f"{qual['quality_bound']['bound_holds']}.",
          f"Pre-registered claim decision: **{qual['decision']}**.", ""]
    L += ["## Table 7 · Attack surface", "",
          "| quantity | value |", "|---|---:|"]
    pred = data["predictable"]
    L += [f"| detection, unpredictable challenges | {pred['unpredictable']['detection_rate']:.3f} |",
          f"| detection, predictable challenges | {pred['predictable']['detection_rate']:.3f} |"]
    cal, eff, err = data["calibration"], data["effect"], data["errors"]
    L += ["", "## Table 8 · Calibration, effect sizes, layered errors", "",
          "| quantity | value |", "|---|---:|",
          f"| distinct continuous test scores | {cal['n_distinct_test_scores']:,} |",
          f"| Brier (raw / Platt) | {cal['brier_raw']:.4f} / {cal['brier_calibrated']:.4f} |",
          f"| ECE (raw / Platt) | {cal['ece_raw']:.4f} / {cal['ece_calibrated']:.4f} |",
          f"| generative parse-failure rate | {cal['generative_parse_failure_rate']} (N/A, no generative model) |",
          f"| Cohen's d (positive-class scores, gbdt vs linear) | {eff['cohens_d_positive_class_scores_gbdt_vs_linear']:.4f} |",
          f"| Cliff's delta | {eff['cliffs_delta_positive_class_scores']:.4f} |",
          f"| ablation x seed interaction | {eff['ablation_by_seed_interaction']} |",
          "", "Layered error (amount quintiles, full held-out test):", "",
          "| stratum | n | fraud rate | precision | recall | AUC-PR |",
          "|---|---:|---:|---:|---:|---:|"]
    for r in err["by_amount_quintile"]:
        ap = r["auc_pr"]
        L.append(f"| {r['stratum']} | {r['n']:,} | {r['fraud_rate']:.4f} | "
                 f"{r['precision']:.4f} | {r['recall']:.4f} | "
                 f"{'—' if ap is None else f'{ap:.4f}'} |")
    # ---- Table 9: MEASURED GPU memory / latency ------------------------------
    g = data["gpu"]
    L += ["", "## Table 9 · Peak device memory and latency (measured with torch on the GPU)", "",
          "| n | materialised MiB | SDPA MiB | PSWA MiB | PSWA-stream MiB | PSWA p50 ms | SDPA p50 ms |",
          "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in g["bench"]["rows"]:
        m = r["modes"]

        def cell(key):
            v = m.get(key, {})
            return (f"{v['transient_mib']:.2f}" if v.get("status") == "ok"
                    else (v.get("status") or "-"))
        L.append(f"| {r['n']} | {cell('full_materialised')} | {cell('full_sdpa')} | "
                 f"{cell('pswa')} | {cell('pswa_stream')} | "
                 f"{(m.get('pswa', {}).get('p50_ms') or float('nan')):.2f} | "
                 f"{(m.get('full_sdpa', {}).get('p50_ms') or float('nan')):.2f} |")
    cl = g["claims"]
    L += ["", f"Device: {g['bench']['device']['name']}, "
              f"{g['bench']['device']['total_memory_bytes'] / 2**30:.1f} GiB, "
              f"torch {g['bench']['device']['torch']}. "
              f"Dense materialised OOMs at n = {cl.get('n_where_dense_oom_but_pswa_runs')}. "
              f"VRAM reduction vs materialised at the largest common n: "
              f"{cl.get('max_vram_reduction_vs_materialised')}. "
              f"Ratio vs SDPA at n={cl.get('largest_n_with_both_measured')}: "
              f"{cl.get('vram_reduction_at_largest_n_vs_sdpa')} "
              "(negative = PSWA uses MORE memory than the fused kernel).", ""]

    # ---- Table 10: local generative baseline --------------------------------
    z = data["llm0"]
    sft = data["llmsft"]
    scl = data["llmscale"]
    L += ["## Table 10 · Local generative baseline (Qwen2.5, this GPU)", "",
          "| system | AUC-PR | AUC-ROC | Recall@FPR=1% | F1 |", "|---|---:|---:|---:|---:|"]
    zc = z["elicitation"]["choice"]
    L.append(f"| GBDT on the same {zc['n_rows']} test rows | "
             f"{zc['gbdt_same_rows']['auc_pr']:.4f} | {zc['gbdt_same_rows']['auc_roc']:.4f} | "
             f"{zc['gbdt_same_rows']['recall_at_fpr_1pct']:.4f} | {zc['gbdt_same_rows']['f1']:.4f} |")
    L.append(f"| Qwen2.5-0.5B zero-shot (logit interface) | "
             f"{zc['llm']['auc_pr']:.4f} | {zc['llm']['auc_roc']:.4f} | "
             f"{zc['llm']['recall_at_fpr_1pct']:.4f} | {zc['llm']['f1']:.4f} |")
    sc = sft["elicitation"]["choice"]
    L.append(f"| Qwen2.5-0.5B + LoRA (continuous targets) | "
             f"{sc['llm']['auc_pr']:.4f} | {sc['llm']['auc_roc']:.4f} | "
             f"{sc['llm']['recall_at_fpr_1pct']:.4f} | {sc['llm']['f1']:.4f} |")
    L += ["", f"Paired difference (GBDT - LLM) on the identical rows: "
              f"{zc['delta_auc_pr_gbdt_minus_llm']:+.4f}, permutation p = "
              f"{zc['permutation']['p_two_sided']:.4g}, bootstrap 95% CI "
              f"[{zc['bootstrap']['delta_auc_pr_ci95'][0]:+.4f}, "
              f"{zc['bootstrap']['delta_auc_pr_ci95'][1]:+.4f}]. "
              f"JSON-interface parse failure rate (zero-shot) = "
              f"{z['json_interface']['parse_failure_rate']:.2%} over "
              f"{z['json_interface']['n_rows']} rows; after LoRA = "
              f"{sft['json_interface']['parse_failure_rate']:.2%}. "
              f"SFT targets: mean {sft['target_stats']['mean']:.4f}, "
              f"sd {sft['target_stats']['sd']:.4f}, "
              f"{sft['target_stats']['n_distinct']} distinct values. "
              f"Measured throughput: "
              f"{z['elicitation'].get('choice_latency', {}).get('throughput_rows_per_s')} "
              f"rows/s (logit interface) vs "
              f"{z['json_interface']['latency'].get('throughput_rows_per_s')} rows/s "
              f"(free-form JSON).", ""]
    L += ["| model | params | load peak GiB | gen peak GiB | AUC-PR | rows/s |",
          "|---|---:|---:|---:|---:|---:|"]
    for r in scl["rows"]:
        if r.get("status") != "ok":
            L.append(f"| {r['model']} | - | - | - | - | - |")
            continue
        L.append(f"| {r['model']} | {r['params']/1e9:.2f}B | {r['load_peak_gib']:.2f} | "
                 f"{r['generation_peak_gib']:.2f} | {r['auc_pr']:.4f} | "
                 f"{r['latency'].get('throughput_rows_per_s')} |")
    L += ["", "The ladder is measured at two points only; any 7B/8B figure would be an "
              "extrapolation and is therefore NOT reported as a measurement.", ""]

    # ---- Table 11: measured EVM gas ----------------------------------------
    gas = data["gas"]
    L += ["## Table 11 · On-chain anchoring cost (executed on py-evm, solc "
          f"{gas['solc']})", "",
          "| operation | gas (measured) |", "|---|---:|",
          f"| contract deployment | {gas['deploy_gas']:,} |",
          f"| one per-transaction `LOG1` commitment | {gas['per_tx_log_gas_mean']:,.0f} |",
          f"| one batch Merkle-root `SSTORE` + event | {gas['batch_root_gas']:,} |",
          f"| on-chain inclusion proof verification (depth {gas['merkle_depth']}) | "
          f"{gas['verify_inclusion_gas']:,} |",
          "", f"Per {gas['batch_size']} transactions: {gas['total_gas_per_tx_design']:,} gas "
              f"(one log each) vs {gas['total_gas_batch_design']:,} gas (one root) = "
              f"**{gas['gas_saving_fraction']:.2%} saving**. "
              f"Reference pricing $/tx: per-tx {gas['cost_reference']['usd_per_tx_per_tx_design']:.6f}, "
              f"batched {gas['cost_reference']['usd_per_tx_batch_design']:.6f} "
              f"(at {gas['cost_reference']['gas_price_gwei']} gwei, "
              f"${gas['cost_reference']['eth_usd']:,}/ETH — assumptions, not quotes).", ""]

    # ---- Table 12: cost sensitivity + row-level inference -------------------
    L += ["## Table 12 · Cost sensitivity (threshold from validation)", "",
          "| c_fn : c_fp | threshold | test cost/tx | recall | FPR |", "|---|---:|---:|---:|---:|"]
    for r in data["cost"]["rows"]:
        L.append(f"| {r['cost_ratio_fn_fp']} | {r['threshold_from_validation']:.4f} | "
                 f"{r['test_cost_per_tx']:.5f} | {r['test_recall']:.4f} | {r['test_fpr']:.4f} |")
    pp = data["perm"]["pairs"]
    L += ["", "Paired randomisation tests (row-level, AUC-PR):", "",
          "| comparison | delta | permutation p | bootstrap CI (rows) | bootstrap CI (entities) |",
          "|---|---:|---:|---|---|"]
    for name, v in pp.items():
        L.append(f"| {name} | {v['permutation']['observed_delta']:+.4f} | "
                 f"{v['permutation']['p_two_sided']:.4g} | "
                 f"[{v['bootstrap_rows']['delta_auc_pr_ci95'][0]:+.4f}, "
                 f"{v['bootstrap_rows']['delta_auc_pr_ci95'][1]:+.4f}] | "
                 f"[{v['bootstrap_entities']['delta_auc_pr_ci95'][0]:+.4f}, "
                 f"{v['bootstrap_entities']['delta_auc_pr_ci95'][1]:+.4f}] |")

    L += ["", "All numbers above are read from `results/<run_id>/*.json`.", ""]
    return "\n".join(L) + "\n"


def stage_figures(state, cfg) -> dict:
    from .common import PAPER_DIR, atomic_write_json, atomic_write_text, utcnow
    from .stages import emit
    results_dir = state.results_dir
    data, hashes = _load_all(results_dir)
    made = build_plots(data)
    chain = {"run_id": state.run_id, "generated_at": utcnow(),
        "generator": "experiments/make_figures.py",
        "generator_sha256": sha256_file(Path(__file__)),
        "policy": "fail-fast; no EXAMPLE DATA; mock inputs rejected",
        "inputs": hashes, "figures": []}
    for name, tags, title in PLOT_SPECS:
        chain["figures"].append({
            "figure": f"figures/{name}", "title": title,
            "sha256": sha256_file(ROOT / "figures" / name),
            "input_files": [REQUIRED[t] for t in tags],
        })
    chain_path = results_dir / "s7_figures" / "ARTIFACT_CHAIN.json"
    atomic_write_json(chain_path, chain)
    atomic_write_json(ROOT / "ARTIFACT_CHAIN.json", chain)

    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_text(PAPER_DIR / "TABLES.md", build_tables(data))
    atomic_write_text(PAPER_DIR / "FIGURE_LEGEND.md",
                      "# Figure provenance\n\n" + "\n".join(
                          f"- `{f['figure']}` <- {', '.join(f['input_files'])}"
                          for f in chain["figures"]) + "\n")

    def tables_unit():
        return {"figures": [f["figure"] for f in chain["figures"]],
                "n_inputs": len(hashes),
                "tables_written": ["paper/TABLES.md", "paper/FIGURE_LEGEND.md"],
                "artifact_chain": "results/<run_id>/s7_figures/ARTIFACT_CHAIN.json",
                "assertions": {"no_example_data": True,
                               "all_inputs_real": all(
                                   not data[t].get("mock", False) for t in data),
                               "chain_complete": len(chain["figures"]) == len(PLOT_SPECS)}}
    return {"s7_figures/plots": emit(
        state, "s7_figures/plots", tables_unit,
        deps=[results_dir / rel for rel in REQUIRED.values()])}
