"""Render the six vector figures of the AIJ manuscript from results JSON.

Every figure is written as PDF (vector, fonts embedded) because Elsevier asks
for vector artwork for line art; no bitmap is produced for the paper.

Usage:  python paper_aij/make_figs.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
sys.path.insert(0, str(ROOT))
from experiments import pswa, attention_bench as ab   # noqa: E402

BLUE, ORANGE, GREEN, RED = "#0072B2", "#E69F00", "#009E73", "#D55E00"
GREY, PURPLE = "#666666", "#CC79A7"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def latest_run() -> Path:
    runs = sorted([p for p in (ROOT / "results").iterdir()
                   if p.is_dir() and (p / "RUN_METADATA.json").exists()])
    return runs[-1]


def load(run: Path, rel: str) -> dict:
    return json.loads((run / f"{rel}.json").read_text(encoding="utf-8"))


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", name)


# --------------------------------------------------------------------- fig 1 --
def fig_mask(run: Path) -> None:
    n, w, g, s = 96, 12, 24, 3
    qi = np.arange(n)[:, None]
    kj = np.arange(n)[None, :]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    titles = ["dense causal", "local window only", "window + periodic anchors"]
    for ax, (mode, title) in zip(axes, [("full", titles[0]),
                                        ("sliding", titles[1]),
                                        ("pswa", titles[2])]):
        m = ab.fine_mask(qi, kj, w, g, s, mode)
        ax.imshow(m.astype(float), cmap="Blues", origin="upper",
                  interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xlabel("key index $j$")
        if ax is axes[0]:
            ax.set_ylabel("query index $i$")
        ax.set_xticks([0, n - 1])
        ax.set_yticks([0, n - 1])
    fig.text(0.5, -0.02,
             f"$w={w}$ (band), $g={g}$ (anchor period, dashed guides), "
             f"$s={s}$ (sinks): every $g$-th key column is visible to all later queries",
             ha="center", fontsize=7, color=GREY)
    for ax in axes[1:]:
        for a in range(g, n, g):
            ax.axvline(a - 0.5, color=RED, lw=0.4, ls=":", alpha=0.8)
    save(fig, "fig1_mask.pdf")


# --------------------------------------------------------------------- fig 2 --
def fig_counting(run: Path) -> None:
    cplx = load(run, "s1_pswa/complexity_curve")
    w, g, s = cplx["w"], cplx["g"], cplx["s"]
    pts = cplx["modes"]["pswa"]["curve"]
    ns = np.array([p["n"] for p in pts], float)
    meas = np.array([p["nnz"] for p in pts], float)
    dense = np.array([p["dense_nnz"] for p in pts], float)
    grid = np.logspace(math.log10(ns.min()), math.log10(ns.max()), 100)
    closed = np.array([sum(min(i + 1, w) + min(i + 1, s) + max(0, (i - w) // g)
                           for i in range(int(n)) ) for n in grid])

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    ax = axes[0]
    ax.loglog(ns, meas, "o", color=BLUE, ms=5, label="measured $|S|$ (window+anchors)")
    ax.loglog(grid, closed, "-", color=BLUE, lw=1, label="closed form, Eq. (3)")
    ax.loglog(grid, grid * (w + s), "--", color=GREEN, lw=1,
              label=r"window only: $n(w{+}s)$, slope 1")
    ax.loglog(ns, dense, "s", color=RED, ms=4, mfc="none", label=r"dense: $n(n{+}1)/2$, slope 2")
    ax.set_xlabel("sequence length $n$")
    ax.set_ylabel("attention entries $|S|$")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25, which="both", lw=0.4)

    ax = axes[1]
    ex = [math.log(b / a) / math.log(nb / na)
          for a, b, na, nb in zip(meas[:-1], meas[1:], ns[:-1], ns[1:])]
    mids = np.sqrt(ns[:-1] * ns[1:])
    wide = np.logspace(math.log10(192), math.log10(65536), 120)
    cf = np.array([sum(min(i + 1, w) + min(i + 1, s) + max(0, (i - w) // g)
                       for i in range(int(n))) for n in wide])
    exp_cf = np.gradient(np.log(cf), np.log(wide))
    ax.semilogx(wide, exp_cf, "-", color=GREY, lw=1,
                label="exponent of Eq. (3)")
    ax.semilogx(mids, ex, "o", color=BLUE, ms=5,
                label="exponent of measured $|S|$")
    ax.axhline(1.0, color=GREEN, ls="--", lw=1, label="linear")
    ax.axhline(2.0, color=RED, ls="--", lw=1, label="quadratic")
    nstar = 2 * g * (w + s)
    ax.axvline(nstar, color=BLUE, ls=":", lw=0.9, alpha=0.8)
    ax.annotate(r"$n^\ast=2g(w{+}s)$" + f"$={nstar}$", xy=(nstar, 1.35),
                xytext=(nstar * 0.13, 1.55), fontsize=6.5, color=BLUE,
                arrowprops=dict(arrowstyle="->", lw=0.7, color=BLUE))
    ax.set_xlabel("sequence length $n$")
    ax.set_ylabel("local exponent $d\\log|S|/d\\log n$")
    ax.set_ylim(0.94, 2.06)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25, which="both", lw=0.4)
    save(fig, "fig2_counting.pdf")


# --------------------------------------------------------------------- fig 3 --
def fig_device(run: Path) -> None:
    gpu = load(run, "s1_pswa/gpu_memory_latency")
    rows = gpu["bench"]["rows"]
    series = [("full_materialised", "dense materialised", "#555555", "s"),
              ("full_sdpa", "fused SDPA", BLUE, "o"),
              ("longformer_global", "local + sinks", GREEN, "^"),
              ("pswa", "window + anchors", ORANGE, "D"),
              ("pswa_stream", "window + anchors, streamed", PURPLE, "v")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    for ax, field, lab in ((axes[0], "transient_mib", "peak transient device memory (MiB)"),
                           (axes[1], "p50_ms", "latency, median of 12 runs (ms)")):
        for key, label, color, marker in series:
            xs, ys = [], []
            for r in rows:
                m = r["modes"].get(key, {})
                if m.get("status") == "ok":
                    xs.append(r["n"])
                    ys.append(max(m[field], 1e-3))
            ax.loglog(xs, ys, marker=marker, color=color, ms=4, lw=1.1, label=label)
        ax.set_xlabel("sequence length $n$")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.25, which="both", lw=0.4)
    oom = [r["n"] for r in rows
           if r["modes"]["full_materialised"].get("status") != "ok"]
    axes[0].set_ylim(2e-2, 2e4)
    if oom:
        last_ok = max(r["n"] for r in rows
                      if r["modes"]["full_materialised"].get("status") == "ok")
        axes[0].annotate("dense: out of memory",
                         xy=(oom[0], 2e3), xytext=(last_ok * 0.16, 7e3),
                         arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8),
                         fontsize=7, color="#555555")
    handles = [plt.Line2D([], [], marker=m, color=cl, ms=4, lw=1.1, label=lab)
               for _, lab, cl, m in series]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.20))
    axes[0].set_title("memory", loc="left")
    axes[1].set_title("latency", loc="left")
    save(fig, "fig3_device.pdf")


# --------------------------------------------------------------------- fig 4 --
def fig_density(run: Path) -> None:
    blk = load(run, "s1_pswa/block_alignment")
    byg = blk["rows_by_g"]
    byb = blk["rows_by_block"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))

    ax = axes[0]
    xs = [r["predicted_density_block_over_g"] + r["window_contribution"] for r in byg]
    ys = [r["density_over_causal"] for r in byg]
    for r, x, y in zip(byg, xs, ys):
        col = BLUE if r["g_multiple_of_block"] else RED
        ax.plot(x, y, "o", color=col, ms=6)
        ax.annotate(f"$g$={r['g']}", (x, y), textcoords="offset points",
                    xytext=(-16, 5), fontsize=6.5, color=col)
    lim = [0, 1.08]
    ax.plot(lim, lim, "-", color=GREY, lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r"predicted $\min(1,B/g)+$ window term")
    ax.set_ylabel("measured block density")
    ax.set_title("density law (n=%d, B=%d)" % (blk["n"], blk["block"]), loc="left")
    ax.plot([], [], "o", color=BLUE, label="aligned")
    ax.plot([], [], "o", color=RED, label="not aligned")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25, lw=0.4)

    ax = axes[1]
    bs = [r["block"] for r in byb]
    infl = [r["block_density"] / r["element_density"] for r in byb]
    ax.loglog(bs, infl, "o-", color=ORANGE, ms=5, lw=1.2, label="measured")
    ceiling = 1.0 / byb[0]["element_density"]
    ax.axhline(ceiling, ls=":", color=GREY, lw=1,
               label="ceiling $1/$element density")
    ax.set_xticks(bs)
    ax.set_xticklabels([str(b) for b in bs])
    ax.minorticks_off()
    ax.set_xlim(min(bs) / 1.4, max(bs) * 1.4)
    ax.set_xlabel("block size $B$")
    ax.set_ylabel("block density / element density")
    ax.set_title("granularity inflation ($g=%d$)" % byb[0]["g"], loc="left")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25, which="both", lw=0.4)
    save(fig, "fig4_density.pdf")


# --------------------------------------------------------------------- fig 5 --
def fig_llm(run: Path) -> None:
    z = load(run, "s8_llm/zero_shot_0_5b")
    f = load(run, "s8_llm/lora_sft_0_5b")
    zc = z["elicitation"]["choice"]
    fc = f["elicitation"]["choice"]
    labels = ["GBDT\n(same rows)", "0.5B\nzero-shot", "0.5B\n+ LoRA (900 rows)"]
    metrics = [("auc_pr", "AUC-PR"), ("auc_roc", "AUC-ROC"), ("f1", "F1 @ 0.5")]
    vals = {
        "auc_pr": [zc["gbdt_same_rows"]["auc_pr"], zc["llm"]["auc_pr"], fc["llm"]["auc_pr"]],
        "auc_roc": [zc["gbdt_same_rows"]["auc_roc"], zc["llm"]["auc_roc"], fc["llm"]["auc_roc"]],
        "f1": [zc["gbdt_same_rows"]["f1"], zc["llm"]["f1"], fc["llm"]["f1"]],
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    x = np.arange(len(labels))
    width = 0.26
    for k, (key, lab) in enumerate(metrics):
        ax.bar(x + (k - 1) * width, vals[key], width, label=lab,
               color=[BLUE, ORANGE, GREEN][k], edgecolor="white", lw=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("held-out score")
    ax.set_ylim(0, 1.0)
    ax.legend(ncol=3, loc="upper right")
    ax.set_title("discrimination", loc="left")
    ax.grid(alpha=0.25, axis="y", lw=0.4)

    ax = axes[1]
    pf = [0.0, 100 * z["json_interface"]["parse_failure_rate"],
          100 * f["json_interface"]["parse_failure_rate"]]
    bars = ax.bar(x, pf, 0.5, color=[GREEN, RED, GREEN], edgecolor="white", lw=0.4)
    for b, v in zip(bars, pf):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}%", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 115)
    ax.set_ylabel("JSON parse-failure rate (%)")
    ax.set_title("format compliance", loc="left")
    ax.grid(alpha=0.25, axis="y", lw=0.4)
    save(fig, "fig5_llm.pdf")


# --------------------------------------------------------------------- fig 6 --
def fig_cost(run: Path) -> None:
    cost = load(run, "s4_stats/cost_sensitivity")
    rows = cost["rows"]
    r = [x["cost_ratio_fn_fp"] for x in rows]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ax = axes[0]
    ax.semilogx(r, [x["threshold_from_validation"] for x in rows], "o-",
                color=BLUE, ms=4, lw=1.2)
    ax.set_xlabel(r"cost ratio $c_{\mathrm{FN}}/c_{\mathrm{FP}}$")
    ax.set_ylabel("threshold chosen on validation")
    ax.grid(alpha=0.25, which="both", lw=0.4)
    ax = axes[1]
    ax.semilogx(r, [x["test_cost_per_tx"] for x in rows], "o-", color=RED, ms=4, lw=1.2,
                label="test cost / transaction")
    ax2 = ax.twinx()
    ax2.semilogx(r, [100 * x["test_fpr"] for x in rows], "s--", color=GREY, ms=3, lw=1,
                 label="FPR (%)")
    ax2.set_ylabel("FPR (%)", color=GREY)
    ax2.spines["right"].set_visible(True)
    ax.set_xlabel(r"cost ratio $c_{\mathrm{FN}}/c_{\mathrm{FP}}$")
    ax.set_ylabel("expected cost / transaction", color=RED)
    ax.grid(alpha=0.25, which="both", lw=0.4)
    axes[0].set_title("threshold responds to cost", loc="left")
    ax.set_title("realised cost and false-alarm rate", loc="left")
    save(fig, "fig6_cost.pdf")


def main() -> int:
    run = latest_run()
    print("figures from", run.name)
    fig_mask(run)
    fig_counting(run)
    fig_device(run)
    fig_density(run)
    fig_llm(run)
    fig_cost(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
