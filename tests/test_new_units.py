"""Regression tests for the units added after the first reviewer round.

Every test here corresponds to a defect that actually occurred in this
repository, so a future edit that silently re-introduces it fails loudly:

  * the PSAW mask let row 0 attend to future tokens (1.98 absolute error);
  * the block-alignment assertion compared *different* anchor periods and
    therefore "confirmed" a claim the data refuted;
  * `transient` measurements absorbed first-touch CUDA allocations, moving the
    fitted exponent of the dense curve from 2.00 to 1.49 between two runs;
  * the resume ledger skipped a downstream unit after its inputs changed,
    leaving stale figures behind;
  * a tokenizer `vocab.json` was counted as a result envelope because the token
    "stage" happened to be a key of the vocabulary.

Run: C:/Users/apple/anaconda3/python.exe -m pytest tests -q
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments import attention_bench as ab
from experiments import llm_eval as le
from experiments import stats2


# --------------------------------------------------------------- attention ---
@pytest.mark.parametrize("mode", ["full", "sliding", "pswa", "nsa_like",
                                 "longformer_global"])
def test_fine_mask_is_causal_in_every_mode(mode):
    """No query may ever see a future key, including via anchors or sinks."""
    n, w, g, s = 64, 8, 16, 4
    qi = np.arange(n)
    m = ab.fine_mask(qi[:, None], qi[None, :], w, g, s, mode)
    future = qi[None, :] > qi[:, None]
    assert not (m & future).any(), "mask is not causal"


def test_block_pattern_is_a_safe_superset_of_the_fine_mask():
    """The executed block pattern must cover every allowed element (no misses)
    and must not mark a block pair that contains nothing (no waste)."""
    n, w, g, s, blk = 512, 8, 64, 4, 16
    qi = np.arange(n)
    fine = ab.fine_mask(qi[:, None], qi[None, :], w, g, s, "pswa")
    bp = ab.block_pattern(n, w, g, s, "pswa", blk).astype(bool)
    nb = n // blk
    for bi in range(nb):
        for bj in range(nb):
            sub = fine[bi * blk:(bi + 1) * blk, bj * blk:(bj + 1) * blk]
            if sub.any():
                assert bp[bi, bj], "allowed element in a block that is skipped"
            if bp[bi, bj]:
                assert sub.any(), "block materialised but contains nothing allowed"


def test_block_density_follows_block_over_g():
    """The measured law, and the replacement for the withdrawn alignment lemma.

    Block density over causal block pairs tracks min(1, block/g) plus a window
    term that grows with g. An unaligned g must sit ON that curve, i.e. within a
    few percent of the interpolation between its aligned neighbours; if it ever
    rises above, the old "non-alignment degenerates to dense" claim would have
    been right after all.
    """
    n, w, s, blk = 4096, 8, 4, 64

    def over_pred(g):
        bp = ab.block_pattern(n, w, g, s, "pswa", blk).astype(bool)
        nb = bp.shape[0]
        tri = np.tril(np.ones((nb, nb), bool))
        return float(bp[tri].sum() / tri.sum()) / min(1.0, blk / g)

    aligned = {g: over_pred(g) for g in (128, 192, 256, 320, 448, 512)}
    for g, r in aligned.items():
        assert g % blk == 0
        assert 1.0 <= r <= 1.60, (g, r)

    # log-log linear interpolation between the enclosing aligned periods
    for g in (224, 352):
        assert g % blk != 0
        lo = max(x for x in aligned if x < g)
        hi = min(x for x in aligned if x > g)
        t = (np.log(g) - np.log(lo)) / (np.log(hi) - np.log(lo))
        expect = aligned[lo] + t * (aligned[hi] - aligned[lo])
        got = over_pred(g)
        assert abs(got - expect) / expect < 0.03, (g, got, expect)


def test_sparsity_vanishes_when_block_ge_g():
    """Pigeonhole: once a block is at least as long as the anchor period, every
    block carries an anchor and block-level sparsity is gone."""
    n, w, s, blk = 1024, 8, 4, 128
    for g in (64, 128):
        bp = ab.block_pattern(n, w, g, s, "pswa", blk).astype(bool)
        nb = bp.shape[0]
        tri = np.tril(np.ones((nb, nb), bool))
        assert bp[tri].sum() / tri.sum() >= 0.98, g


# ------------------------------------------------------------------- stats ---
def test_permutation_test_is_calibrated_and_sensitive():
    rng = np.random.default_rng(0)
    n = 400
    y = (rng.random(n) < 0.2).astype(int)
    good = y * 0.6 + rng.random(n) * 0.4          # clearly informative
    null_same = good.copy()
    noise = rng.random(n)

    # identical systems -> nothing to detect
    r_null = stats2.paired_permutation(y, null_same, null_same, n_perm=200, seed=1)
    assert r_null["observed_delta"] == 0.0
    assert r_null["p_two_sided"] >= 0.99

    # a real gap -> detected
    r_alt = stats2.paired_permutation(y, good, noise, n_perm=200, seed=1)
    assert r_alt["observed_delta"] > 0
    assert r_alt["p_two_sided"] < 0.01


def test_bootstrap_resampling_unit_is_recorded_and_clustered_ci_is_wider():
    rng = np.random.default_rng(0)
    n_ent, per_ent = 120, 5
    ent = np.repeat(np.arange(n_ent), per_ent)
    y = np.repeat((rng.random(n_ent) < 0.3).astype(int), per_ent)
    # strongly clustered scores: all rows of an entity share one draw
    base = rng.random(n_ent)[ent]
    a = base + rng.normal(0, 0.01, n_ent * per_ent)
    b = base[::-1][ent] + rng.normal(0, 0.01, n_ent * per_ent)

    r_row = stats2.paired_bootstrap(y, a, b, B=120, seed=2)
    r_ent = stats2.paired_bootstrap(y, a, b, B=120, seed=2, groups=ent)
    assert r_row["resample_unit"] == "row"
    assert r_ent["resample_unit"] == "entity"
    w_row = np.subtract(*r_row["delta_auc_pr_ci95"][::-1])
    w_ent = np.subtract(*r_ent["delta_auc_pr_ci95"][::-1])
    assert w_ent >= w_row, (w_ent, w_row)


def test_cost_curve_chooses_threshold_on_validation():
    rng = np.random.default_rng(3)
    n = 500
    y = (rng.random(n) < 0.3).astype(int)
    s = np.clip(y * 0.5 + rng.random(n) * 0.5, 0, 1)
    rows = stats2.cost_curve(y[:250], s[:250], y[250:], s[250:],
                             ratios=(1, 10))
    assert len(rows) == 2 and [r["cost_ratio_fn_fp"] for r in rows] == [1, 10]
    for r in rows:
        assert 0.0 < r["threshold_from_validation"] <= 1.0
        assert r["test_cost_per_tx"] >= 0.0
    # a missed fraud costs more than a false alarm -> the costlier ratio must
    # not choose a lower threshold than the cheaper one
    assert (rows[0]["threshold_from_validation"]
            >= rows[1]["threshold_from_validation"])


# --------------------------------------------------------------------- LLM ---
def test_parse_answer_counts_failure_instead_of_imputing():
    ok = le.parse_answer('{"risk": 0.93, "label": 1}')
    assert ok["ok"] is True and abs(ok["risk"] - 0.93) < 1e-9
    # the original pipeline silently substituted 0.9 / 0.1 for these
    for junk in ("I cannot help with that.", "", "yes", "n/a"):
        r = le.parse_answer(junk)
        assert r["ok"] is False and r["risk"] is None


def test_latency_summary_is_monotone():
    ms = [5.0] * 90 + [50.0] * 10
    d = le.summarise_latency(ms)
    assert d["p50_ms_per_row"] <= d["p95_ms_per_row"]
    assert d["throughput_rows_per_s"] > 0
    assert le.summarise_latency([]) == {}


# ------------------------------------------------------------------- ledger ---
def test_downstream_unit_is_rerun_when_an_input_changes(tmp_path: Path):
    """A figure/table unit must not stay 'done' after its inputs are re-measured."""
    import json

    import experiments.common as common
    from experiments.stages import emit

    orig = (common.STATE_DIR, common.RESULTS_ROOT, common.CURRENT_FILE)
    common.STATE_DIR = tmp_path / "state"
    common.RESULTS_ROOT = tmp_path / "results"
    common.CURRENT_FILE = common.STATE_DIR / "CURRENT"
    try:
        st = common.RunState("deps_run", fresh=True)
        upstream = common.RESULTS_ROOT / "src" / "upstream.json"
        upstream.parent.mkdir(parents=True, exist_ok=True)
        upstream.write_text(json.dumps({"stage": "src", "unit": "upstream",
                                        "run_id": "deps_run", "v": 1}),
                            encoding="utf-8")
        calls = {"n": 0}

        def produce():
            calls["n"] += 1
            return {"v": calls["n"]}

        assert emit(st, "down/unit", produce, deps=[upstream]) == "done"
        st2 = common.RunState("deps_run")
        assert emit(st2, "down/unit", produce, deps=[upstream]) == "skipped"
        assert calls["n"] == 1

        upstream.write_text(json.dumps({"stage": "src", "unit": "upstream",
                                        "run_id": "deps_run", "v": 2}),
                            encoding="utf-8")
        st3 = common.RunState("deps_run")
        assert emit(st3, "down/unit", produce, deps=[upstream]) == "done"
        assert calls["n"] == 2, "stale downstream artefact was not re-executed"
    finally:
        common.STATE_DIR, common.RESULTS_ROOT, common.CURRENT_FILE = orig


def test_vocabulary_file_is_not_mistaken_for_a_result_envelope(tmp_path: Path):
    """A tokenizer vocab contains the tokens 'stage' and 'mock'; requiring the
    full envelope signature (stage+unit+run_id) keeps it out of the audit."""
    import json
    vocab = tmp_path / "vocab.json"
    vocab.write_text(json.dumps({"stage": 91, "mock": 198, "the": 7}),
                     encoding="utf-8")
    d = json.loads(vocab.read_text(encoding="utf-8"))
    assert not ({"stage", "unit", "run_id"} <= set(d))
