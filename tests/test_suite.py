"""CPU-only unit tests. Run: .venv/Scripts/python.exe -m pytest tests -q"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from experiments import protocol as pr, pswa, stats
from experiments.common import (RunState, atomic_write_json, load_json,
                                sha256_file)
from experiments import make_figures


def test_pswa_mask_is_causal_and_sparse():
    n, w, g, s = 256, 32, 24, 4
    for mode in pswa.MODES:
        m = pswa.build_mask(n, w, g, s, mode)
        assert m.shape == (n, n)
        assert not m[np.triu_indices(n, 1)].any(), "future positions must be masked"
    full = pswa.build_mask(n, w, g, s, "full")
    sparse = pswa.build_mask(n, w, g, s, "pswa")
    assert sparse.sum() < full.sum()
    # causal: sink tokens are fully visible only to queries at position >= s
    assert sparse[s:, :s].all()
    assert sparse[:, ::g].any()


def test_ablation_is_not_a_noop_and_bound_holds():
    rng = np.random.default_rng(0)
    n, d = 128, 16
    q, k, v = rng.normal(0, 1, (n, d)), rng.normal(0, 1, (n, d)), rng.normal(0, 1, (n, d))
    dense, probs = pswa.full_attention(q, k, v)
    mask = pswa.build_mask(n, 32, 24, 4, "pswa")
    sparse, peak = pswa.sparse_attention(q, k, v, mask, block=32)
    r = pswa.assert_ablation_nontrivial(dense, sparse)
    assert r["nontrivial"] and r["max_abs_diff"] > 1e-6
    assert peak[0] < probs.nbytes + (q @ k.T).nbytes
    b = pswa.quality_bound(probs, v, mask, dense)
    assert b["bound_holds"], b
    assert 0 <= b["rho_mean"] <= 1


def test_merkle_inclusion_proof_and_tamper_rejection():
    leaves = [pr.hx(pr.H(f"leaf{i}".encode())) for i in range(17)]
    layers = pr.merkle_layers(leaves)
    root = pr.hx(layers[-1][0])
    for i in (0, 3, 16):
        proof = pr.inclusion_proof(layers, i)
        assert pr.verify_inclusion(root, leaves[i], proof, i)
        assert not pr.verify_inclusion(root, pr.hx(pr.H(b"evil")), proof, i)
    # wrong index must fail
    assert not pr.verify_inclusion(root, leaves[3], pr.inclusion_proof(layers, 3), 4)


def test_challenge_bound_and_predictable_attack():
    a = pr.simulate_challenges(1000, 0.10, 0.10, 400, 1)
    assert 0 <= a["empirical_escape"] <= 1
    assert abs(a["empirical_escape"] - a["theory_hypergeometric"]) < 0.06
    b = pr.simulate_challenges(1000, 0.05, 0.10, 200, 1, predictable=True)
    assert b["detection_rate"] < a["detection_rate"]
    assert pr.required_q(1000, 0.05, 0.01) > 0


def test_delong_holm_and_bootstrap():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 800)
    s1 = y * 0.6 + rng.normal(0, 1, 800)
    s2 = y * 0.2 + rng.normal(0, 1, 800)
    t = stats.delong_paired_test(y, s1, s2)
    assert t["auc_a"] > t["auc_b"] and t["p_two_sided"] < 0.05
    assert t["ci95"][0] <= t["delta"] <= t["ci95"][1]
    h = stats.holm([0.01, 0.04, 0.5])
    assert h["adjusted"] == sorted(h["adjusted"])
    assert h["reject"][0]
    groups = rng.integers(0, 40, 800)
    cb = stats.cluster_bootstrap(y, s1, groups, B=30, seed=1)
    assert cb["ci95"][0] <= cb["point"] <= cb["ci95"][1]


def test_resume_state_marks_interrupted_units(tmp_path: Path):
    import experiments.common as common
    orig_state = common.STATE_DIR
    orig_results = common.RESULTS_ROOT
    orig_current = common.CURRENT_FILE
    common.STATE_DIR = tmp_path / "state"
    common.RESULTS_ROOT = tmp_path / "results"
    common.CURRENT_FILE = common.STATE_DIR / "CURRENT"
    try:
        st = RunState("unit_test_run", fresh=True)
        t0 = st.begin("stage/unit_a", {"x": 1})
        p = st.results_dir / "stage" / "unit_a.json"
        atomic_write_json(p, {"ok": True})
        st.finish("stage/unit_a", p, t0)
        st.begin("stage/unit_b", {})           # simulate a killed process
        st2 = RunState("unit_test_run")
        assert st2.is_done("stage/unit_a"), "completed unit must be skipped"
        assert not st2.is_done("stage/unit_b")
        assert st2.doc["units"]["stage/unit_b"]["status"] == "interrupted"
        assert st2.doc["interruption_count"] == 1
    finally:
        common.STATE_DIR = orig_state
        common.RESULTS_ROOT = orig_results
        common.CURRENT_FILE = orig_current


def test_figures_fail_fast_without_real_data(tmp_path: Path):
    with pytest.raises(SystemExit) as e:
        make_figures._load_all(tmp_path)
    assert "FAIL-FAST" in str(e.value)


def test_data_download_is_idempotent():
    from experiments import data_io
    rec = data_io.ensure_dataset("ieee_cis")
    assert rec["exists"] and rec["size_bytes"] > 1_000_000
    assert rec["sha256"] == sha256_file(Path(rec["path"]))
    again = data_io.ensure_dataset("ieee_cis")
    assert again["action"] in ("reused", "reused_unpinned")
    assert again["sha256"] == rec["sha256"]
