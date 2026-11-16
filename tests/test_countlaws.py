"""Tests for tools/countlaws.py, the standalone counting tool shipped to readers.

The tool is the reader-facing instrument of the paper, so it must agree with
(i) the mask builder that produced the stored measurements and (ii) the closed
form printed in Proposition 1. Both are checked here, plus the two edges that
have been wrong in earlier drafts: rows above the window (the anchor term must
start exactly at i = w) and the sink term above the window (the window has
already swallowed the first w sinks, so the term is min(s, i-w+1)).

Run: C:/Users/apple/anaconda3/python.exe -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import countlaws as cl  # noqa: E402

from experiments import attention_bench as ab  # noqa: E402


# ------------------------------------------------------------ entry law ---
def test_entry_count_matches_closed_form_on_a_grid():
    """The loop and the closed form are different code paths for one integer."""
    for n in (1, 2, 17, 64, 65, 256, 512, 1024, 2048, 4096):
        for w, g, s in ((64, 96, 4), (8, 16, 4), (12, 24, 3), (1, 1, 0),
                        (4, 7, 2)):
            if w > n:
                continue
            assert cl.entry_count(n, w, g, s) == cl.entry_count_closed(n, w, g, s)


def test_entry_count_matches_the_mask_builder():
    """Count the boolean mask itself; it must equal the law, entry for entry."""
    for n in (32, 128, 256, 512):
        for w, g, s in ((8, 16, 4), (12, 24, 3), (16, 8, 4), (5, 9, 1)):
            qi = np.arange(n)
            m = ab.fine_mask(qi[:, None], qi[None, :], w, g, s, "pswa").astype(bool)
            assert int(m.sum()) == cl.entry_count(n, w, g, s), (n, w, g, s)


def test_entry_law_reproduces_the_stored_reference_point():
    """n=4096, w=64, g=96, s=4 is the point quoted in Section 6 and Table 3."""
    assert cl.entry_count(4096, 64, 96, 4) == 358906


def test_anchor_term_starts_exactly_at_the_window_edge():
    """Row w is the first row with a non-zero anchor term; row w-1 has none."""
    w, g, s = 4, 3, 2
    # rows 0..w-1 are covered by the window alone: i+1 entries each
    assert cl.entry_count(w, w, g, s) == w * (w + 1) // 2
    # row w adds w window + min(s, 1) sink + 0 anchors
    assert cl.entry_count(w + 1, w, g, s) - cl.entry_count(w, w, g, s) == w + 1
    # row w+g adds the first anchor
    row = w + g
    delta = (cl.entry_count(row + 1, w, g, s) - cl.entry_count(row, w, g, s))
    assert delta == w + min(s, row - w + 1) + 1


def test_sink_term_saturates_above_the_window():
    w, g, s = 4, 64, 3
    n = 64
    # every row i >= w+2 has min(s, i-w+1) = s = 3 sinks
    for i in (w + 2, w + 10, n - 1):
        d = cl.entry_count(i + 1, w, g, s) - cl.entry_count(i, w, g, s)
        assert d == w + s + (i - w) // g


def test_anchor_term_is_quadratic_in_n():
    """Doubling n more than doubles the time past the crossover: this is the
    paper's central claim and the reason the linear reading is a small-n artefact.
    """
    w, g, s = 64, 96, 4
    n0 = cl.crossover(w, g, s)
    assert n0 == 2 * g * (w + s) == 13056
    small = cl.entry_count(n0, w, g, s)
    large = cl.entry_count(2 * n0, w, g, s)
    assert large > 2.5 * small, "anchor term failed to dominate above n*"


# ------------------------------------------------------ granularity law ---
def _window_only(n, w, block, s=0):
    """Density with the anchor set switched off: the pure window/tiling term.

    g = 10**9 is large enough that the only 'anchor' hit is block 0, which the
    finite-g run also counts, so the two terms are directly comparable.
    """
    return cl.block_density(n, w, 10 ** 9, s, block)


@pytest.mark.parametrize("n", [8192, 16384, 32768])
@pytest.mark.parametrize("block,g", [(64, 96), (64, 192), (64, 256)])
def test_block_density_approaches_min_one_over_b_over_g(block, g, n):
    """D_blk -> min(1, B/g) + window term, with an explicit O(B/n) error.

    The finite-n residual is negative (anchor blocks overlap window blocks) and
    must shrink like B/n, which is the sense in which the law is asymptotic.
    """
    floor = min(1.0, block / g)
    resid = cl.block_density(n, 1, g, 0, block) - _window_only(n, 1, block) - floor
    assert abs(resid) <= 3.0 * block / n, (block, g, n, resid)


def test_block_density_residual_vanishes_with_n():
    block, g = 64, 192
    floor = min(1.0, block / g)
    res = [abs(cl.block_density(n, 1, g, 0, block) - _window_only(n, 1, block) - floor)
           for n in (8192, 16384, 32768)]
    assert res[0] > res[1] > res[2], res


@pytest.mark.parametrize("block,g", [(256, 192), (128, 128), (64, 32)])
def test_block_density_saturates_when_g_does_not_exceed_b(block, g):
    """If every block contains an anchor, the kernel materialises causal blocks
    without exception, whatever the pattern says. Exact, not asymptotic."""
    assert cl.block_density(8192, 1, g, 0, block) == 1.0


def test_block_density_is_independent_of_alignment():
    """g % B == 0 must not matter; the withdrawn lemma claimed it did.

    192 % 64 == 0 (aligned) and 200 % 64 == 8 (unaligned) each satisfy the same
    law with their own floor.
    """
    n = 16384
    for g in (192, 200):
        floor = min(1.0, 64 / g)
        resid = cl.block_density(n, 1, g, 0, 64) - _window_only(n, 1, 64) - floor
        assert abs(resid) <= 3.0 * 64 / n, (g, resid)
    # and alignment is not a predictor of the *measured* drop in density
    assert cl.block_density(n, 1, 192, 0, 64) > cl.block_density(n, 1, 200, 0, 64)


def test_element_density_decreases_and_inflation_increases():
    w, g, s, B = 64, 96, 4, 64
    dens = [cl.element_density(n, w, g, s) for n in (512, 4096, 32768)]
    assert dens[0] > dens[1] > dens[2]
    infl = [cl.report(n, w, g, s, B)["inflation"] for n in (512, 4096, 32768)]
    assert infl[0] < infl[1] < infl[2]


def test_work_ratio_falls_with_n_but_never_reaches_zero():
    """The tool's practical reading: tiling loses at small n, and at large n the
    floor is set by B/g, not by n. It does not vanish."""
    w, g, s, B = 64, 96, 4, 64
    r = [cl.work_ratio(n, w, g, s, B) for n in (256, 4096, 32768)]
    assert r[0] > 1.0, "at tiny n the pattern cannot beat dense"
    assert r[0] > r[1] > r[2]
    assert 0.60 < r[2] < 0.70, r[2]       # floor ~ B/g + window/block coupling
    assert r[2] > min(1.0, B / g) * 0.9


def test_global_key_budget_grows_linearly():
    """Theorem 3 in one line: periodic anchors give n/g global keys."""
    g, s = 96, 4
    assert cl.report(1024, 64, g, s, 64)["global_keys"] == s + 11
    assert cl.report(9600, 64, g, s, 64)["global_keys"] == s + 100


# -------------------------------------------------------------- CLI/edges ---
def test_cli_runs_and_asserts_exactness(capsys):
    assert cl.main(["--n", "4096", "--w", "64", "--g", "96", "--s", "4"]) == 0
    out = capsys.readouterr().out
    assert "358,906" in out
    assert "crossover n* = 2g(w+s) = 13056" in out


def test_cli_sweep(capsys):
    assert cl.main(["--sweep", "256,32768,8"]) == 0
    lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
    assert len(lines) == 2 + 8 + 1        # header, rule, eight rows, footer
    assert lines[-1].startswith("crossover n*")


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        cl.entry_count(16, 4, 0, 1)
    with pytest.raises(ValueError):
        cl.entry_count(-1, 4, 8, 1)
