"""Tests for the provenance chain: paper number -> result file -> code.

`tools/trace.py` is what the repository promises a reader: for any number printed
in the manuscript, name the JSON field it came from and the function that wrote
it. A promise like that is only worth something if it is checked, so these tests
assert the three properties that make the trace meaningful:

1. every macro in ``numbers.tex`` resolves to a result file (nothing is typed);
2. the value the trace reports is the value the paper prints (no second source
   of truth);
3. every result file maps to a command that exists in the suite (the pointer is
   executable, not decorative).

Run: python -m pytest tests -q
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_aij"
sys.path.insert(0, str(ROOT / "tools"))

import trace as T  # noqa: E402


@pytest.fixture(scope="module")
def doc():
    return T.build()


@pytest.fixture(scope="module")
def numbers_tex():
    return (PAPER / "numbers.tex").read_text(encoding="utf-8")


# ------------------------------------------------------------- completeness --
def test_every_emitted_macro_is_in_numbers_tex(doc, numbers_tex):
    printed = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", numbers_tex))
    assert printed == set(doc["macros"]), (
        "numbers.tex and the build script disagree about the macro set; "
        "run python paper_aij/build_assets.py")


def test_every_macro_resolves_to_a_result_file(doc):
    missing = [n for n, m in doc["macros"].items() if not m["result_file"]]
    assert not missing, f"macros with no provenance: {missing}"


def _result_path(doc, rel: str) -> Path:
    return ROOT / "results" / doc["run_id"] / (rel if rel.endswith(".json") else f"{rel}.json")


def test_every_resolved_file_exists_in_the_run(doc):
    for name, m in doc["macros"].items():
        p = _result_path(doc, m["result_file"])
        assert p.exists(), f"{name} points at a file that is not in the run: {p}"


def test_trace_value_equals_the_value_the_paper_prints(doc, numbers_tex):
    """The trace must not re-derive: it reports the emitted string verbatim."""
    for m in re.finditer(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}", numbers_tex):
        name, value = m.group(1), m.group(2)
        assert doc["macros"][name]["value"] == value


# ------------------------------------------------------------- provenance ----
@pytest.mark.parametrize("macro,expected_file,expected_path", [
    # a plain field: one number, one key
    ("IeeeRows", "data/ieee_cis", "schema.rows"),
    ("GasPerTx", "s3_protocol/onchain_gas", "per_tx_log_gas_mean"),
    ("LlmAucPr", "s8_llm/zero_shot_0_5b", "elicitation.choice.llm.auc_pr"),
    ("RhoMean", "s1_pswa/quality_bound", "quality_bound.rho_mean"),
    ("SftRows", "s8_llm/lora_sft_0_5b", "n_train_rows"),
    # a nested list index
    ("ChalDetLow", "s3_protocol/challenge_grid", "rows.[0].detection_rate"),
])
def test_known_macro_resolves_to_the_expected_field(doc, macro, expected_file,
                                                    expected_path):
    m = doc["macros"][macro]
    assert m["result_file"] == expected_file, (macro, m["result_file"])
    assert m["json_path"] == expected_path, (macro, m["json_path"])


def test_derived_macro_keeps_its_expression(doc):
    """A ratio is not a field, so the trace must show the code, not a path."""
    for macro in ("VramTimesSdpa", "LatSlidingRatio", "AnchorShareAtLargestN"):
        m = doc["macros"][macro]
        assert m["json_path"] in (None, "", "[]"), (macro, m["json_path"])
        assert m["expr"], f"{macro} has neither a path nor an expression"


def test_json_path_actually_opens(doc):
    """Follow three traced paths through the real JSON and compare the value."""
    for macro in ("IeeeRows", "GasPerTx", "LlmAucPr"):
        m = doc["macros"][macro]
        data = json.loads(_result_path(doc, m["result_file"]).read_text(encoding="utf-8"))
        node = data
        for part in m["json_path"].split("."):
            node = node[int(part[1:-1])] if part.startswith("[") else node[part]
        printed = m["value"].replace("{,}", "").replace("\\%", "")
        assert f"{node:.4f}" == printed or f"{node}" == printed \
            or f"{node:,.1f}".replace(",", "") == printed, (macro, node, printed)


# --------------------------------------------------------------- code side ---
def test_unit_table_covers_the_units_the_paper_quotes(doc):
    """>= 30 units is a floor, not a target: it catches a broken parser."""
    assert doc["n_units"] >= 30


def test_every_mapped_unit_has_a_runnable_command(doc):
    for name, m in doc["macros"].items():
        c = m["code"]
        if not c:
            continue
        assert re.fullmatch(r"python -m experiments\.run_all --only \w+", c["command"]), \
            (name, c["command"])
        stage = c["command"].rsplit(" ", 1)[-1]
        assert stage in {"data", "s1_pswa", "s2_gbl", "s3_protocol", "s4_stats",
                         "s5_baselines", "s6_drift", "s8_llm"}, (name, stage)
        mod = ROOT / (c["module"].replace(".", "/") + ".py")
        assert mod.exists(), (name, mod)
        assert f"def {c['function']}(" in mod.read_text(encoding="utf-8"), \
            (name, c["function"])


def test_paper_locations_point_at_lines_that_mention_the_macro(doc):
    checked = 0
    for name, m in doc["macros"].items():
        for loc in m["paper"]:
            if not loc["line"] or loc["file"].startswith("tables/"):
                continue
            text = (PAPER / loc["file"]).read_text(encoding="utf-8").splitlines()
            assert f"\\{name}" in text[loc["line"] - 1], (name, loc)
            checked += 1
    assert checked > 100, "the paper index found suspiciously few uses"


# ------------------------------------------------------------------- CLI -----
def test_cli_macro_lookup_prints_the_provenance(capsys):
    assert T.main(["--macro", "VramTimesSdpa"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["value"] == "669"
    assert out["result_file"] == "s1_pswa/gpu_memory_latency"


def test_cli_rejects_an_unknown_macro(capsys):
    assert T.main(["--macro", "NotAMacro"]) == 1
    assert "unknown macro" in capsys.readouterr().err


# --------------------------------------------------------------- the file ----
def test_traceability_md_is_current():
    """The checked-in table must be what the tool emits today."""
    doc = T.build()
    want = T.markdown(doc)
    have = (PAPER / "TRACEABILITY.md").read_text(encoding="utf-8")
    assert have == want, "paper_aij/TRACEABILITY.md is stale; run python tools/trace.py"
