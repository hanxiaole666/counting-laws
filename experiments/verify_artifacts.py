"""Recompute every manuscript number from raw logs and verify the data chain.

    python -m experiments.verify_artifacts

Emits ARTIFACT_VERIFY.json with PASS / FAIL. A failure is NEVER auto-repaired.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .common import (ROOT, atomic_write_json, load_json, sha256_file, utcnow)


def main() -> int:
    chain_path = ROOT / "ARTIFACT_CHAIN.json"
    meta_path = ROOT / "RUN_METADATA.json"
    checks = []
    ok = True

    def chk(name: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
        if not passed:
            ok = False

    chk("artifact_chain_exists", chain_path.exists())
    chk("run_metadata_exists", meta_path.exists())
    if not chain_path.exists() or not meta_path.exists():
        out = {"verdict": "FAIL", "checks": checks, "generated_at": utcnow()}
        atomic_write_json(ROOT / "ARTIFACT_VERIFY.json", out)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 1

    chain = load_json(chain_path)
    meta = load_json(meta_path)
    run_id = meta["run_id"]
    results = ROOT / "results" / run_id

    # 1. every registered input file still hashes to the recorded value
    for rel, digest in chain["inputs"].items():
        p = results / rel
        chk(f"input_sha256::{rel}", p.exists() and sha256_file(p) == digest,
            "missing" if not p.exists() else "")

    # 2. every figure still hashes to the recorded value
    for f in chain["figures"]:
        p = ROOT / f["figure"]
        chk(f"figure_sha256::{f['figure']}", p.exists() and sha256_file(p) == f["sha256"],
            "missing" if not p.exists() else "")

    # 3. all result JSONs are real (mock == false) and all assertions pass.
    #
    # Only true result envelopes count. This guard is not cosmetic: a LoRA
    # adapter saved under results/s8_llm/lora_adapter/ ships a tokenizer
    # vocab.json, and that map happens to contain the *tokens* "stage" and
    # "mock", so `"stage" in d` alone classified a vocabulary file as a result
    # envelope and its `{"mock": 198}` entry as a mock run. Requiring the full
    # envelope signature (stage + unit + run_id) removes the ambiguity.
    n_results = 0
    assertion_failures = []
    mock_hits = []
    hypothesis_unreported = []
    refuted_hypotheses = []
    for p in sorted(results.rglob("*.json")):
        try:
            d = load_json(p)
        except json.JSONDecodeError:
            continue
        if not isinstance(d, dict) or not {"stage", "unit", "run_id"} <= set(d):
            continue
        n_results += 1
        if d.get("mock"):
            mock_hits.append(str(p.relative_to(ROOT)))
        for k, v in (d.get("assertions") or {}).items():
            if v is not True:
                assertion_failures.append(f"{p.relative_to(ROOT)}::{k}={v}")
        # Hypotheses are allowed to be False (that is a refutation, not a bug),
        # but each must be decided -- never left as None/missing.
        for k, v in (d.get("hypotheses") or {}).items():
            if not isinstance(v, bool):
                hypothesis_unreported.append(f"{p.relative_to(ROOT)}::{k}={v!r}")
            elif v is False:
                refuted_hypotheses.append(f"{p.relative_to(ROOT)}::{k}")
    chk("real_results_present", n_results >= 15, f"n_results={n_results}")
    chk("no_mock_results", not mock_hits, ",".join(mock_hits))
    chk("all_assertions_pass", not assertion_failures, ";".join(assertion_failures[:8]))
    chk("every_hypothesis_is_decided", not hypothesis_unreported,
        ";".join(hypothesis_unreported[:5]))
    chk("refuted_hypotheses_are_listed", True,
        f"{len(refuted_hypotheses)} refuted: " + ";".join(refuted_hypotheses[:6]))

    # 4. no placeholder numbers in tables or raw results
    tables = (ROOT / "paper" / "TABLES.md")
    chk("no_example_data_in_tables",
        tables.exists() and "EXAMPLE DATA" not in tables.read_text(encoding="utf-8"))
    # Only result payloads matter; ARTIFACT_CHAIN.json legitimately mentions
    # the phrase in its policy description and is not a numeric result.
    placeholder_hits = []
    for p in results.rglob("*.json"):
        raw = p.read_text(encoding="utf-8", errors="ignore")
        if '"stage"' in raw and "EXAMPLE DATA" in raw:
            placeholder_hits.append(str(p.relative_to(ROOT)))
    chk("no_example_data_in_results", not placeholder_hits,
        ",".join(placeholder_hits[:5]))

    # 5. recompute headline numbers straight from raw logs
    main = load_json(results / "s5_baselines" / "main_table.json")
    table = (ROOT / "paper" / "TABLES.md").read_text(encoding="utf-8")
    manuscript = (ROOT / "paper" / "MANUSCRIPT.md").read_text(encoding="utf-8")
    for r in main["rows"]:
        chk(f"table_number::{r['system']}::auc_pr",
            f"{r['auc_pr']:.4f}" in table, f"expected {r['auc_pr']:.4f}")

    qual = load_json(results / "s1_pswa" / "quality_bound.json")
    chk("pswa_apparatus_sparse", qual["assertions"]["apparatus_mask_is_sparse"],
        f"nnz reduction measured")
    chk("pswa_quality_bound", qual["assertions"]["quality_bound_holds"],
        f"emp={qual['quality_bound']['empirical_max_abs_diff']:.6g} "
        f"bound={qual['quality_bound']['theoretical_bound_max']:.6g}")
    chk("pswa_claim_status_recorded", "claims" in qual and "decision" in qual,
        qual.get("decision", ""))

    prot = load_json(results / "s3_protocol" / "merkle_inclusion.json")
    chk("third_party_verification", all(
        prot["assertions"][k] for k in prot["assertions"]), "")

    st = load_json(results / "s4_stats" / "seed_sweep.json")
    chk("seeds_actually_change_results", st["assertions"]["seeds_actually_change_results"],
        f"distinct={st['n_distinct_results']}")

    # 5b. GPU numbers must be real device measurements, not host RAM
    gpu_p = results / "s1_pswa" / "gpu_memory_latency.json"
    chk("gpu_unit_present", gpu_p.exists())
    if gpu_p.exists():
        g = load_json(gpu_p)
        dev = g["bench"]["device"]
        chk("gpu_is_cuda", bool(dev.get("cuda")), dev.get("name", ""))
        chk("gpu_measured_flag", g["assertions"].get("cuda_measured") is True, "")
        chk("gpu_records_oom_evidence",
            isinstance(g["claims"].get("n_where_dense_oom_but_pswa_runs"), list),
            str(g["claims"].get("n_where_dense_oom_but_pswa_runs")))
        chk("gpu_vram_table_in_paper",
            f"{g['claims']['max_vram_reduction_vs_materialised']}" not in (None, "None")
            and "Table 9" in table, "Table 9 present")

    # 5c. LLM numbers must come from a real forward pass with counted failures
    llm_p = results / "s8_llm" / "zero_shot_0_5b.json"
    chk("llm_unit_present", llm_p.exists())
    if llm_p.exists():
        llm = load_json(llm_p)
        chk("llm_scores_not_constant",
            llm["assertions"]["scores_are_not_constant"], "")
        chk("llm_parse_failures_counted",
            "parse_failure_rate" in llm["json_interface"], "")
        chk("llm_emits_real_device_vram",
            bool(llm["vram"].get("cuda")), "")
        chk("llm_auc_in_paper",
            f"{llm['elicitation']['choice']['llm']['auc_pr']:.4f}" in table, "")
    sft_p = results / "s8_llm" / "lora_sft_0_5b.json"
    if sft_p.exists():
        sft = load_json(sft_p)
        chk("sft_targets_are_continuous", sft["assertions"]["targets_are_continuous"],
            f"n_distinct={sft['target_stats']['n_distinct']}")
        chk("sft_targets_not_constant", sft["assertions"]["targets_not_constant"],
            f"sd={sft['target_stats']['sd']:.6f}")
        # The LoRA run collapses ranking to chance. That is a finding, and it is
        # only a *reported* finding if the collapse is visible in the manuscript;
        # otherwise the negative result is buried in a JSON nobody reads.
        if llm_p.exists():
            base = load_json(llm_p)
            b_roc = base["elicitation"]["choice"]["llm"]["auc_roc"]
            s_roc = sft["elicitation"]["choice"]["llm"]["auc_roc"]
            chk("lora_collapse_is_in_the_numbers", s_roc < b_roc,
                f"AUC-ROC {b_roc:.4f} -> {s_roc:.4f}")
            chk("lora_collapse_reported_in_paper",
                f"{s_roc:.4f}" in manuscript,
                "post-LoRA AUC-ROC appears in the manuscript text")
        chk("lora_metadata_is_complete",
            sft["json_interface"]["parse_failure_rate"] == 0.0
            and sft["sft"]["trainable_params"] > 0,
            f"parse_failure_rate={sft['json_interface']['parse_failure_rate']}")

    # 5c-bis. a retracted claim must be retracted in the artefact AND in the text
    al_p = results / "s1_pswa" / "block_alignment.json"
    chk("block_alignment_present", al_p.exists())
    if al_p.exists():
        al = load_json(al_p)
        chk("retracted_claim_is_flagged",
            al["assertions"].get("former_lemma_withdrawn") is True, "")
        chk("retraction_reaches_the_manuscript",
            ("withdrawn" in manuscript.lower() or "retract" in manuscript.lower()),
            "manuscript states the retraction")
        if al.get("rows_by_g"):
            rs = [r for r in al["rows_by_g"] if not r["block_ge_g"]]
            chk("alignment_law_fits", all(
                0.95 <= r["ratio_measured_to_predicted"] <= 1.60 for r in rs),
                str([r["ratio_measured_to_predicted"] for r in rs]))

    # 5d. gas must be consistent with the measured per-operation costs
    gas_p = results / "s3_protocol" / "onchain_gas.json"
    chk("onchain_gas_present", gas_p.exists())
    if gas_p.exists():
        gd = load_json(gas_p)
        recomputed = 1 - gd["batch_root_gas"] / (
            gd["per_tx_log_gas_mean"] * gd["batch_size"])
        chk("gas_saving_recomputable", abs(recomputed - gd["gas_saving_fraction"]) < 5e-3,
            f"recomputed={recomputed:.4f} reported={gd['gas_saving_fraction']}")
        chk("gas_measured_on_evm", gd["assertions"]["gas_measured_on_evm"] is True, "")

    # 5e. every unit that used to be blocked and is now measured must say so
    for unit, rel in (("s1_pswa/gpu_vram_scan", "s1_pswa/gpu_memory_latency.json"),
                      ("s5_baselines/llm_scale_scan", "s5_baselines/llm_scale_scan.json")):
        chk(f"former_blocked_unit_measured::{unit}", (results / rel).exists(),
            "superseded by a real measurement")

    # 6. blocked units must be disclosed, not silently filled
    blocked = meta.get("blocked_units", {})
    blocked_doc = ROOT / "paper" / "BLOCKED.md"
    chk("blocked_units_disclosed",
        blocked_doc.exists() and all(k in blocked_doc.read_text(encoding="utf-8")
                                     for k in blocked),
        f"n_blocked={len(blocked)}")

    out = {
        "verdict": "PASS" if ok else "FAIL",
        "run_id": run_id, "generated_at": utcnow(),
        "n_result_files": n_results,
        "n_blocked_units": len(blocked),
        "blocked_units": list(blocked),
        "checks": checks,
        "passed": sum(c["passed"] for c in checks),
        "total": len(checks),
        "honesty": {"cross_family_review": False,
                    "submission_ready": False,
                    "note": "PASS means local artifacts are internally consistent, "
                            "NOT that the paper is accepted or submission-ready."},
    }
    atomic_write_json(ROOT / "ARTIFACT_VERIFY.json", out)
    print(json.dumps({k: out[k] for k in ("verdict", "passed", "total",
                                          "n_result_files", "n_blocked_units")},
                     ensure_ascii=False, indent=2))
    if not ok:
        print("Failed checks:")
        for c in checks:
            if not c["passed"]:
                print(" -", c["check"], c["detail"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
