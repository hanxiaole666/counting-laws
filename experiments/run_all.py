"""Single-command orchestrator with crash-safe checkpoint / resume.

    python -m experiments.run_all                 # auto-resume from last run
    python -m experiments.run_all --fresh          # start a brand-new run
    python -m experiments.run_all --only s1_pswa,s3_protocol
    python -m experiments.run_all --from s4_stats
    python -m experiments.run_all --status
    python -m experiments.run_all --run-id myrun  # resume a named run

Data is downloaded automatically when missing and reused when present
(``experiments.data_io.ensure_dataset``). Killing the process mid-run is safe:
restarting with the same run_id skips completed units and re-runs only the
unit that was interrupted.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .common import (PAPER_DIR, RESULTS_ROOT, ROOT as CROOT, RunState,
                     atomic_write_json, atomic_write_text, environment,
                     load_json, utcnow)
from . import stages

CONFIG = Path(__file__).with_name("config.yaml")

ORDER = ["data", "s1_pswa", "s2_gbl", "s3_protocol", "s4_stats",
         "s5_baselines", "s6_drift", "s8_llm", "s7_figures"]


def _stage_pswa(state, cfg):
    """PSWA masks (CPU) followed by the real CUDA VRAM/latency benchmark."""
    log = stages.stage_pswa(state, cfg)
    from .stages2 import stage_gpu
    log.update(stage_gpu(state, cfg))
    return log


def _stage_stats(state, cfg):
    log = stages.stage_stats(state, cfg)
    from .stages2 import stage_stats_extra
    log.update(stage_stats_extra(state, cfg))
    return log


def _stage_llm(state, cfg):
    from .stages2 import stage_llm
    return stage_llm(state, cfg)


HANDLERS = {
    "data": stages.stage_data,
    "s1_pswa": _stage_pswa,
    "s2_gbl": stages.stage_gbl,
    "s3_protocol": stages.stage_protocol,
    "s4_stats": _stage_stats,
    "s5_baselines": stages.stage_baselines,
    "s6_drift": stages.stage_drift,
    "s8_llm": _stage_llm,
}


def load_cfg() -> dict:
    with open(CONFIG, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def status_table(state: RunState) -> str:
    lines = ["unit".ljust(38) + "status".ljust(14) + "attempts"]
    lines.append("-" * 66)
    for key in sorted(state.doc["units"]):
        r = state.doc["units"][key]
        lines.append(key.ljust(38) + r.get("status", "?").ljust(14)
                     + str(r.get("attempts", "")))
    return "\n".join(lines)


def write_metadata(state: RunState, cfg: dict, started: float,
                   outcomes: dict, errors: dict) -> None:
    env = environment()
    blocked = {k: v for k, v in state.doc["units"].items()
               if v.get("status") == "blocked"}
    meta = {
        "run_id": state.run_id, "started_at": utcnow(),
        "elapsed_s": round(time.time() - started, 2),
        "suite_version": "1.0.0",
        "config": cfg,
        "config_sha256": __import__("hashlib").sha256(
            CONFIG.read_bytes()).hexdigest(),
        "environment": env,
        "data_manifest": (load_json(ROOT / "data" / "MANIFEST.json")
                          if (ROOT / "data" / "MANIFEST.json").exists() else {}),
        "outcomes": outcomes, "errors": errors,
        "blocked_units": {k: {"reason": v["reason"], "resume_cmd": v["resume_cmd"]}
                          for k, v in blocked.items()},
        "summary": state.summary(),
        "honesty": {
            "mock_results_used_in_manuscript": False,
            "example_data_used": False,
            "cross_family_review": False,
            "note": "All reported numbers come from results/<run_id>/*.json. "
                    "Blocked units are NOT filled with estimates.",
        },
    }
    atomic_write_json(ROOT / "RUN_METADATA.json", meta)
    atomic_write_json(state.results_dir / "RUN_METADATA.json", meta)

    lines = ["# Blocked experiments (honest disclosure)", "",
             "These units could NOT be executed on this host. Their numbers are",
             "absent from the manuscript (marked `— (blocked)`), never estimated.",
             ""]
    for k, v in sorted(blocked.items()):
        lines += [f"## `{k}`", "", f"- **Reason**: {v['reason']}",
                  f"- **Resume command**: `{v['resume_cmd']}`", ""]

    # Limitations that are real but are not units: they bound the claims just as
    # hard, so they are listed here rather than left implicit.
    lines += [
        "## Known limitations that are not suite units", "",
        "| item | why it is not measured | what it would take |",
        "|---|---|---|",
        "| ZKML proof of inference | a 0.5B-model proof circuit is infeasible on "
        "6 GiB; claiming it would require a prover-grade GPU | >=80 GiB device or "
        "a circuit for the last layer only |",
        "| 7B / 8B full-precision LLM baseline | needs >=24 GiB of device memory; "
        "the measured ladder stops at 1.5B and any 7B point would be an "
        "extrapolation of a two-point fit | a 24 GiB card, or a separate 4-bit run "
        "reported as its own measurement |",
        "| cross-family independent review | requires reviewers outside this "
        "project; same-family self-review is only provisional | 2-3 external cold "
        "reads, recorded in paper/GAP_TO_ACCEPTANCE.md section C1 |",
        ""]
    atomic_write_text(PAPER_DIR / "BLOCKED.md", "\n".join(lines) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="VERI-FinRisk experiment suite")
    ap.add_argument("--fresh", action="store_true", help="new run_id, ignore previous progress")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--only", default=None, help="comma-separated stage names")
    ap.add_argument("--from", dest="from_stage", default=None)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--force", default=None,
                    help="comma-separated unit keys to invalidate before "
                         "running, e.g. s1_pswa/block_alignment")
    args = ap.parse_args(argv)

    state = RunState(run_id=args.run_id, fresh=args.fresh)
    if args.status:
        print(status_table(state))
        print(json.dumps(state.summary(), ensure_ascii=False, indent=2))
        return 0

    cfg = load_cfg()
    if args.force:
        keys = [k.strip() for k in args.force.split(",") if k.strip()]
        for k in keys:
            ok = state.forget(k)
            print(f"force: {'invalidated' if ok else 'no record for'} {k}")
    todo = ORDER
    if args.only:
        wanted = [x.strip() for x in args.only.split(",")]
        todo = [s for s in ORDER if s in wanted]
    if args.from_stage:
        i = ORDER.index(args.from_stage)
        todo = [s for s in ORDER[i:] if s in todo]
    if args.no_figures:
        todo = [s for s in todo if s != "s7_figures"]

    print(f"run_id = {state.run_id}   resume = {not args.fresh}")
    print(f"stages = {', '.join(todo)}")
    print(f"python = {sys.executable}")
    started = time.time()
    outcomes, errors = {}, {}
    for stage in todo:
        print(f"\n=== STAGE {stage} ===", flush=True)
        try:
            if stage == "s7_figures":
                from .make_figures import stage_figures
                log = stage_figures(state, cfg)
            else:
                log = HANDLERS[stage](state, cfg)
            outcomes[stage] = log
            done = sum(v == "done" for v in log.values())
            skipped = sum(v == "skipped" for v in log.values())
            blocked_n = sum(v == "blocked" for v in log.values())
            print(f"    done={done} skipped(resumed)={skipped} blocked={blocked_n}",
                  flush=True)
        except Exception as exc:
            errors[stage] = repr(exc)
            print(f"    FAILED: {exc}", flush=True)
            traceback.print_exc()
            write_metadata(state, cfg, started, outcomes, errors)
            print("\nFix the error above then re-run the SAME command; completed "
                  "units are skipped automatically.", flush=True)
            return 1

    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    write_metadata(state, cfg, started, outcomes, errors)
    print("\n=== SUMMARY ===")
    print(status_table(state))
    print(json.dumps(state.summary(), ensure_ascii=False))
    print(f"results -> {state.results_dir}")
    print(f"elapsed {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
