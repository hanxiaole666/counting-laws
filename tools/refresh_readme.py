"""Keep the numbers quoted in README.md generated, not typed.

A README is documentation, and documentation rots.  The one part of this
repository that must never rot is the part that quotes a measurement: if the
README says the tool prints 358,906 entries, that string has to be whatever the
tool prints today.

This script rewrites the marked blocks in ``README.md`` from live execution and
from ``results/``.  ``--check`` compares instead of writing and exits non-zero
when a block is stale, which is what the test-suite runs.

Blocks are delimited by

    <!-- BEGIN:<name> -->
    ...
    <!-- END:<name> -->

Usage
    python tools/refresh_readme.py            # rewrite
    python tools/refresh_readme.py --check    # verify only
"""
from __future__ import annotations

import argparse
import functools
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
PAPER = ROOT / "paper_aij"

sys.path.insert(0, str(ROOT / "tools"))


# --------------------------------------------------------------------- blocks
def block_countlaws() -> str:
    """Live output of the reader-facing instrument, both in one fence."""
    cmds = [
        [sys.executable, "tools/countlaws.py", "--n", "4096", "--w", "64",
         "--g", "96", "--s", "4"],
        [sys.executable, "tools/countlaws.py", "--n", "32768", "--w", "64",
         "--g", "256", "--s", "4", "--block", "64"],
    ]
    out = []
    for cmd in cmds:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"{' '.join(cmd)} failed:\n{r.stderr}")
        out.append(r.stdout.rstrip("\n"))
    head = "$ python tools/countlaws.py --n 4096 --w 64 --g 96 --s 4"
    head2 = ("$ python tools/countlaws.py --n 32768 --w 64 --g 256 --s 4 "
             "--block 64   # the setting of the GPU benchmark")
    body = "\n".join([head, out[0], "", head2, out[1]])
    return "```\n" + body + "\n```"


@functools.lru_cache(maxsize=1)
def _stats() -> dict:
    import trace as T
    doc = T.build()
    macros = doc["macros"]
    used = sum(1 for m in macros.values() if m["paper"])
    units = len(T.unit_index())
    # parametrised cases are what a reader counts, so ask the collector rather
    # than counting ``def test_*`` by hand
    tests = 0
    r = subprocess.run([sys.executable, "-m", "pytest", "tests",
                        "--collect-only", "-q"], cwd=ROOT,
                       capture_output=True, text=True)
    m = re.search(r"(\d+) tests? collected", r.stdout)
    if m:
        tests = int(m.group(1))
    run_dir = ROOT / "results" / doc["run_id"]
    results = len(list(run_dir.glob("*/*.json"))) if run_dir.exists() else 0
    summary = {}
    meta = run_dir / "RUN_METADATA.json"
    if meta.exists():
        summary = json.loads(meta.read_text(encoding="utf-8")).get("summary", {})
    counts = summary.get("counts", {})
    return {"run": doc["run_id"], "macros": len(macros), "used": used,
            "units": units, "tests": tests, "results": results,
            "unused": len(macros) - used,
            "done": counts.get("done", 0),
            "blocked": counts.get("blocked", 0),
            "total": summary.get("total_units", 0)}


def block_stats() -> str:
    s = _stats()
    rows = [
        ("source run", f"`{s['run']}`"),
        ("result files", f"{s['results']} JSON under `results/<run_id>/`"),
        ("experiment units",
         f"{s['total']} ({s['done']} completed, {s['blocked']} blocked, all resumable)"),
        ("LaTeX macros generated from results", f"{s['macros']}"),
        ("of which cited in the manuscript", f"{s['used']}"),
        ("test cases (`pytest`)", f"{s['tests']}"),
    ]
    lines = ["| | |", "|---|---|"]
    lines += [f"| {a} | {b} |" for a, b in rows]
    return "\n".join(lines)


def block_trace_sample() -> str:
    """Six rows of the generated provenance table: one per result family."""
    doc_path = PAPER / "traceability.json"
    if not doc_path.exists():
        raise SystemExit("run python tools/trace.py first")
    doc = json.loads(doc_path.read_text(encoding="utf-8"))
    want = ["IeeeFraudPct", "RhoMean", "LargestN", "LlmAucPr",
            "GasSavingPct", "CurveSmallAucPr"]
    L = ["| number in the paper | value | read from | JSON path |", "|---|---|---|---|"]
    for name in want:
        m = doc["macros"].get(name)
        if not m:
            continue
        path = m["json_path"]
        path = f"`{path}`" if path not in (None, "", "[]") else "(derived)"
        rel = m["result_file"]
        rel = rel if rel.endswith(".json") else f"{rel}.json"
        L.append(f"| `\\{name}` | {m['value']} | `{rel}` | {path} |")
    return "\n".join(L)


BLOCKS = {
    "countlaws-demo": block_countlaws,
    "stats": block_stats,
    "trace-sample": block_trace_sample,
}


# ----------------------------------------------------------------------- main
def render(text: str) -> str:
    for name, fn in BLOCKS.items():
        pat = re.compile(
            rf"(<!-- BEGIN:{re.escape(name)} -->\n)(.*?)(<!-- END:{re.escape(name)} -->)",
            re.S)
        if not pat.search(text):
            raise SystemExit(f"README.md has no block {name!r}")
        body = fn()
        text = pat.sub(lambda m: m.group(1) + body + "\n" + m.group(3), text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    old = README.read_text(encoding="utf-8")
    new = render(old)
    if args.check:
        if new != old:
            # show the first differing block, not the whole file
            for i, (a, b) in enumerate(zip(old.splitlines(), new.splitlines()), 1):
                if a != b:
                    print(f"README.md is stale at line {i}:\n  have: {a}\n  want: {b}")
                    break
            else:
                print("README.md differs in length from the regenerated version")
            return 1
        print("README.md blocks are current")
        return 0
    README.write_text(new, encoding="utf-8")
    print("README.md blocks refreshed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
