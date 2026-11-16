"""One entry point per question, on any platform.

`make` is the conventional way to write this down, and there is a Makefile here,
but the reference machine for this artefact is Windows and `make` is not always
installed. So the commands live in this file and the Makefile is a thin wrapper:
there is exactly one implementation of each task, and `python tasks.py <name>`
works in PowerShell, cmd, bash and CI alike.

    python tasks.py                 # list tasks
    python tasks.py check           # everything verifiable without a GPU
    python tasks.py trace           # regenerate the paper→code table
    python tasks.py paper           # rebuild main.pdf from the stored results
    python tasks.py --help

Every task prints the exact command it runs, so a reader can copy it out and
stop using this file at any point. `--dry-run` prints without executing.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
PAPER = "paper_aij"

# name -> (description, [(cwd, argv), ...])
TASKS: dict[str, tuple[str, list[tuple[str, list[str]]]]] = {
    "install": ("CPU dependency set (add install-gpu for the rest)", [
        (".", [PY, "-m", "pip", "install", "-r", "requirements.lock"]),
    ]),
    "install-gpu": ("torch+CUDA, transformers, peft, solc, py-evm", [
        (".", [PY, "-m", "pip", "install", "-r", "requirements.lock",
               "-r", "requirements-gpu.lock"]),
    ]),
    "data": ("download the two datasets if absent, verify sha256", [
        (".", [PY, "-m", "experiments.data_io", "--ensure", "--datasets",
               "ieee_cis,baf"]),
    ]),
    "run": ("the full suite; re-running the same command resumes", [
        (".", [PY, "-m", "experiments.run_all"]),
    ]),
    "run-fresh": ("the full suite from zero (new run id)", [
        (".", [PY, "-m", "experiments.run_all", "--fresh"]),
    ]),
    "status": ("what each unit did, and which units are blocked", [
        (".", [PY, "-m", "experiments.run_all", "--status"]),
    ]),
    "test": ("the test suite", [
        (".", [PY, "-m", "pytest", "tests", "-q"]),
    ]),
    "numbers": ("results → numbers.tex + tables/*.tex", [
        (".", [PY, f"{PAPER}/build_assets.py"]),
    ]),
    "figs": ("results → six vector figures", [
        (".", [PY, f"{PAPER}/make_figs.py"]),
    ]),
    "trace": ("results → paper_aij/TRACEABILITY.md", [
        (".", [PY, "tools/trace.py"]),
    ]),
    "readme": ("regenerate the numbers quoted in README.md", [
        (".", [PY, "tools/refresh_readme.py"]),
    ]),
    "verify": ("tests + hashes + second derivation + provenance", [
        (".", [PY, "-m", "pytest", "tests", "-q"]),
        (".", [PY, "-m", "experiments.verify_artifacts"]),
        (".", [PY, f"{PAPER}/check_numbers.py"]),
        (".", [PY, "tools/trace.py", "--check"]),
    ]),
    "check": ("verify, plus the README's quoted numbers", [
        (".", [PY, "-m", "pytest", "tests", "-q"]),
        (".", [PY, "-m", "experiments.verify_artifacts"]),
        (".", [PY, f"{PAPER}/check_numbers.py"]),
        (".", [PY, "tools/trace.py", "--check"]),
        (".", [PY, "tools/refresh_readme.py", "--check"]),
    ]),
    "paper": ("numbers + figures + pdflatex ×3", [
        (".", [PY, f"{PAPER}/build_assets.py"]),
        (".", [PY, f"{PAPER}/make_figs.py"]),
        (PAPER, ["pdflatex", "-interaction=nonstopmode", "main"]),
        (PAPER, ["bibtex", "main"]),
        (PAPER, ["pdflatex", "-interaction=nonstopmode", "main"]),
        (PAPER, ["pdflatex", "-interaction=nonstopmode", "main"]),
    ]),
    "clean": ("LaTeX and python intermediates (results/ untouched)", []),
    "countlaws": ("the reader's instrument, at the paper's reference setting", [
        (".", [PY, "tools/countlaws.py", "--n", "4096", "--w", "64",
               "--g", "96", "--s", "4"]),
    ]),
}

CLEAN_GLOBS = [f"{PAPER}/main.aux", f"{PAPER}/main.bbl", f"{PAPER}/main.blg",
               f"{PAPER}/main.log", f"{PAPER}/main.out", f"{PAPER}/main.spl"]


def _clean() -> None:
    for rel in CLEAN_GLOBS:
        p = ROOT / rel
        if p.exists():
            p.unlink()
            print(f"rm {rel}")
    for d in list(ROOT.rglob("__pycache__")):
        shutil.rmtree(d, ignore_errors=True)
    prev = ROOT / PAPER / "preview"
    if prev.exists():
        shutil.rmtree(prev, ignore_errors=True)
        print(f"rm {PAPER}/preview")


def run(name: str, dry_run: bool = False) -> int:
    desc, cmds = TASKS[name]
    print(f"# {name}: {desc}")
    for cwd, argv in cmds:
        printable = " ".join(argv)
        print(f"  [{cwd}] {printable}")
        if dry_run:
            continue
        if not (ROOT / cwd).exists():
            print(f"  ! missing directory {cwd}", file=sys.stderr)
            return 1
        r = subprocess.run(argv, cwd=ROOT / cwd)
        if r.returncode != 0:
            print(f"  ! failed: {printable}", file=sys.stderr)
            return r.returncode
    if name == "clean" and not dry_run:
        _clean()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", nargs="?", help="task name; omit to list them")
    ap.add_argument("--dry-run", action="store_true", help="print, do not execute")
    args = ap.parse_args(argv)

    if not args.task:
        width = max(len(k) for k in TASKS)
        print(__doc__.split("Every task prints")[0].strip())
        print()
        for name, (desc, _) in TASKS.items():
            print(f"  {name.ljust(width)}  {desc}")
        return 0

    if args.task not in TASKS:
        print(f"unknown task {args.task!r}; try: {', '.join(TASKS)}", file=sys.stderr)
        return 2
    return run(args.task, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
