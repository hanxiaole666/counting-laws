"""Stage exactly the files that go public, and nothing else.

The working tree is not the artefact. It also holds superseded runs, an earlier
manuscript pipeline, raw datasets and a virtual environment, and every one of
those would be a liability in a public repository: a reader would not know which
numbers the paper uses. So the published tree is *built* from an explicit list
(``paper_aij/github/UPLOAD_MANIFEST.md``), the same way ``numbers.tex`` is built
from the results.

    python tools/stage_release.py                      # write dist/counting-laws
    python tools/stage_release.py --dest /tmp/x        # somewhere else
    python tools/stage_release.py --git                # also git init + commit
    python tools/stage_release.py --verify             # checks on the staged tree

The staged tree gets a ``MANIFEST.sha256`` of every file, so a reader can verify
that what they cloned is what was staged, and the script refuses to overwrite a
destination that contains files it did not write.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = ROOT / "dist" / "counting-laws"

# Tier 1 of the upload manifest, as globs relative to the repository root.
INCLUDE = [
    # front matter and entry points
    "README.md", "Makefile", "tasks.py", "LICENSE", "NOTICE", "DATASHEET.md",
    "CITATION.cff", ".gitignore", ".github/workflows/ci.yml",
    # dependencies and configuration
    "requirements.txt", "requirements.lock", "requirements-gpu.lock",
    "config.yaml",
    # code: the suite, the reader's instrument, the doc generators
    "experiments/*.py", "tools/*.py", "tests/*.py",
    # the run the manuscript is built from (RUN_METADATA.json sits at its root
    # and is what selects the run, so it is not optional)
    "results/run_20261116_094330/RUN_METADATA.json",
    "results/run_20261116_094330/*/*.json",
    # the manuscript and everything that feeds it
    "paper_aij/main.tex", "paper_aij/*.tex", "paper_aij/refs.bib",
    "paper_aij/elsarticle-num.bst", "paper_aij/build_assets.py",
    "paper_aij/make_figs.py", "paper_aij/check_numbers.py",
    "paper_aij/assets.json", "paper_aij/TRACEABILITY.md",
    "paper_aij/traceability.json", "paper_aij/highlights.txt",
    "paper_aij/submission_answers.md", "paper_aij/main.pdf",
    "paper_aij/sections/*.tex", "paper_aij/tables/*.tex",
    "paper_aij/figures/*.pdf", "paper_aij/github/*.md",
    "paper_aij/github/CITATION.cff",
]

# Things that exist in the working tree and must not follow the copy.
EXCLUDE = [
    "results/run_20261116_094330/s8_llm/lora_adapter",   # release asset
    "paper_aij/preview",
    "paper_aij/github/README.md",                        # pointer, not the README
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect() -> list[tuple[Path, Path]]:
    """[(source file, destination path relative to the staged root)]."""
    out: list[Path] = []
    for pat in INCLUDE:
        hits = sorted(ROOT.glob(pat))
        if not hits:
            print(f"  ! nothing matches {pat}", file=sys.stderr)
        out += [p for p in hits if p.is_file()]
    seen, files = set(), []
    for p in out:
        rel = p.relative_to(ROOT)
        if rel in seen:
            continue
        if any(str(rel).startswith(e) for e in EXCLUDE):
            continue
        seen.add(rel)
        files.append(p)

    # If the primary PDF was open in a viewer during the last build, LaTeX writes
    # a second file under a different jobname.  Stage the newest manuscript, not
    # whichever one happened to be writable at the time.
    pairs = [(p, p.relative_to(ROOT)) for p in files]
    alt = ROOT / "paper_aij" / "main_v.pdf"
    main = ROOT / "paper_aij" / "main.pdf"
    if alt.exists() and main.exists() and alt.stat().st_mtime > main.stat().st_mtime:
        print("  note: staging paper_aij/main_v.pdf (newer than main.pdf)")
        pairs = [(alt, Path("paper_aij/main.pdf")) if src == main else (src, rel)
                 for src, rel in pairs]
    return pairs


def stage(dest: Path, dry_run: bool) -> int:
    files = collect()
    if not files:
        print("nothing to stage", file=sys.stderr)
        return 1
    if dest.exists() and any(dest.iterdir()) and not (dest / "MANIFEST.sha256").exists():
        print(f"refusing to write into a non-empty directory I did not create: {dest}",
              file=sys.stderr)
        return 1

    total = 0
    lines = []
    for src, rel in files:
        dst = dest / rel
        total += src.stat().st_size
        lines.append(f"{sha256(src)}  {rel.as_posix()}")
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    if not dry_run:
        (dest / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(files)} files, {total / 2**20:.1f} MiB → {dest}"
          + (" (dry run)" if dry_run else ""))
    return 0


def git_commit(dest: Path, message: str) -> int:
    def run(*args, **kw):
        return subprocess.run(["git", *args], cwd=dest, text=True,
                              capture_output=True, **kw)

    if not (dest / ".git").exists():
        r = run("init", "-b", "main")
        if r.returncode != 0:
            print(r.stderr, file=sys.stderr)
            return r.returncode
    # a missing identity is the usual reason an automated commit fails
    if run("config", "user.email").returncode != 0:
        run("config", "user.email", "hanxiaoleop@163.com")
        run("config", "user.name", "Xiaole Han")
    run("add", "-A")
    r = run("commit", "-m", message)
    if r.returncode != 0 and "nothing to commit" not in (r.stdout + r.stderr):
        print(r.stderr, file=sys.stderr)
        return r.returncode
    log = run("log", "--oneline", "-1").stdout.strip()
    count = run("ls-files").stdout.count("\n")
    print(f"git: {count} files tracked, HEAD {log}")
    return 0


def verify(dest: Path) -> int:
    """Run the public checks against the staged tree, not the working tree."""
    checks = [
        [sys.executable, "tools/countlaws.py", "--n", "4096", "--w", "64",
         "--g", "96", "--s", "4"],
        [sys.executable, "paper_aij/build_assets.py"],
        [sys.executable, "paper_aij/check_numbers.py"],
        [sys.executable, "tools/trace.py", "--check"],
        [sys.executable, "tools/refresh_readme.py", "--check"],
        [sys.executable, "-m", "pytest", "tests", "-q"],
    ]
    bad = 0
    for argv in checks:
        r = subprocess.run(argv, cwd=dest, capture_output=True, text=True)
        ok = r.returncode == 0
        print(f"  {'ok  ' if ok else 'FAIL'} {' '.join(argv[1:])}")
        if not ok:
            bad += 1
            print((r.stdout + r.stderr)[-800:], file=sys.stderr)
    # the suite downloads a dataset and leaves caches behind; the staged tree is
    # a publication, not a working copy, so put it back the way it was
    for junk in ("data", ".pytest_cache"):
        shutil.rmtree(dest / junk, ignore_errors=True)
    for d in list(dest.rglob("__pycache__")):
        shutil.rmtree(d, ignore_errors=True)
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--git", action="store_true", help="git init + first commit")
    ap.add_argument("--verify", action="store_true",
                    help="run the public checks inside the staged tree")
    ap.add_argument("--message", default="counting laws: code, artefact, manuscript")
    args = ap.parse_args(argv)

    rc = stage(args.dest, args.dry_run)
    if rc or args.dry_run:
        return rc
    if args.git:
        rc = git_commit(args.dest, args.message)
        if rc:
            return rc
    if args.verify:
        print("checks on the staged tree:")
        return verify(args.dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
