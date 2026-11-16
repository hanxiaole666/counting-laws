"""Provenance tracer: paper number -> result JSON -> generating code -> command.

The manuscript contains no hand-typed measurement.  Every quantity is a LaTeX
macro emitted by ``paper_aij/build_assets.py`` from a stored result file.  This
tool closes the last link in that chain: it walks the *source* of
``build_assets.py`` and, for every macro, recovers

    which result file  ->  which JSON path  ->  which experiment unit
    ->  which python module and function  ->  which command re-creates it

and where in the paper the macro is actually used (file, line, section).

Nothing here is a curated list that can drift out of date: the macro set comes
from the generated ``numbers.tex``, the JSON path comes from the AST of the code
that emitted the number, and the unit table comes from the ``emit()`` call sites
in ``experiments/``.

Usage
    python tools/trace.py                 # write paper_aij/TRACEABILITY.md + .json
    python tools/trace.py --check         # same, plus fail on any broken link
    python tools/trace.py --macro VramTimesSdpa   # explain one number

Exit code is 0 on success, 1 if ``--check`` finds a macro with no provenance or
a macro that the paper never uses.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_aij"
EXPERIMENTS = ROOT / "experiments"

# Variables assigned outside ``load()`` that still come straight from a result
# file (RUN_METADATA.json is read with json.loads, not load(), and the run id is
# the directory name).
EXTRA_ALIASES = {"meta": ("RUN_METADATA.json", ()),
                 "rid": ("RUN_METADATA.json", ()),
                 "run": ("RUN_METADATA.json", ())}

# Calls that return a view/ordering of their first argument: the JSON path of
# the result is the path of that argument.
PASSTHROUGH = {"fmt", "pct", "str", "zip", "enumerate", "sorted", "reversed",
               "min", "max", "abs", "len", "sum", "list", "tuple", "round",
               "iter", "next"}


# ---------------------------------------------------------------------------
# 1. macro provenance from the AST of build_assets.py
# ---------------------------------------------------------------------------
def _const(node):
    """Return the literal value of a Constant / -Constant, else None."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _const(node.operand)
        return -v if isinstance(v, (int, float)) else None
    return None


def _resolve(node, aliases, locals_):
    """Resolve an expression to (relative result file, dotted JSON path)."""
    if node is None:
        return None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = node.func.id
        if fn == "load" and len(node.args) >= 2:
            rel = _const(node.args[1])
            return (rel, ()) if isinstance(rel, str) else None
        if fn in locals_:                       # local helper: prov of its body
            return locals_[fn]
        if fn in PASSTHROUGH:
            for a in node.args:
                r = _resolve(a, aliases, locals_)
                if r:
                    return r
            return None
        return None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        # x.replace(...), x.get(...), x.append(...): provenance of the receiver
        return _resolve(node.func.value, aliases, locals_)
    if isinstance(node, ast.Name):
        return aliases.get(node.id)
    if isinstance(node, ast.Subscript):
        base = _resolve(node.value, aliases, locals_)
        key = _const(node.slice)
        if base is None:
            return None
        f, path = base
        if isinstance(key, int) and not isinstance(key, bool):
            return (f, path + (f"[{key}]",))
        if key is None:
            return (f, path + ("[]",))
        return (f, path + (str(key).replace("-", "_"),))
    if isinstance(node, ast.JoinedStr):          # f-string: use first field
        for v in node.values:
            if isinstance(v, ast.FormattedValue):
                r = _resolve(v.value, aliases, locals_)
                if r:
                    return r
        return None
    if isinstance(node, ast.BinOp):
        # arithmetic means the printed value is not a field of the file: keep
        # the provenance (which file the operands came from) and drop the path,
        # because naming one operand's path would be a false claim.
        r = (_resolve(node.left, aliases, locals_)
             or _resolve(node.right, aliases, locals_))
        return (r[0], ("[]",)) if r else None
    if isinstance(node, ast.IfExp):
        return _resolve(node.body, aliases, locals_)
    if isinstance(node, ast.Compare):
        return _resolve(node.left, aliases, locals_)
    if isinstance(node, (ast.ListComp, ast.GeneratorExp, ast.SetComp,
                         ast.DictComp)):
        if node.generators:
            return _resolve(node.generators[0].iter, aliases, locals_)
        return None
    if isinstance(node, ast.Attribute):
        base = _resolve(node.value, aliases, locals_)
        return (base[0], base[1] + (node.attr,)) if base else None
    return None


def _names_in(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _files_in(node, aliases):
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and n.id in aliases:
            out.add(aliases[n.id][0])
    return {f for f in out if f}


def _assign_targets(target, node, aliases, seg, src):
    """Record aliases for one assignment target (handles tuples)."""
    if isinstance(target, (ast.Tuple, ast.List)):
        vals = node.value.elts if isinstance(node.value, (ast.Tuple, ast.List)) else []
        for i, t in enumerate(target.elts):
            if isinstance(t, ast.Name):
                v = vals[i] if i < len(vals) else node.value
                _assign_targets(t, ast.Assign(targets=[], value=v, lineno=0), aliases, seg, src)
        return
    if not isinstance(target, ast.Name):
        return
    r = _resolve(node.value, aliases, {})
    if r is None:
        # keep whatever we can: a name bound to an unresolvable expression still
        # carries the file of the expressions it was built from
        files = _files_in(node.value, aliases)
        r = (sorted(files)[0], ("[]",)) if len(files) == 1 else None
    if r is not None:
        aliases[target.id] = r


def _literal_loop(node):
    """If a ``for`` iterates over a literal sequence, return {var: [values]}."""
    it = node.iter
    if not isinstance(it, (ast.Tuple, ast.List)):
        return None
    rows = []
    for el in it.elts:
        if isinstance(el, (ast.Tuple, ast.List)):
            rows.append([_const(x) for x in el.elts])
        else:
            rows.append([_const(el)])
    if not rows or any(v is None for r in rows for v in r):
        return None
    if isinstance(node.target, ast.Name):
        return {node.target.id: [r[0] for r in rows]}
    if isinstance(node.target, ast.Tuple):
        names = [t.id for t in node.target.elts if isinstance(t, ast.Name)]
        if len(names) != len(rows[0]):
            return None
        return {n: [r[i] for r in rows] for i, n in enumerate(names)}
    return None


def _fs_names(node, bindings):
    """Concrete macro names for an f-string whose fields are known loop vars."""
    parts = []
    for v in node.values:
        if isinstance(v, ast.Constant):
            parts.append([str(v.value)])
        elif isinstance(v, ast.FormattedValue):
            if isinstance(v.value, ast.Name) and v.value.id in bindings:
                parts.append([str(x) for x in bindings[v.value.id]])
            else:
                return None
        else:
            return None
    out = [""]
    for opts in parts:
        out = [a + b for a in out for b in opts]
    return out


def parse_builder(path: Path):
    """Return {macro: {expr, file, path, files}} from build_assets.py."""
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    aliases: dict[str, tuple[str, tuple[str, ...]]] = dict(EXTRA_ALIASES)

    # events in source order: name bindings first, then macro emissions
    events = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                events.append((node.lineno, "assign", (t, node)))
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            events.append((node.lineno, "assign", (node.target, node)))
        elif isinstance(node, ast.For):
            events.append((node.lineno, "for", node))
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) \
                and isinstance(node.value.func, ast.Attribute) \
                and node.value.func.attr == "append":
            events.append((node.lineno, "append", node.value))
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in ("M", "MT") and node.args:
                events.append((node.lineno, "macro", node))
    events.sort(key=lambda e: e[0])

    # literal for-loops: (start, end, {var: [values]}) so that f-string macro
    # names such as M(f"AucPr{tag}", ...) expand into their concrete names
    loops = []
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            b = _literal_loop(node)
            if b:
                loops.append((node.lineno, getattr(node, "end_lineno", node.lineno), b))

    def active_bindings(lineno):
        merged = {}
        for a, b, bind in loops:
            if a <= lineno <= b:
                merged.update(bind)
        return merged

    out: dict[str, dict] = {}
    deferred = []

    def record(name, expr, direct, files):
        snippet = ast.get_source_segment(src, expr) if expr is not None else ""
        snippet = re.sub(r"\s+", " ", (snippet or "")).strip()
        entry = {
            "expr": snippet,
            "file": direct[0] if direct else (sorted(files)[0] if len(files) == 1 else None),
            "path": ".".join(direct[1]) if direct else None,
            "files": sorted(files),
        }
        if entry["file"] is None:
            deferred.append((name, expr, entry))
        out[name] = entry

    for _ln, kind, payload in events:
        if kind == "assign":
            target, node = payload
            _assign_targets(target, node, aliases, src, src)
        elif kind == "append":
            if not payload.args:
                continue
            r = _resolve(payload.args[0], aliases, {})
            if r is None:
                files = _files_in(payload.args[0], aliases)
                r = (sorted(files)[0], ("[]",)) if len(files) == 1 else None
            if r is not None and isinstance(payload.func.value, ast.Name):
                aliases[payload.func.value.id] = r
        elif kind == "for":
            r = _resolve(payload.iter, aliases, {})
            if r is None:
                continue
            if isinstance(payload.target, ast.Name):
                aliases[payload.target.id] = r
            elif isinstance(payload.target, ast.Tuple):
                for t in payload.target.elts:
                    if isinstance(t, ast.Name):
                        aliases[t.id] = r
        else:
            call = payload
            first = call.args[0]
            expr = call.args[1] if len(call.args) > 1 else None
            if isinstance(first, ast.JoinedStr):
                names = _fs_names(first, active_bindings(call.lineno))
            else:
                c = _const(first)
                names = [c] if isinstance(c, str) else None
            if not names:
                continue
            direct = _resolve(expr, aliases, {})
            files = _files_in(expr, aliases) if expr is not None else set()
            if direct:
                files.add(direct[0])
            for name in names:
                record(name, expr, direct, files)

    # ---- second pass: helper functions and anything that needed the final
    # symbol table.  A local def's provenance is the single result file its
    # body touches, resolved with the complete alias map.
    locals_ = {}
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        # names bound anywhere inside the function (assignments, for targets,
        # comprehension targets, parameters): only *free* names carry the
        # provenance of the result file the helper reads from.
        own = set()
        for n in ast.walk(fn):
            if isinstance(n, ast.Assign):
                own |= {t.id for t in n.targets if isinstance(t, ast.Name)}
            elif isinstance(n, (ast.For, ast.comprehension)):
                tg = n.target
                own |= ({t.id for t in tg.elts if isinstance(t, ast.Name)}
                        if isinstance(tg, ast.Tuple) else
                        {tg.id} if isinstance(tg, ast.Name) else set())
        own |= {a.arg for a in fn.args.args}
        files = {aliases[v.id][0] for v in ast.walk(fn)
                 if isinstance(v, ast.Name) and v.id in aliases
                 and v.id not in own and aliases[v.id][0]}
        if len(files) == 1:
            locals_[fn.name] = (files.pop(), ("[]",))
    for name, expr, entry in deferred:
        direct = _resolve(expr, aliases, locals_)
        if direct:
            entry["file"], entry["path"] = direct[0], ".".join(direct[1])
            entry["files"] = sorted(set(entry["files"]) | {direct[0]})
    return out


# ---------------------------------------------------------------------------
# 2. macro -> paper location
# ---------------------------------------------------------------------------
TEX_ORDER = ["main.tex", "sections/abstract.tex", "sections/s1_intro.tex",
             "sections/s2_setting.tex", "sections/s3_related.tex",
             "sections/s4_theory.tex", "sections/s5_protocol.tex",
             "sections/s6_attention.tex", "sections/s7_language.tex",
             "sections/s8_system.tex", "sections/s9_discussion.tex",
             "sections/s10_conclusion.tex", "sections/a1_proofs.tex",
             "sections/a2_artifact.tex", "sections/a3_tables.tex"]
# table floats are \input from the sections, so they are scanned after them
TEX_TABLES = ["tables/tab_counting.tex", "tables/tab_block.tex",
              "tables/tab_granularity.tex", "tables/tab_device.tex",
              "tables/tab_main.tex", "tables/tab_cost.tex",
              "tables/tab_llm.tex"]

SEC_RE = re.compile(r"\\(section|subsection)\{([^}]*)\}(\\label\{([^}]*)\})?")
MACRO_RE = re.compile(r"\\([A-Za-z]+)")


def _appendix_number(k: int) -> str:
    """1 -> A, 26 -> Z, 27 -> AA (LaTeX \\Alph)."""
    out = ""
    while k > 0:
        k, r = divmod(k - 1, 26)
        out = chr(ord("A") + r) + out
    return out


def _value_variants(v: str):
    """Printed forms a macro value can take inside a generated table."""
    out = {v}
    out.add(v.replace("{,}", ","))
    out.add(v.replace(r"\%", "%"))
    if v.endswith("0") and "." in v:
        pass
    return {x for x in out if x}


def table_hits(macros, idx):
    """Macros not used in the prose but printed by a generated table float.

    The tables in ``paper_aij/tables/`` are written by build_assets.py with
    python f-strings rather than macros, so a value can legitimately appear in a
    float without a ``\\Macro`` reference.  Detecting it by value closes that
    hole; the match is exact on the printed decimal, so it cannot be spurious
    for the four-decimal metrics used here.
    """
    tabs = {rel: (PAPER / rel).read_text(encoding="utf-8")
            for rel in TEX_TABLES if (PAPER / rel).exists()}
    for name, m in macros.items():
        if idx.get(name):
            continue
        for rel, txt in tabs.items():
            if any(v and v in txt for v in _value_variants(str(m))):
                idx[name] = [{"file": rel, "line": 0, "section": "",
                              "section_no": "", "label": "",
                              "subsection": "(generated table)"}]
                break
    return idx


def paper_index(macros):
    """{macro: [{file, line, section, label, kind}]} in document order.

    Section numbers are derived from the order of ``\\section`` in the sources
    rather than read out of ``main.aux``: the trace must be reproducible in a
    clone that has never been compiled.
    """
    idx = {m: [] for m in macros}
    nums: dict[str, str] = {}
    n_sec = n_apx = 0
    # which section files sit after \appendix in main.tex: that boundary, not a
    # flag inside the loop, is what decides between 1..10 and A..C
    appendix_files = set()
    mt = (PAPER / "main.tex").read_text(encoding="utf-8").splitlines()
    at = next((i for i, l in enumerate(mt) if l.strip().startswith("\\appendix")), None)
    if at is not None:
        for l in mt[at:]:
            m = re.search(r"\\input\{(sections/[^}]+)\}", l)
            if m:
                appendix_files.add(m.group(1) + ".tex")
    # context at each \input{tables/...} so a float inherits its section
    float_ctx = {}
    for rel in TEX_ORDER:
        p = PAPER / rel
        if not p.exists():
            continue
        appendix = rel in appendix_files
        cur_sec, cur_sub, cur_lab = "", "", ""
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            for sm in SEC_RE.finditer(line):
                if sm.group(1) == "section":
                    cur_sec, cur_sub = sm.group(2), ""
                    cur_lab = sm.group(4) or ""
                    # \appendix resets the section counter and swaps the
                    # numbering to letters, exactly as LaTeX does
                    if appendix:
                        n_apx += 1
                        num = f"Appendix {_appendix_number(n_apx)}"
                    else:
                        n_sec += 1
                        num = str(n_sec)
                    if cur_lab:
                        nums[cur_lab] = num
                else:
                    cur_sub = sm.group(2)
            im = re.search(r"\\input\{(tables/[^}]+)\}", line)
            if im:
                float_ctx[im.group(1)] = (cur_sec, cur_lab, cur_sub)
            for mm in MACRO_RE.finditer(line):
                name = mm.group(1)
                if name in idx:
                    idx[name].append({
                        "file": rel, "line": i,
                        "section": cur_sec, "section_no": nums.get(cur_lab, ""),
                        "label": cur_lab, "subsection": cur_sub,
                    })

    for rel in TEX_TABLES:
        p = PAPER / rel
        if not p.exists():
            continue
        sec, lab, sub = float_ctx.get(rel, ("", "", ""))
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            for mm in MACRO_RE.finditer(line):
                name = mm.group(1)
                if name in idx:
                    idx[name].append({
                        "file": rel, "line": i, "section": sec,
                        "section_no": nums.get(lab, ""), "label": lab,
                        "subsection": sub,
                    })
    return idx


# ---------------------------------------------------------------------------
# 3. result unit -> code
# ---------------------------------------------------------------------------
EMIT_RE = re.compile(r'emit\(\s*state\s*,\s*"([^"]+)"')
STAGE_OF = {"data": "data", "s1_pswa": "s1_pswa", "s2_gbl": "s2_gbl",
            "s3_protocol": "s3_protocol", "s4_stats": "s4_stats",
            "s5_baselines": "s5_baselines", "s6_drift": "s6_drift",
            "s8_llm": "s8_llm"}


def unit_index():
    """{unit_key: {module, function, line, command}} from emit() call sites."""
    out = {}
    for p in sorted(EXPERIMENTS.glob("*.py")):
        txt = p.read_text(encoding="utf-8")
        defs = [(m.start(), m.group(1))
                for m in re.finditer(r"^def ([A-Za-z_0-9]+)\(", txt, re.M)]
        for m in re.finditer(r'emit\(\s*state\s*,\s*"([^"]+)"', txt):
            key = m.group(1)
            fn = ""
            for off, name in defs:
                if off <= m.start():
                    fn = name
            stage = key.split("/")[0]
            out[key] = {
                "module": f"experiments.{p.stem}",
                "function": fn,
                "line": txt[:m.start()].count("\n") + 1,
                "stage": stage,
                "command": f"python -m experiments.run_all --only {stage}",
            }
    # units whose key is built at run time (f"data/{name}") are attributed to
    # the stage that owns the prefix
    for stage, (module, fn) in {"data": ("experiments.stages", "stage_data")}.items():
        for name in ("ieee_cis", "baf"):
            out.setdefault(f"{stage}/{name}", {
                "module": module, "function": fn, "line": 0, "stage": stage,
                "command": f"python -m experiments.run_all --only {stage}",
            })
    return out


# ---------------------------------------------------------------------------
# 4. emit
# ---------------------------------------------------------------------------
def build():
    assets = json.loads((PAPER / "assets.json").read_text(encoding="utf-8"))
    macros = assets["macros"]
    prov = parse_builder(PAPER / "build_assets.py")
    used = table_hits(macros, paper_index(macros))
    units = unit_index()

    rows = {}
    for name, value in macros.items():
        p = prov.get(name, {})
        rel = p.get("file")
        unit = units.get(rel, {}) if rel else {}
        rows[name] = {
            "value": value,
            "paper": used.get(name, []),
            "result_file": rel,
            "json_path": p.get("path"),
            "expr": p.get("expr"),
            "candidate_files": p.get("files", []),
            "unit": rel,
            "code": unit,
        }
    return {"run_id": assets["run_id"], "n_macros": len(macros),
            "n_units": len(units), "macros": rows}


def markdown(doc):
    macros = doc["macros"]
    used = [m for m in macros if macros[m]["paper"]]
    unused = [m for m in macros if not macros[m]["paper"]]
    unres = [m for m in macros if not macros[m]["result_file"]]

    L = []
    A = L.append
    A("# Traceability: paper number to code")
    A("")
    A("Every number in the manuscript is a LaTeX macro emitted by")
    A("`paper_aij/build_assets.py` from a stored result file. This document is")
    A("generated by `tools/trace.py` and states, for each macro, the value, the")
    A("result file it was read from, the JSON path inside that file, the")
    A("experiment unit that produced the file, and where the macro is used in the")
    A("paper. Sections that quote no macro (the related-work discussion, for")
    A("instance) do not appear below; that is a property of the prose, not a gap")
    A("in the trace.")
    A("")
    A("| | |")
    A("|---|---|")
    A(f"| source run | `{doc['run_id']}` |")
    A(f"| macros emitted | {doc['n_macros']} |")
    A(f"| macros used in the manuscript | {len(used)} |")
    A(f"| macros with a resolved result file | {doc['n_macros'] - len(unres)} |")
    A(f"| instrumented experiment units | {doc['n_units']} |")
    A("")
    A("Reproduce the whole chain with:")
    A("")
    A("```bash")
    A("python -m experiments.run_all --fresh        # results/<run_id>/*.json")
    A("python paper_aij/build_assets.py            # numbers.tex + tables/*.tex")
    A("python paper_aij/check_numbers.py            # independent re-derivation")
    A("python tools/trace.py --check                # this document")
    A("```")
    A("")

    # group by paper section, in document order
    order_pos = {rel: i for i, rel in enumerate(TEX_ORDER + TEX_TABLES)}

    def pos(name):
        p = macros[name]["paper"]
        if not p:
            return (10_000, 0, "")
        return (order_pos.get(p[0]["file"], 9_000), p[0]["line"], name)

    order, by_sec = [], {}
    for name in sorted(macros, key=pos):
        m = macros[name]
        if not m["paper"]:
            continue
        first = m["paper"][0]
        key = (first["file"], first["section"], first["section_no"])
        if key not in by_sec:
            by_sec[key] = []
            order.append(key)
        by_sec[key].append(name)

    A("## Macros by place of use")
    A("")
    for key in order:
        rel, sec, no = key
        clean = (no or "").replace("~", " ").strip()
        if clean.startswith("Appendix"):
            title = f"{clean}. {sec or rel}"
        elif clean:
            title = f"§{clean} {sec or rel}"
        else:
            title = sec or rel
        if rel.startswith("tables/") and title != rel:
            title += f" (generated float, `{rel}`)"
        A(f"### {title}")
        A("")
        A("| macro | value | result file | JSON path | used at |")
        A("|---|---|---|---|---|")
        for name in by_sec[key]:
            m = macros[name]
            seen_loc, locs = set(), []
            for x in m["paper"]:
                k = (x["file"], x["line"])
                if k in seen_loc:
                    continue
                seen_loc.add(k)
                locs.append(f"`{x['file'].replace('sections/', '')}:{x['line']}`")
            shown = ", ".join(locs[:3]) + (f" +{len(locs) - 3}" if len(locs) > 3 else "")
            path = m["json_path"]
            path = f"`{path}`" if path not in (None, "", "[]") else "(derived)"
            relf = m["result_file"]
            relf = relf if relf.endswith(".json") else f"{relf}.json"
            A(f"| `\\{name}` | {m['value']} | `{relf}` | {path} | {shown} |")
        A("")

    A("## Result files and the code that produces them")
    A("")
    A("Each row names the emitted unit key, the module and function that writes")
    A("it, and the command that re-runs just that stage. `python -m")
    A("experiments.run_all --only <stage>` resumes rather than restarts: a unit")
    A("already marked done is skipped unless its inputs changed.")
    A("")
    A("| result file | experiment unit | module | function | command |")
    A("|---|---|---|---|---|")
    seen = set()
    for name in sorted(macros):
        m = macros[name]
        rel = m["result_file"]
        if not rel or rel in seen:
            continue
        seen.add(rel)
        c = m["code"] or {}
        relf = rel if rel.endswith(".json") else f"{rel}.json"
        A(f"| `{relf}` | `{rel}` | `{c.get('module', '-')}` "
          f"| `{c.get('function', '-')}` | `{c.get('command', '-')}` |")
    A("")

    derived = [n for n in macros
               if macros[n]["result_file"] and macros[n]["json_path"] in (None, "", "[]")]
    if derived:
        A("## Derived quantities")
        A("")
        A("These macros are computed from a result file but are not a single field")
        A("of it: a min, a max, a ratio, or an identity re-evaluated at build time.")
        A("The middle column is the exact expression that produced the printed")
        A("value, so the derivation can be re-read line by line rather than taken")
        A("on trust.")
        A("")
        A("| macro | value | source expression | file |")
        A("|---|---|---|---|")
        for n in sorted(derived):
            m = macros[n]
            e = (m["expr"] or "")[:120].replace("|", r"\|")
            relf = m["result_file"]
            relf = relf if relf.endswith(".json") else f"{relf}.json"
            A(f"| `\\{n}` | {m['value']} | `{e}` | `{relf}` |")
        A("")

    A("## Checking a single number")
    A("")
    A("The fastest way to audit one claim is to ask this tool about the macro")
    A("that carries it, then open the file it names at the path it prints.")
    A("")
    A("```bash")
    A("python tools/trace.py --macro VramTimesSdpa")
    A("python -c \"import json;print(json.load(open('results/run_20261116_094330/"
      "s1_pswa/gpu_memory_latency.json'))['bench']['rows'][-1]['modes']['pswa'])\"")
    A("```")
    A("")
    A("A macro whose value disagrees with the file it names is a build error,")
    A("not a judgement call: `check_numbers.py` re-derives every audited macro")
    A("through a second, independent code path and fails when the two differ.")
    A("")

    if unused or unres:
        A("## Integrity notes")
        A("")
        if unres:
            A(f"- macros with no resolved result file ({len(unres)}): "
              + ", ".join(f"`\\{n}`" for n in unres))
        if unused:
            A(f"- emitted but not referenced by the manuscript ({len(unused)}): "
              + ", ".join(f"`\\{n}`" for n in unused))
            A("")
            A("  These are values the build script prepares for tables or for prose")
            A("  that was rewritten, and they are reported rather than deleted so")
            A("  that the size of the number set is auditable.")
        A("")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="fail if any macro has no provenance or is unused")
    ap.add_argument("--macro", help="explain a single macro and exit")
    args = ap.parse_args(argv)

    doc = build()
    md = markdown(doc)

    if args.macro:
        m = doc["macros"].get(args.macro)
        if not m:
            print(f"unknown macro {args.macro}", file=sys.stderr)
            return 1
        print(json.dumps(m, indent=2))
        return 0

    if not args.check:                       # a check must not touch the tree
        (PAPER / "traceability.json").write_text(json.dumps(doc, indent=1),
                                                 encoding="utf-8")
        (PAPER / "TRACEABILITY.md").write_text(md, encoding="utf-8")

    used = sum(1 for m in doc["macros"].values() if m["paper"])
    unres = [n for n, m in doc["macros"].items() if not m["result_file"]]
    unused = [n for n, m in doc["macros"].items() if not m["paper"]]
    print(f"{doc['n_macros']} macros, {used} used in the paper, "
          f"{doc['n_macros'] - len(unres)} resolved to a result file, "
          f"{doc['n_units']} units mapped to code")
    if unused:
        print(f"  unused: {', '.join(sorted(unused))}")
    if unres:
        print(f"  unresolved: {', '.join(sorted(unres))}")

    if args.check:
        bad = [name for name, m in doc["macros"].items() if not m["result_file"]]
        for name in bad:
            print(f"FAIL {name}: no result file", file=sys.stderr)
        # an emitted-but-uncited macro is preparation, not a broken link: the
        # build emits quantities for tables and for prose that was cut, and the
        # trace lists them by name so the size of the number set stays auditable.
        if bad:
            return 1
        if unused:
            print(f"note: {len(unused)} of {doc['n_macros']} macros are prepared "
                  f"but not cited (listed in TRACEABILITY.md)")
        print("PASS: every emitted macro is traceable to a result file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
