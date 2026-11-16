# counting-laws

**Two counting laws that predict, before you train, whether a "sparse" attention
pattern is actually cheaper and whether a small language model can actually
replace a supervised scorer.**

Artefact for the manuscript *Two counting laws for cost-sensitive decision
systems: anchor-based sparse attention is quadratic, and small language models are
supervision-bound*, submitted to *Artificial Intelligence* (Elsevier).

This repository is small on purpose. The paper's claim is that two integers settle
questions usually settled by benchmarking, and the code that computes them is
standard library only. If you have two minutes, run `tools/countlaws.py`. If you
have an hour, run `experiments/` and rebuild the PDF.

<!-- BEGIN:stats -->
| | |
|---|---|
| source run | `run_20261116_094330` |
| result files | 36 JSON under `results/<run_id>/` |
| experiment units | 36 (35 completed, 1 blocked, all resumable) |
| LaTeX macros generated from results | 277 |
| of which cited in the manuscript | 246 |
| test cases (`pytest`) | 76 |
<!-- END:stats -->

---

## Start here

```bash
git clone https://github.com/hanxiaole666/counting-laws
cd counting-laws
pip install -r requirements.lock          # numpy, pandas, scikit-learn — no GPU

python tools/countlaws.py --n 4096 --w 64 --g 96 --s 4
```

<!-- BEGIN:countlaws-demo -->
```
$ python tools/countlaws.py --n 4096 --w 64 --g 96 --s 4
      n      entries  elem dens  blk dens  inflation  work/dense  glob keys
---------------------------------------------------------------------------
   4096      358,906     4.277%    69.71%      16.3x      70.78%         47

crossover n* = 2g(w+s) = 13056   (below it the window dominates, above it the anchors do)

$ python tools/countlaws.py --n 32768 --w 64 --g 256 --s 4 --block 64   # the setting of the GPU benchmark
      n      entries  elem dens  blk dens  inflation  work/dense  glob keys
---------------------------------------------------------------------------
  32768    4,298,586     0.801%    25.73%      32.1x      25.78%        132

crossover n* = 2g(w+s) = 34816   (below it the window dominates, above it the anchors do)
```
<!-- END:countlaws-demo -->

Read the first line as: a mask that attends to 4.3% of the causal entries still
makes a block-granular kernel execute 69.7% of the causal *blocks*, because the
kernel cannot execute a mask, only a tile. The last column is the point of the
paper: the number of keys visible to every query is what decides whether the cost
is linear or quadratic in `n`, and it is 47 here and 132 at `n = 32768, g = 256`.

## The laws

**Entry law (identity, not a bound).** For a causal sliding window of width `w`,
`s` attention sinks and a periodic anchor set of period `g`:

```
|S(i)| = min(i+1, w) + min(i+1, s) + floor((i-w)/g)_+
|S|    = n(w+s) + (n-w)^2/(2g) + O(n)
```

The anchor term is **quadratic in `n`**, which is why a pattern described as
`O(n(w + n/g))` in the literature is not linear in `n` at fixed `g`. The identity
is checked entry by entry against the mask builder over the whole measured grid,
not against a fitted slope (`max relative error 0.000%`, `tests/test_countlaws.py`).

**Linearity criterion.** Cost is `Θ(n)` iff the *global-key budget* — the number
of keys visible to every later query — is bounded as `n` grows. A fixed set of
global tokens has budget `O(1)`. Periodic anchors have budget `s + ⌈n/g⌉`, which
grows, so the cost is quadratic. This is the entire difference between a
BigBird-style design and periodic anchoring, and it is a count, not a style.

**Granularity law.** A block kernel executes `min(1, B/g)` of the causal key
blocks plus a window term, *independent of whether `g` is a multiple of `B`*.
Verified to three decimals at six settings. The practical reading: at `B = 256`
with `g = 192` the kernel materialises **every** causal block, so a mask
advertised as 0.54% sparse is executed as dense attention.

## Where the numbers in the paper come from

No measurement is typed into the LaTeX. The chain is:

```
experiments/ ──► results/<run_id>/*.json ──► paper_aij/build_assets.py
                                                     │
                                    numbers.tex ◄────┤────► tables/*.tex
                                          │
                                      main.tex  ──►  main.pdf
```

Three programs audit that chain, and each answers a different question:

| program | question it answers |
|---|---|
| `paper_aij/check_numbers.py` | does a **second, independent derivation** of 51 quantities agree with the values in `numbers.tex`? |
| `experiments/verify_artifacts.py` | does every result file match its recorded sha256, and was any of it mocked? |
| `tools/trace.py` | for **any** number in the paper, which JSON field and which function produced it? |

`.github/workflows/ci.yml` runs the tests and the first two audits on every push;
the full suite and the PDF build are `workflow_dispatch` only, because they need
the datasets and a few gigabytes of weights.

### Trace one number

```bash
python tools/trace.py --macro VramTimesSdpa
```

```json
{
 "value": "669",
 "result_file": "s1_pswa/gpu_memory_latency",
 "json_path": "[]",
 "expr": "f\"{cell('pswa', 'transient_mib') / cell('full_sdpa', 'transient_mib'):.0f}\"",
 "paper": [{"file": "sections/abstract.tex", "line": 21, "section": "Abstract"}]
}
```

`json_path: "[]"` marks a *derived* quantity: the expression field then shows the
exact code that combined the two fields, so the derivation is readable rather than
asserted. `python tools/trace.py` regenerates `paper_aij/TRACEABILITY.md`, a full
table of all 277 numbers — value, result file, JSON path, and the file and line
where the paper uses it. A sample, for the device measurements of Section 6:

<!-- BEGIN:trace-sample -->
| number in the paper | value | read from | JSON path |
|---|---|---|---|
| `\IeeeFraudPct` | 3.50 | `data/ieee_cis.json` | `schema.fraud_rate` |
| `\RhoMean` | 0.1592 | `s1_pswa/quality_bound.json` | `quality_bound.rho_mean` |
| `\LargestN` | 32{,}768 | `s1_pswa/gpu_memory_latency.json` | `claims.largest_n_with_both_measured` |
| `\LlmAucPr` | 0.0767 | `s8_llm/zero_shot_0_5b.json` | `elicitation.choice.llm.auc_pr` |
| `\GasSavingPct` | 99.71 | `s3_protocol/onchain_gas.json` | `gas_saving_fraction` |
| `\CurveSmallAucPr` | 0.0355 | `s5_baselines/learning_curve.json` | `rows.[]` |
<!-- END:trace-sample -->

The trace is computed from the AST of `build_assets.py` and from the `emit()`
call sites in `experiments/`, so it cannot drift from the code: if you rename a
field in a result file, `build_assets.py` fails at import, and if you add a macro
without a source, `tools/trace.py --check` fails. The build is also deterministic:
re-running `paper_aij/build_assets.py` in a fresh clone leaves `git status` clean,
so `numbers.tex` in the repository is exactly what the code produces.

## Reproducing the paper

Every block below is also a task: `python tasks.py <name>` (`make <name>` is the
same thing, and the Makefile only wraps `tasks.py`, so there is one
implementation). `python tasks.py` with no argument lists them all, and
`--dry-run` prints commands without running them.

```bash
# 1. data (auto-downloads on first use, verifies sha256, resumes partial files;
#    neither dataset is redistributed here, see NOTICE)
python -m experiments.data_io --ensure --datasets ieee_cis,baf

# 2. the suite: 36 units, crash-safe. Re-running the same command resumes;
#    killing it loses at most the unit in flight.
python -m experiments.run_all --fresh
python -m experiments.run_all                     # resume / skip completed
python -m experiments.run_all --status            # what is done, what is blocked
python -m experiments.run_all --force s1_pswa/block_alignment   # invalidate one unit

# 3. numbers, figures, PDF
python paper_aij/build_assets.py                  # → numbers.tex + 7 tables
python paper_aij/make_figs.py                     # → 6 vector PDFs
cd paper_aij && pdflatex main && bibtex main && pdflatex main && pdflatex main

# 4. audits
python -m pytest tests -q
python -m experiments.verify_artifacts
python paper_aij/check_numbers.py
python tools/trace.py --check
python tools/refresh_readme.py --check            # the blocks above
```

Full suite: about 7 minutes on the reference machine (RTX 2060, 6 GiB), of which
LoRA training is ~136 s. `python tasks.py check` runs every audit, and
`python tasks.py paper` rebuilds the PDF from the stored results.

Hardware: CPU-only runs 31 of 36 units; the GPU, LLM and EVM units fail fast with
the install command rather than substituting an estimate. `requirements-gpu.lock`
adds torch+CUDA, transformers, peft, solc and py-evm.

### Resume semantics

A unit is skipped only if it is marked done **and** the sha256 of every upstream
artefact it declared is unchanged. `--force <unit>` marks one unit invalid without
touching the rest. This is what makes it safe to stop the suite halfway and come
back to it: the state lives in `results/<run_id>/.state.json`, not in memory.

## Repository map

```
tasks.py                   every command in this README as a named task (make wraps it)
tools/stage_release.py     builds the public tree from the upload manifest (this repo)
tools/countlaws.py         the two laws, stdlib only — the reader's instrument
tools/trace.py             paper number → result field → function → command
tools/refresh_readme.py    regenerates the generated blocks in this file

experiments/               the whole suite, one module per instrumented family
  run_all.py                 orchestrator: resume, --only, --from, --status, --force
  data_io.py                 download + sha256 + partial resume, never redistributes
  attention_bench.py         real CUDA peak memory and latency, five implementations
  pswa.py                    mask construction and the closed form
  representation.py          entity sequences, ACF period estimation
  baselines.py               linear / GBDT / cascade, cost-sensitive thresholds
  stats.py stats2.py         DeLong, McNemar, bootstrap, permutation, power
  protocol.py                commitment, challenge, Merkle inclusion
  evm_gas.py                 solc 0.8.20 + py-evm, measured gas (Shanghai)
  llm_eval.py                Qwen2.5 logit scoring, JSON elicitation, LoRA SFT
  stages.py stages2.py       the units themselves, one emit() per result file
  verify_artifacts.py        hash and mock checks over every result file

paper_aij/                 the manuscript and everything that feeds it
  build_assets.py            results → numbers.tex + tables/*.tex (277 macros)
  check_numbers.py           51 macros re-derived by a second code path
  make_figs.py              6 vector figures, Type 42 fonts
  TRACEABILITY.md           generated: every number, its source, its use
  sections/ tables/ figures/ main.tex refs.bib

results/<run_id>/          one JSON per unit, with assertions and hypotheses
tests/                     52 tests; each one is a defect that actually occurred
config.yaml                every parameter quoted in the paper, hashed per run
```

## What is *not* here, and why

- **The datasets.** IEEE-CIS is competition data and BAF is CC BY-NC-SA 4.0; both
  are fetched at run time and verified. `DATASHEET.md` records the terms.
- **Three blocked measurements.** TEE remote attestation, a ZKML inference proof,
  and a 7B–8B language-model baseline could not be run on the reference machine.
  They are absent from the paper with the command that would unblock them
  (`experiments/run_all.py --status` lists them). Nothing is estimated.
- **31 prepared but uncited quantities.** `build_assets.py` emits them for the
  tables and for prose that was cut; `paper_aij/TRACEABILITY.md` lists them by
  name rather than hiding them, so the size of the number set stays auditable.

## The three claims the paper withdraws

The repository is also the record of what did not survive. In the course of the
work, three of the author's own intermediate claims were refuted by measurements
in `results/` — a periodicity that the ACF does not support, a block-alignment
lemma that compared different `g`, and a "small models are within two points"
claim that is off by a factor of seven in AUC-PR. All three are reported in the
paper (§6.7, §7.4) rather than deleted. If you are reading this to decide whether
the apparatus works, that is the evidence we would offer.

## Citing

See `CITATION.cff`. The manuscript's reference list is built from `refs.bib`
(26 entries, each verified against Crossref/arXiv); the repository version and the
paper version are cited as one artefact.

## Licence

Code MIT (`LICENSE`); data and model terms in `NOTICE`; dataset limitations in
`DATASHEET.md`. The LaTeX sources are under the same MIT terms as the code, so the
figures and tables can be reused with attribution.
