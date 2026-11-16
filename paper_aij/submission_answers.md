# Submission to *Artificial Intelligence* (Elsevier) — answers to the AIJ submission questions, and cover letter

Manuscript: **Two counting laws for cost-sensitive decision systems: anchor-based
sparse attention is quadratic, and small language models are supervision-bound**
Author: Xiaole Han (sole author), ORCID 0009-0009-9611-2368. Affiliation:/nBusiness School, Qilu Institute of Technology, Jinan, Shandong 250200, China.
Corresponding author: Xiaole Han, hanxiaoleop@163.com. Article type: Full length
article. Review model: single blind
(AIJ default).
Files in this package: `main.tex` + `sections/` + `refs.bib` (elsarticle source),
`highlights.txt`, this file, and the artefact at
<https://github.com/hanxiaole666/counting-laws>.

Every number quoted below is a macro in `numbers.tex`, generated from
`results/run_20261116_094330/`; the same value appears in the body at the section
cited. `paper_aij/check_numbers.py` re-derives 46 of them through a second code
path and prints PASS. Nothing in this file is typed by hand.

---

## Q1. What is, in one or two sentences, the original contribution of this work?

Two integer quantities — the number of keys visible to *every* query (the
global-key budget), and the block granularity of the kernel that executes a mask —
decide, before anything is trained, whether an attention pattern advertised as
sparse is linear or quadratic in the sequence length; and we give the exact
counting law that computes both, verify it to the last entry, and use it to show
on commodity hardware that the popular periodic-anchor pattern is quadratic while
the same pattern without anchors is faster than fused dense attention.

A second, independent contribution is a dissociation in small language models: on
identical rows a 0.5B model reaches AUC-PR 0.0767 against 0.5295 for
gradient-boosted trees, and low-rank adaptation on 900 rows repairs the output
format completely (100% → 0% parse failures) while driving AUC-ROC from 0.6412 to
0.4582, which shows that the binding constraint is the amount of supervision and
not the interface or the instruction tuning.

## Q2. Why should this contribution be considered important for the field of Artificial Intelligence? Please confirm where you explain this in the paper.

Because both defaults it corrects are *selection rules* that practitioners apply
before they measure anything: "long sequences → use a sparse pattern to get
linear cost" and "few labels → use a pretrained model zero-shot". The paper shows
that the first rule is decided by a count the literature does not record (the
global-key budget, §4.1–4.3, Theorem 3) and that the second is decided by rows
per class rather than by model capability (§7.3). A correction to a selection rule
propagates to every system that used the rule; a further decimal of AUC-PR on one
benchmark does not.

The claim is falsifiable, and it was falsified where it was wrong: three of our
own intermediate claims are reported as refuted, including a block-alignment
lemma that is explicitly withdrawn (§6.7), because a protocol that cannot refute
its author cannot certify anything.

**Where this is explained.** Abstract (both laws and the on-device verdict);
§1.2 "What we establish" (contributions C1–C6); §1.3 "Why this is a contribution
to artificial intelligence" (the argument above, stated as such); §9.1 "Three
counts, and why they belong together"; §9.6 "Relevance to the wider AI audience".

## Q3. What is the most closely related work by others, and how does this work differ? (1–3 citations)

**Closest, and the one the paper is measured against — Elkan (2001), IJCAI:**
C. Elkan, "The foundations of cost-sensitive learning", *Proc. 17th International
Joint Conference on Artificial Intelligence (IJCAI)*, 2001, pp. 973–978.
We adopt this framework unchanged — every threshold in the paper is set on
validation data by expected cost and evaluated once on held-out data — and change
the *object* it is applied to, asking whether a pretrained language model can
stand in for the scorer under that criterion when supervision is scarce. The
answer is a dissociation (interface repaired, discrimination destroyed) that the
framework alone does not predict.

**Closest on the attention side — Baldi & Vershynin (2023), *Artificial
Intelligence*:** P. Baldi, R. Vershynin, "The quarks of attention: Structure and
capacity of neural attention building blocks", *Artificial Intelligence* 319
(2023) 103901. That work analyses what an attention block can *represent*; ours is
the complementary accounting of what a mask *costs to execute*, and the two are
independent: a pattern can be expressive and still quadratic, which is exactly the
case we exhibit.

**Closest on the vocabulary the paper corrects — Zaheer et al. (2020), NeurIPS:**
M. Zaheer et al., "Big Bird: Transformers for longer sequences", *NeurIPS* 33,
2020. Big Bird establishes that window + global + random links keep linear cost,
with a *fixed-size* global set. We keep the window and the sinks and replace the
fixed global set with a *periodic* one — the natural choice when no task-specific
global tokens are known — and show that this single change moves the pattern out
of the linear class. The two results are consistent; together they say that how a
position becomes globally visible is a complexity decision.

Two further anchors for the AI framing: M. Wooldridge, "Artificial Intelligence
requires more than deep learning — but what, exactly?", *Artificial Intelligence*
289 (2020) 103386, and the efficiency taxonomy of Y. Tay et al., "Efficient
transformers: A survey", *ACM Computing Surveys* 55 (6) (2022) 1–28, which states
complexity per pattern at the level of attended entries and for designs as
published; we add the parameter it does not record and then measure the executed
rather than the nominal cost.

**Where the relationship is discussed in detail.** §3.1–3.6 survey the six
relevant strands; §3.7 is titled "The closest prior work, and how this paper
differs" and treats each of the above in its own paragraph, including the tabular
benchmarking line (Gorishniy et al. 2021; Grinsztajn et al. 2022) where we
deliberately *weaken* the supervised baseline so that the comparison favours the
pretrained model.

## Q4. Is your work relevant to AI, and where do you explain the relevance to AI in the paper?

Yes. The paper is about two of the most widely deployed substitutions in applied
AI — a sparse attention pattern in place of dense attention, and a pretrained
language model in place of a trained task model — and it replaces the heuristics
used to choose between them with counts that can be evaluated before training.
Both results are about learning systems, not about a domain application; the
fraud-detection datasets are the substrate on which the second law is measured,
not the subject of the paper.

**Where this is explained.** §1.3, in a subsection with that title, which names
the three strands of AI research the work engages (attention as an analysable
object; when learned models beat strong tabular baselines; cost-sensitive decision
making) and states the position on method; §9.6 restates it for the reader who
arrives from the conclusion; §2 states the decision-theoretic objective that makes
the efficiency question a decision question rather than a benchmark question.

## Q5. Can you confirm that at least the abstract, introduction and conclusion can be appreciated by a general AI audience?

Yes, and this is a design constraint of the writing, not an afterthought.

- The **abstract** states both laws in words before any symbol, gives the on-device
  verdict in one sentence, and closes with the three refuted claims. No equation,
  no dataset-specific term.
- The **introduction** opens with the two defaults as practitioners state them,
  introduces the counting quantity in prose, and defers every formal statement to
  §4. §1.3 and §1.4 fix what is and is not claimed, so a reader outside the
  sub-field can tell what the paper does *not* assert.
- The **conclusion** (§10) restates both laws without notation, names the two
  places where the evidence stops (two model sizes, one GPU), and says what a
  practitioner should do instead — cap the global-key budget; count labelled rows
  per class before choosing a language model.

A reader who works only in, say, planning or knowledge representation needs no
background in efficient transformers: the object of the first law is a mask matrix
and a block size, both defined in §2.2 and drawn in Figure 1, and the object of
the second is a labelled table, defined in §2.3.

## Q6. Additional electronic material that would be made available

Published in ScienceDirect with the paper (not in print), archived at
<https://github.com/hanxiaole666/counting-laws> and mirrored in the submission:

1. **Programs.** The full experiment suite (`experiments/`), one module per
   instrumented family: `attention_bench.py` (device memory/latency),
   `llm_eval.py` (elicitation and LoRA adaptation), `evm_gas.py` (on-chain gas via
   `solc` 0.8.20 and py-evm), `stats2.py` (paired inference), `run_all.py`
   (orchestration with crash-safe resume). Plus the build chain that produces the
   manuscript: `paper_aij/build_assets.py`, `make_figs.py`, `check_numbers.py`.
2. **A provenance table, one line per number in the paper.**
   `paper_aij/TRACEABILITY.md`, generated by `tools/trace.py`, states for each of
   the 277 quantities the manuscript prints: the value, the result file it was read
   from, the JSON path inside that file, the experiment unit and function that
   wrote the file, and the section, file and line where the paper uses it.
   Quantities that are not a single field (ratios, maxima, identities re-evaluated
   at build time) are listed with the expression that produced them. The table is
   computed from the syntax tree of the build script and from the write sites of
   the result files, so it cannot silently disagree with the code; two further
   checks, `check_numbers.py` (second, independent derivation) and
   `verify_artifacts.py` (re-hash every result file, reject substitutes), are the
   reason a referee does not have to take any of this on trust.
3. **A reader's instrument.** `tools/countlaws.py` — standard library only, no
   environment needed — computes both laws and reproduces the reference count
   (n=4096, w=64, g=96, s=4 → 358,906) exactly; it is the object that makes the
   main claim checkable in one command by a referee.
4. **Test data (three derived artefacts).** `patterns/` — the boolean masks of the
   full (n, w, g, s, B) grid, as sparse index lists (synthetic, licence-free);
   `sequences/` — the entity-length distribution and the aggregated
   autocorrelation function of §6.5, i.e. derived statistics only, no transaction
   records; `baf_splits.json` — temporal split boundaries and per-period
   prevalences for the Bank Account Fraud suite, redistributable under
   CC BY-NC-SA 4.0 with attribution.
5. **The trained adapter.** The LoRA adapter of §7.3 (1,081,344 trainable
   parameters, 0.22% of Qwen2.5-0.5B-Instruct, Apache 2.0) with its training
   configuration, so the dissociation can be reproduced without retraining.
6. **Results.** One JSON per unit with measurements, configuration, device,
   assertions and hypotheses; the manuscript reads these at build time, so the
   text cannot drift from the data. The raw IEEE-CIS and BAF data are **not**
   redistributed (competition rules and CC BY-NC-SA respectively); both are
   fetched by a script that verifies checksums, and the download is automatic.
7. **A test suite** (`tests/`, 76 cases) in which every case corresponds to a
   defect that actually occurred in this repository, including the withdrawn
   block-alignment lemma and the two documentation checks that keep the README's
   quoted numbers generated.

Two measurements are reported as **blocked** rather than estimated — hardware
attestation (no SGX/SEV-SNP/TDX device) and proof-carrying inference (no
prover-scale device); each blocked unit stores the command that would unblock it,
and the verification gate reports the blocked set explicitly instead of treating
absence as success.

---

## Cover letter (paste into the "Comments" box)

Dear Editors,

I submit the enclosed manuscript, "Two counting laws for cost-sensitive decision
systems: anchor-based sparse attention is quadratic, and small language models are
supervision-bound", for consideration as a full-length article in *Artificial
Intelligence*.

The paper is a study of two selection rules that applied AI relies on and rarely
audits. The first is that a sparse attention pattern buys linear cost; I give an
exact law for the number of attended entries of the window + sink + anchor family,
prove that linearity holds if and only if the number of globally visible keys is
bounded, and confirm on a 6 GiB GPU that the periodic-anchor variant needs 669×
the transient memory of a fused dense kernel at n=32768 and is 9.0× slower, while
the same pattern without anchors is 2.1× *faster* than fused dense. The cost is
attributable to the anchors, exactly as the law predicts. The second rule is that
a pretrained language model can substitute for a trained scorer when labels are
scarce; on identical held-out rows a 0.5B model scores AUC-PR 0.0767 against
0.5295 for gradient-boosted trees, and adapting it on 900 rows repairs the output
format completely while pushing AUC-ROC to 0.4582 — the constraint is supervision,
not the interface.

Three of my own intermediate claims were refuted by my own audit apparatus and are
reported as such, including one lemma that is explicitly withdrawn (§6.7). Every
number in the manuscript is generated from a stored result file at build time, and
the artefact contains a standard-library tool with which a referee can recompute
the central count in one command.

The work is not under consideration elsewhere, has a single author, and has no
competing interests. I have no conflicts to declare. I would be glad to suggest
referees on request; I have not done so here to avoid any appearance of
pre-selection.

Yours sincerely,
Xiaole Han
Business School, Qilu Institute of Technology, Jinan, Shandong 250200, China
hanxiaoleop@163.com · ORCID 0009-0009-9611-2368

---

## Pre-submission checklist (author-side, delete before uploading)

| Item | Status |
|---|---|
| elsarticle source, `pdflatex`+`bibtex` clean, no undefined references, no overfull boxes | done, 37 pp. |
| Highlights: 5 bullets, longest 78 characters | `highlights.txt` |
| Generative-AI declaration before the reference list | `main.tex` |
| CRediT statement | `main.tex` |
| Competing interest / funding / data availability | `main.tex` |
| Vector figures (PDF, Type 42 fonts) | `figures/*.pdf` |
| Affiliations, e-mail, ORCID, postcode | filled (ORCID 0009-0009-9611-2368) |
| Repository made public before submission | **TODO** |
| `main.bbl` uploaded with the source, or `.bib` only | either |
