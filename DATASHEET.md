# Datasheet for the datasets used in VERI-FinRisk

## Motivation

The datasets provide labelled, time-ordered fraud decisions for evaluating
accuracy–cost–verifiability trade-offs on memory-constrained edge inference.
They are used only for research benchmarking; no production deployment claim
is made.

## Composition

| | IEEE-CIS | BAF |
|---|---:|---:|
| rows | 590,540 | 1,000,000 |
| columns (mirror) | 438 | 36 |
| positive (fraud) | 20,663 (3.499%) | 11,029 (1.103%) |
| split train / val / test | 413,378 / 88,581 / 88,581 | 794,989 / 108,168 / 96,843 |
| time unit | seconds from reference | month index 0–7 |
| `ts` range | 86,400 – 15,811,131 | 0 – 7 |
| source kind | real card transactions | synthetic (source-declared) |
| licence | Kaggle competition rules (mirror: "other") | CC BY-NC-SA 4.0 |

Both mirrors add six schema columns: `dataset`, `split`, `ts`, `label`,
`user_id`, `seen_user`. Splits are cut **strictly by time**; verified monotone
in `ts`. `user_id` is entirely `<NA>` in both mirrors, so entity grouping in
this work is done with `card1` (13,553 distinct values, no missing).

## Collection and preprocessing

- No raw data are shipped. `experiments/data_io.py` downloads from the mirror,
  pins sha256 in `data/MANIFEST.json`, and reuses the file on subsequent runs.
- Features are numeric columns only, selected deterministically from the
  parquet schema (core fields first, Vesta `V*` fields last), pruned to at most
  `experiments/config.yaml: n_features`.
- Missing values are filled with the median **computed on the training split
  only**; the scaler is likewise fitted on train only.
- No target encoding, no future-window derived features, no shuffling across
  splits. Operating point is chosen once on validation at FPR = 1%.

## Model weights (not "data", listed here for completeness)

Two public Apache-2.0 checkpoints are *used*, never redistributed:

| | Qwen2.5-0.5B-Instruct | Qwen2.5-1.5B-Instruct |
|---|---:|---:|
| params | 0.49 B | 1.54 B |
| layers / hidden / KV heads | 24 / 896 / 2 | 28 / 1536 / 2 |
| on-disk (bf16 safetensors) | 954 MB | 2.9 GB |
| downloaded to | `.hf_cache/models/` | `.hf_cache/models/` |
| role | zero-shot + LoRA-SFT elicitation baseline | 2-point scale ladder |

Both are evaluated **zero-shot on the same rows** as every other system, using a
logit-based yes/no elicitation (`p = P(Yes)/(P(Yes)+P(No))` sampled at the single
answer slot) so that no JSON string parsing is on the critical path. Free-form
JSON generation is measured separately and its parse-failure rate is *counted and
reported*, never imputed as 0.9 / 0.1. The LoRA adapters in
`s8_llm/lora_sft_0_5b` regress the per-row **continuous** target produced by the
GBDT committee; they are not trained on hard 0/1 labels.

## Known limitations / biases

- IEEE-CIS fraud labels are highly imbalanced; AUC-ROC is optimistically biased,
  so AUC-PR / Recall@FPR=1% / expected loss are primary.
- `card1` grouping is a proxy for a customer entity; it is not a verified
  identity key and may merge or split real customers.
- The median entity sequence length is 4 (p90 = 46); long-sequence attention
  applies only to a thin tail. This is reported as a negative result.
- BAF is synthetic and includes month-wise concept drift; its protected
  attributes (`customer_age`, `employment_status`, `income`) are used only for
  fairness reporting, not for claiming production fairness.
- Test-period temporal slices from one dataset are NOT independent external
  validation.

## Prohibited uses

- Re-distributing the parquet files (see NOTICE).
- Any commercial use of BAF-derived data under CC BY-NC-SA 4.0.
- Presenting `--mock` / synthetic-only numbers as empirical results.
- Citing the unverified references in `paper/MANUSCRIPT.md`.
