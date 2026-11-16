"""REAL generative evaluation with a locally hosted small LLM.

Replaces the original pipeline's three fatal shortcuts:

* the SFT target was a hard-coded constant (0.90 / 0.05) -> here the target is
  the model's own continuous calibrated probability, per transaction;
* the "generative score" fell back to 0.9 / 0.1 on parse failure, which
  manufactures AUC out of nothing -> here parse failures are COUNTED and the
  affected rows are reported separately, and a logit-based elicitation is
  provided as a parse-free alternative;
* no generative output ever existed -> here a real Qwen2.5 model runs on the
  local GPU and every number comes from its own forward passes.

Two elicitation interfaces are compared on the same rows:

  ``logit`` : single forward pass, p = P(token "1") / (P("0") + P(token "1"))
              at the answer position. No decoding, no parsing, ~1 forward pass.
  ``text``  : greedy generation of a short JSON answer, then regex/JSON parse.
              This is what practitioners actually do, and its parse-failure
              rate is measured rather than silently absorbed.

CPU/GPU note: everything runs in float16 on the device that torch reports; if
CUDA is absent the module falls back to CPU float32 and says so in the result
envelope instead of pretending.
"""
from __future__ import annotations

import json
import math
import os
import re
import statistics
import time

import numpy as np

from . import representation as rp

INSTRUCTION = (
    "You are a card-fraud risk analyst. A single transaction is given as "
    "key=value pairs. Rate the probability that it is fraudulent.\n"
    "Answer with JSON only, no explanation: "
    '{"risk": <number between 0 and 1>, "label": <0 or 1>}\n'
)


def _fmt_row(row) -> str:
    return rp.render(row, "key_value")


def build_prompt(row, shots: list | None = None) -> str:
    parts = [INSTRUCTION]
    if shots:
        for s in shots:
            parts.append(f"Transaction: {_fmt_row(s['row'])}\n"
                         f'{{"risk": {s["risk"]:.4f}, "label": {s["label"]}}}\n')
    parts.append(f"Transaction: {_fmt_row(row)}\nAnswer: ")
    return "".join(parts)


_JSON_RE = re.compile(r"\{[^{}]*\}")
_FLOAT_RE = re.compile(r"(?<![0-9.])((?:0|1)(?:\.\d+)?|\.\d+)(?![0-9])")


def parse_answer(text: str) -> dict:
    """Parse the model's answer. Returns ok=False instead of a fake score."""
    m = _JSON_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(0).replace("'", '"'))
            risk = obj.get("risk")
            if risk is None and "label" in obj:
                risk = float(obj["label"])
            risk = float(risk)
            if math.isfinite(risk):
                risk = min(1.0, max(0.0, risk))
                return {"ok": True, "risk": risk, "label": int(risk >= 0.5),
                        "raw": m.group(0)[:120]}
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    m = _FLOAT_RE.search(text)
    if m:
        try:
            risk = float(m.group(1))
            if 0.0 <= risk <= 1.0:
                return {"ok": True, "risk": risk, "label": int(risk >= 0.5),
                        "raw": text[:120]}
        except ValueError:
            pass
    return {"ok": False, "risk": None, "label": None, "raw": text[:120]}


# ------------------------------------------------------------------ models --
def load_model(model_id: str, dtype: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    if dtype is None:
        dtype = "float16" if torch.cuda.is_available() else "float32"
    td = {"float16": torch.float16, "float32": torch.float32,
          "bfloat16": torch.bfloat16}[dtype]
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=td)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tok, model, device, dtype


def _vram_snapshot(torch) -> dict:
    if not torch.cuda.is_available():
        return {"cuda": False, "allocated_bytes": None}
    return {"cuda": True,
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "peak_bytes": int(torch.cuda.max_memory_allocated()),
            "params_of_device_gib": round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 3)}


def _unwrap(model):
    """Return (backbone, lm_head, hf_model) for plain or PEFT-wrapped models.

    Attribute access is delegated by PEFT, so ``model.model`` is ambiguous:
    for a PeftModel it resolves to the wrapped *CausalLM*, not to the backbone.
    We therefore walk the ``.model`` chain and take the first module that has no
    ``lm_head`` as the backbone, and the innermost module that has one as the
    head. Verified against both plain Qwen2ForCausalLM and PeftModel.
    """
    chain, cur = [], model
    for _ in range(5):
        if cur is None:
            break
        chain.append(cur)
        cur = getattr(cur, "model", None)
    backbone = next((x for x in chain if not hasattr(x, "lm_head")), None)
    head = None
    for x in reversed(chain):
        if hasattr(x, "lm_head"):
            head = x.lm_head
            break
    hf = next((x for x in chain if hasattr(x, "lm_head")), model)
    return backbone, head, hf


def _last_logits(model, enc):
    """Logits at the final position WITHOUT materialising (batch, seq, vocab).

    ``model(**enc).logits`` allocates batch x seq x 151936 floats and OOMs a
    6 GB card at batch 32 (observed: 2.14 GiB single allocation). Running the
    backbone and applying the LM head to the last hidden state keeps the
    transient tensor at O(batch x hidden).
    """
    import torch
    backbone, head, hf = _unwrap(model)
    with torch.no_grad():
        if backbone is None or head is None:
            return hf(**enc).logits[:, -1, :].float()   # safe fallback
        h = backbone(**enc).last_hidden_state[:, -1, :]
        return head(h).float()


def score_choice(tok, model, device, prompts: list[str], pos: str = "Yes",
                 neg: str = "No", batch_size: int = 32, max_len: int = 512,
                 torch_mod=None) -> dict:
    """Parse-free elicitation: p = P(pos) / (P(neg) + P(pos)) at the answer slot."""
    import torch
    torch_mod = torch_mod or torch
    idp = tok.encode(" " + pos, add_special_tokens=False)[0]
    idn = tok.encode(" " + neg, add_special_tokens=False)[0]
    scores, lat = [], []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len).to(device)
        t0 = time.perf_counter()
        logits = _last_logits(model, enc)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1e3 / len(chunk))
        pr = torch.softmax(logits, dim=-1)
        p = (pr[:, idp] / (pr[:, idp] + pr[:, idn] + 1e-12)).cpu().numpy()
        scores.extend(p.tolist())
    return {"scores": scores, "ms_per_row": lat,
            "vram": _vram_snapshot(torch), "pos_token": pos, "neg_token": neg}


def score_logit(tok, model, device, prompts: list[str], batch_size: int = 32,
                torch_mod=None) -> dict:
    """Digit variant: p = P('1') / (P('0') + P('1')). Kept for comparison."""
    return score_choice(tok, model, device, prompts, pos="1", neg="0",
                        batch_size=batch_size, torch_mod=torch_mod)


def score_text(tok, model, device, prompts: list[str], batch_size: int = 8,
               max_new_tokens: int = 24) -> dict:
    """Greedy generation + parse. Parse failures are returned, not papered over."""
    import torch
    scores, oks, raws, lat, fails = [], [], [], [], 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024).to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 do_sample=False, num_beams=1,
                                 pad_token_id=tok.pad_token_id)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1e3 / len(chunk))
        new = gen[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(new, skip_special_tokens=True)
        for t in texts:
            p = parse_answer(t)
            oks.append(p["ok"])
            raws.append(p["raw"])
            if p["ok"]:
                scores.append(p["risk"])
            else:
                fails += 1
                scores.append(np.nan)          # never a fabricated 0.9 / 0.1
    return {"scores": scores, "parse_ok": oks, "raw_samples": raws[:5],
            "parse_failures": fails, "ms_per_row": lat,
            "vram": _vram_snapshot(torch)}


def lora_sft(model_id: str, train_prompts: list[str], train_targets: list[float],
             out_dir: str, epochs: float = 1.0, lr: float = 2e-4,
             batch_size: int = 2, grad_accum: int = 8, max_len: int = 512,
             seed: int = 0, log_every: int = 25) -> dict:
    """LoRA SFT whose target is the CONTINUOUS calibrated score per row."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    td = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=td)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                     task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    answers = [f'{{"risk": {t:.4f}, "label": {int(t >= 0.5)}}}' for t in train_targets]

    def encode(i):
        prompt = train_prompts[i]
        ans = answers[i] + (tok.eos_token or "")
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        a_ids = tok(ans, add_special_tokens=False).input_ids
        ids = (p_ids + a_ids)[:max_len]
        labels = ([-100] * len(p_ids) + a_ids)[:max_len]
        return {"input_ids": ids, "labels": labels,
                "attention_mask": [1] * len(ids)}

    enc_rows = [encode(i) for i in range(len(train_prompts))]
    ds = Dataset.from_dict({
        "input_ids": [r["input_ids"] for r in enc_rows],
        "labels": [r["labels"] for r in enc_rows],
        "attention_mask": [r["attention_mask"] for r in enc_rows]})

    def collate(batch):
        mx = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id
        input_ids = [b["input_ids"] + [pad] * (mx - len(b["input_ids"])) for b in batch]
        labels = [b["labels"] + [-100] * (mx - len(b["labels"])) for b in batch]
        attn = [b["attention_mask"] + [0] * (mx - len(b["attention_mask"])) for b in batch]
        return {"input_ids": torch.tensor(input_ids),
                "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attn)}

    args = TrainingArguments(
        output_dir=out_dir, num_train_epochs=epochs, per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum, learning_rate=lr, logging_steps=log_every,
        save_strategy="no", eval_strategy="no", report_to=[], fp16=torch.cuda.is_available(),
        seed=seed, data_seed=seed, remove_unused_columns=False, optim="adamw_torch")
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collate)
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    train_s = time.time() - t0
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return {"adapter_dir": out_dir, "train_seconds": round(train_s, 1),
            "train_loss": float(result.training_loss),
            "trainable_params": int(trainable), "total_params": int(total),
            "trainable_fraction": float(trainable / total),
            "peak_vram_bytes": (int(torch.cuda.max_memory_allocated())
                                if torch.cuda.is_available() else None)}


def load_peft(adapter_dir: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(adapter_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    base = json.load(open(os.path.join(adapter_dir, "adapter_config.json"),
                          encoding="utf-8"))["base_model_name_or_path"]
    td = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=td)
    model = PeftModel.from_pretrained(model, adapter_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tok, model, device


def summarise_latency(ms_per_row: list[float]) -> dict:
    if not ms_per_row:
        return {}
    v = sorted(ms_per_row)
    return {"p50_ms_per_row": round(statistics.median(v), 2),
            "p95_ms_per_row": round(v[min(len(v) - 1, int(0.95 * len(v)))], 2),
            "throughput_rows_per_s": round(1000.0 / statistics.fmean(v), 2)}
