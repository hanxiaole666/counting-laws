"""OPTIONAL: real GPU peak-VRAM scan for full vs PSWA attention.

    python -m experiments.gpu_scan --resume

This module is the resume target of the `s1_pswa/gpu_vram_scan` blocked unit.
It REFUSES to emit numbers when torch/CUDA is unavailable -- it prints the
exact install command and exits 2. Host-RAM tensor-byte measurements in
`s1_pswa/quality_bound.json` are explicitly NOT a substitute for VRAM.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .common import ROOT, atomic_write_json, environment, utcnow


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ns", default="256,512,1024,2048,4096")
    ap.add_argument("--w", type=int, default=64)
    ap.add_argument("--g", type=int, default=96)
    ap.add_argument("--s", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    env = environment()
    if not env.get("torch") or not env.get("cuda_available"):
        print("BLOCKED: CUDA torch is not available on this host.")
        print("  torch =", env.get("torch"), " cuda =", env.get("cuda_available"))
        print("  GPU reported by driver: NVIDIA GeForce RTX 2060 (6144 MiB), "
              "driver 522.06 -> CUDA 11.8 / cu121 wheels are compatible.")
        print("Install (choose ONE), then re-run this command:")
        print("  uv pip install --python .venv/Scripts/python.exe torch "
              "--index-url https://download.pytorch.org/whl/cu118")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
        print("No VRAM numbers were written; the blocked unit stays blocked.")
        return 2

    import torch
    import numpy as np
    from .pswa import build_mask

    ns = [int(x) for x in args.ns.split(",")]
    d = 64
    rows = []
    for n in ns:
        q = torch.randn(n, d, device="cuda", dtype=torch.float32)
        k = torch.randn(n, d, device="cuda", dtype=torch.float32)
        v = torch.randn(n, d, device="cuda", dtype=torch.float32)
        causal = torch.tril(torch.ones(n, n, dtype=torch.bool, device="cuda"))

        torch.cuda.reset_peak_memory_stats()
        scores = (q @ k.T) / (d ** 0.5)
        scores = scores.masked_fill(~causal, float("-inf"))
        a = torch.softmax(scores, dim=-1)
        out_full = a @ v
        torch.cuda.synchronize()
        full_peak = torch.cuda.max_memory_allocated()
        del scores, a, out_full
        torch.cuda.empty_cache()

        mask = torch.from_numpy(build_mask(n, args.w, args.g, args.s, "pswa")).to("cuda")
        torch.cuda.reset_peak_memory_stats()
        sparse_peak = 0
        out = torch.zeros(n, d, device="cuda", dtype=torch.float32)
        block = 64
        for b in range(0, n, block):
            hi = min(n, b + block)
            cols = torch.nonzero(mask[b:hi].any(dim=0)).flatten()
            if cols.numel() == 0:
                continue
            tile = (q[b:hi] @ k[cols].T) / (d ** 0.5)
            tile = tile.masked_fill(~mask[b:hi][:, cols], float("-inf"))
            out[b:hi] = torch.softmax(tile, dim=-1) @ v[cols]
            sparse_peak = max(sparse_peak, torch.cuda.max_memory_allocated())
        torch.cuda.synchronize()
        sparse_peak = max(sparse_peak, torch.cuda.max_memory_allocated())
        rows.append({"n": n, "full_peak_bytes": int(full_peak),
                     "pswa_peak_bytes": int(sparse_peak),
                     "vram_reduction": 1 - sparse_peak / full_peak})
        del q, k, v, causal, mask, out
        torch.cuda.empty_cache()

    out_path = Path(args.out) if args.out else (
        ROOT / "results" / "gpu_scan.json")
    atomic_write_json(out_path, {
        "generated_at": utcnow(), "gpu": env.get("gpu_name"),
        "torch": env["torch"], "w": args.w, "g": args.g, "s": args.s,
        "rows": rows, "note": "REAL torch.cuda.max_memory_allocated measurements"})
    print(json.dumps({"verdict": "MEASURED", "out": str(out_path),
                      "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
