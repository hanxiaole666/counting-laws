#!/usr/bin/env python3
"""countlaws -- count the cost of an attention pattern before you run it.

Implements the two laws of the paper "Two counting laws for cost-sensitive
decision systems" (Artificial Intelligence):

  entry law        |S| = sum_i [ min(i+1,w) + min(i+1,s) + floor((i-w)/g)_+ ]
  granularity law  D_blk = min(1, B/g) + window term

Standard library only, so a referee can run it without an environment.

    python countlaws.py --n 32768 --w 64 --g 256 --s 4 --block 64
    python countlaws.py --sweep 256,32768,8
"""
from __future__ import annotations

import argparse
import math


def entry_count(n: int, w: int, g: int, s: int) -> int:
    """Exact number of attended entries of the causal window+anchor+sink mask."""
    if min(n, w, g, s) < 0 or g == 0:
        raise ValueError("need n,w,g,s >= 0 and g > 0")
    total = 0
    for i in range(n):
        if i < w:                      # the window already covers every position
            total += i + 1
            continue
        window = w
        sinks = min(s, i - w + 1)      # sinks the window has just dropped
        anchors = (i - w) // g         # multiples of g at or below i-w
        total += window + sinks + anchors
    return total


def entry_count_closed(n: int, w: int, g: int, s: int) -> int:
    """Same quantity in closed form (w rows summed analytically)."""
    total = w * (w + 1) // 2 if n >= w else n * (n + 1) // 2
    for i in range(w, n):
        total += w + min(s, i - w + 1) + (i - w) // g
    return total


def block_density(n: int, w: int, g: int, s: int, block: int) -> float:
    """Fraction of causal key blocks a block-granular kernel must materialise."""
    nblk = (n + block - 1) // block
    hit = 0
    seen = 0
    for bi in range(nblk):
        qs = bi * block
        qe = min(n, qs + block) - 1
        for bj in range(bi + 1):
            ks = bj * block
            ke = min(n, ks + block) - 1
            if ks > qe:
                continue
            seen += 1
            local = ke > qs - w
            anchor = (ke // g) * g >= ks
            sink = ks < s
            hit += bool(local or anchor or sink)
    return hit / seen if seen else 0.0


def element_density(n: int, w: int, g: int, s: int) -> float:
    return entry_count(n, w, g, s) / (n * (n + 1) / 2)


def work_ratio(n: int, w: int, g: int, s: int, block: int) -> float:
    """Block-granular work relative to dense attention (dense = 1.0).

    A kernel multiplies the number of causal block pairs by the density of the
    pattern over those pairs, and each materialised pair costs block^2 entries.
    """
    if not n:
        return 0.0
    nblk = (n + block - 1) // block
    causal_pairs = nblk * (nblk + 1) // 2
    tiles = causal_pairs * block_density(n, w, g, s, block)
    return tiles * block * block / (n * (n + 1) / 2)


def crossover(w: int, g: int, s: int) -> int:
    """Length at which the anchor term equals the window term."""
    return 2 * g * (w + s)


def report(n: int, w: int, g: int, s: int, block: int) -> dict:
    exact = entry_count(n, w, g, s)
    return {
        "n": n, "w": w, "g": g, "s": s, "block": block,
        "entries": exact,
        "entries_closed": entry_count_closed(n, w, g, s),
        "element_density": element_density(n, w, g, s),
        "block_density": block_density(n, w, g, s, block),
        "inflation": block_density(n, w, g, s, block) / element_density(n, w, g, s)
        if element_density(n, w, g, s) else float("inf"),
        "work_vs_dense": work_ratio(n, w, g, s, block),
        "global_keys": s + math.ceil(n / g),
        "crossover_n": crossover(w, g, s),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--w", type=int, default=64)
    ap.add_argument("--g", type=int, default=256)
    ap.add_argument("--s", type=int, default=4)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--sweep", type=str, default=None,
                    help="lo,hi,points: sweep n on a log grid instead of one value")
    a = ap.parse_args(argv)

    if a.sweep:
        lo, hi, pts = (a.sweep.split(",") + ["8"])[:3]
        ns = [int(round(lo_i)) for lo_i in
              [float(lo) * (float(hi) / float(lo)) ** (k / (int(pts) - 1))
               for k in range(int(pts))]]
    else:
        ns = [a.n]

    hdr = f"{'n':>7} {'entries':>12} {'elem dens':>10} {'blk dens':>9} " \
          f"{'inflation':>10} {'work/dense':>11} {'glob keys':>10}"
    print(hdr)
    print("-" * len(hdr))
    for n in ns:
        r = report(n, a.w, a.g, a.s, a.block)
        assert r["entries"] == r["entries_closed"], "entry law is not exact"
        print(f"{n:>7} {r['entries']:>12,} {100 * r['element_density']:>9.3f}% "
              f"{100 * r['block_density']:>8.2f}% {r['inflation']:>9.1f}x "
              f"{100 * r['work_vs_dense']:>10.2f}% {r['global_keys']:>10}")
    print()
    print(f"crossover n* = 2g(w+s) = {crossover(a.w, a.g, a.s)}"
          f"   (below it the window dominates, above it the anchors do)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
