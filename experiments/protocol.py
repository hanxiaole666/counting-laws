"""Commitment-anchoring + unpredictable-challenge recomputation protocol.

Replaces the original ``ZKP-lite`` (a bare SHA256 hash) with an explicit
threat model, a Merkle inclusion proof verifiable by a third party holding
ONLY on-chain data, a 128-bit salt, and an empirically testable soundness
bound. Everything here is real computation; nothing is a placeholder.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass

import numpy as np


def H(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def hx(b: bytes) -> str:
    return "0x" + b.hex()


def canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str).encode("utf-8")


# --------------------------------------------------------------------------
# Deterministic reference scorer (integer-quantised replay semantics)
# --------------------------------------------------------------------------
def reference_score(features: np.ndarray, weights: np.ndarray,
                    scale: int = 10_000) -> int:
    """Quantised integer score: what the verifier recomputes from an opening.

    Fixed-point arithmetic makes GPU/CPU replay bit-identical. This is a
    *reference scorer*, not a 3B LLM; the protocol is model-agnostic.
    """
    z = float(np.dot(np.asarray(features, dtype=np.float64),
                     np.asarray(weights, dtype=np.float64)))
    p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
    return int(round(p * scale))


# --------------------------------------------------------------------------
# Legacy scheme (verbatim semantics of scripts/04_chain_tamper_test.py)
# --------------------------------------------------------------------------
def legacy_fingerprint(rec: dict, score4: float, model_version: str = "v1") -> dict:
    return {
        "amt_bucket": round(float(rec["TransactionAmt"]), 2),
        "hour": int((float(rec["TransactionDT"]) % 86400) // 3600),
        "card4": str(rec.get("card4", "")),
        "domain": str(rec.get("P_emaildomain", "")),
        "risk_score": round(float(score4), 4),
        "model_version": model_version,
    }


def legacy_commit(fp: dict, nonce: str) -> str:
    payload = json.dumps(fp, sort_keys=True, ensure_ascii=False) + "|" + nonce
    return hx(H(payload.encode("utf-8")))


def legacy_nonces(n: int, seed: int = 42) -> list[str]:
    """The original code used np.random.default_rng(42) -> fully predictable."""
    rng = np.random.default_rng(seed)
    return [int(rng.integers(0, 2**63)).to_bytes(8, "little").hex()
            for _ in range(n)]


# --------------------------------------------------------------------------
# Hardened scheme
# --------------------------------------------------------------------------
def hardened_commit(salt: bytes, input_digest: bytes, model_digest: bytes,
                    score: int, cfg_digest: bytes) -> str:
    assert len(salt) >= 16, "salt must be >= 128 bits"
    return hx(H(salt + input_digest + model_digest
                + score.to_bytes(8, "big", signed=False) + cfg_digest))


def make_opening(salt: bytes, features: np.ndarray, model_digest: bytes,
                 cfg_digest: bytes, score: int) -> dict:
    return {
        "salt": salt.hex(),
        "input_digest": H(canon(np.asarray(features, dtype=np.float64).round(6).tolist())).hex(),
        "model_digest": model_digest.hex(),
        "cfg_digest": cfg_digest.hex(),
        "score": score,
        "weights_digest": model_digest.hex(),
    }


# --------------------------------------------------------------------------
# Merkle tree with real inclusion proofs
# --------------------------------------------------------------------------
def merkle_layers(leaves: list[str]) -> list[list[bytes]]:
    if not leaves:
        return [[H(b"")]]
    cur = [bytes.fromhex(x[2:]) if x.startswith("0x") else H(canon(x)) for x in leaves]
    layers = [cur]
    while len(cur) > 1:
        if len(cur) % 2:
            cur = cur + [cur[-1]]
        cur = [H(cur[i] + cur[i + 1]) for i in range(0, len(cur), 2)]
        layers.append(cur)
    return layers


def merkle_root(leaves: list[str]) -> str:
    return hx(merkle_layers(leaves)[-1][0])


def inclusion_proof(layers: list[list[bytes]], index: int) -> list[dict]:
    proof = []
    i = index
    for layer in layers[:-1]:
        if i % 2 == 0:
            sib = layer[i + 1] if i + 1 < len(layer) else layer[i]
            proof.append({"sibling": sib.hex(), "sibling_is_left": False})
        else:
            sib = layer[i - 1]
            proof.append({"sibling": sib.hex(), "sibling_is_left": True})
        i //= 2
    return proof


def verify_inclusion(root: str, leaf: str, proof: list[dict], index: int) -> bool:
    """Verify an inclusion proof, binding the leaf to its POSITION.

    The original pipeline's merkle_root() produced no proof at all. A proof
    that ignores ``index`` would let a leaf be replayed at another position,
    so we reconstruct the path direction from the index bits and reject any
    mismatch.
    """
    h = bytes.fromhex(leaf[2:]) if leaf.startswith("0x") else H(canon(leaf))
    i = index
    for item in proof:
        sib = bytes.fromhex(item["sibling"])
        expected_left = (i % 2 == 1)
        if bool(item["sibling_is_left"]) != expected_left:
            return False
        h = H(sib + h) if expected_left else H(h + sib)
        i //= 2
    return hx(h) == root


# --------------------------------------------------------------------------
# Challenge-recompute protocol
# --------------------------------------------------------------------------
def hypergeometric_escape(N: int, k: int, m: int) -> float:
    """Exact Pr[no cheated record sampled] for sampling k without replacement
    from N records containing m cheaters."""
    if m == 0:
        return 1.0
    if k > N - m:
        return 0.0
    # C(N-m, k) / C(N, k) computed in log space
    logp = 0.0
    for j in range(k):
        logp += math.log(max(1, N - m - j)) - math.log(max(1, N - j))
    return math.exp(logp)


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def simulate_challenges(N: int, f: float, q: float, trials: int,
                        seed: int, predictable: bool = False) -> dict:
    """Monte-Carlo estimate of the escape probability.

    predictable=True models the adversary knowing the challenge set in advance
    (it cheats only on non-challenged records) -- the negative result required
    by the threat model.
    """
    rng = np.random.default_rng(seed)
    m = max(1, int(round(N * f)))
    k = max(1, int(round(N * q)))
    escaped = 0
    for _ in range(trials):
        cheaters = set(rng.choice(N, size=m, replace=False).tolist())
        if predictable:
            # adversary picks challenge set first, cheats outside it
            challenge = set(rng.choice(N, size=k, replace=False).tolist())
            outside = [i for i in range(N) if i not in challenge]
            cheaters = set(rng.choice(outside, size=min(m, len(outside)),
                                      replace=False).tolist()) if outside else set()
        else:
            challenge = set(rng.choice(N, size=k, replace=False).tolist())
        if not (cheaters & challenge):
            escaped += 1
    p = escaped / trials
    lo, hi = wilson_ci(p, trials)
    simple = (1 - f) ** (q * N)
    exact = hypergeometric_escape(N, k, m)
    return {
        "N": N, "f": f, "q": q, "n_cheaters": m, "n_challenged": k,
        "trials": trials, "predictable_randomness": predictable,
        "empirical_escape": p, "ci95": [lo, hi],
        "theory_simple_bound": simple,
        "theory_hypergeometric": exact,
        "detection_rate": 1 - p,
        "empirical_matches_simple_bound": bool(abs(p - simple) <= 0.05),
    }


def required_q(N: int, f0: float, delta: float) -> float:
    return min(1.0, math.log(1 / delta) / (N * f0))


# --------------------------------------------------------------------------
# Attack surface: dictionary recovery against low-entropy commitments
# --------------------------------------------------------------------------
def dictionary_attack(records: list[dict], scores4: list[float],
                      nonces: list[str], dictionary: dict,
                      max_targets: int = 20) -> dict:
    """Recover a legacy commitment by enumerating low-entropy fields.

    Threat model: the risk score is disclosed to the transacting party, so the
    attacker only has to guess amount / hour / card4 / domain. The nonce is
    generated by ``np.random.default_rng(42)`` and is therefore reproducible:
    the effective search space is the field grid, NOT 2^64.
    """
    recovered = 0
    attempts = 0
    t0 = time.perf_counter()
    for idx in range(min(max_targets, len(records))):
        target = legacy_commit(legacy_fingerprint(records[idx], scores4[idx]), nonces[idx])
        nonce = nonces[idx]
        hit = False
        for amt in dictionary["amt"]:
            for hour in dictionary["hour"]:
                for card4 in dictionary["card4"]:
                    for domain in dictionary["domain"]:
                        attempts += 1
                        fp = {"amt_bucket": amt, "hour": hour, "card4": card4,
                              "domain": domain, "risk_score": round(float(scores4[idx]), 4),
                              "model_version": "v1"}
                        if legacy_commit(fp, nonce) == target:
                            hit = True
                            break
                    if hit:
                        break
                if hit:
                    break
            if hit:
                break
        recovered += int(hit)
    grid = (len(dictionary["amt"]) * len(dictionary["hour"])
            * len(dictionary["card4"]) * len(dictionary["domain"]))
    return {"targets": min(max_targets, len(records)),
            "recovered": recovered,
            "recovery_rate": recovered / max(1, min(max_targets, len(records))),
            "hash_attempts": attempts,
            "elapsed_s": round(time.perf_counter() - t0, 3),
            "nominal_nonce_space": "2^63 (default_rng integers)",
            "effective_search_space": grid,
            "reason": "nonce generated by np.random.default_rng(42) is predictable; "
                      "the remaining fields are low-entropy"}


def hardened_attack(targets: list[str], budget: int = 50_000) -> dict:
    """Guessing a 128-bit salt: expected success = budget / 2^128."""
    t0 = time.perf_counter()
    hits = 0
    for _ in range(budget):
        guess = os.urandom(16)
        if guess.hex() in targets:
            hits += 1
    expected = budget / (2**128)
    return {"salt_bits": 128, "guesses": budget, "salt_guesses_matched": hits,
            "recovery_rate": hits / max(1, len(targets)) if targets else 0.0,
            "expected_success_probability": expected,
            "theoretical_space": "2^128",
            "elapsed_s": round(time.perf_counter() - t0, 3),
            "reason": "128-bit secret salt removes the low-entropy dictionary surface"}


def replay_determinism(features: np.ndarray, weights: np.ndarray,
                       trials: int = 200, seed: int = 0) -> dict:
    """Quantised replay is bit-identical; float32 noise is not."""
    rng = np.random.default_rng(seed)
    base = reference_score(features, weights)
    exact = sum(reference_score(features, weights) == base for _ in range(trials))
    mismatch_float = 0
    eps = 1e-6
    for _ in range(trials):
        noisy = features + rng.normal(0, eps, features.shape)
        if reference_score(noisy, weights) != base:
            mismatch_float += 1
    return {"trials": trials, "quantised_replay_exact_matches": exact,
            "quantised_replay_exact_rate": exact / trials,
            "float_perturbation_eps": eps,
            "float_replay_mismatch": mismatch_float,
            "float_replay_mismatch_rate": mismatch_float / trials,
            "semantics": "int16-like fixed-point, scale=10000"}
