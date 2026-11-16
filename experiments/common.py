"""Shared paths, hashing, atomic writes and the checkpoint/resume state machine.

Design contract (ARIS experiment-audit):
  * every stage is split into small *units* (stage/unit key);
  * a unit is marked ``running`` BEFORE work starts and ``done`` only AFTER its
    result file exists and has been hashed;
  * if the process is killed, the leftover ``running`` marker becomes
    ``interrupted`` on the next launch and that unit is re-executed;
  * completed units are skipped on resume, unless --fresh is given;
  * every result JSON carries run_id, mock flag, seed and source hashes.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_ROOT = ROOT / "results"
FIG_DIR = ROOT / "figures"
PAPER_DIR = ROOT / "paper"
STATE_DIR = ROOT / ".aris" / "state"
LOG_DIR = ROOT / ".aris" / "logs"
CURRENT_FILE = STATE_DIR / "CURRENT"

STAGES = ["data", "s1_pswa", "s2_gbl", "s3_protocol", "s4_stats",
          "s5_baselines", "s6_drift", "s7_figures"]

STATUS_DONE = "done"
STATUS_RUNNING = "running"
STATUS_FAILED = "failed"
STATUS_BLOCKED = "blocked"
STATUS_INTERRUPTED = "interrupted"


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                     default=str).encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def load_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(o):
    """numpy scalars/arrays -> plain Python, without importing numpy."""
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    if hasattr(o, "tolist"):
        try:
            return o.tolist()
        except Exception:
            pass
    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:
            pass
    if isinstance(o, Path):
        return str(o)
    return str(o)


def normalize_json(obj):
    """Round-trip through JSON so np.bool_ becomes a real bool, not "True"."""
    return json.loads(json.dumps(obj, ensure_ascii=False, default=_json_default))


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def environment() -> dict:
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "utc": utcnow(),
    }
    try:
        import numpy
        env["numpy"] = numpy.__version__
    except ImportError:
        env["numpy"] = None
    try:
        import pandas
        env["pandas"] = pandas.__version__
    except ImportError:
        env["pandas"] = None
    try:
        import sklearn
        env["sklearn"] = sklearn.__version__
    except ImportError:
        env["sklearn"] = None
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["gpu_name"] = (torch.cuda.get_device_name(0)
                           if torch.cuda.is_available() else None)
    except ImportError:
        env["torch"] = None
        env["cuda_available"] = False
        env["gpu_name"] = None
    return env


class RunState:
    """Persistent, crash-safe per-unit progress ledger."""

    def __init__(self, run_id: str | None = None, fresh: bool = False):
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        if run_id is None and not fresh and CURRENT_FILE.exists():
            run_id = CURRENT_FILE.read_text(encoding="utf-8").strip()
        self.run_id = run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.path = STATE_DIR / f"{self.run_id}.json"
        self.results_dir = RESULTS_ROOT / self.run_id
        self.results_dir.mkdir(parents=True, exist_ok=True)
        if fresh or not self.path.exists():
            self.doc = {
                "run_id": self.run_id,
                "created_at": utcnow(),
                "suite_version": "1.0.0",
                "units": {},
                "interruption_count": 0,
            }
            atomic_write_json(self.path, self.doc)
        else:
            self.doc = load_json(self.path)
        # A leftover 'running' marker means the previous process died.
        interrupted = 0
        for key, rec in self.doc["units"].items():
            if rec.get("status") == STATUS_RUNNING:
                rec["status"] = STATUS_INTERRUPTED
                rec["interrupted_at"] = utcnow()
                interrupted += 1
        if interrupted:
            self.doc["interruption_count"] += interrupted
            self._flush()
        atomic_write_text(CURRENT_FILE, self.run_id)

    def _flush(self) -> None:
        atomic_write_json(self.path, self.doc)

    def begin(self, key: str, params: dict | None = None) -> float:
        rec = self.doc["units"].get(key, {})
        rec.update({"status": STATUS_RUNNING, "started_at": utcnow(),
                    "params_hash": sha256_obj(params or {}),
                    "attempts": int(rec.get("attempts", 0)) + 1})
        self.doc["units"][key] = rec
        self._flush()
        return time.time()

    def finish(self, key: str, result_path: Path, started: float,
               extra: dict | None = None) -> None:
        if not result_path.exists():
            raise FileNotFoundError(f"unit {key} produced no result: {result_path}")
        rec = self.doc["units"].setdefault(key, {})
        try:
            rel = str(result_path.relative_to(ROOT)).replace("\\", "/")
        except ValueError:
            # external result roots (e.g. pytest tmp_path) are still valid
            rel = result_path.resolve().as_posix()
        rec.update({"status": STATUS_DONE, "finished_at": utcnow(),
                    "elapsed_s": round(time.time() - started, 3),
                    "result": rel,
                    "result_sha256": sha256_file(result_path)})
        if extra:
            rec.update(extra)
        self._flush()

    def fail(self, key: str, message: str) -> None:
        rec = self.doc["units"].setdefault(key, {})
        rec.update({"status": STATUS_FAILED, "failed_at": utcnow(),
                    "error": str(message)[:1000]})
        self._flush()

    def blocked(self, key: str, reason: str, resume_cmd: str) -> None:
        rec = self.doc["units"].setdefault(key, {})
        rec.update({"status": STATUS_BLOCKED, "reason": reason,
                    "resume_cmd": resume_cmd, "recorded_at": utcnow()})
        self._flush()

    def forget(self, key: str) -> bool:
        """Invalidate one completed unit so the next run re-executes it.

        Motivated by a real incident: ``s1_pswa/block_alignment`` was marked
        ``done`` while its ``assertions`` block encoded a spurious comparison.
        Re-running the unit by hand and reusing the old state would have left
        the wrong assertion in place, so invalidation has to be explicit and
        visible in the ledger (``invalidated_at`` + ``attempts``).
        """
        rec = self.doc["units"].get(key)
        if not rec:
            return False
        rec.pop("result_sha256", None)
        rec.pop("finished_at", None)
        rec["status"] = STATUS_INTERRUPTED
        rec["invalidated_at"] = utcnow()
        rec["invalidate_reason"] = "manual --force"
        self._flush()
        return True

    def is_done(self, key: str) -> bool:
        rec = self.doc["units"].get(key)
        if not rec or rec.get("status") != STATUS_DONE:
            return False
        p = ROOT / rec["result"]
        if not p.exists():
            return False
        return sha256_file(p) == rec["result_sha256"]

    def unit_result(self, key: str):
        rec = self.doc["units"].get(key, {})
        if rec.get("status") != STATUS_DONE:
            return None
        return load_json(ROOT / rec["result"])

    def deps_unchanged(self, key: str, dep_hashes: dict | None) -> bool:
        """True if every upstream artifact this unit consumed is unchanged.

        Without this, resuming a run skips the figure/table stage even after an
        upstream number has been re-measured, silently leaving stale figures in
        the repository. That happened for real: `s1_pswa/gpu_memory_latency` was
        re-run (its quadratic exponent moved 1.49 -> 2.00) and the figure stage
        reported `skipped(resumed)=1`. A downstream unit is therefore only
        complete if its inputs still hash to what they hashed when it ran.
        """
        if not dep_hashes:
            return True
        rec = self.doc["units"].get(key, {})
        stored = rec.get("deps_sha")
        if not isinstance(stored, dict):
            return False                      # ran before dependency tracking
        return stored == dep_hashes

    def summary(self) -> dict:
        counts = {}
        for rec in self.doc["units"].values():
            counts[rec.get("status", "unknown")] = counts.get(rec.get("status", "unknown"), 0) + 1
        return {"run_id": self.run_id, "counts": counts,
                "interruptions": self.doc["interruption_count"],
                "total_units": len(self.doc["units"])}


def result_envelope(stage: str, unit: str, run_id: str, mock: bool,
                    seed: int | None = None, **extra) -> dict:
    """Every emitted result carries provenance. Never silently 'real'."""
    d = {"stage": stage, "unit": unit, "run_id": run_id,
         "mock": bool(mock), "seed": seed, "generated_at": utcnow(),
         "suite_version": "1.0.0"}
    d.update(extra)
    return d


def require(condition: bool, message: str) -> None:
    """Fail-fast assertion used instead of silent fallbacks."""
    if not condition:
        raise AssertionError(message)
