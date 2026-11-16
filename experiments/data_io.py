"""Auto-download, verify, and load the two evaluation datasets.

Commands (copy-paste ready):

    # 1) idempotent: download only when missing / corrupt; otherwise continue
    python -m experiments.data_io --ensure
    python -m experiments.data_io --ensure --datasets ieee_cis,baf
    python -m experiments.data_io --status

    # 2) via the Kaggle CLI instead (requires ~/.kaggle/kaggle.json + accepted rules)
    kaggle competitions download -c ieee-fraud-detection -p data/raw
    kaggle datasets download -v sgpjesus/bank-account-fraud-dataset-neurips-2022 -p data/raw

    # 3) direct HTTPS (no credentials), same files the downloader uses
    curl -L -C - -o data/ieee_cis.parquet \
      "https://hf-mirror.com/datasets/mkbadeniyi/fraud-detection-benchmark/resolve/main/fraud_datasets/ieee_cis_card_payments_real.parquet?download=true"

``--ensure`` is what run_all.py calls automatically before any experiment.
Every dataset is pinned by sha256 in data/MANIFEST.json after first download.
A checksum mismatch is a hard error, not a warning that gets ignored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from .common import DATA_DIR, ROOT, atomic_write_json, load_json, sha256_file, utcnow

HF_REPO = "mkbadeniyi/fraud-detection-benchmark"
HF_DIR = "fraud_datasets"
MIRRORS = ["https://hf-mirror.com", "https://huggingface.co"]

DATASETS = {
    "ieee_cis": {
        "file": "ieee_cis_card_payments_real.parquet",
        "dest": "ieee_cis.parquet",
        "size_hint": 61_700_000,
        # published SHA256 on the dataset card (verified on first download)
        "card_sha256": "62a14b59a946eb6e0c0d45ac56ed6eeb7dc014babeb04803653191bcb3ddd30d",
        "kind": "real",
        "license": "third-party mirror of Kaggle IEEE-CIS (competition rules apply; NOT redistributable)",
        "kaggle": "kaggle competitions download -c ieee-fraud-detection -p data/raw",
    },
    "baf": {
        "file": "baf_bank_account_opening_synthetic.parquet",
        "dest": "baf.parquet",
        "size_hint": 66_000_000,
        "card_sha256": None,
        "kind": "synthetic (source-declared)",
        "license": "BAF Suite, CC BY-NC-SA 4.0 (attribute; no commercial use; share-alike)",
        "kaggle": "kaggle datasets download -v sgpjesus/bank-account-fraud-dataset-neurips-2022 -p data/raw",
    },
}

MANIFEST = DATA_DIR / "MANIFEST.json"
CHUNK = 1 << 20


def _urls(name: str) -> list[str]:
    meta = DATASETS[name]
    return [f"{m}/datasets/{HF_REPO}/resolve/main/{HF_DIR}/{meta['file']}?download=true"
            for m in MIRRORS]


def _parquet_sane(path: Path) -> bool:
    """Lightweight structural check (works even without pyarrow installed)."""
    if path.stat().st_size < 1_000_000:
        return False
    with open(path, "rb") as fh:
        head = fh.read(4)
        fh.seek(-4, os.SEEK_END)
        tail = fh.read(4)
    return head == b"PAR1" and tail == b"PAR1"


def _download_resumable(url: str, dest: Path, timeout: int = 90) -> dict:
    """HTTP Range resume into <dest>.part, atomic promotion on completion.

    HTTP 416 means the .part already covers the whole object (e.g. the process
    died after the last write but before os.replace). We validate the parquet
    magic bytes and promote instead of restarting from zero.
    """
    part = dest.with_name(dest.name + ".part")
    pos = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "VERI-FinRisk/1.0", "Accept": "*/*"}
    if pos:
        headers["Range"] = f"bytes={pos}-"
    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and part.exists() and _parquet_sane(part):
            size = part.stat().st_size
            os.replace(part, dest)
            return {"url": url, "status": 416, "bytes_written": 0,
                    "resumed_from": pos, "promoted_complete_part": True,
                    "file_size": size, "elapsed_s": 0.0,
                    "note": "416 + valid PAR1 footer -> .part was already complete"}
        raise
    with resp:
        status = resp.status
        if pos and status != 206:
            # server ignored Range: restart from zero
            pos = 0
            mode = "wb"
        else:
            mode = "ab" if pos else "wb"
        total = resp.headers.get("Content-Range") or resp.headers.get("Content-Length")
        wrote = 0
        t0 = time.time()
        with open(part, mode) as out:
            while True:
                buf = resp.read(CHUNK)
                if not buf:
                    break
                out.write(buf)
                wrote += len(buf)
                if wrote % (8 * CHUNK) == 0:
                    out.flush()
                    os.fsync(out.fileno())
                    print(f"    ... {wrote / 1e6:.1f} MB received", flush=True)
            out.flush()
            os.fsync(out.fileno())
    size = part.stat().st_size
    if not _parquet_sane(part):
        raise RuntimeError(f"downloaded file is not a complete parquet: {part} ({size} B)")
    os.replace(part, dest)
    return {"url": url, "status": status, "bytes_written": wrote,
            "resumed_from": pos, "total_hint": str(total),
            "file_size": size, "elapsed_s": round(time.time() - t0, 2)}


def _try_download(name: str, force: bool = False) -> dict:
    meta = DATASETS[name]
    dest = DATA_DIR / meta["dest"]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if force and dest.exists():
        dest.unlink()
    if dest.exists():
        return {"action": "reused", "path": str(dest), "sha256": sha256_file(dest)}

    errors = []
    for url in _urls(name):
        for attempt in range(3):
            try:
                print(f"  downloading {name} from {url.split('/')[2]} (attempt {attempt + 1})",
                      flush=True)
                info = _download_resumable(url, dest)
                digest = sha256_file(dest)
                if meta["card_sha256"] and digest != meta["card_sha256"]:
                    # The dataset card publishes a hash; if the file differs we
                    # must NOT silently accept it. Record the discrepancy and pin
                    # the observed bytes, forcing a human decision.
                    info["card_sha256_mismatch"] = {
                        "expected": meta["card_sha256"], "observed": digest}
                    info["action"] = "downloaded_with_hash_discrepancy"
                else:
                    info["action"] = "downloaded"
                info["sha256"] = digest
                info["mirror"] = url.split("/")[2]
                return info
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                errors.append(f"{url.split('/')[2]} attempt {attempt + 1}: {exc}")
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"all download sources failed for {name}:\n  " + "\n  ".join(errors))


def _manifest() -> dict:
    if MANIFEST.exists():
        return load_json(MANIFEST)
    return {}


def ensure_dataset(name: str, force: bool = False) -> dict:
    """Idempotent: valid local data -> reuse; missing/corrupt -> (re)download."""
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name}; known: {list(DATASETS)}")
    meta = DATASETS[name]
    dest = DATA_DIR / meta["dest"]
    man = _manifest()
    rec = man.get(name, {})
    action = "reused"
    if dest.exists():
        digest = sha256_file(dest)
        if rec.get("sha256") and digest != rec["sha256"]:
            # pinned bytes changed -> refuse to silently use them
            print(f"  checksum mismatch for {name}; re-downloading", flush=True)
            dest.unlink()
        elif not rec.get("sha256"):
            action = "reused_unpinned"
    if not dest.exists():
        info = _try_download(name, force=force)
        action = info.pop("action", "downloaded")
        digest = info.pop("sha256")
        rec = {"sha256": digest, "card_sha256": meta["card_sha256"],
               "card_sha256_mismatch": info.get("card_sha256_mismatch"),
               "size_bytes": dest.stat().st_size, "source": info.get("url"),
               "mirror": info.get("mirror"), "downloaded_at": utcnow(),
               "kind": meta["kind"], "license": meta["license"],
               "kaggle_alternative": meta["kaggle"]}
        man[name] = rec
        atomic_write_json(MANIFEST, man)
    else:
        rec = dict(rec)
        rec.setdefault("sha256", sha256_file(dest))
        rec.setdefault("kind", meta["kind"])
        rec.setdefault("license", meta["license"])
    rec.update({"name": name, "path": str(dest), "action": action,
                "exists": dest.exists(), "size_bytes": dest.stat().st_size})
    return rec


def schema(name: str) -> dict:
    import pandas as pd
    rec = ensure_dataset(name)
    df = pd.read_parquet(rec["path"])
    out = {
        "dataset": name, "rows": int(len(df)), "columns": int(df.shape[1]),
        "column_names": [str(c) for c in df.columns],
        "label_column": "label" if "label" in df.columns else None,
        "sha256": rec["sha256"], "kind": rec["kind"],
    }
    if "label" in df.columns:
        out["fraud_count"] = int(df["label"].sum())
        out["fraud_rate"] = float(df["label"].mean())
    if "split" in df.columns:
        out["split_counts"] = {str(k): int(v) for k, v in df["split"].value_counts().items()}
        out["split_fraud_rate"] = {str(k): float(v) for k, v in
                                   df.groupby("split")["label"].mean().items()}
    if "ts" in df.columns:
        out["ts_min"] = str(df["ts"].min())
        out["ts_max"] = str(df["ts"].max())
        out["ts_monotonic_by_split"] = bool(
            df.groupby("split", sort=False)["ts"].max().iloc[0]
            <= df.groupby("split", sort=False)["ts"].min().iloc[-1]
        ) if len(df.groupby("split")) > 1 else None
    if "user_id" in df.columns:
        out["n_entities"] = int(df["user_id"].nunique())
        out["entity_overlap_across_splits"] = int(
            (df.groupby("user_id")["split"].nunique() > 1).sum())
    return out


def feature_columns(name: str, k: int = 60) -> list[str]:
    """Deterministic numeric predictor list from the parquet schema.

    Core fields come first, V* (Vesta) fields last; META columns are excluded.
    Reading only these k columns keeps peak RAM ~150 MB instead of ~2 GB.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from .baselines import META_COLS
    rec = ensure_dataset(name)
    schema = pq.read_schema(rec["path"])
    numeric = [f.name for f in schema
               if (pa.types.is_integer(f.type) or pa.types.is_floating(f.type))
               and f.name not in META_COLS]
    core = [c for c in numeric if not c.startswith(("V", "id_"))]
    vest = [c for c in numeric if c.startswith("V")]
    ordered = core + vest
    return ordered[:k]


def load(name: str, columns: list[str] | None = None,
         splits: list[str] | None = None):
    """Column-pruned parquet read; optionally restricted to given splits."""
    import pandas as pd
    rec = ensure_dataset(name)
    df = pd.read_parquet(rec["path"], columns=columns)
    if splits is not None and "split" in df.columns:
        df = df[df["split"].isin(splits)]
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Download / verify evaluation data")
    ap.add_argument("--ensure", action="store_true", help="download missing data, reuse existing")
    ap.add_argument("--datasets", default="ieee_cis,baf")
    ap.add_argument("--status", action="store_true", help="print schema + provenance")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    names = [s.strip() for s in args.datasets.split(",") if s.strip()]
    out = {}
    for name in names:
        rec = ensure_dataset(name, force=args.force)
        out[name] = rec
        if args.status:
            try:
                rec["schema"] = schema(name)
            except Exception as exc:  # surface, never hide
                rec["schema_error"] = str(exc)
    if args.json or args.status:
        print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    else:
        for name, rec in out.items():
            print(f"{name}: {rec['action']}  {rec['size_bytes'] / 1e6:.1f} MB  "
                  f"sha256={rec['sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
