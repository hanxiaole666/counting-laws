"""Resume targets for the two remaining blocked units.

    python -m experiments.optional_scans chain
    python -m experiments.optional_scans llm

Each probe checks its real prerequisites and, if they are missing, prints the
exact commands needed and exits 2. No gas / latency / accuracy number is ever
invented. When prerequisites ARE present, implement the measurement against the
real service and write results/optional_scans.json; that path is intentionally
left explicit rather than stubbed with fake values.
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys

from .common import ROOT, atomic_write_json, environment, utcnow


def _probe_chain() -> int:
    missing = []
    for cmd in ("node", "npx", "solc"):
        if shutil.which(cmd) is None:
            missing.append(cmd)
    for mod in ("web3", "solcx"):
        if importlib.util.find_spec(mod) is None:
            missing.append(f"python:{mod}")
    if missing:
        print("BLOCKED: on-chain measurement prerequisites missing ->",
              ", ".join(missing))
        print("Install, then re-run `python -m experiments.optional_scans chain`:")
        print("  npm install --save-dev hardhat")
        print("  pip install web3 py-solc-x")
        print("  npx hardhat node   # local EVM; deploy contracts/AnchorRoot.sol")
        print("  docker compose up hardhat   # alternative")
        print("No Gas numbers were written.")
        return 2
    print("Prerequisites present. Deploy the contract and record REAL receipts;")
    print("this repository does not ship a fabricated gas model.")
    return 0


def _probe_llm() -> int:
    env = environment()
    missing = []
    for mod in ("transformers", "peft", "bitsandbytes", "accelerate"):
        if importlib.util.find_spec(mod) is None:
            missing.append(f"python:{mod}")
    if not env.get("torch"):
        missing.append("torch")
    if missing:
        print("BLOCKED: LLM scale-scan prerequisites missing ->", ", ".join(missing))
        print("Install, download weights, then re-run "
              "`python -m experiments.optional_scans llm`:")
        print("  uv pip install --python .venv/Scripts/python.exe torch transformers "
              "peft accelerate bitsandbytes")
        print("  # huggingface.co is unreachable from this host; use a mirror:")
        print("  HF_ENDPOINT=https://hf-mirror.com huggingface-cli download "
              "Qwen/Qwen2.5-1.5B-Instruct")
        print("  # 7B-Int4 weights are ~4-5 GB; check the 20 GB free disk first.")
        print("No accuracy / latency numbers were written.")
        return 2
    print("Prerequisites present. Run the real quantisation ladder; do NOT reuse")
    print("host-RAM or proxy-model numbers as LLM results.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", choices=["chain", "llm"])
    args = ap.parse_args(argv)
    return _probe_chain() if args.target == "chain" else _probe_llm()


if __name__ == "__main__":
    sys.exit(main())
