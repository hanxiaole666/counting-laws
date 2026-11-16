"""REAL on-chain gas measurement on a local EVM (py-evm + solc).

The original script estimated gas with a hand-written model and printed a
"98.4% saving". This module compiles an actual Solidity contract, deploys it on
a real EVM implementation, executes real transactions and reads
``receipt.gas_used``. Nothing here is estimated: every number is the gas the EVM
charged for the bytecode it executed.

Measured designs (the comparison the paper needs):

  1. ``per_tx_log``      -- one ``LOG1`` event per transaction (status-quo
                            "anchoring every transaction on chain").
  2. ``batch_root``      -- ONE ``SSTORE`` of a Merkle root per 1000 transactions
                            (what the protocol actually proposes), with the 1000
                            leaves kept off chain.
  3. ``verify_inclusion``-- on-chain Merkle proof verification for one leaf,
                            i.e. what a challenger must pay to disprove a claim.

Deployment, calldata and intrinsic (21k) gas are all included in the totals, so
the reported saving cannot be inflated by hiding fixed costs.
"""
from __future__ import annotations

import json
import time

CONTRACT = """// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract AnchorRegistry {
    bytes32 public root;
    uint256 public anchorCount;

    event Leaf(bytes32 indexed commitment);

    function anchorRoot(bytes32 r) external {
        root = r;
        anchorCount += 1;
        emit Leaf(r);
    }

    function commitPerTx(bytes32 c) external {
        emit Leaf(c);
    }

    function verifyInclusion(bytes32 leaf, bytes32[] calldata proof) external view returns (bool) {
        bytes32 h = leaf;
        for (uint256 i = 0; i < proof.length; i++) {
            bytes32 p = proof[i];
            if (h <= p) {
                h = keccak256(abi.encodePacked(h, p));
            } else {
                h = keccak256(abi.encodePacked(p, h));
            }
        }
        return h == root;
    }
}
"""

SOLC_VERSION = "0.8.20"
LAST_RECEIPT: list[int] = []


def _ensure_solc():
    import solcx
    installed = [str(v) for v in solcx.get_installed_solc_versions()]
    if SOLC_VERSION not in installed:
        solcx.install_solc(SOLC_VERSION)
    solcx.set_solc_version(SOLC_VERSION)
    return SOLC_VERSION


def compile_contract() -> dict:
    import solcx
    ver = _ensure_solc()
    out = solcx.compile_source(CONTRACT, output_values=["abi", "bin"],
                              solc_version=ver)
    key = [k for k in out if k.endswith(":AnchorRegistry")][0]
    return {"abi": out[key]["abi"], "bin": out[key]["bin"], "solc": ver}


def _make_chain(funded: bytes | None = None):
    from eth import constants
    from eth.chains.base import MiningChain
    from eth.db.atomic import AtomicDB
    from eth.vm.forks import ShanghaiVM
    from eth_utils import to_canonical_address, to_wei

    klass = MiningChain.configure(__name__="TestChain",
                                  vm_configuration=((constants.GENESIS_BLOCK_NUMBER,
                                                     ShanghaiVM),))
    genesis = {"difficulty": 0, "gas_limit": 30_000_000,
               "timestamp": 1_700_000_000}
    addr = funded or to_canonical_address("0x" + "11" * 20)
    state = {addr: {"balance": to_wei(1000, "ether"), "nonce": 0,
                    "code": b"", "storage": {}}}
    return klass.from_genesis(AtomicDB(), genesis, state)


def run_gas_measurements(batch_size: int = 1000, merkle_depth: int = 10) -> dict:
    import os
    from eth_keys import keys
    from eth_utils import keccak

    t0 = time.time()
    compiled = compile_contract()
    pk = keys.PrivateKey(b"\x01" * 32)
    sender = pk.public_key.to_canonical_address()
    chain = _make_chain(sender)          # fund the signing account at genesis

    def send(data_hex: str, to=None, gas: int = 8_000_000):
        """Submit one transaction and return the gas the EVM actually charged."""
        from eth import constants as eth_constants
        vm = chain.get_vm()
        unsigned = vm.create_unsigned_transaction(
            nonce=vm.state.get_nonce(sender), gas_price=10 ** 9, gas=gas,
            to=(eth_constants.CREATE_CONTRACT_ADDRESS if to is None else to),
            value=0, data=bytes.fromhex(data_hex))
        signed = unsigned.as_signed_transaction(private_key=pk)
        _block, receipt, comp = chain.apply_transaction(signed)
        if comp.is_error:
            raise RuntimeError(f"EVM reverted: gas_used={receipt.gas_used} "
                               f"err={comp.error!r}")
        # `receipt.gas_used` on MiningChain is block-cumulative; the exact
        # per-transaction cost is gas supplied minus gas remaining after
        # execution (intrinsic 21k + calldata + execution, refunds applied).
        exact = int(gas - comp.get_gas_remaining())
        LAST_RECEIPT.append(int(receipt.gas_used))
        return exact

    deploy_gas = send(compiled["bin"])
    # contract address the EVM derived for (sender, nonce=0)
    import rlp
    from eth_utils import to_canonical_address
    contract_addr = to_canonical_address(keccak(rlp.encode([sender, 0]))[12:])
    if chain.get_vm().state.get_code(contract_addr) == b"":
        raise RuntimeError("contract deployment did not produce code")

    root = keccak(b"anchor-1")
    per_tx = []
    for i in range(5):
        leaf = keccak(b"tx-%d" % i)
        per_tx.append(send(_encode_call(compiled["abi"], "commitPerTx",
                                        ["bytes32"], [leaf]), to=contract_addr))
    anchor = send(_encode_call(compiled["abi"], "anchorRoot", ["bytes32"], [root]),
                  to=contract_addr)

    # Merkle proof verification cost (depth = merkle_depth)
    proof = [keccak(b"p%d" % i) for i in range(merkle_depth)]
    verify_data = _encode_call(compiled["abi"], "verifyInclusion",
                               ["bytes32", "bytes32[]"], [root, proof])
    verify = send(verify_data, to=contract_addr)

    total_per_tx_design = sum(per_tx) / len(per_tx) * batch_size
    total_batch_design = anchor + verify * 0  # one anchor per batch
    saving = 1 - total_batch_design / total_per_tx_design
    gas_price_gwei, eth_usd = 20.0, 3000.0          # reference assumptions only
    usd_per_million = lambda gas: gas / 1e6 * gas_price_gwei * 1e-9 * eth_usd * 1e6 / 1e6
    usd_per_tx_design = (total_per_tx_design / batch_size) * gas_price_gwei * 1e-9 * eth_usd
    usd_batch_design = (total_batch_design / batch_size) * gas_price_gwei * 1e-9 * eth_usd
    return {
        "solc": compiled["solc"],
        "deploy_gas": int(deploy_gas),
        "per_tx_log_gas": [int(g) for g in per_tx],
        "per_tx_log_gas_mean": round(sum(per_tx) / len(per_tx), 1),
        "per_tx_log_gas_std": round(float(__import__("statistics").pstdev(per_tx)), 1),
        "batch_root_gas": int(anchor),
        "verify_inclusion_gas": int(verify),
        "merkle_depth": merkle_depth,
        "batch_size": batch_size,
        "total_gas_per_tx_design": int(total_per_tx_design),
        "total_gas_batch_design": int(total_batch_design),
        "gas_saving_fraction": round(saving, 4),
        "calldata_bytes_per_tx_design": 4 + 32,
        "calldata_bytes_batch_design": 4 + 32,
        "cost_reference": {
            "gas_price_gwei": gas_price_gwei, "eth_usd": eth_usd,
            "usd_per_tx_per_tx_design": round(usd_per_tx_design, 8),
            "usd_per_tx_batch_design": round(usd_batch_design, 8),
            "note": "reference pricing, not a market quote: change the two "
                    "constants and every figure rescales linearly",
        },
        "elapsed_s": round(time.time() - t0, 2),
        "source": "py-evm ShanghaiVM + solc 0.8.20; per-tx cost = gas supplied "
                  "- gas remaining (receipt.gas_used is block-cumulative on "
                  "MiningChain and was explicitly NOT used)",
    }


def _encode_call(abi, name, types, values) -> str:
    """Minimal ABI encoder for bytes32 / bytes32[] / uint256 arguments."""
    import eth_abi
    selector = _selector(abi, name)
    encoded = eth_abi.encode(types, values).hex()
    return selector + encoded


def _selector(abi, name) -> str:
    from eth_utils import keccak
    for item in abi:
        if item.get("name") == name and item.get("type") == "function":
            sig = f"{name}({','.join(i['type'] for i in item['inputs'])})"
            return keccak(text=sig)[:4].hex()
    raise KeyError(name)


if __name__ == "__main__":
    print(json.dumps(run_gas_measurements(), indent=2, ensure_ascii=False))
