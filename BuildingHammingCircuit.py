#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from itertools import combinations
from typing import List, Optional, Tuple

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, transpile

try:
    from qiskit import AncillaRegister
except Exception:
    AncillaRegister = None  # type: ignore


def multi_qubit_cost(qc: QuantumCircuit, allow_ccx: bool) -> int:
    """Count multi-qubit gates:
       - if allow_ccx: CX + CCX
       - else: CX only (since CCX should be decomposed away)
    """
    ops = qc.count_ops()
    cx = int(ops.get("cx", 0))
    ccx = int(ops.get("ccx", 0))
    return cx + ccx if allow_ccx else cx


def _need_clean_ancillas_for_mode(num_controls: int, mode: str) -> int:
    """Conservative ancilla requirements for common mcx modes."""
    mode = mode.lower()
    if mode in ("v-chain", "v-chain-dirty"):
        return max(0, num_controls - 2)
    if mode in ("noancilla", "recursion"):
        # For <=4 controls, recursion is typically 0-ancilla; if not, Qiskit will error and we skip.
        return 0
    raise ValueError(f"Unknown mcx mode: {mode}")


def build_hw2_oracle_with_mcx_mode(
    mcx_mode: str,
    n_ctrl: int,
    n_anc_total: int,
) -> QuantumCircuit:
    """Build HW=2 oracle using qc.mcx(..., mode=mcx_mode), leaving decomposition to Qiskit."""
    if n_ctrl != 4:
        raise ValueError("This script currently targets n_ctrl=4 (HW=2 on 4 controls).")

    c = QuantumRegister(n_ctrl, "c")
    t = QuantumRegister(1, "t")

    use_anc = (n_anc_total > 0 and AncillaRegister is not None)
    if use_anc:
        anc = AncillaRegister(n_anc_total, "anc")
        qc = QuantumCircuit(c, t, anc, name=f"HW2_mcx_{mcx_mode}")
    else:
        qc = QuantumCircuit(c, t, name=f"HW2_mcx_{mcx_mode}")
        anc = None  # type: ignore

    controls = [c[i] for i in range(n_ctrl)]
    target = t[0]

    need = _need_clean_ancillas_for_mode(n_ctrl, mcx_mode)
    if need > 0:
        if not use_anc:
            raise ValueError(f"Mode {mcx_mode} needs ancillas but n_anc_total=0.")
        if need > n_anc_total:
            raise ValueError(f"Mode {mcx_mode} needs {need} ancillas but only {n_anc_total} provided.")
        anc_list = [anc[i] for i in range(need)]
    else:
        anc_list = None

    # HW=2 patterns: choose which 2 controls are 1; the rest are 0 (open-controls via X)
    for ones in combinations(range(n_ctrl), 2):
        zeros = [i for i in range(n_ctrl) if i not in ones]

        for i in zeros:
            qc.x(c[i])

        qc.mcx(controls, target, ancilla_qubits=anc_list, mode=mcx_mode)

        for i in zeros:
            qc.x(c[i])

    return qc


def try_transpile(qc: QuantumCircuit, seed: int, opt_level: int, allow_ccx: bool) -> QuantumCircuit:
    """Transpile to the requested basis."""
    if allow_ccx:
        basis = ["x", "cx", "ccx", "id"]
    else:
        basis = ["u", "cx", "id"]  # force CCX to be decomposed away

    return transpile(
        qc,
        basis_gates=basis,
        optimization_level=opt_level,
        seed_transpiler=seed,
    )


def dump_qasm_best(qc: QuantumCircuit, out_base: str) -> str:
    """Export to QASM3 if available; else QASM2. Returns output filename."""
    try:
        from qiskit import qasm3  # type: ignore

        out = out_base + ".qasm3"
        qasm3.dump(qc, out)
        return out
    except Exception:
        pass

    try:
        from qiskit import qasm2  # type: ignore

        out = out_base + ".qasm"
        qasm2.dump(qc, out)
        return out
    except Exception:
        try:
            from qiskit import qasm2  # type: ignore

            out = out_base + ".qasm"
            text = qasm2.dumps(qc)
            with open(out, "w", encoding="utf-8") as f:
                f.write(text)
            return out
        except Exception as e:
            raise RuntimeError(f"Failed to export QASM2/QASM3: {e}") from e


def parse_seeds(seeds_str: str) -> List[int]:
    seeds_str = seeds_str.strip()
    if not seeds_str:
        return [0]
    parts = [p.strip() for p in seeds_str.split(",")]
    out = []
    for p in parts:
        if p:
            out.append(int(p))
    return out if out else [0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anc", type=int, default=16, help="Total clean ancillas to allocate (default: 16).")
    ap.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated transpiler seeds (default: 0,1,2).")
    ap.add_argument("--opt", type=int, default=3, choices=[0, 1, 2, 3], help="transpile optimization_level (default: 3).")
    ap.add_argument("--out", type=str, default="hw2_best_u_cx_ccx", help="Output filename base (no extension).")
    ap.add_argument(
        "--no-ccx",
        action="store_true",
        help="Disallow CCX in final circuit (basis becomes ['u','cx','id']; cost becomes CX only).",
    )
    args = ap.parse_args()

    allow_ccx = not args.no_ccx

    print("Qiskit version:", getattr(qiskit, "__version__", "unknown"))
    print("Requested ancillas:", args.anc)
    print("Seeds:", args.seeds)
    print("Optimization level:", args.opt)
    print("Allow CCX in output:", allow_ccx)

    n_ctrl = 4
    modes = ["noancilla", "recursion", "v-chain", "v-chain-dirty"]
    seeds = parse_seeds(args.seeds)

    best: Optional[Tuple[int, str, int, QuantumCircuit]] = None  # (cost, mode, seed, circuit)

    print("\nPer-try results:")
    if allow_ccx:
        print("  (mode, seed) -> U, CX, CCX, (CX+CCX), depth")
    else:
        print("  (mode, seed) -> U, CX, CCX(should be 0), (CX), depth")

    for mode in modes:
        for seed in seeds:
            try:
                qc = build_hw2_oracle_with_mcx_mode(
                    mcx_mode=mode,
                    n_ctrl=n_ctrl,
                    n_anc_total=args.anc,
                )
                tqc = try_transpile(qc, seed=seed, opt_level=args.opt, allow_ccx=allow_ccx)

                ops = tqc.count_ops()
                u_cnt = int(ops.get("u", 0))
                cx_cnt = int(ops.get("cx", 0))
                ccx_cnt = int(ops.get("ccx", 0))
                cost = multi_qubit_cost(tqc, allow_ccx=allow_ccx)

                print(
                    f"[ok] mode={mode:12s} seed={seed:3d}  "
                    f"U={u_cnt:5d}  CX={cx_cnt:5d}  CCX={ccx_cnt:5d}  "
                    f"cost={cost:5d}  depth={tqc.depth():5d}"
                )

                if best is None or cost < best[0]:
                    best = (cost, mode, seed, tqc)

            except Exception as e:
                print(f"[skip] mode={mode:12s} seed={seed:3d} -> {type(e).__name__}: {e}")

    if best is None:
        raise RuntimeError("All modes failed. Check your Qiskit install/API compatibility.")

    cost, mode, seed, tqc = best
    ops = tqc.count_ops()

    print("\nBEST RESULT")
    print("  mode:", mode)
    print("  seed:", seed)
    print("  U:", int(ops.get("u", 0)))
    print("  CX:", int(ops.get("cx", 0)))
    print("  CCX:", int(ops.get("ccx", 0)))
    if allow_ccx:
        print("  Total (CX+CCX):", int(ops.get("cx", 0)) + int(ops.get("ccx", 0)))
    else:
        print("  Total (CX):", int(ops.get("cx", 0)))
    print("  depth:", tqc.depth())
    print("  total qubits:", tqc.num_qubits)

    out_file = dump_qasm_best(tqc, args.out)
    print("Saved:", out_file)


if __name__ == "__main__":
    main()
