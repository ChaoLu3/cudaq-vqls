#!/usr/bin/env python
"""Generate LCU datasets for the VQLS solver.

Builds the linear system A x = b for a problem case, decomposes A into a Linear
Combination of Unitaries (Pauli strings) via FWHT, verifies the decomposition,
and writes a JSON dataset that ``run_vqls.py`` consumes.

Examples
--------
# One tridiagonal system, 10 qubits, drop terms below 1e-2:
python generate_lcu.py --case tridiagonal --size 10 --tol 0.01

# A range of tridiagonal sizes:
python generate_lcu.py --case tridiagonal --size 3-10 --tol 0.01

# Hele-Shaw velocity field on a 4x4 grid:
python generate_lcu.py --case hele-shaw --size 4 --var velocity --tol 0.01

Output: data/<dataset>.json with keys {coeffs, strings, matrix, b, tol,
num_qubits, num_terms, gen_time}.
"""

import argparse
import json
import os
import time

import numpy as np

from vqls import lcu, problems


def _parse_sizes(text):
    """Parse '10' or '3-10' into a list of ints."""
    if "-" in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(text)]


def generate_one(case, size, var, tol, data_dir, verify=True):
    if case == "tridiagonal":
        A, b = problems.tridiagonal_system(size)
    else:
        A, b = problems.hele_shaw_system(size, size, var=var)

    num_qubits = int(round(np.log2(A.shape[0])))
    print(f"[{case} size={size} var={var} tol={tol}] "
          f"matrix {A.shape}, {num_qubits} qubits")

    t0 = time.time()
    coeffs, strings = lcu.lcu_decompose(A, tol=tol)
    gen_time = time.time() - t0
    print(f"  LCU terms: {len(coeffs)}   decomposition time: {gen_time:.3f}s")

    if verify:
        lcu.verify_lcu(A, coeffs, strings)

    imag = max((abs(complex(c).imag) for c in coeffs), default=0.0)
    print(f"  max |imag(coeff)| = {imag:.2e}"
          + ("  (Hermitian A -> real coeffs)" if imag < 1e-9 else "  (complex coeffs)"))

    # Coefficients are complex in general; JSON has no complex type, so store each
    # as an [real, imag] pair. run_vqls.py reconstructs complex from these.
    data = {
        "case": case,
        "tol": tol,
        "num_qubits": num_qubits,
        "num_terms": len(coeffs),
        "coeffs": [[float(np.real(c)), float(np.imag(c))] for c in coeffs],
        "strings": strings,
        "matrix": A.tolist(),
        "b": np.asarray(b).tolist(),
        "gen_time": gen_time,
    }

    os.makedirs(data_dir, exist_ok=True)
    out = os.path.join(data_dir, problems.dataset_filename(case, size, var, tol))
    with open(out, "w") as fh:
        json.dump(data, fh)
    print(f"  saved -> {out}")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", choices=["tridiagonal", "hele-shaw"], default="tridiagonal")
    p.add_argument("--size", default="3",
                   help="Qubit count for tridiagonal, or grid dim n (n x n) for "
                        "hele-shaw. Accepts a range like '3-10'.")
    p.add_argument("--var", choices=["velocity", "pressure"], default="velocity",
                   help="Hele-Shaw field (ignored for tridiagonal).")
    p.add_argument("--tol", type=float, default=0.01,
                   help="LCU truncation: drop Pauli terms with |coeff| <= tol.")
    p.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "data"))
    p.add_argument("--no-verify", action="store_true")
    args = p.parse_args()

    for size in _parse_sizes(args.size):
        generate_one(args.case, size, args.var, args.tol, args.data_dir,
                     verify=not args.no_verify)


if __name__ == "__main__":
    main()
