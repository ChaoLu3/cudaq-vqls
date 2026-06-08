#!/usr/bin/env python
"""DVQLS solver driver (MPI + multi-GPU, CUDA-Q mqpu).

Loads an LCU dataset, optimizes a hardware-efficient ansatz to solve A x = b, and
writes the recovered solution, fidelity, and a results plot.

Parallelism
-----------
The cost function needs, for every ordered pair of LCU terms (l, l') and every
qubit j, a Hadamard-test expectation value (real + imag -> 2 circuits each): this
is O(L^2 * nq) circuits per optimizer step, with L = number of Pauli terms. The
flat list of (l, l') pairs is *strided across MPI ranks*; within a rank, circuits
are submitted asynchronously and round-robined across that rank's GPUs (QPUs).
Each rank reduces locally, then an MPI all-reduce combines partial sums.

Run (single node, 4 GPUs)
-------------------------
    python generate_lcu.py --case tridiagonal --size 10 --tol 0.01
    mpirun -np 4 python run_dvqls.py --case tridiagonal --size 10 --tol 0.01

Under SLURM use ``srun`` (see slurm/). One MPI rank per GPU is the usual choice.
"""

import argparse
import json
import os
import time

import numpy as np
from mpi4py import MPI
from scipy.optimize import minimize

import cudaq
from cudaq import spin

from dvqls import kernels, problems, state_prep
from dvqls.utils import MinimizeStopper, fidelity

PAULI_TO_INT = {"X": 1, "Y": 2, "Z": 3, "I": 4}  # matches kernels.controlled_pauli


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", choices=["tridiagonal", "hele-shaw"], default="tridiagonal")
    p.add_argument("--size", type=int, default=3,
                   help="Qubit count (tridiagonal) or grid dim n (hele-shaw).")
    p.add_argument("--var", choices=["velocity", "pressure"], default="velocity")
    p.add_argument("--tol", type=float, default=0.01, help="LCU tolerance of the dataset.")
    p.add_argument("--backend", choices=["dense", "mps"], default="dense",
                   help="b-vector state preparation backend.")
    p.add_argument("--layers", type=int, default=3, help="Ansatz layers (3*nq params each).")
    p.add_argument("--optimizer", choices=["COBYLA", "L-BFGS-B"], default="COBYLA")
    p.add_argument("--maxiter", type=int, default=1000)
    p.add_argument("--precision", choices=["fp32", "fp64"], default="fp64")
    p.add_argument("--target", default="nvidia",
                   help="CUDA-Q target. 'nvidia' for GPU (mqpu); 'qpp-cpu' to "
                        "debug on CPU without a GPU.")
    p.add_argument("--max-sec", type=float, default=3600 * 10, help="Optimizer wall-clock cap.")
    p.add_argument("--shots", type=int, default=10 ** 6, help="Shots for solution read-out.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "data"))
    p.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "results"))
    # MPS-only knobs
    p.add_argument("--mps-layers", type=int, default=5)
    p.add_argument("--mps-chi", type=int, default=512)
    p.add_argument("--mps-fidelity", type=float, default=0.99)
    return p.parse_args()


def load_dataset(args, comm):
    """Rank 0 reads the JSON dataset; broadcast the small fields to all ranks.

    The (potentially large) dense matrix stays on rank 0 -- it is only needed at
    the end to compute the reference classical solution.
    """
    rank = comm.Get_rank()
    payload, matrix = None, None
    if rank == 0:
        fname = problems.dataset_filename(args.case, args.size, args.var, args.tol)
        path = os.path.join(args.data_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Dataset not found: {path}\nGenerate it first, e.g.:\n"
                f"  python generate_lcu.py --case {args.case} --size {args.size} "
                f"--var {args.var} --tol {args.tol}")
        with open(path) as fh:
            data = json.load(fh)
        matrix = np.array(data["matrix"])
        # coeffs are stored as [real, imag] pairs (complex in general).
        coeffs = [complex(re, im) for re, im in data["coeffs"]]
        payload = {
            "coeffs": coeffs,
            "strings": data["strings"],
            "b": data["b"],
            "num_qubits": int(data["num_qubits"]),
        }
        print(f"[rank 0] loaded {path}: {payload['num_qubits']} qubits, "
              f"{len(coeffs)} LCU terms")
    payload = comm.bcast(payload, root=0)
    return payload, matrix


def make_cost(num_qubits, coeffs, mapped_paulis, ham, comm, qpu_count, cost_history):
    """Build the distributed cost closure for scipy.optimize.minimize."""
    rank, nranks = comm.Get_rank(), comm.Get_size()
    pairs = [(l, lp) for l in range(len(coeffs)) for lp in range(len(coeffs))]

    def cost(theta):
        qpu_id = 0
        exp_futures, psi_futures = [], []

        # --- submit (this rank's stride of (l, l') pairs), async across GPUs ---
        for idx in range(rank, len(pairs), nranks):
            l, lp = pairs[idx]
            for j in range(num_qubits):  # numerator terms (one controlled-Z per qubit)
                fut = []
                for is_imag in (False, True):
                    fut.append(cudaq.observe_async(
                        kernels.hadamard_test, ham, num_qubits, is_imag, j,
                        theta, mapped_paulis[l], mapped_paulis[lp], qpu_id=qpu_id))
                    qpu_id = (qpu_id + 1) % qpu_count
                exp_futures.append((l, lp, fut))
            fut = []  # normalization term (<psi|psi>), j = -1
            for is_imag in (False, True):
                fut.append(cudaq.observe_async(
                    kernels.hadamard_test, ham, num_qubits, is_imag, -1,
                    theta, mapped_paulis[l], mapped_paulis[lp], qpu_id=qpu_id))
                qpu_id = (qpu_id + 1) % qpu_count
            psi_futures.append((l, lp, fut))

        # --- collect local partial sums ---
        local_num = 0.0 + 0.0j
        for l, lp, fut in exp_futures:
            term = fut[0].get().expectation() + 1.0j * fut[1].get().expectation()
            local_num += coeffs[l] * np.conj(coeffs[lp]) * term
        local_psi = 0.0 + 0.0j
        for l, lp, fut in psi_futures:
            term = fut[0].get().expectation() + 1.0j * fut[1].get().expectation()
            local_psi += coeffs[l] * np.conj(coeffs[lp]) * term

        # --- global reduction ---
        total_num = comm.allreduce(local_num, op=MPI.SUM)
        total_psi = comm.allreduce(local_psi, op=MPI.SUM)

        eps = 1e-16
        cost_val = 0.5 - 0.5 * np.real(total_num / (num_qubits * total_psi + eps))
        if rank == 0:
            cost_history.append(cost_val)
            print(f"  cost = {cost_val:.8f}")
        return cost_val

    return cost


def recover_and_report(args, result, num_qubits, matrix, b_vec, cost_history):
    """Rank 0: sample the optimized ansatz, compute fidelity, save JSON + plot."""
    counts = cudaq.sample(kernels.prepare_solution, result.x, num_qubits,
                          shots_count=args.shots)
    quantum = np.zeros(2 ** num_qubits, dtype=complex)
    for bitstring, mag in counts.items():
        quantum[int(bitstring, 2)] = np.sqrt(mag / args.shots)

    classical = np.linalg.solve(matrix, b_vec)
    classical = classical / np.linalg.norm(classical)

    # Read-out endianness: compare both bit orderings, keep the better match.
    fid_normal = fidelity(quantum, classical)
    fid_rev = fidelity(quantum[::-1], classical)
    if fid_rev > fid_normal:
        quantum = quantum[::-1]
    fid = max(fid_normal, fid_rev)
    print(f"Fidelity (DVQLS vs classical): {fid:.6f}")

    os.makedirs(args.out_dir, exist_ok=True)
    tag = problems.dataset_filename(args.case, args.size, args.var, args.tol)[:-5]
    with open(os.path.join(args.out_dir, f"{tag}_result.json"), "w") as fh:
        json.dump({
            "result_param": result.x.tolist(),
            "fidelity": fid,
            "nfev": int(result.nfev),
            "final_cost": float(cost_history[-1]) if cost_history else None,
            "cost_history": [float(c) for c in cost_history],
        }, fh)

    _plot(args, tag, num_qubits, quantum, classical, cost_history)


def _plot(args, tag, num_qubits, quantum, classical, cost_history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.2))
    labels = [format(i, f"0{num_qubits}b") for i in range(2 ** num_qubits)]
    x = np.arange(len(labels))
    ax1.bar(x - 0.2, np.abs(classical), 0.4, label="Classical", color="#0071C5")
    ax1.bar(x + 0.2, np.abs(quantum), 0.4, label="DVQLS", color="#76B900")
    ax1.set_xlabel("Computational basis state")
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"Solution ({num_qubits} qubits)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=90, fontsize=6)
    ax1.legend()

    ax2.plot(range(1, len(cost_history) + 1), cost_history, color="#76B900", marker="o", ms=3)
    ax2.set_xlabel("Function evaluation")
    ax2.set_ylabel("Cost")
    ax2.set_title("Learning curve")
    fig.tight_layout()
    out = os.path.join(args.out_dir, f"{tag}_results.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved plot -> {out}")


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if args.target == "nvidia":
        cudaq.set_target("nvidia", option=f"mqpu,{args.precision}")
    else:
        cudaq.set_target(args.target)
    qpu_count = max(cudaq.get_target().num_qpus(), 1)
    if rank == 0:
        print(f"target: {args.target}   MPI ranks: {comm.Get_size()}   "
              f"QPUs per rank: {qpu_count}   precision: {args.precision}")

    # Warm up the JIT so it doesn't pollute the first cost evaluation's timing.
    @cudaq.kernel
    def _warm_up():
        q = cudaq.qubit()
        x(q)
    cudaq.sample(_warm_up)

    payload, matrix = load_dataset(args, comm)
    num_qubits = payload["num_qubits"]
    coeffs = payload["coeffs"]
    mapped_paulis = [[PAULI_TO_INT[ch] for ch in s] for s in payload["strings"]]

    # Build U_b (auto-generated to match num_qubits) and inject into kernels.
    state_prep.prepare_b(payload["b"], num_qubits, backend=args.backend, comm=comm,
                         max_num_layers=args.mps_layers, chi_max=args.mps_chi,
                         target_fidelity=args.mps_fidelity)

    ham = spin.z(0)  # Hadamard test reads <Z> on the ancilla (allocated as qubit 0)

    np.random.seed(args.seed)
    n_params = kernels.num_params(num_qubits, args.layers)
    init = np.random.normal(0, 0.01, n_params)

    cost_history = []
    cost = make_cost(num_qubits, coeffs, mapped_paulis, ham, comm, qpu_count, cost_history)
    stopper = MinimizeStopper(max_sec=args.max_sec)

    if rank == 0:
        print(f"Optimizing: {args.optimizer}, {n_params} params, maxiter={args.maxiter}")
    t0 = time.time()
    try:
        result = minimize(cost, init, method=args.optimizer,
                          options={"maxiter": args.maxiter},
                          callback=(stopper.callback if rank == 0 else None))
    except StopIteration:
        if rank == 0:
            print("Stopped: wall-clock limit reached.")
        return
    elapsed = time.time() - t0

    if rank == 0:
        print(f"Done. nfev={result.nfev}  total={elapsed:.2f}s  "
              f"avg/eval={elapsed / max(result.nfev, 1):.3f}s")
        recover_and_report(args, result, num_qubits, matrix,
                           np.array(payload["b"], dtype=float), cost_history)


if __name__ == "__main__":
    main()
