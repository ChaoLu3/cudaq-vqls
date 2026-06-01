# VQLS-CUDAQ

A **Variational Quantum Linear Solver (VQLS)** built on NVIDIA **CUDA-Q**, designed
to run at scale on multi-GPU / multi-node systems (developed on NERSC Perlmutter).
It solves linear systems `A x = b` for two physics problems:

- **`tridiagonal`** — a symmetric tridiagonal Toeplitz matrix (a standard,
  well-conditioned VQLS benchmark).
- **`hele-shaw`** — the finite-difference operator of 2D Hele-Shaw (Stokes) flow,
  for either the pressure or the velocity field.

The matrix `A` is expanded as a **Linear Combination of Unitaries (LCU)** —
`A = Σ_l c_l P_l` with Pauli strings `P_l` — and a hardware-efficient ansatz is
optimized so that `A|x⟩ ∝ |b⟩`.

---

## How VQLS works here

VQLS minimizes a cost that is zero when the prepared state `|x(θ)⟩ = V(θ)|0⟩`
satisfies `A|x⟩ ∝ |b⟩`:


$$
C_L(\theta)
=
1 - \frac{1}{n}
\sum_{j=1}^{n}
\frac{
\langle x(\theta) \vert
A^\dagger U_b P_j U_b^\dagger A
\vert x(\theta) \rangle
}{
\langle x(\theta) \vert A^\dagger A \vert x(\theta) \rangle
}
$$

Every overlap is estimated by a **Hadamard test**: an ancilla is put in
superposition, the controlled operations are applied, and `⟨Z⟩` on the ancilla
yields the real part (an extra `Rz(−π/2)` on the ancilla gives the imaginary part).
Because `A = Σ_l c_l P_l`, each overlap decomposes into terms over **ordered pairs
of Pauli strings `(l, l′)`** and over each qubit `j` — that double sum is where the
parallel work comes from (see *Parallelism* below).

---

## Repository layout

```
vqls_cudaq/
├── README.md
├── requirements.txt
├── generate_lcu.py          # build A,b and decompose into LCU (CPU, no GPU needed)
├── run_vqls.py              # MPI + multi-GPU solver driver (the main entry point)
├── vqls/                    # the package
│   ├── problems.py          # tridiagonal + Hele-Shaw systems; dataset naming
│   ├── lcu.py               # FWHT-based LCU decomposition + verification
│   ├── kernels.py           # CUDA-Q kernels: ansatz, controlled-Pauli, Hadamard test
│   ├── state_prep.py        # |b> preparation: dense-unitary and MPS backends
│   ├── utils.py             # fidelity, optimizer time-limit callback
│   └── _generated/          # auto-generated U_b kernels (created at runtime)
├── slurm/
│   ├── single_node.sh       # 1 node, 4 GPUs
│   ├── strong_scaling.sh    # fixed problem, sweep ranks  -> speed-up
│   └── weak_scaling.sh      # grow problem + nodes together -> constant work/GPU
├── data/                    # generated LCU datasets (*.json)
└── results/                 # solver outputs (*_result.json, *_results.pdf)
```

---

## Installation

On NERSC Perlmutter (or any system with NVIDIA GPUs):

```bash
module load conda cudatoolkit
conda create --prefix ./.conda python=3.12 -y
conda activate ./.conda
pip install -r requirements.txt
```

`cudaq` provides the `nvidia` GPU simulator with the **mqpu** (multi-QPU) platform.
`quimb` is only needed for the optional MPS state-prep backend.

---

## Quickstart

```bash
# 1. Generate an LCU dataset (CPU only — safe to run on a login node)
python generate_lcu.py --case tridiagonal --size 10 --tol 0.01

# 2a. NERSC interactive test on GPUs via srun (nvidia mqpu target).
#     --gpus-per-node is required, or the tasks get no GPU ("no CUDA-capable device"):
srun -A m5097 -C gpu -q interactive -N 1 -n 4 --gpus-per-node=4 --gpu-bind=none -t 00:10:00 \
    python -u run_vqls.py --case tridiagonal --size 10 --tol 0.01 --target nvidia

# 2b. Or, off-scheduler / single workstation with GPUs:
mpirun -np 4 python run_vqls.py --case tridiagonal --size 10 --tol 0.01

# Debug the whole pipeline on CPU without a GPU:
python run_vqls.py --case tridiagonal --size 4 --tol 0.01 --target qpp-cpu
```

For batch jobs, use the scripts in `slurm/` (which call `srun` inside `sbatch`).

Outputs land in `results/`: a JSON with the optimized parameters, fidelity, and
cost history, plus a PDF comparing the VQLS solution to the classical one.

---

## The two problems

| Case | `--size` means | Matrix | RHS `b` |
|------|----------------|--------|---------|
| `tridiagonal` | qubit count `K` | `2^K × 2^K`, diag 1, off-diag −1/3 | `e₀` (|0…0⟩) |
| `hele-shaw` | grid dim `n` (n×n) | finite-difference operator, padded to a power of two and symmetrized to Hermitian | analytic pressure/velocity forcing |

The non-Hermitian Hele-Shaw operators are embedded via the standard
`[[0, A†], [A, 0]]` dilation (handled in `problems.py`), which makes `A` Hermitian
at the cost of one extra qubit. This is the validated path; note the LCU
decomposition itself no longer needs this (it handles non-Hermitian `A` with
complex coefficients), but the dilation keeps the solve well-posed and the
solution real for comparison against the classical reference.

---

## LCU generation (`generate_lcu.py`)

`A` is decomposed into Pauli strings with a **Fast Walsh-Hadamard Transform** over
the diagonals of `A` — far cheaper than the naive `O(4ⁿ)` projection, and
parallelized across CPU cores via shared memory.

The decomposition is **general**: it produces the exact (complex) Pauli
coefficients of any `2ⁿ × 2ⁿ` matrix, Hermitian or not. (For Hermitian `A` the
coefficients come out real.) The per-diagonal FWHT uses a `(+i)^{#Y}` phase so it
decomposes `A` itself; an earlier sign convention silently decomposed `Aᵀ`, which
agrees only for symmetric `A`.

```bash
python generate_lcu.py --case tridiagonal --size 3-10 --tol 0.01   # a range of sizes
python generate_lcu.py --case hele-shaw --size 4 --var velocity --tol 0.01
```

- `--tol` truncates the expansion: terms with `|c_k| ≤ tol` are dropped. **Larger
  `tol` ⇒ fewer Pauli terms ⇒ cheaper, lower-accuracy cost.** This single knob
  controls the solver's workload (work scales as `O(L²)` in the term count `L`).
- Each run prints `max |imag(coeff)|` (≈0 confirms a Hermitian `A`) and a
  verification of the decomposition (trace and Frobenius-norm identities).

Datasets are written to `data/<name>.json` with keys
`{coeffs, strings, matrix, b, tol, num_qubits, num_terms}`, where each coefficient
is stored as a `[real, imag]` pair (JSON has no complex type) and reloaded as a
complex number by the solver.

---

## State preparation: the `U_b` problem and the fix

VQLS needs a unitary `U_b` with `U_b|0⟩ = |b⟩` that can be applied **and adjointed,
controlled,** inside the Hadamard test. The catch in CUDA-Q: a custom operation
(`cudaq.register_operation`) has a **fixed arity decided when it is registered**, so
a single hand-written `U_b` kernel cannot serve different qubit counts — the
original code had to manually un-comment the matching branch for each problem size.

This package removes that footgun: **`U_b` is generated automatically to match the
exact qubit count** and injected into the kernel module at runtime. There is also
an automatic fast path, plus two general backends:

- **Basis-state fast path (automatic).** If `b` is a computational basis state
  `|k⟩` — which the `tridiagonal` RHS `b = e₀` always is — `U_b` is just an X on
  each set bit of `k` (identity for `k = 0`). No custom op, so it runs at **any
  qubit count on any backend**. This is detected automatically and overrides the
  chosen backend.

| Backend | How | Trade-off |
|---------|-----|-----------|
| `dense` (default) | Pad `|b⟩` to a full `2ⁿ × 2ⁿ` unitary (`|b⟩` as its first column) and register it as one custom op. | Exact, but **cuStateVec caps a custom matrix op at 6 qubits** (`dim ≤ 64`), so on the `nvidia` backend a *non-basis* `b` is limited to **≤ 6 qubits**. (Raises a clear error otherwise.) |
| `mps` | Approximate `|b⟩` as a bond-2 MPS and compile it into 1- and 2-qubit gates (Ran et al., [arXiv:1908.07958](https://arxiv.org/abs/1908.07958)). | Each gate has fixed small arity *regardless of system size*, so it scales past the 6-qubit dense ceiling. **Experimental — check the printed encoding fidelity per `b`.** |

So: `tridiagonal` runs at any size via the basis fast path; Hele-Shaw (a general
`b`) uses `dense` up to 6 qubits, or `mps` beyond that.

```bash
python run_vqls.py ... --backend dense          # default; auto basis fast path for tridiagonal
python run_vqls.py ... --backend mps --mps-layers 5 --mps-fidelity 0.99
```

Generation is **MPI-safe**: each rank writes its own rank-local module file (for
`mps`, the gate layers are computed on rank 0 and broadcast so every rank registers
identical operations), so there is no shared-file race.

> Implementation note: the dense `U_b` passes qubits **high→low** to the custom op.
> NumPy's `kron` convention (qubit 0 as the most-significant bit) is the reverse of
> CUDA-Q's register order; passing them reversed makes `U_b|0⟩ = |b⟩` exact.

---

## The solver and its parallelism (`run_vqls.py`)

The cost function requires, **per optimizer step**, a Hadamard-test expectation for
every ordered pair of LCU terms `(l, l′)` and every qubit `j` (real + imaginary ⇒
two circuits each): `O(L² · n)` circuits, where `L` is the number of Pauli terms.

The driver distributes this work on three levels:

1. **Across MPI ranks** — the flat list of `(l, l′)` pairs is *strided by rank*
   (`pairs[rank::nranks]`); each rank reduces its share, then `MPI.allreduce(SUM)`
   combines the partial numerator and normalization sums.
2. **Across GPUs within a rank** — circuits are submitted with
   `cudaq.observe_async(..., qpu_id=…)`, round-robined over that rank's QPUs (the
   `nvidia` `mqpu` platform exposes each visible GPU as a virtual QPU).
3. A `warm_up` kernel is sampled once at startup so JIT compilation does not
   pollute the first cost evaluation's timing.

The classical optimizer is SciPy `minimize` (`COBYLA` by default; `L-BFGS-B`
available). A wall-clock cap (`--max-sec`) ends long runs cleanly with the best
parameters found.

### Key CLI flags (`run_vqls.py`)

| Flag | Default | Meaning |
|------|---------|---------|
| `--case` | `tridiagonal` | `tridiagonal` or `hele-shaw` |
| `--size` | `10` | qubit count (tridiagonal) or grid dim (hele-shaw) |
| `--var` | `velocity` | Hele-Shaw field: `velocity` / `pressure` |
| `--tol` | `0.01` | LCU tolerance of the dataset to load |
| `--backend` | `dense` | `b`-encoding: `dense` / `mps` |
| `--layers` | `7` | ansatz layers (3·n params per layer) |
| `--optimizer` | `COBYLA` | `COBYLA` / `L-BFGS-B` |
| `--maxiter` | `1000` | optimizer iterations |
| `--precision` | `fp64` | `fp32` / `fp64` (nvidia target) |
| `--target` | `nvidia` | `nvidia` (GPU/mqpu) or `qpp-cpu` (CPU debug) |
| `--shots` | `10⁶` | shots for solution read-out |

---

## Running at scale (SLURM)

Edit `RELEASE_DIR` and `CONDA_ENV` at the top of each script, then:

### Single node
```bash
sbatch slurm/single_node.sh        # 1 node, 4 GPUs, 4 ranks
```

### Strong scaling — *fixed problem, more resources*
```bash
sbatch slurm/strong_scaling.sh
```
Holds `size` and `tol` constant (so total work is fixed) and sweeps the rank count
within one allocation (`--ntasks-per-node` = 4, 8, 16, 32). With `--gpu-bind=none`
each rank exposes all 4 GPUs as virtual QPUs; oversubscribing ranks per GPU raises
throughput up to saturation. Compare the `avg/eval` time printed by each step —
falling time = good strong scaling.

### Weak scaling — *grow problem and resources together*
```bash
bash slurm/weak_scaling.sh         # run on a LOGIN node; it submits the jobs
```
Work scales as `O(L²)` in the LCU term count, and a smaller `--tol` keeps more
terms. The script pairs each tolerance with a node count so work-per-GPU stays
roughly flat, and submits one job each. Flat `avg/eval` across jobs = good weak
scaling.

> Rank/GPU layout follows the validated Perlmutter pattern: `--gpu-bind=none` with
> several ranks per node. Tune `--ntasks-per-node` and the tol→node pairing to your
> problem and queue.

---

## Outputs

For a run tagged `<case>_<size>_<var>_tol<tol>`:

- `results/<tag>_result.json` — optimized parameters, final fidelity vs. the
  classical solution, `nfev`, and the full cost history.
- `results/<tag>_results.pdf` — left: VQLS vs. classical solution amplitudes;
  right: the optimizer learning curve.

Fidelity is reported as the better of both bit-orderings (the sampled solution and
the NumPy classical solution can differ by endianness at read-out).

---

## Validation status

- **LCU decomposition** — verified exactly against trace and Frobenius-norm
  identities for both problem families.
- **Dense `U_b`** — verified `U_b|0⟩ = |b⟩` to fidelity 1.0 (CPU simulator).
- **MPS `U_b`** — verified high fidelity on small cases; accuracy depends on `b`
  and the bond/layer budget. Treat as experimental and check the printed encoding
  fidelity for your problem before trusting results.
- **End-to-end solve** — the full pipeline (dataset → `U_b` → distributed cost →
  optimization → read-out) converges:
  - CPU simulator (`qpp-cpu`), 2-qubit tridiagonal: **fidelity 0.9998**.
  - GPU (`nvidia` mqpu, A100) via `srun`, single rank: **0.18 s / cost-eval**.
  - GPU, **4 MPI ranks** (distributed striding + `allreduce`), 3-qubit
    tridiagonal: **fidelity 0.9973** over 400 iterations.

---

## Notes & limitations

- The `dense` backend is capped at **6 qubits on the `nvidia` backend** (cuStateVec
  limits a custom matrix op to `dim ≤ 64`); for a non-basis-state `b` beyond that,
  use `--backend mps`. A basis-state `b` (e.g. tridiagonal) bypasses this entirely
  via the automatic X-gate fast path and runs at any size.
- The ansatz is hardware-efficient (RY-RZ-RY + CZ chain); expressivity may limit
  fidelity on harder systems — increase `--layers`.
- Pauli integer encoding inside the kernels is `X=1, Y=2, Z=3, I=4`.
- The Hadamard test measures `⟨Z⟩` on the ancilla, allocated as qubit 0; the
  Hamiltonian is therefore `spin.z(0)`.
