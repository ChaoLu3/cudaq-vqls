"""Right-hand-side state preparation: build a unitary U_b with U_b|0> = |b>.

Two backends, both solving the same headache -- a CUDA-Q custom operation has a
*fixed arity* fixed at registration time, so a single hand-written ``U_b`` kernel
cannot cover different qubit counts. Here ``U_b`` is generated automatically to
exactly match ``num_qubits`` and injected into ``vqls.kernels``:

* ``dense`` : pad |b> to a full 2^n x 2^n unitary (|b> as its first column) and
  register it as one big custom op. Exact, but the dense matrix limits this to
  ~12-14 qubits. This is the validated default.

* ``mps``   : approximate |b> by a bond-2 Matrix Product State and compile it into
  a sequence of 1- and 2-qubit gates (Ran et al., arXiv:1908.07958). Every gate
  has fixed small arity *independent of system size*, so it scales past the dense
  ceiling. Marked experimental -- check the printed encoding fidelity per problem.

Generation is MPI-safe: each rank writes its own rank-local module file (from data
it already holds, broadcast for MPS), so there is no shared-file race. The custom
operations are registered in every rank's process-local registry.
"""

import importlib.util
import os

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import null_space

import cudaq


# ----------------------------------------------------------------------------
# Generated-kernel plumbing
# ----------------------------------------------------------------------------
def _write_and_import(src, module_name, gen_dir):
    """Write ``src`` to ``gen_dir/module_name.py`` and import its ``U_b`` kernel.

    Writing a real file (rather than exec-ing a string) is required because
    CUDA-Q inspects kernel source via ``inspect.getsource``.
    """
    os.makedirs(gen_dir, exist_ok=True)
    path = os.path.join(gen_dir, module_name + ".py")
    with open(path, "w") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.U_b


def _inject_u_b(u_b):
    """Make ``u_b`` visible to the Hadamard-test kernel as ``kernels.U_b``."""
    from . import kernels
    kernels.U_b = u_b


# ----------------------------------------------------------------------------
# Basis-state fast path
# ----------------------------------------------------------------------------
# cuStateVec (the nvidia backend) caps a single registered custom-matrix
# operation at dim <= 64, i.e. at most 6 qubits. So the dense backend cannot run
# beyond 6 qubits on GPU.
DENSE_MAX_QUBITS_GPU = 6


def _basis_index(b_vec, atol=1e-9):
    """If b is (proportional to) a single computational basis state |k>, return k;
    otherwise None. The tridiagonal RHS b = e_0 is the k = 0 case."""
    v = np.asarray(b_vec, dtype=np.complex128)
    nz = np.flatnonzero(np.abs(v) > atol)
    return int(nz[0]) if len(nz) == 1 else None


def _build_basis(k, num_qubits, rank, gen_dir):
    """U_b|0...0> = |k> via an X on each set bit of k (none for k = 0 -> identity).
    No custom op, so this scales to any qubit count and runs on any backend.
    CUDA-Q indexes the statevector with qubit 0 as the most-significant bit, so
    qubit q corresponds to bit (num_qubits - 1 - q) of k."""
    flips = [q for q in range(num_qubits) if (k >> (num_qubits - 1 - q)) & 1]
    body = "\n".join(f"    x(qreg[{q}])" for q in flips) or "    pass"
    src = ("import cudaq\n\n\n@cudaq.kernel\n"
           "def U_b(nq: int, qreg: cudaq.qvector):\n" + body + "\n")
    return _write_and_import(src, f"ub_basis_{num_qubits}q_r{rank}", gen_dir)


# ----------------------------------------------------------------------------
# Dense backend
# ----------------------------------------------------------------------------
def state_to_unitary(b_vec):
    """Register the custom op ``Unitary_b_vec``: a unitary whose first column is
    the normalized ``b_vec`` (orthonormal complement from the null space)."""
    v = np.asarray(b_vec, dtype=np.complex128)
    nrm = np.linalg.norm(v)
    if not np.isclose(nrm, 1.0):
        v = v / nrm
    v = v.reshape(-1, 1)
    U = np.hstack((v, null_space(v.conj().T)))
    cudaq.register_operation("Unitary_b_vec", np.ascontiguousarray(U))
    return U


def _build_dense(b_vec, num_qubits, rank, gen_dir):
    state_to_unitary(b_vec)  # registered identically on every rank
    # Reversed qubit order: NumPy's kron convention (|b> built with qubit 0 as the
    # most significant bit) is the opposite of CUDA-Q's register indexing. Passing
    # the qubits high->low makes U_b|0> == |b> exactly (verified to fidelity 1.0).
    args = ", ".join(f"qreg[{i}]" for i in range(num_qubits - 1, -1, -1))
    src = (
        "import cudaq\n\n\n"
        "@cudaq.kernel\n"
        "def U_b(nq: int, qreg: cudaq.qvector):\n"
        f"    Unitary_b_vec({args})\n"
    )
    return _write_and_import(src, f"ub_dense_{num_qubits}q_r{rank}", gen_dir)


# ----------------------------------------------------------------------------
# MPS backend
# ----------------------------------------------------------------------------
def _gram_schmidt(matrix: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Orthonormalize columns; near-zero columns are replaced by random vectors
    so the result is a genuine unitary block."""
    num_rows, num_columns = matrix.shape
    unitary = np.zeros((num_rows, num_columns), dtype=np.complex128)
    basis: list = []
    for j in range(num_columns):
        col = matrix[:, j]
        if np.allclose(col, 0):
            col = np.random.uniform(-1, 1, num_rows) + 1j * np.random.uniform(-1, 1, num_rows)
        for bvec in basis:
            col = col - (bvec.conj().T @ col) * bvec
        if np.linalg.norm(col) < 1e-12:
            col = np.random.uniform(-1, 1, num_rows) + 1j * np.random.uniform(-1, 1, num_rows)
            for bvec in basis:
                col = col - (bvec.conj().T @ col) * bvec
        unitary[:, j] = col / np.linalg.norm(col)
        basis.append(unitary[:, j])
    return unitary


class MPSEncoder:
    """Approximate a statevector by a bond-2 MPS and return the unitary layers
    that build it from |0...0> (Ran's iterative-disentangling construction)."""

    def __init__(self, target_fidelity=0.99):
        self.target_fidelity = target_fidelity

    def _generate_layer(self, mps):
        num_sites = mps.L
        layer = []
        for i, tensor in enumerate(reversed(mps.arrays)):
            i = num_sites - i - 1
            if i == 0:
                d_right, d = tensor.shape
                tensor = tensor.reshape((1, d_right, d))
            if i == num_sites - 1:
                d_left, d = tensor.shape
                tensor = tensor.reshape((d_left, 1, d))
            tensor = np.swapaxes(tensor, 1, 2)
            d_left, d, d_right = tensor.shape
            isometry = tensor.reshape((d * d_left, d_right))
            qubits = reversed(range(i - int(np.ceil(np.log2(d_left))), i + 1))
            qubits = [abs(q - num_sites + 1) for q in qubits]
            matrix = np.zeros((isometry.shape[0], isometry.shape[0]), dtype=isometry.dtype)
            matrix[:, : isometry.shape[1]] = isometry
            layer.append((qubits, _gram_schmidt(matrix)))
        return layer

    def encode(self, statevector, max_num_layers=5, chi_max=512):
        import quimb.tensor as qtn

        mps = qtn.MatrixProductState.from_dense(statevector)
        mps = qtn.tensor_1d_compress.tensor_network_1d_compress(mps, max_bond=chi_max)
        mps.permute_arrays()
        mps.compress(form="left", max_bond=chi_max)
        mps.left_canonicalize(normalize=True)

        disentangled = mps.copy(deep=True)
        unitary_layers = []
        zero = np.zeros((2 ** mps.L,), dtype=np.complex128)
        zero[0] = 1.0

        fidelity = 0.0
        for layer_index in range(max_num_layers):
            compressed = disentangled.copy(deep=True)
            compressed.normalize()
            compressed.compress(form="left", max_bond=2)
            layer = self._generate_layer(compressed)
            unitary_layers.append(layer)
            for i, _ in enumerate(layer):
                inverse = layer[-(i + 1)][1].conj().T
                if inverse.shape[0] == 4:
                    disentangled.gate_split_(inverse, (i - 1, i))
                else:
                    disentangled.gate_(inverse, (i), contract=True)
            disentangled = qtn.tensor_1d_compress.tensor_network_1d_compress(
                disentangled, max_bond=chi_max)
            fidelity = np.abs(np.vdot(disentangled.to_dense().ravel(), zero))
            if fidelity >= self.target_fidelity:
                print(f"  [mps] reached |<0|disentangled>| = {fidelity:.5f} "
                      f"in {layer_index + 1} layer(s).")
                break
        else:
            print(f"  [mps] stopped at |<0|disentangled>| = {fidelity:.5f} "
                  f"after {max_num_layers} layers.")

        unitary_layers.reverse()  # apply forward to |0...0> to build |b>
        return unitary_layers


def _build_mps(b_vec, num_qubits, rank, gen_dir, comm,
               max_num_layers, chi_max, target_fidelity):
    layers = None
    if rank == 0:
        layers = MPSEncoder(target_fidelity).encode(
            np.asarray(b_vec, dtype=np.complex128), max_num_layers, chi_max)
    if comm is not None:
        layers = comm.bcast(layers, root=0)

    body, op_idx = [], 0
    for layer in layers:
        for qubits, unitary in layer:
            name = f"mps_op_{op_idx}"
            cudaq.register_operation(name, np.ascontiguousarray(unitary))
            targets = ", ".join(f"qreg[{q}]" for q in list(qubits)[::-1])
            body.append(f"    {name}({targets})")
            op_idx += 1
    if not body:
        body = ["    pass"]
    src = ("import cudaq\n\n\n"
           "@cudaq.kernel\n"
           "def U_b(nq: int, qreg: cudaq.qvector):\n"
           + "\n".join(body) + "\n")
    return _write_and_import(src, f"ub_mps_{num_qubits}q_r{rank}", gen_dir)


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------
def prepare_b(b_vec, num_qubits, backend="dense", comm=None, gen_dir=None,
              max_num_layers=5, chi_max=512, target_fidelity=0.99):
    """Register the state-prep op(s), generate the matching ``U_b`` kernel, and
    inject it into ``vqls.kernels``. Call once on every rank before optimizing.

    Returns the generated ``U_b`` kernel (also reachable as ``kernels.U_b``).
    """
    rank = comm.Get_rank() if comm is not None else 0
    if gen_dir is None:
        gen_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_generated")

    # Fast path: if b is a computational basis state (e.g. tridiagonal's b = e_0),
    # U_b is just X gates -- no custom op, so it works at any size on any backend.
    k = _basis_index(b_vec)
    if k is not None:
        if rank == 0:
            print(f"  [state_prep] b = |{k}> (basis state) -> trivial X-gate U_b")
        u_b = _build_basis(k, num_qubits, rank, gen_dir)
    elif backend == "dense":
        # cuStateVec caps custom matrix ops at 6 qubits; fail clearly, not cryptically.
        if "nvidia" in cudaq.get_target().name and num_qubits > DENSE_MAX_QUBITS_GPU:
            raise ValueError(
                f"--backend dense builds a {num_qubits}-qubit custom operation, but the "
                f"nvidia (cuStateVec) backend caps custom matrix ops at "
                f"{DENSE_MAX_QUBITS_GPU} qubits (dim<=64). For a non-basis-state b at "
                f">{DENSE_MAX_QUBITS_GPU} qubits, use --backend mps.")
        u_b = _build_dense(b_vec, num_qubits, rank, gen_dir)
    elif backend == "mps":
        u_b = _build_mps(b_vec, num_qubits, rank, gen_dir, comm,
                         max_num_layers, chi_max, target_fidelity)
    else:
        raise ValueError(f"Unknown backend {backend!r}; use 'dense' or 'mps'.")

    _inject_u_b(u_b)
    return u_b
