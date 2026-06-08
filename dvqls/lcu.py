"""Decompose a matrix A into a Linear Combination of Unitaries (LCU):

    A = sum_k coeffs[k] * P_k,

where each ``P_k`` is a tensor product of Pauli operators (a "Pauli string"
like ``"IXYZ"``). DVQLS needs A in this form so the controlled-A in the
Hadamard test reduces to controlled Paulis.

The decomposition uses a Fast Walsh-Hadamard Transform (FWHT) over the diagonals
of A, which is far cheaper than the naive O(4^n) projection onto every Pauli
basis element, and parallelizes across CPU cores via shared memory. Terms with
``|coeff| <= tol`` are dropped; a larger ``tol`` yields fewer Pauli terms (and a
cheaper, lower-accuracy cost function).
"""

import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory

import numpy as np


def _fwht_inplace(vec):
    """In-place Fast Walsh-Hadamard Transform of a 1D array (length a power of 2)."""
    n = vec.shape[0]
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(h):
                u = vec[i + j]
                v = vec[i + h + j]
                vec[i + j] = u + v
                vec[i + h + j] = u - v
        h <<= 1
    return vec


def _bits_to_pauli_string(a, b, n_qubits):
    """Map the (a, b) bit pair to a Pauli string (00->I, 10->X, 01->Z, 11->Y)."""
    s = []
    for i in range(n_qubits):
        ai = (a >> i) & 1
        bi = (b >> i) & 1
        if ai == 0 and bi == 0:
            s.append("I")
        elif ai == 1 and bi == 0:
            s.append("X")
        elif ai == 0 and bi == 1:
            s.append("Z")
        else:
            s.append("Y")
    return "".join(reversed(s))


def _process_a_range_shared(shm_name, shape, dtype, a_start, a_end, tol):
    """Worker: decompose the diagonals offset by [a_start, a_end) of the matrix
    held in shared memory."""
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        matrix = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        N = matrix.shape[0]
        n_qubits = int(math.log2(N))
        idx = np.arange(N, dtype=np.int64)
        coeffs_local, strings_local = [], []
        for a in range(a_start, a_end):
            f = matrix[idx, idx ^ a].astype(np.complex128).copy()
            _fwht_inplace(f)
            for b in range(N):
                # s_count = number of Y's in this Pauli (positions where a&b both set).
                # The (+s) phase gives the exact coefficient of A itself; using (-s)
                # would instead yield the coefficient of A^T (it differs only on terms
                # with an odd number of Y's, which vanish for symmetric A). Coefficients
                # are complex in general -- real only when A is Hermitian.
                s_count = bin(a & b).count("1")
                alpha = (1j) ** (s_count) * f[b] / N
                if abs(alpha) > tol:
                    coeffs_local.append(alpha.item())
                    strings_local.append(_bits_to_pauli_string(a, b, n_qubits))
    finally:
        shm.close()
    return coeffs_local, strings_local


def lcu_decompose(matrix, n_procs=None, tol=1e-2, chunk_size=None):
    """Decompose ``matrix`` into (coeffs, pauli_strings) via FWHT.

    Parameters
    ----------
    matrix : (N, N) ndarray, N a power of 2.
    n_procs : worker processes (defaults to all CPUs).
    tol : drop Pauli terms with ``|coeff| <= tol``.
    chunk_size : diagonals per task (defaults to a balanced split).

    Returns
    -------
    (coeffs, strings) : list[complex], list[str].
    """
    matrix = np.asarray(matrix)
    N = matrix.shape[0]
    if n_procs is None:
        n_procs = os.cpu_count()
    if chunk_size is None:
        chunk_size = max(1, N // (4 * n_procs))

    shm = shared_memory.SharedMemory(create=True, size=matrix.nbytes)
    try:
        shm_matrix = np.ndarray(matrix.shape, dtype=matrix.dtype, buffer=shm.buf)
        shm_matrix[:] = matrix[:]
        ranges = [(i, min(i + chunk_size, N)) for i in range(0, N, chunk_size)]
        coeffs, strings = [], []
        with ProcessPoolExecutor(max_workers=n_procs) as ex:
            futures = [
                ex.submit(_process_a_range_shared, shm.name, matrix.shape,
                          matrix.dtype, lo, hi, tol)
                for lo, hi in ranges
            ]
            for fut in as_completed(futures):
                c_local, s_local = fut.result()
                coeffs.extend(c_local)
                strings.extend(s_local)
        return coeffs, strings
    finally:
        shm.close()
        shm.unlink()


def verify_lcu(matrix, coeffs, strings, tol=1e-3, verbose=True):
    """Sanity-check a decomposition via trace and Frobenius norm identities.

    Uses Pauli orthogonality: ``||A||_F^2 = N * sum_k |coeff_k|^2`` and
    ``trace(A) = N * sum(coeffs of the all-identity term)``.
    Returns True if both match within ``tol``.
    """
    matrix = np.asarray(matrix)
    N = matrix.shape[0]

    trace_recon = sum(c for c, s in zip(coeffs, strings) if set(s) == {"I"})
    trace_ok = np.isclose(np.trace(matrix), N * trace_recon, atol=tol)

    norm_recon = N * sum(np.abs(c) ** 2 for c in coeffs)
    norm_orig = np.linalg.norm(matrix, "fro") ** 2
    norm_ok = np.isclose(norm_orig, norm_recon, atol=tol)

    if verbose:
        rel = (abs(norm_orig) - abs(norm_recon)) / abs(norm_orig) if norm_orig else 0.0
        print(f"  [verify] trace match: {bool(trace_ok)}  "
              f"||A||_F^2 orig={norm_orig:.6g} recon={norm_recon:.6g} rel_diff={rel:.2e}")
    return bool(trace_ok and norm_ok)
