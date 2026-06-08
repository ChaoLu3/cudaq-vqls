"""Linear systems A x = b that the solver targets.

Two problem families are provided:

* ``tridiagonal``  : a symmetric tridiagonal Toeplitz matrix (diag 1, off-diag -1/3),
  a standard, well-conditioned DVQLS benchmark. ``b = e_0`` (first basis state).
* ``hele-shaw``    : the finite-difference operator of 2D Hele-Shaw (Stokes) flow,
  for either the ``pressure`` or ``velocity`` field. The raw operator is padded to
  a power-of-two size and symmetrized to a Hermitian system (required by DVQLS).
"""

import math

import numpy as np


def dataset_filename(case, size, var="velocity", tol=0.01):
    """Canonical LCU dataset filename, shared by the generator and the solver.

    ``size`` is the qubit count for ``tridiagonal`` and the grid dimension n
    (n x n) for ``hele-shaw``.
    """
    if case == "tridiagonal":
        return f"tridiagonal_{size}q_tol{tol}.json"
    if case == "hele-shaw":
        return f"hele-shaw_{size}x{size}_{var}_tol{tol}.json"
    raise ValueError(f"Unknown case {case!r}; use 'tridiagonal' or 'hele-shaw'.")


# ----------------------------------------------------------------------------
# Tridiagonal Toeplitz
# ----------------------------------------------------------------------------
def tridiagonal_toeplitz(n, a=1.0, b=-1.0 / 3.0, c=-1.0 / 3.0):
    """Dense n x n symmetric tridiagonal Toeplitz matrix (main diag ``a``,
    super-diagonal ``b``, sub-diagonal ``c``)."""
    A = a * np.eye(n)
    if n > 1:
        A += b * np.eye(n, k=1) + c * np.eye(n, k=-1)
    return A


def tridiagonal_system(num_qubits):
    """Return (A, b) for an ``num_qubits``-qubit tridiagonal benchmark.

    ``A`` is 2**num_qubits square; ``b`` is the |0...0> basis state (e_0).
    """
    n = 2 ** num_qubits
    A = tridiagonal_toeplitz(n)
    b = np.zeros(n)
    b[0] = 1.0
    return A, b


# ----------------------------------------------------------------------------
# 2D Hele-Shaw flow (finite-difference operators)
# ----------------------------------------------------------------------------
def _ij2idx(row_i, col_j, ncols):
    return row_i * ncols + col_j


def hele_shaw_analytic(x, y, p_in, p_out, length, depth, mu):
    """Analytical Hele-Shaw pressure/velocity profiles (for the RHS forcing)."""
    P = p_in + (p_out - p_in) * x / length
    U = -(0.5 / mu) * ((p_out - p_in) * y * (depth - y) / length)
    return P, U


def laplacian_pressure(P, p_left, p_right, dx, dy):
    """2nd-order finite-difference Laplacian for the pressure field with
    Dirichlet inlet/outlet boundaries. Returns (operator, rhs)."""
    ny, nx = P.shape
    LP = np.zeros((ny * nx, ny * nx))
    b = np.zeros(ny * nx)
    for i in range(ny):
        for j in range(nx):
            ii = _ij2idx(i, j, nx)
            if j == 0:
                LP[ii, _ij2idx(i, j, nx)] += 1
                b[ii] = p_left
            elif j == nx - 1:
                LP[ii, _ij2idx(i, j, nx)] += 1
                b[ii] = p_right
            elif i == 0:
                LP[ii, _ij2idx(i, j, nx)] += -2 / dx ** 2
                LP[ii, _ij2idx(i, j - 1, nx)] += 1 / dx ** 2
                LP[ii, _ij2idx(i, j + 1, nx)] += 1 / dx ** 2
                LP[ii, _ij2idx(i, j, nx)] += 1 / dy ** 2
                LP[ii, _ij2idx(i + 1, j, nx)] += -2 / dy ** 2
                LP[ii, _ij2idx(i + 2, j, nx)] += 1 / dy ** 2
            elif i == ny - 1:
                LP[ii, _ij2idx(i, j, nx)] += -2 / dx ** 2
                LP[ii, _ij2idx(i, j - 1, nx)] += 1 / dx ** 2
                LP[ii, _ij2idx(i, j + 1, nx)] += 1 / dx ** 2
                LP[ii, _ij2idx(i, j, nx)] += 1 / dy ** 2
                LP[ii, _ij2idx(i - 1, j, nx)] += -2 / dy ** 2
                LP[ii, _ij2idx(i - 2, j, nx)] += 1 / dy ** 2
            else:
                LP[ii, _ij2idx(i, j, nx)] += -2 / dx ** 2
                LP[ii, _ij2idx(i, j - 1, nx)] += 1 / dx ** 2
                LP[ii, _ij2idx(i, j + 1, nx)] += 1 / dx ** 2
                LP[ii, _ij2idx(i, j, nx)] += -2 / dy ** 2
                LP[ii, _ij2idx(i - 1, j, nx)] += 1 / dy ** 2
                LP[ii, _ij2idx(i + 1, j, nx)] += 1 / dy ** 2
    return LP, b


def laplacian_velocity_x(U, u_top, u_bottom, P, dx, dy):
    """2nd-order finite-difference momentum operator for the x-velocity with the
    pressure gradient as forcing. Returns (operator, rhs)."""
    ny, nx = U.shape
    LP = np.zeros((ny * nx, ny * nx))
    b = np.zeros(ny * nx)
    for i in range(ny):
        for j in range(nx):
            ii = _ij2idx(i, j, nx)
            if i == 0:
                LP[ii, _ij2idx(i, j, nx)] += 1
                b[ii] = u_bottom
            elif i == ny - 1:
                LP[ii, _ij2idx(i, j, nx)] += 1
                b[ii] = u_top
            elif j == 0:
                LP[ii, _ij2idx(i, j, nx)] += 1 / dx ** 2
                LP[ii, _ij2idx(i, j + 1, nx)] += -2 / dx ** 2
                LP[ii, _ij2idx(i, j + 2, nx)] += 1 / dx ** 2
                LP[ii, _ij2idx(i, j, nx)] += -2 / dy ** 2
                LP[ii, _ij2idx(i - 1, j, nx)] += 1 / dy ** 2
                LP[ii, _ij2idx(i + 1, j, nx)] += 1 / dy ** 2
                b[ii] = (P[i, j + 1] - P[i, j]) / dx
            elif j == nx - 1:
                LP[ii, _ij2idx(i, j, nx)] += 1 / dx ** 2
                LP[ii, _ij2idx(i, j - 1, nx)] += -2 / dx ** 2
                LP[ii, _ij2idx(i, j - 2, nx)] += 1 / dx ** 2
                LP[ii, _ij2idx(i, j, nx)] += -2 / dy ** 2
                LP[ii, _ij2idx(i - 1, j, nx)] += 1 / dy ** 2
                LP[ii, _ij2idx(i + 1, j, nx)] += 1 / dy ** 2
                b[ii] = (P[i, j] - P[i, j - 1]) / dx
            else:
                LP[ii, _ij2idx(i, j, nx)] += -2 / dx ** 2
                LP[ii, _ij2idx(i, j - 1, nx)] += 1 / dx ** 2
                LP[ii, _ij2idx(i, j + 1, nx)] += 1 / dx ** 2
                LP[ii, _ij2idx(i, j, nx)] += -2 / dy ** 2
                LP[ii, _ij2idx(i - 1, j, nx)] += 1 / dy ** 2
                LP[ii, _ij2idx(i + 1, j, nx)] += 1 / dy ** 2
                b[ii] = 0.5 * (P[i, j + 1] - P[i, j - 1]) / dx
    return LP, b


def _pad_to_pow2(A, b):
    """Pad (A, b) with an identity block so the dimension is a power of two
    (a hard requirement for a qubit register)."""
    n = A.shape[0]
    target = 1 if n == 0 else 2 ** math.ceil(math.log2(n))
    if target == n:
        return A, b
    pad = target - n
    A = np.pad(A, ((0, pad), (0, pad)), mode="constant")
    b = np.pad(b, (0, pad), mode="constant")
    np.fill_diagonal(A[-pad:, -pad:], 1.0)  # keep the padded block non-singular
    return A, b


def _make_hermitian(A, b):
    """Return a Hermitian system. If A is already Hermitian it is returned with
    a small RHS offset (keeps the solution state non-trivial); otherwise the
    standard [[0, A^H], [A, 0]] dilation is used."""
    n = A.shape[0]
    if np.allclose(A, A.conj().T):
        return A, b + 5e-2 * np.linalg.norm(b)
    A_h = np.block([[np.zeros((n, n)), A.conj().T], [A, np.zeros((n, n))]])
    b_h = np.concatenate([b, np.zeros(n)]) + 5e-2 * np.linalg.norm(b)
    return A_h, b_h


def hele_shaw_system(nx, ny, var="velocity"):
    """Return (A, b) for the 2D Hele-Shaw problem on an ``nx`` x ``ny`` grid.

    ``var`` selects the field: ``"pressure"`` or ``"velocity"``. The operator is
    padded to a power-of-two size and symmetrized so DVQLS can solve it.
    Requires ``ny >= 3`` for the 2nd-order stencil in y.
    """
    if ny < 3 and nx > 2:
        raise ValueError("Hele-Shaw needs ny >= 3 for the 2nd-order y-stencil.")

    p_in, p_out = 200.0, 0.0
    u_top, u_bottom = 0.0, 0.0
    length, depth, mu = 1.0, 1.0, 1.0

    x = np.linspace(0, length, nx)
    y = np.linspace(0, depth, ny)
    dx, dy = x[1] - x[0], y[1] - y[0]
    xx, yy = np.meshgrid(x, y)
    P_analytic, _ = hele_shaw_analytic(xx, yy, p_in, p_out, length, depth, mu)

    if var == "pressure":
        P = np.zeros((ny, nx))
        P[:, 0], P[:, -1] = p_in, p_out
        A, b = laplacian_pressure(P, p_in, p_out, dx, dy)
    elif var == "velocity":
        U = np.zeros((ny, nx))
        U[0, :], U[-1, :] = u_bottom, u_top
        A, b = laplacian_velocity_x(U, u_top, u_bottom, P_analytic, dx, dy)
    else:
        raise ValueError("var must be 'pressure' or 'velocity'.")

    A, b = _pad_to_pow2(A, b)
    A, b = _make_hermitian(A, b)
    return A, b
