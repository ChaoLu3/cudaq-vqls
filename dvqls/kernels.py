"""CUDA-Q kernels for DVQLS.

The cost function is evaluated with a Hadamard test: for each ordered pair of
LCU Pauli terms (l, l') we estimate <0| U_b^dag A_l'^dag (Z_j) A_l U_b |0>-type
overlaps via the ancilla expectation value <Z_ancilla>. The imaginary part is
obtained by an extra Rz(-pi/2) on the ancilla.

The right-hand-side preparation ``U_b`` is *not* defined here. It is injected at
runtime as a module global by ``dvqls.state_prep`` (dense or MPS backend) because
its exact form depends on the qubit count and on ``b``. ``hadamard_test`` and the
adjoint below resolve ``U_b`` lazily, at trace time, so the global must be set
before the first call.

Pauli integer encoding (matches ``controlled_pauli``): X=1, Y=2, Z=3, I=4.
"""

import cudaq
import numpy as np

# Right-hand-side state-prep unitary, injected by dvqls.state_prep.set_u_b(...).
U_b = None


def num_params(num_qubits, num_layers):
    """Parameter count for the hardware-efficient ansatz: 3 rotations/qubit/layer."""
    return 3 * num_qubits * num_layers


@cudaq.kernel
def controlled_pauli(paulis: list[int], a: cudaq.qubit, qreg: cudaq.qvector):
    """Apply the Pauli string ``paulis`` to ``qreg`` controlled on ancilla ``a``
    (this is the controlled-A_l of one LCU term)."""
    for idx, pauli_id in enumerate(paulis):
        if pauli_id == 1:
            x.ctrl(a, qreg[idx])
        elif pauli_id == 2:
            y.ctrl(a, qreg[idx])
        elif pauli_id == 3:
            z.ctrl(a, qreg[idx])


@cudaq.kernel
def ansatz(nq: int, weights: list[float], q: cudaq.qvector):
    """Hardware-efficient ansatz: per layer, RY-RZ-RY on every qubit followed by
    a linear CZ entangling chain. Uses ``3 * nq`` parameters per layer."""
    num_param_qubit = 3
    num_param_layer = num_param_qubit * nq
    layers = len(weights) // num_param_layer
    for layer in range(layers):
        for idx in range(nq):
            base = layer * num_param_layer + idx * num_param_qubit
            ry(weights[base], q[idx])
            rz(weights[base + 1], q[idx])
            ry(weights[base + 2], q[idx])
        for i in range(nq - 1):
            cz(q[i], q[i + 1])


@cudaq.kernel
def hadamard_test(nq: int, part: bool, j: int, weights: list[float],
                  paulis: list[int], paulis2: list[int]):
    """One Hadamard-test circuit for the DVQLS cost.

    ``part``  : False -> real part, True -> imaginary part (adds Rz(-pi/2)).
    ``j``     : qubit index for the controlled-Z; ``j == -1`` skips it, giving
                the <psi|psi> normalization term instead of the numerator term.
    ``paulis``/``paulis2`` : the two LCU Pauli strings A_l and A_l'.

    Measuring <Z> on the ancilla yields the desired overlap.
    """
    ancilla = cudaq.qubit()
    q = cudaq.qvector(nq)

    h(ancilla)
    if part:
        rz(-np.pi / 2, ancilla)

    ansatz(nq, weights, q)
    controlled_pauli(paulis, ancilla, q)
    cudaq.adjoint(U_b, nq, q)
    if j != -1:
        z.ctrl(ancilla, q[j])
    U_b(nq, q)
    cudaq.adjoint(controlled_pauli, paulis2, ancilla, q)
    h(ancilla)


@cudaq.kernel
def prepare_solution(weights: list[float], nq: int):
    """Prepare the candidate solution |x> = ansatz(weights)|0> for read-out."""
    q = cudaq.qvector(nq)
    ansatz(nq, weights, q)
