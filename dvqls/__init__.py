"""DVQLS: a Variational Quantum Linear Solver built on NVIDIA CUDA-Q.

Public modules
--------------
problems     : build the linear systems (tridiagonal Toeplitz, 2D Hele-Shaw).
lcu          : decompose a matrix into a Linear Combination of Unitaries (Pauli strings).
kernels      : the CUDA-Q Hadamard-test kernel, controlled-Pauli, and the ansatz.
state_prep   : encode the right-hand-side |b> (dense-unitary or MPS backend).
utils        : fidelity and the optimizer time-limit callback.
"""

__version__ = "1.0.0"
