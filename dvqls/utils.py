"""Small helpers shared across the solver: state fidelity and a wall-clock
time limit for the classical optimizer."""

import time
import warnings

import numpy as np


def fidelity(q_solution, c_solution):
    """State fidelity |<q|c>|^2 between a quantum and classical solution vector.

    Both inputs are normalized internally, so the magnitudes need not match.
    """
    q = np.asarray(q_solution)
    c = np.asarray(c_solution)
    q = q / np.linalg.norm(q)
    c = c / np.linalg.norm(c)
    return float(np.real(np.abs(q.conj().dot(c)) ** 2))


class TookTooLong(Warning):
    """Raised (as a warning) when the optimizer exceeds its wall-clock budget."""


class MinimizeStopper:
    """SciPy ``minimize`` callback that aborts once ``max_sec`` has elapsed.

    Pass ``stopper.callback`` to ``scipy.optimize.minimize(..., callback=...)``.
    It raises ``StopIteration`` so a long run ends cleanly with the best point
    found so far rather than being killed by the scheduler.
    """

    def __init__(self, max_sec=3600):
        self.max_sec = max_sec
        self.start_time = time.time()

    def callback(self, *args, **kwargs):
        if time.time() - self.start_time > self.max_sec:
            warnings.warn("Optimization exceeded time limit.", TookTooLong)
            raise StopIteration
