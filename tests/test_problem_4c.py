"""Testovi za Zadatak 4c: sile pomoću Numba-e (forces_numba.py).

Pokrenite ih sa:  pytest tests/test_problem_4c.py
Dok zadatak nije urađen, ovi testovi padaju (FAILED). Kad sve prođe (passed),
zadatak je gotov. Prvo pokretanje traje duže, jer Numba prevodi kod.
"""

import numpy as np
import pytest
from force_checks import ALL_CHECKS, check_agrees_with_python, make_test_config

from mdsim import forces_numba, forces_python


def test_kernel_is_compiled_with_njit():
    # @njit turns the function into a Numba "dispatcher", which keeps the
    # original Python function in .py_func.
    assert hasattr(forces_numba._forces_kernel, "py_func")


def test_agrees_with_forces_python():
    check_agrees_with_python(forces_numba)


@pytest.mark.parametrize("check", ALL_CHECKS, ids=lambda c: c.__name__[len("check_"):])
def test_forces_numba(check):
    check(forces_numba)


def test_accepts_non_contiguous_input():
    pos, box, s, e = make_test_config()
    wide = np.zeros((len(pos), 3))
    wide[:, :2] = pos
    f, pe = forces_numba.compute_forces(wide[:, :2], box, s, e)  # a non-contiguous view
    f_ref, pe_ref = forces_python.compute_forces(pos, box, s, e)
    assert np.allclose(f, f_ref, rtol=1e-10, atol=1e-12)
    assert pe == pytest.approx(pe_ref, rel=1e-10)
