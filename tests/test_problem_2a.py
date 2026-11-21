"""Testovi za Zadatak 2a: sile pomoću NumPy-ja (forces_numpy.py).

Pokrenite ih sa:  pytest tests/test_problem_2a.py
Dok zadatak nije urađen, ovi testovi padaju (FAILED). Kad sve prođe (passed),
zadatak je gotov.
"""

import warnings

import pytest
from force_checks import ALL_CHECKS, check_agrees_with_python, make_test_config

from mdsim import forces_numpy


def test_agrees_with_forces_python():
    check_agrees_with_python(forces_numpy)


@pytest.mark.parametrize("check", ALL_CHECKS, ids=lambda c: c.__name__[len("check_"):])
def test_forces_numpy(check):
    check(forces_numpy)


def test_gives_no_warnings():
    # A warning here usually means a division by zero for pairs that don't interact.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        forces_numpy.compute_forces(*make_test_config())
