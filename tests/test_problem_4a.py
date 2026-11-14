"""Testovi za Zadatak 4a: sile pomoću trećeg Njutnovog zakona (forces_newton3.py).

Pokrenite ih sa:  pytest tests/test_problem_4a.py
Dok zadatak nije urađen, ovi testovi padaju (FAILED). Kad sve prođe (passed),
zadatak je gotov.
"""

import pytest
from force_checks import ALL_CHECKS, check_agrees_with_python

from mdsim import forces_newton3


def test_agrees_with_forces_python():
    check_agrees_with_python(forces_newton3)


@pytest.mark.parametrize("check", ALL_CHECKS, ids=lambda c: c.__name__[len("check_"):])
def test_forces_newton3(check):
    check(forces_newton3)
