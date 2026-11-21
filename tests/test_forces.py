"""Tests for the pure Python force versions, the ones students are given.

The checks themselves are in force_checks.py; the Problem 2a and 2b tests run the
same checks on the faster versions."""

import pytest
from force_checks import ALL_CHECKS, check_agrees_with_python

from mdsim import forces_newton3, forces_python


@pytest.mark.parametrize("check", ALL_CHECKS, ids=lambda c: c.__name__[len("check_"):])
def test_forces_python(check):
    check(forces_python)


@pytest.mark.parametrize("check", ALL_CHECKS, ids=lambda c: c.__name__[len("check_"):])
def test_forces_newton3(check):
    check(forces_newton3)


def test_newton3_agrees_with_python():
    check_agrees_with_python(forces_newton3)
