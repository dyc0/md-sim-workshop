"""Tests for the pure Python force version, the one students are given.

The checks themselves are in force_checks.py; the Problem 4 tests run the
same checks on the faster versions."""

import pytest
from force_checks import ALL_CHECKS

from mdsim import forces_python


@pytest.mark.parametrize("check", ALL_CHECKS, ids=lambda c: c.__name__[len("check_"):])
def test_forces_python(check):
    check(forces_python)
