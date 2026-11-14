"""Checks that every force version must pass, shared by test_forces.py and the
Problem 4 tests. Each check takes the force module to test."""

import numpy as np
import pytest


def make_test_config(n_side=6, dim=2, seed=1):
    """Two-species test configuration: a slightly shaken square/cubic lattice.

    Spacing 1.25, sigma alternating 1.0 / 1.2, epsilon alternating 1.0 / 0.8.
    Returns (positions, box, sigmas, epsilons).
    """
    rng = np.random.default_rng(seed)
    spacing = 1.25
    grid = np.indices([n_side] * dim).reshape(dim, -1).T.astype(float)
    box = np.full(dim, spacing * n_side)
    positions = (grid + 0.5) * spacing + rng.normal(0.0, 0.05, size=grid.shape)
    positions %= box
    n = len(positions)
    sigmas = np.where(np.arange(n) % 2 == 0, 1.0, 1.2)
    epsilons = np.where(np.arange(n) % 2 == 0, 1.0, 0.8)
    return positions, box, sigmas, epsilons


def potential_energy(module, positions, box, sigmas, epsilons):
    return module.compute_forces(positions, box, sigmas, epsilons)[1]


def check_two_particle_energy_and_force_match_formula(module):
    r = 1.5
    pos = np.array([[2.0, 5.0], [2.0 + r, 5.0]])
    box = np.array([10.0, 10.0])
    f, pe = module.compute_forces(pos, box, np.ones(2), np.ones(2))
    assert pe == pytest.approx(4 * (r**-12 - r**-6) - module.SHIFT, rel=1e-12)
    assert f[0, 0] == pytest.approx(24 / r**2 * (2 * r**-12 - r**-6) * (-r), rel=1e-12)
    assert f[0, 1] == 0.0
    assert np.allclose(f[1], -f[0])


def check_shift_constant(module):
    assert module.SHIFT == pytest.approx(4 * (0.4**12 - 0.4**6))


def check_returns_array_and_float(module):
    f, pe = module.compute_forces(*make_test_config())
    assert isinstance(f, np.ndarray) and f.shape == (36, 2) and f.dtype == np.float64
    assert isinstance(pe, float)


def check_newtons_third_law(module):
    f, _ = module.compute_forces(*make_test_config())
    assert np.allclose(f.sum(axis=0), 0.0, atol=1e-10)


def check_force_is_minus_gradient_of_energy(module):
    positions, box, sigmas, epsilons = make_test_config()
    f, _ = module.compute_forces(positions, box, sigmas, epsilons)
    h = 1e-6
    for i, k in [(0, 0), (7, 1), (20, 0), (35, 1)]:
        plus = positions.copy()
        minus = positions.copy()
        plus[i, k] += h
        minus[i, k] -= h
        numeric = (potential_energy(module, plus, box, sigmas, epsilons)
                   - potential_energy(module, minus, box, sigmas, epsilons)) / (2 * h)
        assert numeric == pytest.approx(-f[i, k], rel=1e-5, abs=1e-6)


def check_cutoff(module):
    sigmas = np.array([1.0, 1.4])  # sigma_ij = 1.2, cutoff 3.0
    box = np.array([20.0, 20.0])
    beyond = np.array([[5.0, 5.0], [5.0 + 3.0 + 1e-9, 5.0]])
    f, pe = module.compute_forces(beyond, box, sigmas, np.ones(2))
    assert np.all(f == 0.0)
    assert pe == 0.0
    inside = np.array([[5.0, 5.0], [5.0 + 3.0 - 1e-9, 5.0]])
    f, pe = module.compute_forces(inside, box, sigmas, np.ones(2))
    assert abs(pe) < 1e-6
    assert f[0, 0] != 0.0


def check_mixing_rule_for_epsilon(module):
    pos = np.array([[2.0, 5.0], [3.5, 5.0]])
    box = np.array([10.0, 10.0])
    _, pe_one = module.compute_forces(pos, box, np.ones(2), np.ones(2))
    _, pe_mixed = module.compute_forces(pos, box, np.ones(2), np.array([1.0, 4.0]))
    assert pe_mixed == pytest.approx(2.0 * pe_one)  # sqrt(1 * 4) = 2


def check_minimum_image_across_boundary(module):
    pos = np.array([[0.5, 5.0], [9.5, 5.0]])
    box = np.array([10.0, 10.0])
    f, _ = module.compute_forces(pos, box, np.ones(2), np.ones(2))
    assert f[0, 0] == pytest.approx(24.0)  # r = 1: pushed apart across the edge
    assert f[1, 0] == pytest.approx(-24.0)


ALL_CHECKS = [
    check_two_particle_energy_and_force_match_formula,
    check_shift_constant,
    check_returns_array_and_float,
    check_newtons_third_law,
    check_force_is_minus_gradient_of_energy,
    check_cutoff,
    check_mixing_rule_for_epsilon,
    check_minimum_image_across_boundary,
]


def check_agrees_with_python(module):
    from mdsim import forces_python
    for cfg in (make_test_config(), make_test_config(n_side=5, dim=3)):
        f_ref, pe_ref = forces_python.compute_forces(*cfg)
        f, pe = module.compute_forces(*cfg)
        assert f.shape == f_ref.shape
        assert np.allclose(f, f_ref, rtol=1e-10, atol=1e-12)
        assert pe == pytest.approx(pe_ref, rel=1e-10)
