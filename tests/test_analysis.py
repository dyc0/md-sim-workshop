"""Tests for mdsim.analysis: the concentration profile."""

import subprocess
import sys
import warnings

import numpy as np

from mdsim.analysis import concentration_profile


# ---------- concentration profile ----------

def test_concentration_profile_hand_built():
    box = np.array([4.0, 1.0])
    pos = np.array([[0.5, 0.5], [0.6, 0.5], [1.5, 0.5], [1.6, 0.5], [2.5, 0.5]])
    species = np.array([0, 0, 0, 1, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        centers, frac = concentration_profile(pos, species, box, n_bins=4)
    assert np.allclose(centers, [0.5, 1.5, 2.5, 3.5])
    assert np.allclose(frac[:3], [1.0, 0.5, 0.0])
    assert np.isnan(frac[3])


def test_concentration_profile_position_equal_to_box_goes_in_last_bin():
    box = np.array([4.0, 1.0])
    pos = np.array([[4.0, 0.5], [0.5, 0.5]])  # x == box can come out of `positions % box`
    species = np.array([0, 1])
    _, frac = concentration_profile(pos, species, box, n_bins=4)
    assert frac[3] == 1.0
    assert frac[0] == 0.0


def test_concentration_profile_along_y():
    box = np.array([1.0, 4.0])
    pos = np.array([[0.5, 0.5], [0.5, 0.6], [0.5, 1.5], [0.5, 1.6], [0.5, 2.5]])
    species = np.array([0, 0, 0, 1, 1])
    centers, frac = concentration_profile(pos, species, box, n_bins=4, axis=1)
    assert np.allclose(centers, [0.5, 1.5, 2.5, 3.5])
    assert np.allclose(frac[:3], [1.0, 0.5, 0.0])


# ---------- unwrapping ----------

# ---------- imports ----------

def test_analysis_does_not_import_matplotlib():
    out = subprocess.run(
        [sys.executable, "-c", "import sys, mdsim.analysis; print('matplotlib' in sys.modules)"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert out == "False"
