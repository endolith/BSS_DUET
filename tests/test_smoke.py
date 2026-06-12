"""Smoke tests — expand when refactoring bss.py into a package."""

import numpy as np


def test_find_peaks_import():
    import find_peaks  # noqa: F401


def test_bss_epsilon_constant():
    import bss

    assert bss.EPSILON > 0


def test_find_peak_indices_runs():
    from find_peaks import find_peak_indices

    grid = np.zeros((5, 5))
    grid[2, 2] = 1.0
    peak_indices = find_peak_indices(grid, n_peaks=1)
    assert peak_indices is not None
