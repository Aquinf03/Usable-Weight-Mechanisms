"""Unit tests for coverage and triggers (no GPU)."""

from __future__ import annotations

import numpy as np

from atlas.mount.coverage import sparse_write_coverage, tile_svd_weight_coverage
from atlas.mount.strategies import tile_svd_mounts
from atlas.mount.trigger import mode_trigger_coeffs


def test_weight_coverage_increases_with_modes():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((32, 64))
    c1 = tile_svd_weight_coverage(w, tile_size=16, modes_per_tile=1)
    c2 = tile_svd_weight_coverage(w, tile_size=16, modes_per_tile=4)
    assert 0 < c1["weight_energy_fraction"] <= 1
    assert c2["weight_energy_fraction"] >= c1["weight_energy_fraction"]


def test_sparse_write_coverage_perfect_basis():
    d0 = np.array([1.0, 0.0, 0.0])
    d1 = np.array([0.0, 1.0, 0.0])
    dirs = np.stack([d0, d1, np.array([0.0, 0.0, 1.0])])
    writes = np.array([[2.0, 3.0, 0.0], [1.0, -1.0, 0.0]])
    cov = sparse_write_coverage(writes, dirs, k_active=2)
    assert cov["write_energy_fraction"] > 0.99


def test_trigger_coeffs_match_right_sv():
    rng = np.random.default_rng(1)
    w = rng.standard_normal((16, 40))
    mounts = tile_svd_mounts(w, tile_size=20, modes_per_tile=1)
    assert "trigger_vector" in mounts[0].meta
    v = mounts[0].meta["trigger_vector"]
    start = mounts[0].meta["col_start"]
    end = mounts[0].meta["col_end"]
    inter = np.zeros((1, 40))
    inter[0, start:end] = v
    coeffs = mode_trigger_coeffs(inter, mounts)
    assert coeffs[0, 0] > 0.9
