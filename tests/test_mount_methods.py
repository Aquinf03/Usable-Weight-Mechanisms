"""Unit tests for mount strategies + unembed lens (no GPU)."""

from __future__ import annotations

import numpy as np

from atlas.mount.lenses import apply_rms_norm, unembed_lens
from atlas.mount.strategies import (
    column_sample_mounts,
    random_baseline_mounts,
    tile_svd_mounts,
    whole_matrix_svd_mounts,
)


def _synthetic_weight(out: int = 64, inp: int = 100, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((out, inp)).astype(np.float64)


def test_whole_matrix_svd_count():
    w = _synthetic_weight()
    mounts = whole_matrix_svd_mounts(w, top_k=5)
    assert len(mounts) == 5
    assert all(abs(np.linalg.norm(m.direction) - 1.0) < 1e-6 for m in mounts)


def test_tile_svd_count():
    w = _synthetic_weight(inp=100)
    mounts = tile_svd_mounts(w, tile_size=25, modes_per_tile=2)
    assert len(mounts) == 8
    assert "trigger_vector" in mounts[0].meta


def test_column_sample_count():
    w = _synthetic_weight(inp=100)
    mounts = column_sample_mounts(w, n=10, seed=1)
    assert len(mounts) == 10


def test_random_baseline_count():
    mounts = random_baseline_mounts(32, n=7, seed=2)
    assert len(mounts) == 7


def test_rms_norm_scales():
    d = np.ones(4)
    w = np.array([1.0, 2.0, 3.0, 4.0])
    out = apply_rms_norm(d, w)
    assert out[3] > out[0]


def test_unembed_lens_shape():
    direction = np.zeros(32)
    direction[3] = 1.0
    lm_head = np.eye(32)
    scores = unembed_lens(direction, lm_head)
    assert scores.shape == (32,)
    assert int(np.argmax(scores)) == 3
