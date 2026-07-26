"""Unit tests for mechanism write-prediction (no GPU)."""

from __future__ import annotations

import numpy as np

from atlas.mount.mechanism import (
    is_proven,
    mechanism_summary,
    score_mount_mechanism,
    tile_writes,
)
from atlas.mount.strategies import tile_svd_mounts


def test_rank1_tile_has_energy_lift():
    rng = np.random.default_rng(0)
    d_model, d_mlp = 32, 64
    u = rng.standard_normal(d_model)
    u /= np.linalg.norm(u)
    v = rng.standard_normal(32)
    v /= np.linalg.norm(v)
    sigma = 2.5
    W = np.zeros((d_model, d_mlp))
    W[:, 0:32] = sigma * np.outer(u, v)

    mounts = tile_svd_mounts(W, tile_size=32, modes_per_tile=1)
    n = 40
    inter = rng.standard_normal((n, d_mlp)) * 0.05
    a = rng.standard_normal(n)
    inter[:, 0:32] += a[:, None] * v[None, :]
    writes = inter @ W.T

    scores = score_mount_mechanism(mounts, W=W, intermediate=inter, writes=writes)
    top = next(s for s in scores if s["col_start"] == 0)
    assert top["svd_identity_ok"]
    assert top["energy_lift_over_random"] > 0.3
    assert top["tile_kept_modes_r2"] > 0.9
    assert is_proven(top)

    summary = mechanism_summary(scores)
    assert summary["n_proven"] >= 1


def test_tile_writes_shape():
    W = np.eye(8, 16)
    x = np.ones((3, 16))
    tw = tile_writes(W, x, 0, 8)
    assert tw.shape == (3, 8)
