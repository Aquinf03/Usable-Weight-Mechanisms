"""Mount chunking strategies on a single weight matrix."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RawMount:
    method: str
    mount_id: str
    direction: np.ndarray
    meta: dict


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v * 0.0
    return v / n


def _as_numpy(W) -> np.ndarray:
    if hasattr(W, "detach"):
        return W.detach().float().cpu().numpy()
    return np.asarray(W, dtype=np.float64)


def whole_matrix_svd_mounts(
    W,
    *,
    top_k: int = 16,
    side: str = "output",
) -> list[RawMount]:
    """SVD on the full matrix; top left singular vectors = output-space modes."""
    w = _as_numpy(W)
    u, _s, _vt = np.linalg.svd(w, full_matrices=False)
    k = min(top_k, u.shape[1])
    mounts: list[RawMount] = []
    for i in range(k):
        mounts.append(
            RawMount(
                method="whole_matrix_svd",
                mount_id=f"whole:sv{i}",
                direction=_normalize(u[:, i]),
                meta={"sv_index": i, "side": side},
            )
        )
    return mounts


def tile_svd_mounts(
    W,
    *,
    tile_size: int = 512,
    modes_per_tile: int = 2,
    axis: str = "input",
) -> list[RawMount]:
    """SVD each input tile; keep top write modes per tile.

    direction = left SV (write); meta['trigger_vector'] = right SV.
    """
    w = _as_numpy(W)
    if axis != "input":
        w = w.T
    n_tiles = (w.shape[1] + tile_size - 1) // tile_size
    mounts: list[RawMount] = []
    for t in range(n_tiles):
        start = t * tile_size
        end = min(start + tile_size, w.shape[1])
        block = w[:, start:end]
        u, s, vt = np.linalg.svd(block, full_matrices=False)
        k = min(modes_per_tile, u.shape[1])
        for i in range(k):
            mounts.append(
                RawMount(
                    method="tile_svd",
                    mount_id=f"tile:{t}:sv{i}",
                    direction=_normalize(u[:, i]),
                    meta={
                        "tile_index": t,
                        "sv_index": i,
                        "singular_value": float(s[i]) if i < s.size else 0.0,
                        "col_start": start,
                        "col_end": end,
                        "axis": axis,
                        "trigger_vector": np.asarray(vt[i], dtype=np.float64).copy(),
                    },
                )
            )
    return mounts


def column_sample_mounts(
    W,
    *,
    n: int = 64,
    seed: int = 0,
    prefer_high_norm: bool = True,
) -> list[RawMount]:
    """Sample individual columns as write directions."""
    w = _as_numpy(W)
    rng = np.random.default_rng(seed)
    norms = np.linalg.norm(w, axis=0)
    if prefer_high_norm:
        probs = norms / (norms.sum() + 1e-12)
        idx = rng.choice(w.shape[1], size=min(n * 4, w.shape[1]), replace=False, p=probs)
        idx = idx[np.argsort(norms[idx])[::-1][:n]]
    else:
        idx = rng.choice(w.shape[1], size=min(n, w.shape[1]), replace=False)
    mounts: list[RawMount] = []
    for i, col in enumerate(sorted(int(x) for x in idx)):
        mounts.append(
            RawMount(
                method="column_sample",
                mount_id=f"col:{col}",
                direction=_normalize(w[:, col]),
                meta={
                    "col_index": col,
                    "col_norm": float(norms[col]),
                    "rank_in_sample": i,
                },
            )
        )
    return mounts


def random_baseline_mounts(dim: int, *, n: int = 64, seed: int = 0) -> list[RawMount]:
    """Isotropic random residual directions (null baseline)."""
    rng = np.random.default_rng(seed)
    mounts: list[RawMount] = []
    for i in range(n):
        mounts.append(
            RawMount(
                method="random_baseline",
                mount_id=f"rand:{i}",
                direction=_normalize(rng.standard_normal(dim)),
                meta={"seed": seed, "index": i},
            )
        )
    return mounts
