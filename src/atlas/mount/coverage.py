"""Coverage: how much of W and real MLP writes do mounts explain?"""

from __future__ import annotations

from typing import Any

import numpy as np

from atlas.mount.strategies import _as_numpy, tile_svd_mounts


def tile_svd_weight_coverage(
    W,
    *,
    tile_size: int = 512,
    modes_per_tile: int = 2,
) -> dict[str, Any]:
    """Frobenius energy of W retained by per-tile truncated SVD."""
    w = _as_numpy(W)
    total = float(np.sum(w * w))
    kept = 0.0
    n_tiles = (w.shape[1] + tile_size - 1) // tile_size
    for t in range(n_tiles):
        start = t * tile_size
        end = min(start + tile_size, w.shape[1])
        block = w[:, start:end]
        u, s, vt = np.linalg.svd(block, full_matrices=False)
        k = min(modes_per_tile, s.size)
        recon = (u[:, :k] * s[:k]) @ vt[:k, :]
        kept += float(np.sum(recon * recon))
    frac = kept / (total + 1e-12)
    return {
        "total_frobenius_sq": round(total, 4),
        "kept_frobenius_sq": round(kept, 4),
        "weight_energy_fraction": round(frac, 4),
        "n_tiles": n_tiles,
        "modes_per_tile": modes_per_tile,
        "n_modes": n_tiles * modes_per_tile,
    }


def sparse_write_coverage(
    writes: np.ndarray,
    directions: np.ndarray,
    *,
    k_active: int = 8,
) -> dict[str, Any]:
    """Fraction of write energy explained by top-k mounts (least-squares).

    writes: (n_tokens, d) residual writes (e.g. MLP outputs)
    directions: (n_mounts, d) unit write directions

    For each token, pick the k mounts with largest |<w, d_i>|, then solve
    least-squares w ≈ D_k a. Report 1 - ||w - D_k a||^2 / ||w||^2.
    """
    w = np.asarray(writes, dtype=np.float64)
    d = np.asarray(directions, dtype=np.float64)
    if w.ndim == 1:
        w = w.reshape(1, -1)
    norms = np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
    d = d / norms
    coeffs = w @ d.T
    n_tok = w.shape[0]
    k = min(k_active, d.shape[0])
    explained_energy = 0.0
    total = float(np.sum(w * w))
    residual_energy = 0.0
    top_mass = 0.0
    for i in range(n_tok):
        c = coeffs[i]
        row_w = w[i]
        if k < c.size:
            idx = np.argpartition(np.abs(c), -k)[-k:]
        else:
            idx = np.arange(c.size)
        Dk = d[idx].T  # (dim, k)
        # least squares: min ||Dk a - w||
        a, *_ = np.linalg.lstsq(Dk, row_w, rcond=None)
        recon = Dk @ a
        resid = row_w - recon
        explained_energy += float(np.sum(recon * recon))
        residual_energy += float(np.sum(resid * resid))
        abs_c = np.abs(c)
        top_mass += float(np.abs(c[idx]).sum() / (abs_c.sum() + 1e-12))
    # Prefer residual-based fraction (always in [0, 1])
    frac = 1.0 - residual_energy / (total + 1e-12)
    frac = float(np.clip(frac, 0.0, 1.0))
    return {
        "n_tokens": n_tok,
        "n_mounts": int(d.shape[0]),
        "k_active": k,
        "write_energy_fraction": round(frac, 4),
        "mean_topk_coeff_mass": round(top_mass / max(n_tok, 1), 4),
        "total_write_energy": round(total, 4),
        "residual_write_energy": round(residual_energy, 4),
    }


def mount_directions_from_weight(
    W,
    *,
    tile_size: int = 512,
    modes_per_tile: int = 2,
) -> tuple[np.ndarray, list]:
    mounts = tile_svd_mounts(W, tile_size=tile_size, modes_per_tile=modes_per_tile)
    dirs = np.stack([m.direction for m in mounts], axis=0)
    return dirs, mounts
