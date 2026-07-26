"""Mechanism cards: prove (trigger v, write u, strength σ) on real MLP writes.

Important: corr(a, <tile_write,u>)≈1 and slope≈σ are SVD *identities*
(always true if u,v are singular vectors of W_tile). They are sanity checks only.

Real proof metrics:
  - mode_energy_fraction: share of tile-write energy along this mode
  - lift vs random direction in the same tile
  - tile_kept_r2: energy explained by all kept modes in the tile together
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from atlas.mount.strategies import _as_numpy
from atlas.mount.trigger import mode_trigger_coeffs


def tile_writes(
    W,
    intermediate: np.ndarray,
    col_start: int,
    col_end: int,
) -> np.ndarray:
    """Residual write contributed by columns [start:end]: (n_tokens, d_model)."""
    w = _as_numpy(W).astype(np.float32, copy=False)
    x = np.asarray(intermediate, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    block = w[:, col_start:col_end]
    return x[:, col_start:col_end] @ block.T


def score_mount_mechanism(
    mounts: list[RawMount],
    *,
    W,
    intermediate: np.ndarray,
    writes: np.ndarray,
) -> list[dict[str, Any]]:
    x = np.asarray(intermediate, dtype=np.float32)
    full_writes = np.asarray(writes, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if full_writes.ndim == 1:
        full_writes = full_writes.reshape(1, -1)

    w_np = _as_numpy(W)
    in_dim = int(w_np.shape[1])

    coeffs = mode_trigger_coeffs(x, mounts)
    tile_cache: dict[tuple[int, int], np.ndarray] = {}

    # Group mounts by tile for multi-mode reconstruction
    by_tile: dict[tuple[int, int], list[int]] = defaultdict(list)
    for j, m in enumerate(mounts):
        by_tile[(int(m.meta["col_start"]), int(m.meta["col_end"]))].append(j)

    tile_r2: dict[tuple[int, int], float] = {}
    for (start, end), idxs in by_tile.items():
        # Full-span tile == full writes — reuse to avoid a second giant matmul
        if start == 0 and end >= in_dim:
            tw = full_writes
        else:
            tw = tile_writes(W, x, start, end)
        tile_cache[(start, end)] = tw
        tw_e = float(np.sum(tw.astype(np.float64) * tw)) + 1e-12
        # Chunked residual so whole-matrix tiles don't allocate 2× (n, d)
        n_tok, d_model = tw.shape
        chunk = 4096
        resid_e = 0.0
        us = []
        sigs = []
        for j in idxs:
            m = mounts[j]
            sigs.append(float(m.meta.get("singular_value") or 0.0))
            u = np.asarray(m.direction, dtype=np.float32).ravel()
            us.append(u / (np.linalg.norm(u) + 1e-12))
        for i0 in range(0, n_tok, chunk):
            i1 = min(n_tok, i0 + chunk)
            pred = np.zeros((i1 - i0, d_model), dtype=np.float32)
            for j, u, sigma in zip(idxs, us, sigs):
                a = coeffs[i0:i1, j].astype(np.float32, copy=False)
                pred += (a * np.float32(sigma))[:, None] * u[None, :]
            r = tw[i0:i1] - pred
            resid_e += float(np.sum(r.astype(np.float64) * r))
            del pred, r
        tile_r2[(start, end)] = float(np.clip(1.0 - resid_e / tw_e, -1.0, 1.0))

    rows: list[dict[str, Any]] = []
    for j, m in enumerate(mounts):
        start = int(m.meta["col_start"])
        end = int(m.meta["col_end"])
        sigma = float(m.meta.get("singular_value") or 0.0)
        u = np.asarray(m.direction, dtype=np.float32).ravel()
        u = u / (np.linalg.norm(u) + 1e-12)
        a = np.asarray(coeffs[:, j], dtype=np.float32)
        tw = tile_cache[(start, end)]

        proj = tw @ u  # = σ a  (SVD identity, up to numerics)

        if float(a.std()) < 1e-12 or float(proj.std()) < 1e-12:
            corr = 0.0
        else:
            corr = float(np.corrcoef(a.astype(np.float64), proj.astype(np.float64))[0, 1])
        slope = float(np.dot(a, proj) / (float(np.dot(a, a)) + 1e-12))
        slope_err = abs(slope - sigma) / (abs(sigma) + 1e-12)
        svd_ok = bool(corr > 0.99 and slope_err < 0.05)

        tw_norm_sq = np.sum(tw * tw, axis=1) + np.float32(1e-12)
        mode_e = (a * np.float32(sigma)) ** 2
        mode_frac = mode_e / tw_norm_sq
        mean_mode_frac = float(np.mean(mode_frac))

        # Random baseline: energy along a random unit vector (seeded per mount)
        mount_rng = np.random.default_rng(10007 + j * 997)
        rand_u = mount_rng.standard_normal(u.shape[0]).astype(np.float32)
        rand_u /= np.linalg.norm(rand_u) + 1e-12
        rand_frac = float(np.mean((tw @ rand_u) ** 2 / tw_norm_sq))
        lift = mean_mode_frac - rand_frac

        # Share of *full* MLP write energy along u (not just tile)
        fw_norm_sq = np.sum(full_writes * full_writes, axis=1) + np.float32(1e-12)
        full_frac = float(np.mean((full_writes @ u) ** 2 / fw_norm_sq))

        abs_a = np.abs(a)
        med = float(np.median(abs_a))
        strong = abs_a >= max(med, 1e-12)
        if not np.any(strong):
            strong = np.ones(a.shape[0], dtype=bool)

        # Cosine without allocating full (n, d) pred: use proj/||tw|| vs σ a / ||tw||
        # pred = (a*σ) u  ⇒ cos(pred, tw) = (a*σ)(tw·u) / (||pred|| ||tw||)
        # ||pred|| = |a*σ|, tw·u = proj
        denom = (np.abs(a * np.float32(sigma)) * np.sqrt(tw_norm_sq)) + np.float32(1e-12)
        cos_tok = (a * np.float32(sigma) * proj) / denom
        cos_strong = float(np.mean(cos_tok[strong]))

        rows.append(
            {
                "mount_id": m.mount_id,
                "tile_index": m.meta.get("tile_index"),
                "sv_index": m.meta.get("sv_index"),
                "col_start": start,
                "col_end": end,
                "singular_value": round(sigma, 6),
                "peak_abs_trigger": round(float(abs_a.max()), 6),
                "mean_abs_trigger": round(float(abs_a.mean()), 6),
                # sanity (SVD identity)
                "svd_identity_corr": round(corr, 4),
                "svd_identity_slope_err": round(slope_err, 4),
                "svd_identity_ok": svd_ok,
                # real proof
                "mode_energy_fraction": round(mean_mode_frac, 4),
                "random_energy_fraction": round(rand_frac, 4),
                "energy_lift_over_random": round(lift, 4),
                "full_write_energy_fraction": round(full_frac, 4),
                "tile_kept_modes_r2": round(tile_r2[(start, end)], 4),
                "mean_cosine_pred_tile_strong": round(cos_strong, 4),
                "n_strong_tokens": int(strong.sum()),
            }
        )
    return rows


def is_proven(sc: dict[str, Any], *, min_lift: float = 0.02) -> bool:
    """Proven = SVD wiring checks out AND mode carries more write energy than chance."""
    return bool(
        sc.get("svd_identity_ok")
        and sc.get("energy_lift_over_random", 0) >= min_lift
    )


def mechanism_summary(scores: list[dict[str, Any]]) -> dict[str, Any]:
    if not scores:
        return {"n_mounts": 0}

    def mean(key: str) -> float:
        return round(sum(float(s[key]) for s in scores) / len(scores), 4)

    proven = sum(1 for s in scores if is_proven(s))
    return {
        "n_mounts": len(scores),
        "n_proven": proven,
        "proven_rate": round(proven / len(scores), 4),
        "mean_mode_energy_fraction": mean("mode_energy_fraction"),
        "mean_energy_lift_over_random": mean("energy_lift_over_random"),
        "mean_tile_kept_modes_r2": mean("tile_kept_modes_r2"),
        "mean_full_write_energy_fraction": mean("full_write_energy_fraction"),
        "frac_svd_identity_ok": round(
            sum(1 for s in scores if s.get("svd_identity_ok")) / len(scores), 4
        ),
    }
