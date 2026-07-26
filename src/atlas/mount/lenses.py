"""Token lenses for residual-space write directions."""

from __future__ import annotations

import numpy as np


def apply_rms_norm(
    direction: np.ndarray, weight: np.ndarray | None, eps: float = 1e-6
) -> np.ndarray:
    """Approximate final RMSNorm: scale by weight / ||d||_rms."""
    d = np.asarray(direction, dtype=np.float64).ravel()
    if weight is None:
        return d
    w = np.asarray(weight, dtype=np.float64).ravel()
    if w.shape[0] != d.shape[0]:
        raise ValueError(f"rms weight {w.shape} incompatible with direction {d.shape}")
    rms = float(np.sqrt(np.mean(d * d) + eps))
    return (d / rms) * w


def unembed_lens(
    direction: np.ndarray,
    lm_head: np.ndarray,
    *,
    rms_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Logit bump if direction is added to residual (vocab scores)."""
    d = apply_rms_norm(direction, rms_weight)
    w = np.asarray(lm_head, dtype=np.float64)
    if w.shape[1] == d.shape[0]:
        return w @ d
    if w.shape[0] == d.shape[0]:
        return d @ w
    raise ValueError(f"lm_head {w.shape} incompatible with direction {d.shape}")
