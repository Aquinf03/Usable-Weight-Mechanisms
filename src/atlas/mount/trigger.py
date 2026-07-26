"""Trigger: when does a tile_svd mount fire on real MLP intermediates?"""

from __future__ import annotations

import numpy as np

from atlas.mount.strategies import RawMount


def mode_trigger_coeffs(
    intermediate: np.ndarray,
    mounts: list[RawMount],
) -> np.ndarray:
    """Project MLP intermediate onto each mode's right singular vector.

    intermediate: (n_tokens, d_mlp)
    returns: (n_tokens, n_mounts) trigger coefficients
    """
    x = np.asarray(intermediate, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    n_tok = x.shape[0]
    out = np.zeros((n_tok, len(mounts)), dtype=np.float64)
    for j, m in enumerate(mounts):
        meta = m.meta
        v = meta.get("trigger_vector")
        start = int(meta.get("col_start", 0))
        end = int(meta.get("col_end", x.shape[1]))
        if v is None:
            out[:, j] = x[:, start:end].mean(axis=1)
            continue
        v = np.asarray(v, dtype=np.float64).ravel()
        width = end - start
        if v.size != width:
            v = v[:width] if v.size > width else np.pad(v, (0, width - v.size))
        out[:, j] = x[:, start:end] @ v
    return out


DEFAULT_TRIGGER_TEXTS = [
    "The capital of France is Paris, a major European city.",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "Photosynthesis converts light energy into chemical energy in plants.",
    "In 1969, Neil Armstrong became the first person to walk on the Moon.",
    "SELECT users.id, users.name FROM users WHERE users.active = 1;",
    "The mitochondria is the powerhouse of the cell.",
    "Shakespeare wrote Hamlet, Macbeth, and Romeo and Juliet.",
    "Gradient descent minimizes a loss by stepping opposite the gradient.",
    "The Amazon rainforest is located primarily in Brazil.",
    "HTTP status 404 means the requested resource was not found.",
]
