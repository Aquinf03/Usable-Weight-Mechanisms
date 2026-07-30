"""Effective-path weight maps for sites where raw W is the wrong mount object."""

from __future__ import annotations

import numpy as np

from atlas.mount.strategies import RawMount, _as_numpy, _normalize, tile_svd_mounts

GATE_POOLS: tuple[str, ...] = ("mean", "mixture_k4", "compose_down")
V_POOLS: tuple[str, ...] = ("lstsq", "compose_o")


def pool_gate_vector(gate_act: np.ndarray, pool: str = "mean") -> np.ndarray:
    """Corpus-mean gate vector g_bar used in W* = diag(g_bar) W_up.

    ``pool`` is accepted for API symmetry with effective-path trial names;
    all current trials use the corpus mean for a single g_bar.
    """
    _ = pool
    g = np.asarray(gate_act, dtype=np.float64)
    if g.ndim == 1:
        g = g.reshape(1, -1)
    g_bar = np.maximum(g.mean(axis=0), 0.0)
    if float(np.linalg.norm(g_bar)) < 1e-12:
        g_bar = np.ones_like(g_bar)
    return g_bar.astype(np.float64)


def mean_gate_effective_weight(
    W_up,
    gate_act: np.ndarray,
    *,
    pool: str = "mean",
) -> np.ndarray:
    """Build W* = diag(g_bar) @ W_up from corpus-mean post-activation gate."""
    w = _as_numpy(W_up).astype(np.float64, copy=False)
    g = np.asarray(gate_act, dtype=np.float64)
    if g.ndim == 1:
        g = g.reshape(1, -1)
    if g.shape[-1] != w.shape[0]:
        raise ValueError(
            f"gate_act out-dim {g.shape[-1]} != W_up rows {w.shape[0]}"
        )
    g_bar = pool_gate_vector(g, pool=pool)
    return (g_bar[:, None] * w).astype(np.float32)


def effective_linear_writes(W_star, intermediate: np.ndarray) -> np.ndarray:
    """W_star @ x token rows for SVD-identity / tile geometry."""
    w = _as_numpy(W_star).astype(np.float32, copy=False)
    x = np.asarray(intermediate, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x @ w.T


def cluster_gate_prototypes(
    gate_act: np.ndarray,
    *,
    k: int = 4,
    seed: int = 0,
    max_iter: int = 25,
) -> list[np.ndarray]:
    """K-means prototypes of post-activation gate vectors (non-negative)."""
    g = np.asarray(gate_act, dtype=np.float64)
    if g.ndim == 1:
        g = g.reshape(1, -1)
    n = g.shape[0]
    k = max(1, min(int(k), n))
    rng = np.random.default_rng(seed)
    norms = np.linalg.norm(g, axis=1)
    p = norms + 1e-8
    p = p / p.sum()
    init_idx = rng.choice(n, size=k, replace=False, p=p)
    centers = np.maximum(g[init_idx].copy(), 0.0)
    for _ in range(max_iter):
        g2 = np.sum(g * g, axis=1, keepdims=True)
        c2 = np.sum(centers * centers, axis=1, keepdims=True).T
        dist = g2 + c2 - 2.0 * (g @ centers.T)
        labels = np.argmin(dist, axis=1)
        new_centers = centers.copy()
        for j in range(k):
            mask = labels == j
            if not np.any(mask):
                new_centers[j] = np.maximum(g[int(rng.integers(0, n))], 0.0)
            else:
                new_centers[j] = np.maximum(g[mask].mean(axis=0), 0.0)
        if float(np.max(np.abs(new_centers - centers))) < 1e-8:
            centers = new_centers
            break
        centers = new_centers
    out: list[np.ndarray] = []
    for j in range(k):
        c = centers[j]
        if float(np.linalg.norm(c)) < 1e-12:
            c = np.ones_like(c)
        out.append(c.astype(np.float64))
    return out


def composed_gate_down_weight(
    W_up,
    W_down,
    gate_act: np.ndarray,
    *,
    pool: str = "mean",
) -> np.ndarray:
    """W_dagger = W_down @ diag(g_bar) @ W_up  (d_model x d_model residual path)."""
    w_star = mean_gate_effective_weight(W_up, gate_act, pool=pool)
    w_down = _as_numpy(W_down).astype(np.float64, copy=False)
    return (w_down @ w_star).astype(np.float32)


def mixture_tile_svd_mounts(
    W_up,
    gate_act: np.ndarray,
    *,
    k: int = 4,
    tile_size: int = 512,
    modes_per_tile: int = 2,
    n_target: int | None = None,
) -> list[RawMount]:
    """Union tile-SVD mounts across K gate-prototype effective maps."""
    w = _as_numpy(W_up).astype(np.float64, copy=False)
    protos = cluster_gate_prototypes(gate_act, k=k)
    mounts: list[RawMount] = []
    for pi, g_bar in enumerate(protos):
        w_star = (g_bar[:, None] * w).astype(np.float32)
        part = tile_svd_mounts(
            w_star, tile_size=tile_size, modes_per_tile=modes_per_tile
        )
        for m in part:
            mounts.append(
                RawMount(
                    method="tile_svd",
                    mount_id=f"proto{pi}:{m.mount_id}",
                    direction=_normalize(np.asarray(m.direction, dtype=np.float64)),
                    meta={
                        **dict(m.meta),
                        "gate_proto": pi,
                        "mount_space": "mixture_gate_up",
                    },
                )
            )
    mounts.sort(key=lambda m: -float(m.meta.get("singular_value") or 0.0))
    if n_target is not None and len(mounts) > n_target:
        mounts = mounts[:n_target]
    return mounts


def lstsq_effective_weight(
    intermediate: np.ndarray,
    writes: np.ndarray,
    *,
    ridge: float = 1e-3,
) -> np.ndarray:
    """Fit W_eff so writes ~= intermediate @ W_eff.T (ridge least squares).

    Used for attn.v: attention mixing is not a fixed linear map of the current
    token; the corpus least-squares map is the best single matrix explaining
    mixed-v from v_proj inputs.
    """
    x = np.asarray(intermediate, dtype=np.float64)
    y = np.asarray(writes, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"token count mismatch x={x.shape[0]} y={y.shape[0]}")
    d_in = x.shape[1]
    xtx = x.T @ x + float(ridge) * np.eye(d_in, dtype=np.float64)
    xty = x.T @ y
    wt = np.linalg.solve(xtx, xty)
    return wt.T.astype(np.float32)


def gqa_expand_kv_weight(
    W_v,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> np.ndarray:
    """Repeat each KV-head block of W_v to match q-head layout for o_proj."""
    w = _as_numpy(W_v).astype(np.float64, copy=False)
    expect = num_kv_heads * head_dim
    if w.shape[0] != expect:
        raise ValueError(
            f"W_v rows {w.shape[0]} != num_kv_heads*head_dim {expect}"
        )
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    n_rep = num_heads // num_kv_heads
    blocks = w.reshape(num_kv_heads, head_dim, w.shape[1])
    expanded = np.repeat(blocks, n_rep, axis=0)
    return expanded.reshape(num_heads * head_dim, w.shape[1]).astype(np.float32)


def composed_v_o_weight(
    W_v,
    W_o,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> np.ndarray:
    """W_dagger = W_o @ expand_GQA(W_v)  (d_model x d_model; ignores attn mixing)."""
    w_exp = gqa_expand_kv_weight(
        W_v,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    w_o = _as_numpy(W_o).astype(np.float64, copy=False)
    if w_o.shape[1] != w_exp.shape[0]:
        raise ValueError(
            f"W_o cols {w_o.shape[1]} != expanded V rows {w_exp.shape[0]}"
        )
    return (w_o @ w_exp).astype(np.float32)


def parse_gate_pools(spec: str | None) -> list[str]:
    """Parse --gate-pools CSV for mlp.up; default mean, mixture_k4, compose_down."""
    if spec is None or not str(spec).strip():
        return gate_pool_fallbacks()
    out: list[str] = []
    for part in str(spec).split(","):
        p = part.strip().lower()
        if not p:
            continue
        if p not in GATE_POOLS and not p.startswith("mixture_k"):
            raise ValueError(f"unknown gate pool {p!r}; known={list(GATE_POOLS)}")
        out.append(p)
    if not out:
        raise ValueError("empty --gate-pools")
    return out


def gate_pool_fallbacks(primary: str = "mean") -> list[str]:
    """Ordered effective-path trials for mlp.up (primary first)."""
    order = ["mean", "mixture_k4", "compose_down"]
    if primary in order:
        return [primary] + [p for p in order if p != primary]
    if primary.startswith("mixture_k") or primary in GATE_POOLS:
        return [primary] + [p for p in order if p != primary]
    return order


def parse_v_pools(spec: str | None) -> list[str]:
    """Parse effective-path trials for attn.v; default lstsq, compose_o."""
    if spec is None or not str(spec).strip():
        return v_pool_fallbacks()
    out: list[str] = []
    for part in str(spec).split(","):
        p = part.strip().lower()
        if not p:
            continue
        if p not in V_POOLS:
            raise ValueError(f"unknown v pool {p!r}; known={list(V_POOLS)}")
        out.append(p)
    if not out:
        raise ValueError("empty v pools")
    return out


def v_pool_fallbacks(primary: str = "lstsq") -> list[str]:
    order = ["lstsq", "compose_o"]
    if primary in order:
        return [primary] + [p for p in order if p != primary]
    return [primary] + order
