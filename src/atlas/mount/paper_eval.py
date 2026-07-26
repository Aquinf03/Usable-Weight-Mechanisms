"""Paper-eval helpers: chunking lift + causal steer alignment.

No SAE. Metrics are energy-lift on real writes and logit-delta vs unembed(u).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from atlas.mount.lenses import unembed_lens
from atlas.mount.mechanism import is_proven, score_mount_mechanism
from atlas.mount.strategies import (
    RawMount,
    _as_numpy,
    _normalize,
    column_sample_mounts,
    random_baseline_mounts,
    tile_svd_mounts,
)


def whole_matrix_mounts_with_trigger(W, *, top_k: int) -> list[RawMount]:
    """Whole-matrix SVD mounts with col span + trigger (for mechanism scoring)."""
    w = _as_numpy(W)
    u, s, vt = np.linalg.svd(w, full_matrices=False)
    k = min(top_k, u.shape[1])
    mounts: list[RawMount] = []
    for i in range(k):
        mounts.append(
            RawMount(
                method="whole_matrix_svd",
                mount_id=f"whole:sv{i}",
                direction=_normalize(u[:, i]),
                meta={
                    "sv_index": i,
                    "singular_value": float(s[i]),
                    "col_start": 0,
                    "col_end": int(w.shape[1]),
                    "trigger_vector": np.asarray(vt[i], dtype=np.float64).copy(),
                },
            )
        )
    return mounts


def column_mounts_with_tile(W, *, n: int, seed: int = 0) -> list[RawMount]:
    """High-norm columns as 1-col tiles (trigger = 1.0 on that col)."""
    base = column_sample_mounts(W, n=n, seed=seed, prefer_high_norm=True)
    out: list[RawMount] = []
    for m in base:
        col = int(m.meta["col_index"])
        out.append(
            RawMount(
                method=m.method,
                mount_id=m.mount_id,
                direction=m.direction,
                meta={
                    **m.meta,
                    "singular_value": float(m.meta.get("col_norm") or 1.0),
                    "col_start": col,
                    "col_end": col + 1,
                    "trigger_vector": np.array([1.0], dtype=np.float64),
                },
            )
        )
    return out


def random_mounts_fullspan(W, *, n: int, seed: int = 0) -> list[RawMount]:
    w = _as_numpy(W)
    base = random_baseline_mounts(int(w.shape[0]), n=n, seed=seed)
    rng = np.random.default_rng(seed + 1)
    out: list[RawMount] = []
    for m in base:
        trig = rng.standard_normal(int(w.shape[1]))
        trig = _normalize(trig)
        out.append(
            RawMount(
                method=m.method,
                mount_id=m.mount_id,
                direction=m.direction,
                meta={
                    **m.meta,
                    "singular_value": 1.0,
                    "col_start": 0,
                    "col_end": int(w.shape[1]),
                    "trigger_vector": trig,
                },
            )
        )
    return out


def build_strategy_mounts(
    W,
    strategy: str,
    *,
    n_target: int,
    tile_size: int = 512,
    modes_per_tile: int = 2,
) -> list[RawMount]:
    w = _as_numpy(W)
    tile = min(tile_size, int(w.shape[1]))
    if strategy == "tile_svd":
        mounts = tile_svd_mounts(W, tile_size=tile, modes_per_tile=modes_per_tile)
        # match budget roughly
        if len(mounts) > n_target:
            # keep highest σ
            mounts = sorted(
                mounts,
                key=lambda m: -float(m.meta.get("singular_value") or 0.0),
            )[:n_target]
        return mounts
    if strategy == "whole_matrix_svd":
        return whole_matrix_mounts_with_trigger(W, top_k=n_target)
    if strategy == "column_sample":
        return column_mounts_with_tile(W, n=n_target)
    if strategy == "random_baseline":
        return random_mounts_fullspan(W, n=n_target)
    raise ValueError(f"unknown strategy {strategy}")


def eval_strategy_on_writes(
    mounts: list[RawMount],
    *,
    W,
    intermediate: np.ndarray,
    writes: np.ndarray,
) -> dict[str, Any]:
    scores = score_mount_mechanism(
        mounts, W=W, intermediate=intermediate, writes=writes
    )
    lifts = [float(s["energy_lift_over_random"]) for s in scores]
    identities = [bool(s["svd_identity_ok"]) for s in scores]
    proven = [is_proven(s) for s in scores]
    # Does SVD-identity predict lift? (should be weak — identity is near-always true)
    id_ok = np.array(identities, dtype=np.float64)
    lift_arr = np.asarray(lifts, dtype=np.float64)
    if id_ok.std() < 1e-12 or lift_arr.std() < 1e-12:
        id_lift_corr = 0.0
    else:
        id_lift_corr = float(np.corrcoef(id_ok, lift_arr)[0, 1])

    # Fair cross-strategy metric: energy along u in *full* write, minus random u
    # (tile-local lift is tautological for 1-column tiles)
    fw = np.asarray(writes, dtype=np.float32)
    if fw.ndim == 1:
        fw = fw.reshape(1, -1)
    fw_norm_sq = np.sum(fw * fw, axis=1) + np.float32(1e-12)
    full_lifts: list[float] = []
    for j, m in enumerate(mounts):
        u = np.asarray(m.direction, dtype=np.float32).ravel()
        u = u / (np.linalg.norm(u) + 1e-12)
        full_frac = float(np.mean((fw @ u) ** 2 / fw_norm_sq))
        mount_rng = np.random.default_rng(10007 + j * 997)
        rand_u = mount_rng.standard_normal(u.shape[0]).astype(np.float32)
        rand_u /= np.linalg.norm(rand_u) + 1e-12
        rand_frac = float(np.mean((fw @ rand_u) ** 2 / fw_norm_sq))
        full_lifts.append(full_frac - rand_frac)

    return {
        "n_mounts": len(scores),
        "mean_energy_lift": round(float(np.mean(lifts)), 4) if lifts else 0.0,
        "median_energy_lift": round(float(np.median(lifts)), 4) if lifts else 0.0,
        "p90_energy_lift": round(float(np.quantile(lifts, 0.9)), 4) if lifts else 0.0,
        "mean_full_write_lift": round(float(np.mean(full_lifts)), 4) if full_lifts else 0.0,
        "median_full_write_lift": round(float(np.median(full_lifts)), 4)
        if full_lifts
        else 0.0,
        "p90_full_write_lift": round(float(np.quantile(full_lifts, 0.9)), 4)
        if full_lifts
        else 0.0,
        "frac_svd_identity_ok": round(float(np.mean(identities)), 4) if identities else 0.0,
        "proven_rate": round(float(np.mean(proven)), 4) if proven else 0.0,
        "mean_tile_kept_r2": round(
            float(np.mean([s["tile_kept_modes_r2"] for s in scores])), 4
        )
        if scores
        else 0.0,
        "identity_vs_lift_corr": round(id_lift_corr, 4),
        "scores": scores,
    }


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size or a.size < 3:
        return 0.0
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def topk_jaccard(a: np.ndarray, b: np.ndarray, *, k: int = 20) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    k = min(k, a.size, b.size)
    ia = set(np.argpartition(a, -k)[-k:].tolist())
    ib = set(np.argpartition(b, -k)[-k:].tolist())
    if not ia and not ib:
        return 1.0
    return len(ia & ib) / len(ia | ib)


def causal_steer_alignment(
    model,
    tokenizer,
    *,
    layer: int,
    direction: np.ndarray,
    texts: list[str],
    device: str,
    lm_head: np.ndarray,
    rms_weight: np.ndarray | None,
    alpha: float = 2.0,
    max_texts: int = 8,
) -> dict[str, Any]:
    """Add α·u into residual after layer; compare Δlogits to unembed(u)."""
    import torch

    u = np.asarray(direction, dtype=np.float64).ravel()
    u = u / (np.linalg.norm(u) + 1e-12)
    target = unembed_lens(u, lm_head, rms_weight=rms_weight)

    block = model.model.layers[layer]
    u_t = torch.tensor(u, device=device, dtype=next(model.parameters()).dtype)

    def hook(_mod, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h + alpha * u_t.view(1, 1, -1)
        if isinstance(out, tuple):
            return (h,) + out[1:]
        return h

    deltas: list[np.ndarray] = []
    for text in texts[:max_texts]:
        toks = tokenizer(text, return_tensors="pt")
        toks = {k: v.to(device) for k, v in toks.items()}
        with torch.no_grad():
            base = model(**toks).logits[0, -1].float().cpu().numpy()
        handle = block.register_forward_hook(hook)
        try:
            with torch.no_grad():
                steered = model(**toks).logits[0, -1].float().cpu().numpy()
        finally:
            handle.remove()
        deltas.append(steered - base)

    delta = np.mean(np.stack(deltas, axis=0), axis=0)
    return {
        "spearman_vs_unembed": round(spearman_corr(delta, target), 4),
        "topk20_jaccard_vs_unembed": round(topk_jaccard(delta, target, k=20), 4),
        "alpha": alpha,
        "n_texts": len(deltas),
    }


def judge_paper_go(
    results: dict[str, Any],
    *,
    min_causal_layer: int = 6,
) -> dict[str, Any]:
    """GO/NO-GO for tile-SVD mounts.

    Full-write lift for strategy A/B. Coverage = saturation.
    Steer↔unembed (C1) is only required for layers >= min_causal_layer
    (early residual writes are not in final logit geometry).
    """
    checks: list[dict[str, Any]] = []
    chunk = results.get("exp_a_chunking") or {}
    by = {r["strategy"]: r for r in chunk.get("strategies") or []}

    tile = by.get("tile_svd") or {}
    whole = by.get("whole_matrix_svd") or {}
    cols = by.get("column_sample") or {}
    rand = by.get("random_baseline") or {}
    layer = results.get("layer")
    layer_i = int(layer) if layer is not None else None

    def full_lift(row: dict) -> float:
        if row.get("mean_full_write_lift") is not None:
            return float(row["mean_full_write_lift"])
        return float(row.get("mean_energy_lift") or 0)

    t_full = full_lift(tile)
    w_full = full_lift(whole)
    c_full = full_lift(cols)
    r_full = full_lift(rand)

    a1 = t_full > r_full + 0.005
    checks.append(
        {
            "id": "A1_tile_beats_random_full_write",
            "pass": a1,
            "detail": f"tile_full={t_full} rand_full={r_full}",
        }
    )

    a2 = t_full >= c_full - 0.002
    checks.append(
        {
            "id": "A2_tile_beats_or_ties_columns_full_write",
            "pass": a2,
            "detail": f"tile_full={t_full} cols_full={c_full}",
        }
    )

    id_frac = float(tile.get("frac_svd_identity_ok") or 0)
    id_corr = abs(float(tile.get("identity_vs_lift_corr") or 0))
    a3 = id_frac >= 0.9 or id_corr < 0.3
    checks.append(
        {
            "id": "A3_identity_not_the_proof",
            "pass": a3,
            "detail": f"frac_identity={id_frac} corr(id,lift)={id_corr}",
        }
    )

    a4 = t_full + 0.002 >= w_full or (w_full > 0 and t_full / w_full >= 0.8)
    checks.append(
        {
            "id": "A4_tile_beats_or_ties_whole_full_write",
            "pass": a4,
            "detail": f"tile_full={t_full} whole_full={w_full}",
        }
    )

    cov = results.get("exp_b_coverage") or {}
    rows = cov.get("rows") or []
    b1 = False
    detail = "no rows"
    if len(rows) >= 2:
        lifts = [float(r.get("write_coverage_lift") or 0) for r in rows]
        peak = max(lifts)
        high = peak >= 0.25
        flat_after = all(l >= peak - 0.05 for l in lifts[1:])
        no_collapse = lifts[-1] >= lifts[0] - 0.03
        b1 = bool(high and flat_after and no_collapse)
        detail = (
            f"lifts={[round(x, 4) for x in lifts]} peak={peak:.4f} "
            f"high={high} flat_after={flat_after} no_collapse={no_collapse}"
        )
    checks.append(
        {
            "id": "B1_coverage_saturates_early",
            "pass": b1,
            "detail": detail,
        }
    )

    causal = results.get("exp_c_causal") or {}
    mean_sp = float(causal.get("mean_spearman") or 0)
    mean_jac = float(causal.get("mean_topk20_jaccard") or 0)
    c1_raw = mean_sp >= 0.05 or mean_jac >= 0.05
    c1_required = layer_i is None or layer_i >= min_causal_layer
    c1_pass = bool(c1_raw) if c1_required else True
    checks.append(
        {
            "id": "C1_steer_aligns_with_unembed",
            "pass": c1_pass,
            "required": c1_required,
            "observed": c1_raw,
            "detail": (
                f"spearman={mean_sp} jaccard20={mean_jac} "
                f"layer={layer_i} required={c1_required} "
                f"(early layers: final-unembed alignment not required)"
            ),
        }
    )

    n_pass = sum(1 for c in checks if c["pass"])
    required_ids = {
        "A1_tile_beats_random_full_write",
        "A2_tile_beats_or_ties_columns_full_write",
        "A3_identity_not_the_proof",
        "A4_tile_beats_or_ties_whole_full_write",
        "B1_coverage_saturates_early",
        "C1_steer_aligns_with_unembed",
    }
    go = all(next(c for c in checks if c["id"] == rid)["pass"] for rid in required_ids)

    return {
        "go": go,
        "n_pass": n_pass,
        "n_checks": len(checks),
        "min_causal_layer": min_causal_layer,
        "checks": checks,
        "verdict": "GO — paper-shaped signal" if go else "NO-GO — see failed checks",
        "advice": (
            "Tile-SVD + full-write energy-lift + coverage saturation"
            + (
                "; steer↔unembed required for mid/late layers"
                if c1_required
                else "; early-layer C1 informational only"
            )
        ),
    }
