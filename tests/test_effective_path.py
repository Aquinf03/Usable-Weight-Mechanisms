"""Tests for mean-gate effective W* mounts (no GPU)."""

from __future__ import annotations

import numpy as np

from atlas.mount.effective import effective_linear_writes, mean_gate_effective_weight
from atlas.mount.paper_eval import eval_strategy_on_writes, judge_paper_go
from atlas.mount.sites import get_site
from atlas.mount.strategies import tile_svd_mounts


def test_mlp_up_mount_space_is_mean_gate():
    spec = get_site("mlp.up")
    assert spec.score_space == "gated_up"
    assert spec.mount_space == "mean_gate_up"


def test_attn_v_mount_space_is_lstsq():
    spec = get_site("attn.v")
    assert spec.score_space == "mixed_v"
    assert spec.mount_space == "lstsq_mixed_v"


def test_mean_gate_effective_weight_scales_rows():
    W = np.ones((4, 3), dtype=np.float32)
    gate = np.array([[1.0, 2.0, 0.0, 4.0], [1.0, 2.0, 0.0, 4.0]], dtype=np.float32)
    W_star = mean_gate_effective_weight(W, gate, pool="mean")
    assert W_star.shape == W.shape
    np.testing.assert_allclose(W_star[0], 1.0)
    np.testing.assert_allclose(W_star[1], 2.0)
    np.testing.assert_allclose(W_star[2], 0.0)
    np.testing.assert_allclose(W_star[3], 4.0)


def test_mean_gate_mounts_lift_on_gated_writes():
    """When g≈ḡ, mounts of W*=diag(ḡ)W explain z=g⊙(Wx)."""
    rng = np.random.default_rng(0)
    d_out, d_in = 32, 64
    u = rng.standard_normal(d_out).astype(np.float32)
    u /= np.linalg.norm(u)
    v = rng.standard_normal(d_in).astype(np.float32)
    v /= np.linalg.norm(v)
    sigma = 2.5
    W = (sigma * np.outer(u, v)).astype(np.float32)

    n = 100
    a = rng.standard_normal(n).astype(np.float32)
    inter = (a[:, None] * v[None, :]).astype(np.float32)
    up = inter @ W.T
    # Nearly-constant gate -> W* ≈ c W, z ≈ c up
    gate_act = np.full_like(up, 1.7)
    z = gate_act * up
    W_star = mean_gate_effective_weight(W, gate_act, pool="mean")
    lin = effective_linear_writes(W_star, inter)

    mounts = tile_svd_mounts(W_star, tile_size=32, modes_per_tile=1)
    summary = eval_strategy_on_writes(
        mounts,
        W=W_star,
        intermediate=inter,
        writes=z,
        linear_writes=lin,
    )
    assert summary["mean_full_write_lift"] > 0.25

    # Raw-W mounts scored on z (old protocol) should be weaker when gate is
    # structured - here gate is flat so compare vs isotropic score tensor.
    noise = rng.standard_normal(z.shape).astype(np.float32)
    noise *= float(np.linalg.norm(z)) / (float(np.linalg.norm(noise)) + 1e-12)
    bad = eval_strategy_on_writes(
        mounts,
        W=W_star,
        intermediate=inter,
        writes=noise,
        linear_writes=lin,
    )
    assert bad["mean_full_write_lift"] < summary["mean_full_write_lift"]


def test_mean_gate_sparse_gate_beats_raw_w_mounts():
    """Sparse gate: W* focuses SVD on open coords; raw W spreads across all rows."""
    rng = np.random.default_rng(3)
    d_out, d_in = 64, 64
    # Diffuse W: energy in all rows
    W = rng.standard_normal((d_out, d_in)).astype(np.float32)
    # Strong structured signal only in rows 0:8
    u = rng.standard_normal(8).astype(np.float32)
    u /= np.linalg.norm(u)
    v = rng.standard_normal(d_in).astype(np.float32)
    v /= np.linalg.norm(v)
    W[:8, :] += 5.0 * np.outer(u, v)

    n = 160
    a = rng.standard_normal(n).astype(np.float32)
    inter = (a[:, None] * v[None, :]).astype(np.float32)
    # Add isotropic input noise so closed rows also get energy in raw up
    inter = inter + 0.3 * rng.standard_normal(inter.shape).astype(np.float32)
    up = inter @ W.T
    gate_act = np.zeros_like(up)
    gate_act[:, :8] = 3.0
    z = gate_act * up

    W_star = mean_gate_effective_weight(W, gate_act, pool="mean")
    lin_star = effective_linear_writes(W_star, inter)
    mounts_star = tile_svd_mounts(W_star, tile_size=32, modes_per_tile=2)
    star = eval_strategy_on_writes(
        mounts_star,
        W=W_star,
        intermediate=inter,
        writes=z,
        linear_writes=lin_star,
    )

    mounts_raw = tile_svd_mounts(W, tile_size=32, modes_per_tile=2)
    raw = eval_strategy_on_writes(
        mounts_raw,
        W=W,
        intermediate=inter,
        writes=z,
        linear_writes=up,
    )
    assert star["mean_full_write_lift"] > raw["mean_full_write_lift"] + 0.05


def test_judge_go_with_strong_mean_gate_summary():
    """Sanity: strong lifts + coverage pass non-residual GO."""
    payload = {
        "layer": 12,
        "residual_write": False,
        "score_space": "gated_up",
        "mount_space": "mean_gate_up",
        "exp_a_chunking": {
            "strategies": [
                {
                    "strategy": "tile_svd",
                    "mean_full_write_lift": 0.12,
                    "frac_svd_identity_ok": 1.0,
                    "identity_vs_lift_corr": 0.0,
                },
                {"strategy": "whole_matrix_svd", "mean_full_write_lift": 0.08},
                {"strategy": "column_sample", "mean_full_write_lift": 0.02},
                {"strategy": "random_baseline", "mean_full_write_lift": 0.0},
            ]
        },
        "exp_b_coverage": {
            "rows": [
                {"write_coverage_lift": 0.22},
                {"write_coverage_lift": 0.28},
                {"write_coverage_lift": 0.30},
                {"write_coverage_lift": 0.29},
                {"write_coverage_lift": 0.28},
            ]
        },
        "exp_c_causal": {"skipped": True},
    }
    v = judge_paper_go(payload)
    assert v["go"] is True


def test_judge_effective_path_b1_floor_and_a1_margin():
    """Early-layer mean-gate peaks ~0.09-0.14 pass at effective floor 0.08."""
    payload = {
        "layer": 3,
        "residual_write": False,
        "score_space": "gated_up",
        "mount_space": "mean_gate_up",
        "exp_a_chunking": {
            "strategies": [
                {
                    "strategy": "tile_svd",
                    "mean_full_write_lift": 0.004,
                    "frac_svd_identity_ok": 1.0,
                    "identity_vs_lift_corr": 0.0,
                },
                {"strategy": "whole_matrix_svd", "mean_full_write_lift": 0.003},
                {"strategy": "column_sample", "mean_full_write_lift": 0.001},
                {"strategy": "random_baseline", "mean_full_write_lift": 0.0},
            ]
        },
        "exp_b_coverage": {
            "rows": [
                {"write_coverage_lift": 0.09},
                {"write_coverage_lift": 0.095},
                {"write_coverage_lift": 0.10},
                {"write_coverage_lift": 0.098},
                {"write_coverage_lift": 0.097},
            ]
        },
        "exp_c_causal": {"skipped": True},
    }
    v = judge_paper_go(payload)
    assert v["go"] is True
    b1 = next(c for c in v["checks"] if c["id"] == "B1_coverage_saturates_early")
    assert "floor=0.08" in b1["detail"]


def test_cluster_and_compose_shapes():
    from atlas.mount.effective import (
        cluster_gate_prototypes,
        composed_gate_down_weight,
        composed_v_o_weight,
        gate_pool_fallbacks,
        lstsq_effective_weight,
        mixture_tile_svd_mounts,
        v_pool_fallbacks,
    )

    rng = np.random.default_rng(0)
    gate = np.abs(rng.standard_normal((80, 16))).astype(np.float32)
    protos = cluster_gate_prototypes(gate, k=4)
    assert len(protos) == 4
    assert protos[0].shape == (16,)
    W_up = rng.standard_normal((16, 32)).astype(np.float32)
    W_down = rng.standard_normal((8, 16)).astype(np.float32)
    W_comp = composed_gate_down_weight(W_up, W_down, gate, pool="mean")
    assert W_comp.shape == (8, 32)
    mounts = mixture_tile_svd_mounts(
        W_up, gate, k=3, tile_size=16, modes_per_tile=1, n_target=12
    )
    # 2 input tiles × 1 mode × 3 protos = 6 (n_target only caps, doesn't invent)
    assert len(mounts) == 6
    assert all(m.direction.shape == (16,) for m in mounts)
    assert gate_pool_fallbacks()[0] == "mean"
    assert "mixture_k4" in gate_pool_fallbacks()
    assert "compose_down" in gate_pool_fallbacks()

    x = rng.standard_normal((50, 12)).astype(np.float32)
    # Planted linear map + noise
    W_true = rng.standard_normal((8, 12)).astype(np.float32)
    y = x @ W_true.T + 0.01 * rng.standard_normal((50, 8)).astype(np.float32)
    W_hat = lstsq_effective_weight(x, y, ridge=1e-4)
    assert W_hat.shape == (8, 12)
    assert float(np.mean((y - x @ W_hat.T) ** 2)) < 0.01

    W_v = rng.standard_normal((4 * 8, 16)).astype(np.float32)  # 4 kv heads
    W_o = rng.standard_normal((16, 8 * 8)).astype(np.float32)  # 8 q heads
    W_vo = composed_v_o_weight(
        W_v, W_o, num_heads=8, num_kv_heads=4, head_dim=8
    )
    assert W_vo.shape == (16, 16)
    assert v_pool_fallbacks() == ["lstsq", "compose_o"]


def test_judge_lstsq_v_uses_effective_floor():
    payload = {
        "layer": 12,
        "residual_write": False,
        "score_space": "mixed_v",
        "mount_space": "lstsq_mixed_v",
        "exp_a_chunking": {
            "strategies": [
                {
                    "strategy": "tile_svd",
                    "mean_full_write_lift": 0.05,
                    "frac_svd_identity_ok": 1.0,
                    "identity_vs_lift_corr": 0.0,
                },
                {"strategy": "whole_matrix_svd", "mean_full_write_lift": 0.02},
                {"strategy": "column_sample", "mean_full_write_lift": 0.01},
                {"strategy": "random_baseline", "mean_full_write_lift": 0.0},
            ]
        },
        "exp_b_coverage": {
            "rows": [
                {"write_coverage_lift": 0.09},
                {"write_coverage_lift": 0.10},
                {"write_coverage_lift": 0.11},
                {"write_coverage_lift": 0.105},
                {"write_coverage_lift": 0.10},
            ]
        },
        "exp_c_causal": {"skipped": True},
    }
    v = judge_paper_go(payload)
    assert v["go"] is True
    b1 = next(c for c in v["checks"] if c["id"] == "B1_coverage_saturates_early")
    assert "floor=0.08" in b1["detail"]
