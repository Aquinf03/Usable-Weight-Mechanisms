"""Unit tests for paper_eval (no GPU)."""

from __future__ import annotations

import numpy as np

from atlas.mount.paper_eval import (
    build_strategy_mounts,
    eval_strategy_on_writes,
    judge_paper_go,
    spearman_corr,
    topk_jaccard,
)


def test_spearman_and_jaccard():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman_corr(a, b) > 0.99
    assert topk_jaccard(a, b, k=2) == 1.0


def test_build_strategies_same_budget():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((32, 64))
    tile = build_strategy_mounts(W, "tile_svd", n_target=8, tile_size=16, modes_per_tile=2)
    n = len(tile)
    whole = build_strategy_mounts(W, "whole_matrix_svd", n_target=n, tile_size=16)
    cols = build_strategy_mounts(W, "column_sample", n_target=n, tile_size=16)
    rand = build_strategy_mounts(W, "random_baseline", n_target=n, tile_size=16)
    assert len(whole) == n
    assert len(cols) == n
    assert len(rand) == n
    assert "col_start" in tile[0].meta
    assert "trigger_vector" in whole[0].meta


def test_eval_strategy_smoke():
    rng = np.random.default_rng(1)
    d_out, d_in = 16, 32
    W = rng.standard_normal((d_out, d_in))
    # make one strong rank-1 tile signal
    u = rng.standard_normal(d_out)
    u /= np.linalg.norm(u)
    v = rng.standard_normal(16)
    v /= np.linalg.norm(v)
    W[:, :16] += 5.0 * np.outer(u, v)
    mounts = build_strategy_mounts(
        W, "tile_svd", n_target=4, tile_size=16, modes_per_tile=2
    )
    n_tok = 20
    inter = rng.standard_normal((n_tok, d_in))
    # fire along v in first tile
    inter[:, :16] += 3.0 * v
    writes = inter @ W.T
    summary = eval_strategy_on_writes(mounts, W=W, intermediate=inter, writes=writes)
    assert summary["n_mounts"] == len(mounts)
    assert "mean_energy_lift" in summary


def test_judge_go_and_nogo():
    good = {
        "layer": 12,
        "exp_a_chunking": {
            "strategies": [
                {
                    "strategy": "tile_svd",
                    "mean_full_write_lift": 0.04,
                    "mean_energy_lift": 0.08,
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
                {"write_coverage_lift": 0.48},
                {"write_coverage_lift": 0.50},
                {"write_coverage_lift": 0.49},
            ]
        },
        "exp_c_causal": {"mean_spearman": 0.2, "mean_topk20_jaccard": 0.1},
    }
    v = judge_paper_go(good)
    assert v["go"] is True

    bad = {
        "layer": 12,
        "exp_a_chunking": {
            "strategies": [
                {
                    "strategy": "tile_svd",
                    "mean_full_write_lift": 0.0,
                    "mean_energy_lift": 0.0,
                    "frac_svd_identity_ok": 1.0,
                    "identity_vs_lift_corr": 0.0,
                },
                {"strategy": "whole_matrix_svd", "mean_full_write_lift": 0.1},
                {"strategy": "column_sample", "mean_full_write_lift": 0.05},
                {"strategy": "random_baseline", "mean_full_write_lift": 0.0},
            ]
        },
        "exp_b_coverage": {
            "rows": [{"write_coverage_lift": 0.1}, {"write_coverage_lift": 0.02}]
        },
        "exp_c_causal": {"mean_spearman": 0.0, "mean_topk20_jaccard": 0.0},
    }
    v2 = judge_paper_go(bad)
    assert v2["go"] is False


def test_judge_saturation_and_fair_columns():
    payload = {
        "layer": 12,
        "exp_a_chunking": {
            "strategies": [
                {
                    "strategy": "tile_svd",
                    "mean_full_write_lift": 0.03,
                    "mean_energy_lift": 0.03,
                    "frac_svd_identity_ok": 1.0,
                    "identity_vs_lift_corr": 0.0,
                },
                {"strategy": "whole_matrix_svd", "mean_full_write_lift": 0.01},
                {
                    "strategy": "column_sample",
                    "mean_full_write_lift": 0.02,
                    "mean_energy_lift": 0.999,
                },
                {"strategy": "random_baseline", "mean_full_write_lift": 0.0},
            ]
        },
        "exp_b_coverage": {
            "rows": [
                {"write_coverage_lift": 0.4818},
                {"write_coverage_lift": 0.4969},
                {"write_coverage_lift": 0.496},
                {"write_coverage_lift": 0.4905},
                {"write_coverage_lift": 0.4846},
            ]
        },
        "exp_c_causal": {"mean_spearman": 0.173, "mean_topk20_jaccard": 0.006},
    }
    v = judge_paper_go(payload)
    assert v["go"] is True
    assert any(c["id"] == "B1_coverage_saturates_early" and c["pass"] for c in v["checks"])


def test_early_layer_waives_causal():
    payload = {
        "layer": 0,
        "exp_a_chunking": {
            "strategies": [
                {
                    "strategy": "tile_svd",
                    "mean_full_write_lift": 0.12,
                    "frac_svd_identity_ok": 1.0,
                    "identity_vs_lift_corr": 0.0,
                },
                {"strategy": "whole_matrix_svd", "mean_full_write_lift": 0.01},
                {"strategy": "column_sample", "mean_full_write_lift": 0.005},
                {"strategy": "random_baseline", "mean_full_write_lift": 0.0},
            ]
        },
        "exp_b_coverage": {
            "rows": [
                {"write_coverage_lift": 0.28},
                {"write_coverage_lift": 0.30},
                {"write_coverage_lift": 0.29},
            ]
        },
        "exp_c_causal": {"mean_spearman": -0.01, "mean_topk20_jaccard": 0.0},
    }
    v = judge_paper_go(payload)
    assert v["go"] is True
    c1 = next(c for c in v["checks"] if c["id"] == "C1_steer_aligns_with_unembed")
    assert c1["required"] is False
    assert c1["observed"] is False
    assert c1["pass"] is True
