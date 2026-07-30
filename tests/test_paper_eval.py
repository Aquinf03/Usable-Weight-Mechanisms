"""Unit tests for paper_eval (no GPU)."""

from __future__ import annotations

import numpy as np
import pytest

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


def test_b1_passes_few_tile_creeping_coverage():
    """attn.o-style: peak late but m=1/2 already within 0.08 of peak."""
    payload = {
        "layer": 0,
        "residual_write": True,
        "exp_a_chunking": {
            "strategies": [
                {
                    "strategy": "tile_svd",
                    "mean_full_write_lift": 0.14,
                    "frac_svd_identity_ok": 1.0,
                    "identity_vs_lift_corr": 0.0,
                },
                {"strategy": "whole_matrix_svd", "mean_full_write_lift": 0.07},
                {"strategy": "column_sample", "mean_full_write_lift": 0.005},
                {"strategy": "random_baseline", "mean_full_write_lift": 0.0},
            ]
        },
        "exp_b_coverage": {
            "rows": [
                {"write_coverage_lift": 0.467},
                {"write_coverage_lift": 0.554},
                {"write_coverage_lift": 0.604},
                {"write_coverage_lift": 0.615},
                {"write_coverage_lift": 0.616},
            ]
        },
        "exp_c_causal": {"mean_spearman": -0.05, "mean_topk20_jaccard": 0.0},
    }
    v = judge_paper_go(payload)
    b1 = next(c for c in v["checks"] if c["id"] == "B1_coverage_saturates_early")
    assert b1["pass"] is True
    assert v["go"] is True  # L0 -> C1 waived


def test_b1_non_residual_uses_lower_peak_floor():
    """attn.q/k-style: peak ~0.19 fails residual floor 0.25 but passes 0.15."""
    payload = {
        "layer": 14,
        "residual_write": False,
        "exp_a_chunking": {
            "strategies": [
                {
                    "strategy": "tile_svd",
                    "mean_full_write_lift": 0.03,
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
                {"write_coverage_lift": 0.111},
                {"write_coverage_lift": 0.180},
                {"write_coverage_lift": 0.192},
                {"write_coverage_lift": 0.190},
                {"write_coverage_lift": 0.188},
            ]
        },
        "exp_c_causal": {"skipped": True},
    }
    v = judge_paper_go(payload)
    b1 = next(c for c in v["checks"] if c["id"] == "B1_coverage_saturates_early")
    assert b1["pass"] is True
    assert "floor=0.15" in b1["detail"]
    assert v["go"] is True


def test_b1_allows_mild_decline_after_early_peak():
    """attn.k + finer tiles: peak at m=1, then ~0.03-0.04 dilution by m=16."""
    payload = {
        "layer": 0,
        "residual_write": False,
        "exp_a_chunking": {
            "strategies": [
                {
                    "strategy": "tile_svd",
                    "mean_full_write_lift": 0.25,
                    "frac_svd_identity_ok": 1.0,
                    "identity_vs_lift_corr": 0.0,
                },
                {"strategy": "whole_matrix_svd", "mean_full_write_lift": 0.04},
                {"strategy": "column_sample", "mean_full_write_lift": 0.05},
                {"strategy": "random_baseline", "mean_full_write_lift": 0.0},
            ]
        },
        "exp_b_coverage": {
            "rows": [
                {"write_coverage_lift": 0.555},
                {"write_coverage_lift": 0.550},
                {"write_coverage_lift": 0.543},
                {"write_coverage_lift": 0.530},
                {"write_coverage_lift": 0.522},
            ]
        },
        "exp_c_causal": {"skipped": True},
    }
    v = judge_paper_go(payload)
    b1 = next(c for c in v["checks"] if c["id"] == "B1_coverage_saturates_early")
    assert b1["pass"] is True
    assert v["go"] is True


def test_b1_non_residual_early_window_includes_m4():
    """attn.q L8-style: peak late but m=4 already within 0.08 of peak."""
    payload = {
        "layer": 8,
        "residual_write": False,
        "exp_a_chunking": {
            "strategies": [
                {
                    "strategy": "tile_svd",
                    "mean_full_write_lift": 0.0152,
                    "frac_svd_identity_ok": 1.0,
                    "identity_vs_lift_corr": 0.0,
                },
                {"strategy": "whole_matrix_svd", "mean_full_write_lift": 0.0173},
                {"strategy": "column_sample", "mean_full_write_lift": 0.0033},
                {"strategy": "random_baseline", "mean_full_write_lift": 0.0},
            ]
        },
        "exp_b_coverage": {
            "rows": [
                {"write_coverage_lift": 0.047},
                {"write_coverage_lift": 0.116},
                {"write_coverage_lift": 0.159},
                {"write_coverage_lift": 0.202},
                {"write_coverage_lift": 0.213},
            ]
        },
        "exp_c_causal": {"skipped": True},
    }
    v = judge_paper_go(payload)
    b1 = next(c for c in v["checks"] if c["id"] == "B1_coverage_saturates_early")
    assert b1["pass"] is True
    assert "early_window=3" in b1["detail"]
    assert v["go"] is True


def test_a4_non_residual_allows_small_absolute_gap():
    """attn.q L22 @ tile=64: 0.0074 vs 0.0100 should not false-fail A4."""
    payload = {
        "layer": 22,
        "residual_write": False,
        "exp_a_chunking": {
            "strategies": [
                {
                    "strategy": "tile_svd",
                    "mean_full_write_lift": 0.0074,
                    "frac_svd_identity_ok": 1.0,
                    "identity_vs_lift_corr": 0.0,
                },
                {"strategy": "whole_matrix_svd", "mean_full_write_lift": 0.0100},
                {"strategy": "column_sample", "mean_full_write_lift": 0.0060},
                {"strategy": "random_baseline", "mean_full_write_lift": 0.0},
            ]
        },
        "exp_b_coverage": {
            "rows": [
                {"write_coverage_lift": 0.130},
                {"write_coverage_lift": 0.230},
                {"write_coverage_lift": 0.298},
                {"write_coverage_lift": 0.324},
                {"write_coverage_lift": 0.327},
            ]
        },
        "exp_c_causal": {"skipped": True},
    }
    v = judge_paper_go(payload)
    a4 = next(c for c in v["checks"] if c["id"] == "A4_tile_beats_or_ties_whole_full_write")
    assert a4["pass"] is True
    assert v["go"] is True


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


def test_causal_steer_injects_on_site_module():
    """Site-local C hooks inject_module, not the full decoder block."""
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from atlas.mount.paper_eval import causal_steer_alignment

    class HookableLinear(torch.nn.Module):
        def __init__(self, d: int):
            super().__init__()
            self.d = d

        def forward(self, x):
            return x

    class TinyLM(torch.nn.Module):
        def __init__(self, d: int, vocab: int):
            super().__init__()
            self.site = HookableLinear(d)
            self.block = HookableLinear(d)
            self.lm_head = torch.nn.Linear(d, vocab, bias=False)
            # Legacy fallback path uses model.model.layers[layer]
            self.model = SimpleNamespace(layers=[self.block])

        def forward(self, input_ids=None, **_kw):
            # Ignore tokens; produce a fixed residual then logits.
            b, t = input_ids.shape
            h = torch.zeros(b, t, self.site.d)
            h = self.site(h)
            h = self.block(h)
            logits = self.lm_head(h)
            return SimpleNamespace(logits=logits)

    class FakeTok:
        def __call__(self, text, return_tensors="pt"):
            return {"input_ids": torch.tensor([[1, 2, 3]])}

    d, vocab = 8, 16
    model = TinyLM(d, vocab)
    model.eval()
    u = np.zeros(d, dtype=np.float64)
    u[0] = 1.0
    lm_head = model.lm_head.weight.detach().float().cpu().numpy()

    row = causal_steer_alignment(
        model,
        FakeTok(),
        layer=0,
        direction=u,
        texts=["hi"],
        device="cpu",
        lm_head=lm_head,
        rms_weight=None,
        alpha=2.0,
        max_texts=1,
        inject_module=model.site,
        inject_at="attn.o",
    )
    assert row["inject_at"] == "attn.o"
    assert row["n_texts"] == 1
    # Steered last-token logits should move along lm_head @ u
    assert abs(row["spearman_vs_unembed"]) > 0.5


def test_causal_steer_legacy_block_fallback():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from atlas.mount.paper_eval import causal_steer_alignment

    class Hookable(torch.nn.Module):
        def forward(self, x):
            return x

    class TinyLM(torch.nn.Module):
        def __init__(self, d: int, vocab: int):
            super().__init__()
            self.block = Hookable()
            self.lm_head = torch.nn.Linear(d, vocab, bias=False)
            self.model = SimpleNamespace(layers=[self.block])
            self.d = d

        def forward(self, input_ids=None, **_kw):
            b, t = input_ids.shape
            h = torch.zeros(b, t, self.d)
            h = self.block(h)
            return SimpleNamespace(logits=self.lm_head(h))

    class FakeTok:
        def __call__(self, text, return_tensors="pt"):
            return {"input_ids": torch.tensor([[1, 2, 3]])}

    d, vocab = 8, 16
    model = TinyLM(d, vocab)
    model.eval()
    u = np.zeros(d)
    u[0] = 1.0
    row = causal_steer_alignment(
        model,
        FakeTok(),
        layer=0,
        direction=u,
        texts=["hi"],
        device="cpu",
        lm_head=model.lm_head.weight.detach().float().cpu().numpy(),
        rms_weight=None,
        alpha=2.0,
        max_texts=1,
    )
    assert row["inject_at"] == "block:L0"
    assert abs(row["spearman_vs_unembed"]) > 0.5
