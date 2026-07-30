"""Unit tests for linear site registry (no GPU)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from atlas.mount.paper_eval import judge_paper_go
from atlas.mount.sites import (
    ALL_LINEAR_SITES,
    PAPER_DEFAULT_SITES,
    RESIDUAL_WRITE_SITES,
    SITES,
    default_tile_size,
    get_site,
    min_coverage_peak,
    normalize_site_name,
    parse_sites,
    resolve_module,
    site_slug,
)


def _fake_gemma_layer():
    """Nested modules matching Gemma-style mlp.* / self_attn.* resolvers."""
    projs = {
        "down_proj": object(),
        "gate_proj": object(),
        "up_proj": object(),
        "o_proj": object(),
        "q_proj": object(),
        "k_proj": object(),
        "v_proj": object(),
    }
    mlp = SimpleNamespace(
        down_proj=projs["down_proj"],
        gate_proj=projs["gate_proj"],
        up_proj=projs["up_proj"],
    )
    self_attn = SimpleNamespace(
        o_proj=projs["o_proj"],
        q_proj=projs["q_proj"],
        k_proj=projs["k_proj"],
        v_proj=projs["v_proj"],
    )
    return SimpleNamespace(mlp=mlp, self_attn=self_attn), projs


def test_residual_defaults():
    assert PAPER_DEFAULT_SITES == ALL_LINEAR_SITES
    assert set(RESIDUAL_WRITE_SITES) == {"mlp.down", "attn.o"}
    assert all(get_site(s).residual_write for s in RESIDUAL_WRITE_SITES)
    assert len(ALL_LINEAR_SITES) == 7
    non_res = [s for s in ALL_LINEAR_SITES if s not in RESIDUAL_WRITE_SITES]
    assert len(non_res) == 5
    assert all(not get_site(s).residual_write for s in non_res)


def test_site_aware_tile_and_b1_floors():
    assert default_tile_size("mlp.down") == 512
    assert default_tile_size("mlp.gate") == 512
    assert default_tile_size("attn.q") == 128
    assert default_tile_size("attn.k") == 256
    assert default_tile_size("attn.v") == 256
    assert min_coverage_peak(True) == 0.25
    assert min_coverage_peak(False) == 0.15
    assert min_coverage_peak(False, mount_space="mean_gate_up") == 0.08
    assert min_coverage_peak(False, mount_space="compose_gate_down") == 0.08
    assert min_coverage_peak(False, mount_space="lstsq_mixed_v") == 0.08
    assert get_site("mlp.up").score_space == "gated_up"
    assert get_site("attn.v").score_space == "mixed_v"
    assert get_site("attn.v").mount_space == "lstsq_mixed_v"


def test_parse_sites_presets_and_aliases():
    assert parse_sites("residual") == ["mlp.down", "attn.o"]
    assert parse_sites("paper") == list(ALL_LINEAR_SITES)
    assert parse_sites("all") == list(ALL_LINEAR_SITES)
    assert parse_sites("mlp.down,attn.o") == ["mlp.down", "attn.o"]
    assert parse_sites("down,o") == ["mlp.down", "attn.o"]
    assert normalize_site_name("o_proj") == "attn.o"
    assert site_slug("mlp.down") == "mlp_down"
    assert site_slug("attn.o") == "attn_o"


def test_parse_sites_unknown():
    with pytest.raises(KeyError):
        parse_sites("mlp.foo")


def test_all_resolvers_on_mock_model():
    layer0, projs = _fake_gemma_layer()
    model = SimpleNamespace(model=SimpleNamespace(layers=[layer0]))
    expected = {
        "mlp.down": projs["down_proj"],
        "mlp.gate": projs["gate_proj"],
        "mlp.up": projs["up_proj"],
        "attn.o": projs["o_proj"],
        "attn.q": projs["q_proj"],
        "attn.k": projs["k_proj"],
        "attn.v": projs["v_proj"],
    }
    assert set(SITES) == set(expected)
    for name, mod in expected.items():
        assert resolve_module(model, 0, name) is mod
        assert resolve_module(model, 0, get_site(name)) is mod


def test_steer_resolvers_prefer_post_norm():
    """Gemma-2: C injects after post-attn / post-FF RMSNorm, not on o/down_proj."""
    from atlas.mount.sites import resolve_steer_module

    post_attn = object()
    post_ff = object()
    down = object()
    o = object()
    layer0 = SimpleNamespace(
        mlp=SimpleNamespace(down_proj=down),
        self_attn=SimpleNamespace(o_proj=o),
        post_attention_layernorm=post_attn,
        post_feedforward_layernorm=post_ff,
    )
    model = SimpleNamespace(model=SimpleNamespace(layers=[layer0]))
    assert resolve_steer_module(model, 0, "attn.o") is post_attn
    assert resolve_steer_module(model, 0, "mlp.down") is post_ff


def test_steer_resolvers_fallback_without_post_norm():
    from atlas.mount.sites import resolve_steer_module

    down = object()
    o = object()
    layer0 = SimpleNamespace(
        mlp=SimpleNamespace(down_proj=down),
        self_attn=SimpleNamespace(o_proj=o),
    )
    model = SimpleNamespace(model=SimpleNamespace(layers=[layer0]))
    assert resolve_steer_module(model, 0, "attn.o") is o
    assert resolve_steer_module(model, 0, "mlp.down") is down


def test_judge_skips_c_for_non_residual():
    payload = {
        "layer": 12,
        "site": "attn.q",
        "residual_write": False,
        "exp_a_chunking": {
            "strategies": [
                {
                    "strategy": "tile_svd",
                    "mean_full_write_lift": 0.04,
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
        "exp_c_causal": {"skipped": True, "reason": "non_residual_write"},
    }
    v = judge_paper_go(payload)
    assert v["go"] is True
    c1 = next(c for c in v["checks"] if c["id"] == "C1_steer_aligns_with_unembed")
    assert c1["skipped"] is True
    assert c1["required"] is False
    assert c1["pass"] is True


def test_judge_requires_c_for_attn_o_mid_layer():
    """attn.o is a residual write: C1 required when L >= 6."""
    base_a = {
        "strategies": [
            {
                "strategy": "tile_svd",
                "mean_full_write_lift": 0.04,
                "frac_svd_identity_ok": 1.0,
                "identity_vs_lift_corr": 0.0,
            },
            {"strategy": "whole_matrix_svd", "mean_full_write_lift": 0.02},
            {"strategy": "column_sample", "mean_full_write_lift": 0.01},
            {"strategy": "random_baseline", "mean_full_write_lift": 0.0},
        ]
    }
    base_b = {
        "rows": [
            {"write_coverage_lift": 0.48},
            {"write_coverage_lift": 0.50},
            {"write_coverage_lift": 0.49},
        ]
    }
    weak_c = {"mean_spearman": 0.0, "mean_topk20_jaccard": 0.0, "inject_at": "attn.o"}
    fail = {
        "layer": 12,
        "site": "attn.o",
        "residual_write": True,
        "exp_a_chunking": base_a,
        "exp_b_coverage": base_b,
        "exp_c_causal": weak_c,
    }
    v = judge_paper_go(fail)
    assert v["go"] is False
    c1 = next(c for c in v["checks"] if c["id"] == "C1_steer_aligns_with_unembed")
    assert c1["required"] is True
    assert c1["pass"] is False

    ok = {
        **fail,
        "exp_c_causal": {
            "mean_spearman": 0.2,
            "mean_topk20_jaccard": 0.1,
            "inject_at": "attn.o",
        },
    }
    v2 = judge_paper_go(ok)
    assert v2["go"] is True
    c1b = next(c for c in v2["checks"] if c["id"] == "C1_steer_aligns_with_unembed")
    assert c1b["required"] is True
    assert c1b["pass"] is True
