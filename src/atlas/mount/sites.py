"""Linear sites in Gemma-style decoder layers.

Paper suite default: **all seven** linear maps per layer.
- Residual writes (`mlp.down`, `attn.o`): full A/B + depth-aware C
  (steer after Gemma-2 post-sublayer RMSNorm).
- Other maps (gate/up, q/k/v): A/B only; Experiment C skipped.
- ``mlp.up``: tile-SVD on mean-gate effective
  ``W*=diag(mean act(gate)) @ W_up``; score on gated product.
- ``attn.v``: tile-SVD on lstsq effective map ``x->mixed_v`` (compose-through-o
  fallback); score on mixed-v / o writes.

Use ``--sites residual`` to restrict to residual writes only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable


ModuleResolver = Callable[[object, int], object]


@dataclass(frozen=True)
class SiteSpec:
    """One linear map inside a transformer block."""

    name: str
    resolver: ModuleResolver
    residual_write: bool
    family: str
    description: str
    # Where Experiment C adds αu so Δresidual ≈ αu (Gemma-2 has post-sublayer RMSNorm).
    steer_resolver: ModuleResolver | None = None
    # What tensor A/B/coverage score against.
    #   module   - raw module outputs (default)
    #   gated_up - act(gate) ⊙ up  (mlp.up)
    #   mixed_v  - attention-mixed values in v_proj out dim (attn.v)
    score_space: str = "module"
    # Which matrix tile-SVD mounts are built from.
    #   module          - weight of ``resolver`` (default)
    #   mean_gate_up    - diag(mean act(gate)) @ W_up  (mlp.up)
    #   lstsq_mixed_v   - ridge lstsq map x -> mixed_v  (attn.v)
    mount_space: str = "module"


def _mlp_down(model, layer: int):
    return model.model.layers[layer].mlp.down_proj


def _mlp_gate(model, layer: int):
    return model.model.layers[layer].mlp.gate_proj


def _mlp_up(model, layer: int):
    return model.model.layers[layer].mlp.up_proj


def _attn_o(model, layer: int):
    return model.model.layers[layer].self_attn.o_proj


def _attn_q(model, layer: int):
    return model.model.layers[layer].self_attn.q_proj


def _attn_k(model, layer: int):
    return model.model.layers[layer].self_attn.k_proj


def _attn_v(model, layer: int):
    return model.model.layers[layer].self_attn.v_proj


def _steer_mlp_down(model, layer: int):
    """Hook post-FF RMSNorm output (Gemma-2) so αu hits residual, not pre-norm."""
    block = model.model.layers[layer]
    if hasattr(block, "post_feedforward_layernorm"):
        return block.post_feedforward_layernorm
    return block.mlp.down_proj


def _steer_attn_o(model, layer: int):
    """Hook post-attention RMSNorm output (Gemma-2) so αu hits residual, not pre-norm."""
    block = model.model.layers[layer]
    if hasattr(block, "post_attention_layernorm"):
        return block.post_attention_layernorm
    return block.self_attn.o_proj


SITES: dict[str, SiteSpec] = {
    "mlp.down": SiteSpec(
        name="mlp.down",
        resolver=_mlp_down,
        residual_write=True,
        family="mlp",
        description="MLP down_proj: residual write Δh = W x (post gate×up)",
        steer_resolver=_steer_mlp_down,
    ),
    "attn.o": SiteSpec(
        name="attn.o",
        resolver=_attn_o,
        residual_write=True,
        family="attn",
        description="Attention o_proj: residual write of head outputs",
        steer_resolver=_steer_attn_o,
    ),
    "mlp.gate": SiteSpec(
        name="mlp.gate",
        resolver=_mlp_gate,
        residual_write=False,
        family="mlp",
        description="MLP gate_proj (pre-nonlinearity; not a residual write)",
    ),
    "mlp.up": SiteSpec(
        name="mlp.up",
        resolver=_mlp_up,
        residual_write=False,
        family="mlp",
        description=(
            "MLP up: tile-SVD on mean-gate effective W*=diag(ḡ)W_up; "
            "A/B scored on gated product act(gate)⊙up"
        ),
        score_space="gated_up",
        mount_space="mean_gate_up",
    ),
    "attn.q": SiteSpec(
        name="attn.q",
        resolver=_attn_q,
        residual_write=False,
        family="attn",
        description="Attention q_proj (writes into attention internals)",
    ),
    "attn.k": SiteSpec(
        name="attn.k",
        resolver=_attn_k,
        residual_write=False,
        family="attn",
        description="Attention k_proj (writes into attention internals)",
    ),
    "attn.v": SiteSpec(
        name="attn.v",
        resolver=_attn_v,
        residual_write=False,
        family="attn",
        description=(
            "Attention v: tile-SVD on lstsq effective x->mixed_v "
            "(compose-through-o fallback); score mixed-v / o writes"
        ),
        score_space="mixed_v",
        mount_space="lstsq_mixed_v",
    ),
}

# Aliases used in CLI / paper prose
SITE_ALIASES: dict[str, str] = {
    "down": "mlp.down",
    "mlp_down": "mlp.down",
    "o": "attn.o",
    "attn_o": "attn.o",
    "o_proj": "attn.o",
    "gate": "mlp.gate",
    "up": "mlp.up",
    "q": "attn.q",
    "k": "attn.k",
    "v": "attn.v",
}

RESIDUAL_WRITE_SITES: tuple[str, ...] = tuple(
    name for name, spec in SITES.items() if spec.residual_write
)
ALL_LINEAR_SITES: tuple[str, ...] = tuple(SITES.keys())
# Paper default: every linear map; C is residual-only in the judge.
PAPER_DEFAULT_SITES: tuple[str, ...] = ALL_LINEAR_SITES


def normalize_site_name(name: str) -> str:
    raw = name.strip().lower().replace(" ", "")
    if raw in SITE_ALIASES:
        return SITE_ALIASES[raw]
    if raw in SITES:
        return raw
    raise KeyError(
        f"unknown site {name!r}; known={list(SITES)} aliases={list(SITE_ALIASES)}"
    )


def get_site(name: str) -> SiteSpec:
    return SITES[normalize_site_name(name)]


def default_tile_size(site: str) -> int:
    """Site-aware input-tile width for tile-SVD mounts.

    Residual / MLP maps use 512. ``attn.k``/``attn.v`` use 256. ``attn.q`` uses
    128: mid/late Q layers can be diffuse enough that 256-wide tiles lose to
    whole-matrix SVD (A4) and saturate coverage late (B1).
    """
    name = normalize_site_name(site)
    if name == "attn.q":
        return 128
    if name in ("attn.k", "attn.v"):
        return 256
    return 512


def tile_size_fallbacks(site: str, primary: int) -> list[int]:
    """Smaller tiles to try if the primary width is NO-GO."""
    name = normalize_site_name(site)
    if name == "attn.q":
        return [t for t in (128, 64) if t < primary]
    if name in ("mlp.up", "attn.v"):
        return [t for t in (256, 128, 64) if t < primary]
    return []


def min_coverage_peak(
    residual_write: bool,
    *,
    mount_space: str | None = None,
) -> float:
    """B1 absolute coverage peak required for GO.

    Residual writes: 0.25 (strong residual energy explained).
    Other linear maps: 0.15 - Q/K/gate live in non-residual spaces where
    absolute explainable energy is lower; saturation (early ≈ peak) still required.
    Effective mounts (``mean_gate_up`` / compose-down / ``lstsq_mixed_v`` /
    compose-o): **0.08** - corpus effective maps explain a thinner slice than
    raw residual writes; saturation still required.
    """
    if residual_write:
        return 0.25
    ms = (mount_space or "").strip().lower()
    if ms in {
        "mean_gate_up",
        "mixture_gate_up",
        "compose_gate_down",
        "lstsq_mixed_v",
        "compose_v_o",
    }:
        return 0.08
    return 0.15


def b1_early_window(residual_write: bool) -> int:
    """How many leading modes-sweep points count as 'early' for B1.

    Residual: first 2 (m=1,2). Non-residual Q/K-style spaces: first 3 (m=1,2,4)
    - coverage can need ~4 modes/tile before the plateau, without late m=16 growth.
    """
    return 2 if residual_write else 3



def parse_sites(spec: str | None, *, default: Iterable[str] | None = None) -> list[str]:
    """Parse site list.

    Accepts comma-separated names, or presets:
      all | all_linear | paper | default  -> all 7 linear sites
      residual | residual_writes          -> mlp.down, attn.o
    """
    if spec is None or not str(spec).strip():
        base = list(default) if default is not None else list(PAPER_DEFAULT_SITES)
        return [normalize_site_name(s) for s in base]

    raw = str(spec).strip().lower()
    if raw in {"all", "all_linear", "linear", "paper", "default"}:
        return list(ALL_LINEAR_SITES)
    if raw in {"residual", "residual_writes"}:
        return list(RESIDUAL_WRITE_SITES)

    out: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(normalize_site_name(part))
    # preserve order, unique
    seen: set[str] = set()
    uniq: list[str] = []
    for name in out:
        if name not in seen:
            seen.add(name)
            uniq.append(name)
    if not uniq:
        raise ValueError("no sites parsed from --sites")
    return uniq


def resolve_module(model, layer: int, site: str | SiteSpec):
    spec = site if isinstance(site, SiteSpec) else get_site(site)
    return spec.resolver(model, layer)


def resolve_steer_module(model, layer: int, site: str | SiteSpec):
    """Module whose output is added into residual for Experiment C.

    On Gemma-2, linear site outputs pass through post-attn / post-FF RMSNorm
    before the residual add. Injecting on ``o_proj`` / ``down_proj`` therefore
    does *not* add αu to the stream. Prefer ``steer_resolver`` when set.
    """
    spec = site if isinstance(site, SiteSpec) else get_site(site)
    if spec.steer_resolver is not None:
        return spec.steer_resolver(model, layer)
    return spec.resolver(model, layer)


def site_slug(name: str) -> str:
    """Filesystem-safe site id (mlp.down -> mlp_down)."""
    return normalize_site_name(name).replace(".", "_")
