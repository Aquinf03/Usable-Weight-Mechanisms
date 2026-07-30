"""Shared helpers for paper experiment scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from atlas.mount.sites import SiteSpec, get_site, resolve_module


def parse_layers(spec: str, *, n_layers: int | None = None) -> list[int]:
    """Parse '12', '0,6,12', '0-25', or 'all' (defaults to 26 for Gemma-2-2B)."""
    raw = spec.strip().lower()
    if raw == "all":
        n = n_layers if n_layers is not None else 26
        return list(range(n))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def load_texts(path: Path | None, default: list[str]) -> list[str]:
    """Load probe texts from JSONL ({text|content|body}) or raw lines."""
    if path is None or not path.exists():
        return list(default)
    texts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            texts.append(line.strip())
            continue
        if isinstance(obj, str):
            texts.append(obj)
            continue
        texts.append(
            (obj.get("text") or obj.get("content") or obj.get("body") or "").strip()
        )
    return [t for t in texts if t]


def collect_linear_io(model, tokenizer, texts: list[str], module, device: str):
    """Capture input/output token rows for a linear module (W @ x)."""
    import torch

    inputs: list = []
    outputs: list = []

    def hook(_mod, inp, out):
        x = inp[0]
        inputs.append(x.detach().float().reshape(-1, x.shape[-1]).cpu())
        outputs.append(out.detach().float().reshape(-1, out.shape[-1]).cpu())

    handle = module.register_forward_hook(hook)
    try:
        for text in texts:
            toks = tokenizer(text, return_tensors="pt")
            toks = {k: v.to(device) for k, v in toks.items()}
            with torch.no_grad():
                model(**toks)
    finally:
        handle.remove()

    return torch.cat(inputs, dim=0).numpy(), torch.cat(outputs, dim=0).numpy()


def collect_gated_up_io(
    model, tokenizer, texts: list[str], layer: int, device: str
):
    """Inputs to up_proj; score writes = act(gate) ⊙ up; also residual down writes.

    Returns ``(x, z, up, gate_act, residual)`` where ``gate_act = act(gate)``
    (post-nonlinearity), ``z`` is the gated product, and ``residual`` is
    ``down_proj`` outputs (for compose-through-down mounts).
    """
    import torch

    mlp = model.model.layers[layer].mlp
    act_fn = mlp.act_fn
    gate_outs: list = []
    up_ins: list = []
    up_outs: list = []
    down_outs: list = []

    def gate_hook(_mod, _inp, out):
        gate_outs.append(out.detach().float().reshape(-1, out.shape[-1]).cpu())

    def up_hook(_mod, inp, out):
        x = inp[0]
        up_ins.append(x.detach().float().reshape(-1, x.shape[-1]).cpu())
        up_outs.append(out.detach().float().reshape(-1, out.shape[-1]).cpu())

    def down_hook(_mod, _inp, out):
        down_outs.append(out.detach().float().reshape(-1, out.shape[-1]).cpu())

    h_g = mlp.gate_proj.register_forward_hook(gate_hook)
    h_u = mlp.up_proj.register_forward_hook(up_hook)
    h_d = mlp.down_proj.register_forward_hook(down_hook)
    try:
        for text in texts:
            toks = tokenizer(text, return_tensors="pt")
            toks = {k: v.to(device) for k, v in toks.items()}
            with torch.no_grad():
                model(**toks)
    finally:
        h_g.remove()
        h_u.remove()
        h_d.remove()

    g = torch.cat(gate_outs, dim=0)
    u = torch.cat(up_outs, dim=0)
    x = torch.cat(up_ins, dim=0)
    residual = torch.cat(down_outs, dim=0)
    with torch.no_grad():
        gate_act = act_fn(g)
        gated = gate_act * u
    return (
        x.numpy(),
        gated.numpy(),
        u.numpy(),
        gate_act.numpy(),
        residual.numpy(),
    )


def gqa_average_to_kv_dim(
    o_proj_input: "Any",
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
):
    """Collapse attention output (q-head dim) into v_proj out-dim via GQA mean.

    o_proj input is (N, num_heads * head_dim). For each KV head, average the
    ``num_heads // num_kv_heads`` query-head slots that share that KV value.
    Result: (N, num_kv_heads * head_dim) - same space as v_proj writes.
    """
    import torch

    t = o_proj_input
    if not isinstance(t, torch.Tensor):
        t = torch.asarray(t)
    n_rep = num_heads // num_kv_heads
    if num_heads * head_dim != t.shape[-1]:
        raise ValueError(
            f"o_proj input dim {t.shape[-1]} != num_heads*head_dim "
            f"{num_heads * head_dim}"
        )
    return (
        t.reshape(-1, num_kv_heads, n_rep, head_dim)
        .mean(dim=2)
        .reshape(-1, num_kv_heads * head_dim)
    )


def collect_mixed_v_io(
    model, tokenizer, texts: list[str], layer: int, device: str
):
    """Inputs to v_proj; mixed-v scores; o_proj outputs for compose-o.

    Returns ``(x, mixed, v, o_writes, attn_cfg)`` where ``mixed`` is
    GQA-averaged o_proj inputs (v out-dim) and ``o_writes`` are o_proj outputs.
    """
    import torch

    attn = model.model.layers[layer].self_attn
    num_heads = int(attn.config.num_attention_heads)
    num_kv = int(
        getattr(attn.config, "num_key_value_heads", None) or num_heads
    )
    head_dim = int(
        getattr(attn, "head_dim", None)
        or (attn.config.hidden_size // num_heads)
    )
    attn_cfg = {
        "num_heads": num_heads,
        "num_kv_heads": num_kv,
        "head_dim": head_dim,
    }

    v_ins: list = []
    v_outs: list = []
    mixed: list = []
    o_outs: list = []

    def v_hook(_mod, inp, out):
        x = inp[0]
        v_ins.append(x.detach().float().reshape(-1, x.shape[-1]).cpu())
        v_outs.append(out.detach().float().reshape(-1, out.shape[-1]).cpu())

    def o_hook(_mod, inp, out):
        flat = inp[0].detach().float().reshape(-1, inp[0].shape[-1]).cpu()
        mixed.append(
            gqa_average_to_kv_dim(
                flat,
                num_heads=num_heads,
                num_kv_heads=num_kv,
                head_dim=head_dim,
            )
        )
        o_outs.append(out.detach().float().reshape(-1, out.shape[-1]).cpu())

    h_v = attn.v_proj.register_forward_hook(v_hook)
    h_o = attn.o_proj.register_forward_hook(o_hook)
    try:
        for text in texts:
            toks = tokenizer(text, return_tensors="pt")
            toks = {k: v.to(device) for k, v in toks.items()}
            with torch.no_grad():
                model(**toks)
    finally:
        h_v.remove()
        h_o.remove()

    return (
        torch.cat(v_ins, dim=0).numpy(),
        torch.cat(mixed, dim=0).numpy(),
        torch.cat(v_outs, dim=0).numpy(),
        torch.cat(o_outs, dim=0).numpy(),
        attn_cfg,
    )


def collect_site_io(
    model,
    tokenizer,
    texts: list[str],
    layer: int,
    site: str | SiteSpec,
    device: str,
):
    """Hook a named site.

    Returns ``(intermediate, score_writes, linear_writes, extras)``.
    ``extras`` may include ``gate_act`` for mean-gate effective mounts.
    """
    spec = site if isinstance(site, SiteSpec) else get_site(site)
    if spec.score_space == "gated_up":
        x, z, up, gate_act, residual = collect_gated_up_io(
            model, tokenizer, texts, layer, device
        )
        return x, z, up, {
            "gate_act": gate_act,
            "residual_writes": residual,
        }
    if spec.score_space == "mixed_v":
        x, mixed, v, o_writes, attn_cfg = collect_mixed_v_io(
            model, tokenizer, texts, layer, device
        )
        return x, mixed, v, {
            "o_writes": o_writes,
            "attn_cfg": attn_cfg,
        }
    inter, writes = collect_linear_io(
        model, tokenizer, texts, resolve_module(model, layer, spec), device
    )
    return inter, writes, writes, {}


def collect_mlp_intermediates_and_writes(
    model,
    tokenizer,
    texts: list[str],
    layer: int,
    device: str,
):
    """Hook mlp.down: intermediates (input) and residual writes (output)."""
    inter, writes, _lin, _ex = collect_site_io(
        model, tokenizer, texts, layer, "mlp.down", device
    )
    return inter, writes


def get_site_weight(model, layer: int, site: str | SiteSpec):
    """Return the weight tensor for a named site."""
    return resolve_module(model, layer, site).weight


def load_model(model_id: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available - using cpu")
        device = "cpu"
    print(f"Loading {model_id} on {device}...")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map=device if device == "cuda" else None
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device if device == "cuda" else None,
        )
    if device == "cpu":
        model = model.to(device)
    model.eval()
    return model, tokenizer, device
