"""Shared helpers for paper experiment scripts."""

from __future__ import annotations

import json
from pathlib import Path


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


def collect_mlp_intermediates_and_writes(
    model,
    tokenizer,
    texts: list[str],
    layer: int,
    device: str,
):
    """Hook mlp.down: intermediates (input) and residual writes (output)."""
    mod = model.model.layers[layer].mlp.down_proj
    return collect_linear_io(model, tokenizer, texts, mod, device)


def load_model(model_id: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available — using cpu")
        device = "cpu"
    print(f"Loading {model_id} on {device} …")
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
