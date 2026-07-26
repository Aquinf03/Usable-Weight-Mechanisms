"""Run paper GO/NO-GO experiments for tile-SVD mounts.

Exp A — chunking A/B (energy lift on real writes)
Exp B — coverage vs modes_per_tile
Exp C — causal steer Δlogits vs unembed(u)

Single layer:
  python scripts/run_paper_experiments.py --layer 12 --device cuda

All 26 Gemma-2-2B layers (load model once):
  python scripts/run_paper_experiments.py --layers all --device cuda \\
    --texts data/corpus/train.jsonl \\
    --out-dir data/eval/paper_experiments_all
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from mount_runtime import (  # noqa: E402
    collect_mlp_intermediates_and_writes,
    load_model,
    load_texts,
    parse_layers,
)
from atlas.mount.coverage import (  # noqa: E402
    mount_directions_from_weight,
    sparse_write_coverage,
    tile_svd_weight_coverage,
)
from atlas.mount.paper_eval import (  # noqa: E402
    build_strategy_mounts,
    causal_steer_alignment,
    eval_strategy_on_writes,
    judge_paper_go,
)
from atlas.mount.trigger import DEFAULT_TRIGGER_TEXTS  # noqa: E402


def parse_modes(spec: str) -> list[int]:
    return sorted({int(x.strip()) for x in spec.split(",") if x.strip()})


def _subsample(inter, writes, max_tokens: int):
    n_all = int(writes.shape[0])
    if max_tokens and n_all > max_tokens:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(n_all, size=max_tokens, replace=False))
        inter = np.asarray(inter, dtype=np.float32)[idx]
        writes = np.asarray(writes, dtype=np.float32)[idx]
        return inter, writes, n_all
    return (
        np.asarray(inter, dtype=np.float32),
        np.asarray(writes, dtype=np.float32),
        n_all,
    )


def run_one_layer(
    *,
    model,
    tokenizer,
    device: str,
    texts: list[str],
    layer: int,
    out_dir: Path,
    args,
    exps: set[str],
    lm_head: np.ndarray | None,
    rms,
) -> dict:
    W = model.model.layers[layer].mlp.down_proj.weight
    inter, writes = collect_mlp_intermediates_and_writes(
        model, tokenizer, texts, layer, device
    )
    inter, writes, n_all = _subsample(inter, writes, args.max_tokens)
    print(
        f"\n######## L{layer} mlp.down  W={tuple(W.shape)}  "
        f"tokens={writes.shape[0]}/{n_all}"
        f"{' (subsampled)' if writes.shape[0] < n_all else ''} ########"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {
        "model": args.model,
        "layer": layer,
        "site": "mlp.down",
        "n_tokens": int(writes.shape[0]),
        "n_tokens_full": n_all,
        "n_texts": len(texts),
    }

    if "A" in exps:
        print("=== Exp A: chunking A/B ===")
        tile_probe = build_strategy_mounts(
            W,
            "tile_svd",
            n_target=10**9,
            tile_size=args.tile_size,
            modes_per_tile=args.modes_per_tile,
        )
        n_target = len(tile_probe)
        print(f"  budget n_mounts={n_target}")
        strategies = []
        for name in ("tile_svd", "whole_matrix_svd", "column_sample", "random_baseline"):
            mounts = build_strategy_mounts(
                W,
                name,
                n_target=n_target,
                tile_size=args.tile_size,
                modes_per_tile=args.modes_per_tile,
            )
            summary = eval_strategy_on_writes(
                mounts, W=W, intermediate=inter, writes=writes
            )
            row = {"strategy": name, **{k: v for k, v in summary.items() if k != "scores"}}
            strategies.append(row)
            print(
                f"  {name:18} full_lift={row['mean_full_write_lift']:.4f} "
                f"tile_lift={row['mean_energy_lift']:.4f} "
                f"proven={row['proven_rate']:.2f}"
            )
        results["exp_a_chunking"] = {
            "n_target": n_target,
            "tile_size": args.tile_size,
            "modes_per_tile": args.modes_per_tile,
            "strategies": strategies,
        }
        csv_path = out_dir / "chunking.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            keys = [
                "strategy",
                "n_mounts",
                "mean_full_write_lift",
                "median_full_write_lift",
                "p90_full_write_lift",
                "mean_energy_lift",
                "median_energy_lift",
                "p90_energy_lift",
                "proven_rate",
                "frac_svd_identity_ok",
                "mean_tile_kept_r2",
                "identity_vs_lift_corr",
            ]
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in strategies:
                w.writerow({k: row.get(k) for k in keys})

    if "B" in exps:
        print("=== Exp B: coverage vs modes ===")
        rows = []
        for m in parse_modes(args.modes_sweep):
            w_cov = tile_svd_weight_coverage(
                W, tile_size=args.tile_size, modes_per_tile=m
            )
            dirs, _mounts = mount_directions_from_weight(
                W, tile_size=args.tile_size, modes_per_tile=m
            )
            write_cov = sparse_write_coverage(
                writes, dirs, k_active=min(8, dirs.shape[0])
            )
            rng = np.random.default_rng(0)
            rand = rng.standard_normal(dirs.shape)
            rand /= np.linalg.norm(rand, axis=1, keepdims=True) + 1e-12
            rand_cov = sparse_write_coverage(
                writes, rand, k_active=min(8, dirs.shape[0])
            )
            lift = float(write_cov["write_energy_fraction"]) - float(
                rand_cov["write_energy_fraction"]
            )
            row = {
                "modes_per_tile": m,
                "n_mounts": int(dirs.shape[0]),
                "weight_energy_fraction": w_cov.get("weight_energy_fraction"),
                "write_energy_fraction": write_cov.get("write_energy_fraction"),
                "write_energy_random": rand_cov.get("write_energy_fraction"),
                "write_coverage_lift": round(lift, 4),
            }
            rows.append(row)
            print(
                f"  m={m:2d} n={row['n_mounts']:4d} "
                f"lift={row['write_coverage_lift']:.3f}"
            )
        results["exp_b_coverage"] = {"rows": rows}
        with (out_dir / "coverage.csv").open("w", encoding="utf-8", newline="") as f:
            keys = list(rows[0].keys()) if rows else []
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in rows:
                w.writerow(row)

    if "C" in exps:
        print("=== Exp C: causal steer vs unembed ===")
        mounts = build_strategy_mounts(
            W,
            "tile_svd",
            n_target=10**9,
            tile_size=args.tile_size,
            modes_per_tile=args.modes_per_tile,
        )
        scored = eval_strategy_on_writes(
            mounts, W=W, intermediate=inter, writes=writes
        )
        paired = sorted(
            zip(mounts, scored["scores"]),
            key=lambda x: -float(x[1]["energy_lift_over_random"]),
        )[: args.n_steer]

        if lm_head is None:
            lm_head = model.lm_head.weight.detach().float().cpu().numpy()

        causal_rows = []
        for m, sc in paired:
            row = causal_steer_alignment(
                model,
                tokenizer,
                layer=layer,
                direction=m.direction,
                texts=texts,
                device=device,
                lm_head=lm_head,
                rms_weight=rms,
                alpha=args.steer_alpha,
                max_texts=min(8, len(texts)),
            )
            row.update(
                {
                    "mount_id": m.mount_id,
                    "energy_lift": sc["energy_lift_over_random"],
                    "singular_value": sc["singular_value"],
                }
            )
            causal_rows.append(row)
            print(
                f"  {m.mount_id:16} sp={row['spearman_vs_unembed']:.3f} "
                f"jac20={row['topk20_jaccard_vs_unembed']:.3f}"
            )
        mean_sp = float(np.mean([r["spearman_vs_unembed"] for r in causal_rows]))
        mean_jac = float(np.mean([r["topk20_jaccard_vs_unembed"] for r in causal_rows]))
        results["exp_c_causal"] = {
            "mean_spearman": round(mean_sp, 4),
            "mean_topk20_jaccard": round(mean_jac, 4),
            "alpha": args.steer_alpha,
            "n_mounts": len(causal_rows),
            "rows": causal_rows,
        }
        with (out_dir / "causal.jsonl").open("w", encoding="utf-8") as f:
            for row in causal_rows:
                f.write(json.dumps(row) + "\n")
        print(f"  mean spearman={mean_sp:.4f}")

    verdict = judge_paper_go(results)
    results["verdict"] = verdict
    (out_dir / "summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (out_dir / "verdict.json").write_text(
        json.dumps(verdict, indent=2), encoding="utf-8"
    )
    print(f"L{layer} VERDICT: {verdict['verdict']}  ({verdict['n_pass']}/{verdict['n_checks']})")
    return results


def _summary_row(results: dict) -> dict:
    by = {
        r["strategy"]: r
        for r in (results.get("exp_a_chunking") or {}).get("strategies") or []
    }
    tile = by.get("tile_svd") or {}
    whole = by.get("whole_matrix_svd") or {}
    cols = by.get("column_sample") or {}
    rand = by.get("random_baseline") or {}
    cov_rows = (results.get("exp_b_coverage") or {}).get("rows") or []
    peak = max((float(r.get("write_coverage_lift") or 0) for r in cov_rows), default=0.0)
    causal = results.get("exp_c_causal") or {}
    verdict = results.get("verdict") or {}
    return {
        "layer": results.get("layer"),
        "go": bool(verdict.get("go")),
        "tile_full_lift": tile.get("mean_full_write_lift"),
        "whole_full_lift": whole.get("mean_full_write_lift"),
        "cols_full_lift": cols.get("mean_full_write_lift"),
        "rand_full_lift": rand.get("mean_full_write_lift"),
        "coverage_peak": round(peak, 4),
        "mean_spearman": causal.get("mean_spearman"),
        "n_pass": verdict.get("n_pass"),
        "n_checks": verdict.get("n_checks"),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Paper GO/NO-GO experiments")
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Single layer (legacy). Ignored if --layers is set.",
    )
    parser.add_argument(
        "--layers",
        default=None,
        help="Comma/range/'all' e.g. all or 0-25 or 0,6,12,18,25",
    )
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--modes-per-tile", type=int, default=2)
    parser.add_argument("--modes-sweep", default="1,2,4,8,16")
    parser.add_argument("--n-steer", type=int, default=8)
    parser.add_argument("--steer-alpha", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--texts", type=Path, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "eval" / "paper_experiments",
    )
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--experiments", default="A,B,C")
    args = parser.parse_args(argv)
    exps = {e.strip().upper() for e in args.experiments.split(",") if e.strip()}

    if args.layers:
        n_layers = None
        # gemma-2-2b default inside parse_layers('all')
        layers = parse_layers(args.layers, n_layers=26 if args.layers.strip().lower() == "all" else None)
    elif args.layer is not None:
        layers = [args.layer]
    else:
        layers = [12]

    texts = load_texts(args.texts, DEFAULT_TRIGGER_TEXTS)
    model, tokenizer, device = load_model(args.model, args.device)
    lm_head = model.lm_head.weight.detach().float().cpu().numpy()
    rms = None
    if hasattr(model.model, "norm") and hasattr(model.model.norm, "weight"):
        rms = model.model.norm.weight.detach().float().cpu().numpy()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    aggregate_rows = []
    n_go = 0

    for layer in layers:
        layer_dir = (
            args.out_dir / f"L{layer}" if len(layers) > 1 else args.out_dir
        )
        results = run_one_layer(
            model=model,
            tokenizer=tokenizer,
            device=device,
            texts=texts,
            layer=layer,
            out_dir=layer_dir,
            args=args,
            exps=exps,
            lm_head=lm_head,
            rms=rms,
        )
        row = _summary_row(results)
        aggregate_rows.append(row)
        if row["go"]:
            n_go += 1

    if len(layers) > 1:
        agg_csv = args.out_dir / "layers.csv"
        with agg_csv.open("w", encoding="utf-8", newline="") as f:
            keys = list(aggregate_rows[0].keys())
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in aggregate_rows:
                w.writerow(row)
        agg = {
            "model": args.model,
            "layers": layers,
            "n_layers": len(layers),
            "n_go": n_go,
            "go_rate": round(n_go / max(len(layers), 1), 4),
            "rows": aggregate_rows,
            "verdict": (
                f"GO on {n_go}/{len(layers)} layers"
                if n_go == len(layers)
                else f"PARTIAL — GO on {n_go}/{len(layers)} layers"
            ),
        }
        (args.out_dir / "aggregate_verdict.json").write_text(
            json.dumps(agg, indent=2), encoding="utf-8"
        )
        print("\n========== ALL LAYERS ==========")
        print(agg["verdict"])
        print(f"Wrote {agg_csv}")
        print(f"Wrote {args.out_dir / 'aggregate_verdict.json'}")


if __name__ == "__main__":
    main()
