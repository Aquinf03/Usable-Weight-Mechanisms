"""Run paper GO/NO-GO experiments for tile-SVD mounts.

Exp A: chunking A/B (full-write energy lift)
Exp B: coverage vs modes_per_tile
Exp C: causal steer vs unembed(u) [residual-write sites only]

  python scripts/build_corpus.py --out data/corpus/train.jsonl
  python scripts/run_paper_experiments.py --layers all --sites all \
    --device cuda --texts data/corpus/train.jsonl \
    --out-dir data/eval/paper_experiments_all

Retry prior NO-GOs only:
  python scripts/run_paper_experiments.py --retry-failures DIR \
    --device cuda --texts data/corpus/train.jsonl

Rejudge summaries after a judge-only change (no GPU):
  python scripts/rejudge_paper_results.py DIR
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
sys.path.insert(0, str(ROOT / "src"))

from mount_runtime import (  # noqa: E402
    collect_site_io,
    get_site_weight,
    load_model,
    load_texts,
    parse_layers,
)
from atlas.mount.coverage import (  # noqa: E402
    mount_directions_from_weight,
    sparse_write_coverage,
    tile_svd_weight_coverage,
)
from atlas.mount.effective import (  # noqa: E402
    composed_gate_down_weight,
    composed_v_o_weight,
    effective_linear_writes,
    lstsq_effective_weight,
    mean_gate_effective_weight,
    mixture_tile_svd_mounts,
    parse_gate_pools,
    parse_v_pools,
)
from atlas.mount.paper_eval import (  # noqa: E402
    build_strategy_mounts,
    causal_steer_alignment,
    eval_strategy_on_writes,
    judge_paper_go,
)
from atlas.mount.sites import (  # noqa: E402
    PAPER_DEFAULT_SITES,
    default_tile_size,
    get_site,
    parse_sites,
    resolve_steer_module,
    site_slug,
    tile_size_fallbacks,
)
from atlas.mount.trigger import DEFAULT_TRIGGER_TEXTS  # noqa: E402


def _weight_shape(W) -> tuple[int, ...]:
    """Shape of a weight tensor without forcing a CUDA-to-host copy."""
    if hasattr(W, "shape"):
        return tuple(int(x) for x in W.shape)
    return tuple(np.asarray(W).shape)


def _mount_weight_and_score_io(
    spec,
    W_raw,
    extras: dict,
    inter,
    writes,
    pool: str,
    *,
    W_down=None,
    W_o=None,
):
    """Build mount matrix + score tensors for one effective-path trial.

    Returns ``(W, inter_s, writes_s, linear_s, mask, mount_space_label, mixture_k)``.
    """
    if spec.mount_space == "lstsq_mixed_v":
        if pool == "compose_o":
            if W_o is None:
                raise ValueError("compose_o requires W_o")
            cfg = extras.get("attn_cfg") or {}
            o_writes = extras.get("o_writes")
            if o_writes is None:
                raise ValueError("compose_o requires extras['o_writes']")
            W = composed_v_o_weight(
                W_raw,
                W_o,
                num_heads=int(cfg["num_heads"]),
                num_kv_heads=int(cfg["num_kv_heads"]),
                head_dim=int(cfg["head_dim"]),
            )
            return (
                W,
                inter,
                np.asarray(o_writes, dtype=np.float32),
                effective_linear_writes(W, inter),
                None,
                "compose_v_o",
                None,
            )
        # default / lstsq: corpus least-squares map x -> mixed_v
        W = lstsq_effective_weight(inter, writes)
        return (
            W,
            inter,
            writes,
            effective_linear_writes(W, inter),
            None,
            "lstsq_mixed_v",
            None,
        )

    if spec.mount_space != "mean_gate_up":
        return W_raw, inter, writes, writes, None, spec.mount_space, None
    gate_act = extras.get("gate_act")
    if gate_act is None:
        raise ValueError("mean_gate_up mount_space requires extras['gate_act']")

    mixture_k = None
    if pool.startswith("mixture_k"):
        mixture_k = int(pool.split("mixture_k", 1)[1] or "4")
        W = mean_gate_effective_weight(W_raw, gate_act, pool="mean")
        mount_label = "mixture_gate_up"
        score_writes = writes
    elif pool == "compose_down":
        if W_down is None:
            raise ValueError("compose_down requires W_down")
        residual = extras.get("residual_writes")
        if residual is None:
            raise ValueError("compose_down requires extras['residual_writes']")
        W = composed_gate_down_weight(W_raw, W_down, gate_act, pool="mean")
        mount_label = "compose_gate_down"
        score_writes = residual
        mixture_k = None
    else:
        W = mean_gate_effective_weight(W_raw, gate_act, pool=pool)
        mount_label = "mean_gate_up"
        score_writes = writes

    linear_s = effective_linear_writes(W, inter)
    return W, inter, score_writes, linear_s, None, mount_label, mixture_k


def parse_modes(spec: str) -> list[int]:
    return sorted({int(x.strip()) for x in spec.split(",") if x.strip()})


def _subsample(inter, writes, max_tokens: int, linear_writes=None, extras=None):
    extras = dict(extras or {})
    n_all = int(writes.shape[0])
    if max_tokens and n_all > max_tokens:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(n_all, size=max_tokens, replace=False))
        inter = np.asarray(inter, dtype=np.float32)[idx]
        writes = np.asarray(writes, dtype=np.float32)[idx]
        if linear_writes is not None:
            linear_writes = np.asarray(linear_writes, dtype=np.float32)[idx]
        for key in ("gate_act", "residual_writes", "o_writes"):
            if key in extras and extras[key] is not None:
                extras[key] = np.asarray(extras[key], dtype=np.float32)[idx]
        return inter, writes, linear_writes, extras, n_all
    return (
        np.asarray(inter, dtype=np.float32),
        np.asarray(writes, dtype=np.float32),
        None
        if linear_writes is None
        else np.asarray(linear_writes, dtype=np.float32),
        extras,
        n_all,
    )


def _run_exp_ab(
    *,
    results: dict,
    W,
    inter,
    writes,
    linear_writes,
    tile_size: int,
    args,
    out_dir: Path,
    exps: set[str],
    mixture_k: int | None = None,
    W_up_raw=None,
    gate_act=None,
) -> None:
    """Fill Exp A/B fields on ``results`` for one tile width."""
    if "A" in exps:
        print("=== Exp A: chunking A/B ===")
        if mixture_k is not None:
            if W_up_raw is None or gate_act is None:
                raise ValueError("mixture mounts need W_up_raw and gate_act")
            # Fair budget: same #mounts as mean-gate tile-SVD, not K× that.
            tile_probe = build_strategy_mounts(
                W,
                "tile_svd",
                n_target=10**9,
                tile_size=tile_size,
                modes_per_tile=args.modes_per_tile,
            )
            n_target = len(tile_probe)
        else:
            tile_probe = build_strategy_mounts(
                W,
                "tile_svd",
                n_target=10**9,
                tile_size=tile_size,
                modes_per_tile=args.modes_per_tile,
            )
            n_target = len(tile_probe)
        print(f"  budget n_mounts={n_target} tile_size={tile_size}")
        strategies = []
        for name in ("tile_svd", "whole_matrix_svd", "column_sample", "random_baseline"):
            if name == "tile_svd" and mixture_k is not None:
                mounts = mixture_tile_svd_mounts(
                    W_up_raw,
                    gate_act,
                    k=mixture_k,
                    tile_size=tile_size,
                    modes_per_tile=args.modes_per_tile,
                    n_target=n_target,
                )
            else:
                mounts = build_strategy_mounts(
                    W,
                    name,
                    n_target=n_target,
                    tile_size=tile_size,
                    modes_per_tile=args.modes_per_tile,
                )
            summary = eval_strategy_on_writes(
                mounts,
                W=W,
                intermediate=inter,
                writes=writes,
                linear_writes=linear_writes,
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
            "tile_size": tile_size,
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
            if mixture_k is not None:
                mounts = mixture_tile_svd_mounts(
                    W_up_raw,
                    gate_act,
                    k=mixture_k,
                    tile_size=tile_size,
                    modes_per_tile=m,
                    n_target=len(
                        build_strategy_mounts(
                            W,
                            "tile_svd",
                            n_target=10**9,
                            tile_size=tile_size,
                            modes_per_tile=m,
                        )
                    ),
                )
                dirs = np.stack([mm.direction for mm in mounts], axis=0)
                w_cov = {"weight_energy_fraction": None}
            else:
                w_cov = tile_svd_weight_coverage(
                    W, tile_size=tile_size, modes_per_tile=m
                )
                dirs, _mounts = mount_directions_from_weight(
                    W, tile_size=tile_size, modes_per_tile=m
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
        results["exp_b_coverage"] = {"rows": rows, "tile_size": tile_size}
        with (out_dir / "coverage.csv").open("w", encoding="utf-8", newline="") as f:
            keys = list(rows[0].keys()) if rows else []
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in rows:
                w.writerow(row)


def run_one_site_layer(
    *,
    model,
    tokenizer,
    device: str,
    texts: list[str],
    layer: int,
    site_name: str,
    out_dir: Path,
    args,
    exps: set[str],
    lm_head: np.ndarray | None,
    rms,
) -> dict:
    spec = get_site(site_name)
    W_raw = get_site_weight(model, layer, spec)
    inter, writes, linear_writes, extras = collect_site_io(
        model, tokenizer, texts, layer, spec, device
    )
    inter, writes, linear_writes, extras, n_all = _subsample(
        inter,
        writes,
        args.max_tokens,
        linear_writes=linear_writes,
        extras=extras,
    )
    primary = (
        int(args.tile_size)
        if args.tile_size is not None
        else default_tile_size(spec.name)
    )
    tile_candidates = [primary]
    if args.tile_size is None:
        tile_candidates.extend(tile_size_fallbacks(spec.name, primary))
    if spec.mount_space == "mean_gate_up":
        pool_candidates = parse_gate_pools(getattr(args, "gate_pools", None))
    elif spec.mount_space == "lstsq_mixed_v":
        # Reuse --gate-pools CLI for v effective-path trials when set.
        raw_pools = getattr(args, "gate_pools", None)
        pool_candidates = parse_v_pools(raw_pools)
    else:
        pool_candidates = ["mean"]

    W_down = None
    if spec.mount_space == "mean_gate_up" and "compose_down" in pool_candidates:
        W_down = get_site_weight(model, layer, "mlp.down")
    W_o = None
    if spec.mount_space == "lstsq_mixed_v" and "compose_o" in pool_candidates:
        W_o = get_site_weight(model, layer, "attn.o")

    print(
        f"\n######## L{layer} {spec.name}  W_raw={_weight_shape(W_raw)}  "
        f"tokens={writes.shape[0]}/{n_all}"
        f"{' (subsampled)' if writes.shape[0] < n_all else ''} "
        f"residual_write={spec.residual_write} "
        f"score_space={spec.score_space} mount_space={spec.mount_space} "
        f"gate_pools={pool_candidates} tile_candidates={tile_candidates} ########"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {
        "model": args.model,
        "layer": layer,
        "site": spec.name,
        "site_family": spec.family,
        "residual_write": spec.residual_write,
        "score_space": spec.score_space,
        "mount_space": spec.mount_space,
        "n_tokens": int(writes.shape[0]),
        "n_tokens_full": n_all,
        "n_texts": len(texts),
    }

    best: dict | None = None
    for pool in pool_candidates:
        (
            W,
            inter_s,
            writes_s,
            linear_s,
            _mask,
            mount_label,
            mixture_k,
        ) = _mount_weight_and_score_io(
            spec,
            W_raw,
            extras,
            inter,
            writes,
            pool,
            W_down=W_down,
            W_o=W_o,
        )
        n_score = int(writes_s.shape[0])
        if spec.mount_space in ("mean_gate_up", "lstsq_mixed_v"):
            print(
                f"\n=== gate_pool={pool}  score_tokens={n_score}/{writes.shape[0]}  "
                f"W_mount={_weight_shape(W)} mount_space={mount_label} ==="
            )
        for tile_size in tile_candidates:
            if len(tile_candidates) > 1:
                print(f"\n--- trying tile_size={tile_size} ---")
            trial = dict(results)
            trial["gate_pool"] = pool
            trial["n_score_tokens"] = n_score
            trial["mount_space"] = mount_label
            _run_exp_ab(
                results=trial,
                W=W,
                inter=inter_s,
                writes=writes_s,
                linear_writes=linear_s,
                tile_size=tile_size,
                args=args,
                out_dir=out_dir,
                exps=exps,
                mixture_k=mixture_k,
                W_up_raw=W_raw if mixture_k is not None else None,
                gate_act=extras.get("gate_act") if mixture_k is not None else None,
            )
            if "C" in exps and not spec.residual_write:
                trial["exp_c_causal"] = {
                    "skipped": True,
                    "reason": "non_residual_write",
                    "mean_spearman": None,
                    "mean_topk20_jaccard": None,
                    "n_mounts": 0,
                    "rows": [],
                }
            trial_verdict = judge_paper_go(trial)
            trial["verdict"] = trial_verdict
            trial["tile_size_used"] = tile_size
            trial["gate_pool_used"] = pool
            if best is None or int(trial_verdict.get("n_pass") or 0) >= int(
                (best.get("verdict") or {}).get("n_pass") or 0
            ):
                best = trial
            if trial_verdict.get("go"):
                break
        if best is not None and (best.get("verdict") or {}).get("go"):
            break

    assert best is not None
    results = best
    tile_size = int(results.get("tile_size_used") or primary)
    # Restore mount W / score tensors for optional Exp C from the winning trial.
    win_pool = str(results.get("gate_pool_used") or pool_candidates[0])
    W, inter, writes, linear_writes, _mask, _ml, _mk = _mount_weight_and_score_io(
        spec,
        W_raw,
        extras,
        inter,
        writes,
        win_pool,
        W_down=W_down,
        W_o=W_o,
    )

    if "C" in exps:
        if not spec.residual_write:
            print(
                "=== Exp C: SKIPPED "
                f"(site {spec.name} is not a residual write; "
                "steer↔unembed does not apply) ==="
            )
            results["exp_c_causal"] = {
                "skipped": True,
                "reason": "non_residual_write",
                "mean_spearman": None,
                "mean_topk20_jaccard": None,
                "n_mounts": 0,
                "rows": [],
            }
        else:
            print("=== Exp C: causal steer vs unembed ===")
            mounts = build_strategy_mounts(
                W,
                "tile_svd",
                n_target=10**9,
                tile_size=tile_size,
                modes_per_tile=args.modes_per_tile,
            )
            scored = eval_strategy_on_writes(
                mounts,
                W=W,
                intermediate=inter,
                writes=writes,
                linear_writes=linear_writes,
            )
            paired = sorted(
                zip(mounts, scored["scores"]),
                key=lambda x: -float(x[1]["energy_lift_over_random"]),
            )[: args.n_steer]

            if lm_head is None:
                lm_head = model.lm_head.weight.detach().float().cpu().numpy()

            steer_module = resolve_steer_module(model, layer, spec)
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
                    inject_module=steer_module,
                    inject_at=f"{spec.name}:post_norm",
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
            mean_jac = float(
                np.mean([r["topk20_jaccard_vs_unembed"] for r in causal_rows])
            )
            results["exp_c_causal"] = {
                "skipped": False,
                "inject_at": f"{spec.name}:post_norm",
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
    print(
        f"L{layer} {spec.name} VERDICT: {verdict['verdict']}  "
        f"({verdict['n_pass']}/{verdict['n_checks']})"
        f"  tile_size={tile_size}"
        f"  gate_pool={results.get('gate_pool_used')}"
    )
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
        "site": results.get("site"),
        "site_family": results.get("site_family"),
        "residual_write": results.get("residual_write"),
        "layer": results.get("layer"),
        "go": bool(verdict.get("go")),
        "tile_full_lift": tile.get("mean_full_write_lift"),
        "whole_full_lift": whole.get("mean_full_write_lift"),
        "cols_full_lift": cols.get("mean_full_write_lift"),
        "rand_full_lift": rand.get("mean_full_write_lift"),
        "coverage_peak": round(peak, 4),
        "mean_spearman": causal.get("mean_spearman"),
        "c_skipped": bool(causal.get("skipped")),
        "n_pass": verdict.get("n_pass"),
        "n_checks": verdict.get("n_checks"),
    }


def _out_dir_for(
    base: Path,
    *,
    site: str,
    layer: int,
    n_sites: int,
    n_layers: int,
    force_nested: bool = False,
) -> Path:
    """Keep legacy flat layout for single site+layer; nest otherwise."""
    if force_nested or (n_sites > 1 and n_layers > 1):
        return base / site_slug(site) / f"L{layer}"
    if n_sites == 1 and n_layers == 1:
        return base
    if n_sites == 1:
        return base / f"L{layer}"
    if n_layers == 1:
        return base / site_slug(site)
    return base / site_slug(site) / f"L{layer}"


def _truthy_go(val) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in {"1", "true", "yes"}


def _load_nogo_jobs(src: Path) -> list[tuple[str, int]]:
    """Load (site, layer) pairs that previously failed GO."""
    csv_path = src / "sites.csv"
    jobs: list[tuple[str, int]] = []
    if csv_path.exists():
        with csv_path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if _truthy_go(row.get("go")):
                    continue
                jobs.append((str(row["site"]), int(row["layer"])))
    else:
        agg_path = src / "aggregate_verdict.json"
        if not agg_path.exists():
            raise FileNotFoundError(
                f"need {csv_path} or {agg_path} under --retry-failures"
            )
        agg = json.loads(agg_path.read_text(encoding="utf-8"))
        for row in agg.get("rows") or []:
            if _truthy_go(row.get("go")):
                continue
            jobs.append((str(row["site"]), int(row["layer"])))
    # stable unique
    seen: set[tuple[str, int]] = set()
    out: list[tuple[str, int]] = []
    for job in jobs:
        if job in seen:
            continue
        seen.add(job)
        out.append(job)
    return out


def _load_existing_rows(out_dir: Path) -> list[dict]:
    csv_path = out_dir / "sites.csv"
    if not csv_path.exists():
        return []
    with csv_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if "layer" in r and r["layer"] not in (None, ""):
            r["layer"] = int(r["layer"])
        if "go" in r:
            r["go"] = _truthy_go(r["go"])
        if "residual_write" in r:
            r["residual_write"] = _truthy_go(r["residual_write"])
    return rows


def _merge_summary_rows(old: list[dict], new: list[dict]) -> list[dict]:
    def key(r: dict) -> tuple[str, int]:
        return (str(r["site"]), int(r["layer"]))

    by = {key(r): dict(r) for r in old}
    for r in new:
        by[key(r)] = dict(r)
    return sorted(by.values(), key=lambda r: (str(r["site"]), int(r["layer"])))


def _write_aggregate(
    out_dir: Path,
    aggregate_rows: list[dict],
    *,
    model: str,
    sites: list[str],
    layers: list[int],
) -> dict:
    n_jobs = len(aggregate_rows)
    n_go = sum(1 for r in aggregate_rows if r.get("go"))
    agg_csv = out_dir / "sites.csv"
    if aggregate_rows:
        keys: list[str] = []
        seen: set[str] = set()
        for row in aggregate_rows:
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        with agg_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for row in aggregate_rows:
                w.writerow({k: row.get(k) for k in keys})
        if len(sites) == 1:
            layers_csv = out_dir / "layers.csv"
            with layers_csv.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                w.writeheader()
                for row in aggregate_rows:
                    w.writerow({k: row.get(k) for k in keys})

    by_site: dict[str, dict] = {}
    for row in aggregate_rows:
        s = str(row["site"])
        bucket = by_site.setdefault(s, {"n": 0, "n_go": 0, "layers": []})
        bucket["n"] += 1
        if row.get("go"):
            bucket["n_go"] += 1
        bucket["layers"].append(row)

    residual_rows = [r for r in aggregate_rows if r.get("residual_write")]
    other_rows = [r for r in aggregate_rows if not r.get("residual_write")]

    def _tier(rows: list[dict]) -> dict:
        n = len(rows)
        ng = sum(1 for r in rows if r.get("go"))
        return {
            "n_site_layers": n,
            "n_go": ng,
            "go_rate": round(ng / max(n, 1), 4),
        }

    agg = {
        "model": model,
        "sites": sites,
        "layers": layers,
        "n_site_layers": n_jobs,
        "n_go": n_go,
        "go_rate": round(n_go / max(n_jobs, 1), 4),
        "by_tier": {
            "residual_write_abc": _tier(residual_rows),
            "other_linear_ab_only": _tier(other_rows),
        },
        "by_site": {
            s: {
                "n_layers": info["n"],
                "n_go": info["n_go"],
                "go_rate": round(info["n_go"] / max(info["n"], 1), 4),
                "residual_write": get_site(s).residual_write,
                "family": get_site(s).family,
            }
            for s, info in by_site.items()
        },
        "rows": aggregate_rows,
        "verdict": (
            f"GO on {n_go}/{n_jobs} site-layers"
            if n_go == n_jobs
            else f"PARTIAL: GO on {n_go}/{n_jobs} site-layers"
        ),
    }
    (out_dir / "aggregate_verdict.json").write_text(
        json.dumps(agg, indent=2), encoding="utf-8"
    )
    print("\n========== ALL SITE-LAYERS ==========")
    print(agg["verdict"])
    rt = agg["by_tier"]["residual_write_abc"]
    ot = agg["by_tier"]["other_linear_ab_only"]
    print(f"  residual A/B/C : {rt['n_go']}/{rt['n_site_layers']} GO")
    print(f"  other A/B only : {ot['n_go']}/{ot['n_site_layers']} GO")
    for s, info in agg["by_site"].items():
        print(
            f"  {s:10} {info['n_go']}/{info['n_layers']} GO "
            f"(residual_write={info['residual_write']})"
        )
    print(f"Wrote {agg_csv}")
    print(f"Wrote {out_dir / 'aggregate_verdict.json'}")
    return agg


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
    parser.add_argument(
        "--sites",
        default="all",
        help=(
            "Sites to evaluate. Presets: all|paper (7 linear maps), "
            "residual (mlp.down,attn.o). Or comma list: mlp.down,attn.o"
        ),
    )
    parser.add_argument(
        "--retry-failures",
        type=Path,
        default=None,
        help=(
            "Only re-run NO-GO site-layers from a prior out-dir "
            "(reads sites.csv / aggregate_verdict.json). Merges into --out-dir."
        ),
    )
    parser.add_argument(
        "--gate-pools",
        default=None,
        help=(
            "Effective-path trials. mlp.up default: mean,mixture_k4,compose_down. "
            "attn.v default: lstsq,compose_o."
        ),
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help=(
            "Input tile width for tile-SVD. Default: site-aware "
            "(512 for residual/MLP; 256 for attn.q/k/v)."
        ),
    )
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

    force_nested = False
    if args.retry_failures is not None:
        jobs = _load_nogo_jobs(args.retry_failures)
        if not jobs:
            print(f"No NO-GO site-layers in {args.retry_failures}; nothing to run.")
            return
        sites = sorted({s for s, _ in jobs})
        layers = sorted({L for _, L in jobs})
        force_nested = True
        if args.out_dir == ROOT / "data" / "eval" / "paper_experiments":
            # Default: write back into the prior tree.
            args.out_dir = args.retry_failures
        print(f"Retrying {len(jobs)} NO-GO site-layers from {args.retry_failures}:")
        for s, L in jobs:
            print(f"  {s} L{L}")
    else:
        sites = parse_sites(args.sites, default=PAPER_DEFAULT_SITES)
        if args.layers:
            layers = parse_layers(
                args.layers,
                n_layers=26 if args.layers.strip().lower() == "all" else None,
            )
        elif args.layer is not None:
            layers = [args.layer]
        else:
            layers = [12]
        jobs = [(s, L) for s in sites for L in layers]

    texts = load_texts(args.texts, DEFAULT_TRIGGER_TEXTS)
    model, tokenizer, device = load_model(args.model, args.device)
    lm_head = model.lm_head.weight.detach().float().cpu().numpy()
    rms = None
    if hasattr(model.model, "norm") and hasattr(model.model.norm, "weight"):
        rms = model.model.norm.weight.detach().float().cpu().numpy()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prior_rows = (
        _load_existing_rows(args.out_dir) if args.retry_failures is not None else []
    )
    new_rows: list[dict] = []
    n_sites_layout = max(len(sites), 2 if force_nested else 1)
    n_layers_layout = max(len(layers), 2 if force_nested else 1)

    for site, layer in jobs:
        job_dir = _out_dir_for(
            args.out_dir,
            site=site,
            layer=layer,
            n_sites=n_sites_layout,
            n_layers=n_layers_layout,
            force_nested=force_nested,
        )
        results = run_one_site_layer(
            model=model,
            tokenizer=tokenizer,
            device=device,
            texts=texts,
            layer=layer,
            site_name=site,
            out_dir=job_dir,
            args=args,
            exps=exps,
            lm_head=lm_head,
            rms=rms,
        )
        new_rows.append(_summary_row(results))

    aggregate_rows = (
        _merge_summary_rows(prior_rows, new_rows)
        if args.retry_failures is not None
        else new_rows
    )
    if len(aggregate_rows) > 1 or args.retry_failures is not None:
        _write_aggregate(
            args.out_dir,
            aggregate_rows,
            model=args.model,
            sites=sorted({str(r["site"]) for r in aggregate_rows}),
            layers=sorted({int(r["layer"]) for r in aggregate_rows}),
        )


if __name__ == "__main__":
    main()
