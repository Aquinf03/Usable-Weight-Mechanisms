"""Re-run the paper GO judge on existing summary.json trees (no GPU).

Usage:
  python scripts/rejudge_paper_results.py data/eval/paper_qk_retune
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atlas.mount.paper_eval import judge_paper_go  # noqa: E402
from atlas.mount.sites import get_site  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out_dir", type=Path, help="Experiment root with */L*/summary.json")
    args = p.parse_args()
    root: Path = args.out_dir

    summaries = sorted(root.glob("**/summary.json"))
    if not summaries:
        raise SystemExit(f"no summary.json under {root}")

    rows: list[dict] = []
    n_go = 0
    for path in summaries:
        results = json.loads(path.read_text(encoding="utf-8"))
        verdict = judge_paper_go(results)
        results["verdict"] = verdict
        path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        (path.parent / "verdict.json").write_text(
            json.dumps(verdict, indent=2), encoding="utf-8"
        )
        site = str(results.get("site") or "")
        layer = results.get("layer")
        go = bool(verdict.get("go"))
        if go:
            n_go += 1
        rows.append(
            {
                "site": site,
                "layer": layer,
                "residual_write": bool(results.get("residual_write")),
                "go": go,
                "n_pass": verdict.get("n_pass"),
                "n_checks": verdict.get("n_checks"),
                "verdict": verdict.get("verdict"),
            }
        )
        print(
            f"L{layer} {site}: {'GO' if go else 'NO-GO'} "
            f"({verdict.get('n_pass')}/{verdict.get('n_checks')})"
        )

    n = len(rows)
    by_site: dict[str, dict] = {}
    for row in rows:
        s = str(row["site"])
        b = by_site.setdefault(s, {"n": 0, "n_go": 0})
        b["n"] += 1
        if row["go"]:
            b["n_go"] += 1

    residual_rows = [r for r in rows if r.get("residual_write")]
    other_rows = [r for r in rows if not r.get("residual_write")]

    def _tier(rs: list[dict]) -> dict:
        nn = len(rs)
        ng = sum(1 for r in rs if r["go"])
        return {"n_site_layers": nn, "n_go": ng, "go_rate": round(ng / max(nn, 1), 4)}

    agg = {
        "n_site_layers": n,
        "n_go": n_go,
        "go_rate": round(n_go / max(n, 1), 4),
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
        "rows": rows,
        "verdict": (
            f"GO on {n_go}/{n} site-layers"
            if n_go == n
            else f"PARTIAL: GO on {n_go}/{n} site-layers"
        ),
    }
    with (root / "sites.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)
    (root / "aggregate_verdict.json").write_text(
        json.dumps(agg, indent=2), encoding="utf-8"
    )
    print(agg["verdict"])
    for s, info in sorted(agg["by_site"].items()):
        print(f"  {s:10} {info['n_go']}/{info['n_layers']} GO")


if __name__ == "__main__":
    main()
