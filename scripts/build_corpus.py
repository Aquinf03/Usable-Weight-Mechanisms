"""Build a real paper corpus (WikiText-2 raw) as JSONL.

Not a toy probe list. Downloads public Wikipedia-derived text and writes
enough documents that a 16,384-token subsample is meaningful.

Usage (from repo root):

  pip install datasets
  python scripts/build_corpus.py --out data/corpus/train.jsonl

PowerShell:

  python scripts/build_corpus.py --out data/corpus/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _clean(text: str) -> str:
    t = text.strip()
    # Drop WikiText section headers like " = Title = "
    if re.fullmatch(r"=+[^=].*=+", t):
        return ""
    return t


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Build WikiText-2 JSONL corpus")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/corpus/train.jsonl"),
        help="Output JSONL path",
    )
    p.add_argument(
        "--min-chars",
        type=int,
        default=80,
        help="Drop lines shorter than this (headers / junk)",
    )
    p.add_argument(
        "--max-docs",
        type=int,
        default=5000,
        help="Cap number of documents (WikiText-2 train is large enough)",
    )
    p.add_argument(
        "--target-chars",
        type=int,
        default=400_000,
        help="Stop once this many characters are written (~enough for >>16k tokens)",
    )
    args = p.parse_args(argv)

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "Missing dependency. Run:\n  pip install datasets\nthen re-run this script."
        ) from e

    print("Downloading WikiText-2 (raw) train split from Hugging Face...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_docs = 0
    n_chars = 0
    with args.out.open("w", encoding="utf-8") as f:
        for row in ds:
            text = _clean(row.get("text") or "")
            if len(text) < args.min_chars:
                continue
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            n_docs += 1
            n_chars += len(text)
            if n_docs >= args.max_docs or n_chars >= args.target_chars:
                break

    if n_docs == 0:
        raise SystemExit("Wrote 0 documents - dataset load failed or filters too strict")

    print(f"Wrote {n_docs} docs / {n_chars:,} chars -> {args.out}")
    print("Next:")
    print(
        "  python scripts/run_paper_experiments.py --layers all --sites residual "
        "--device cuda --texts data/corpus/train.jsonl "
        "--out-dir data/eval/paper_experiments_residual"
    )


if __name__ == "__main__":
    main()
