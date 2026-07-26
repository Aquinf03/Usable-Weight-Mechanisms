# Finding Usable Weight Mechanisms with Tiled SVD

**Aquin Labs** research code for the paper *Finding Usable Weight Mechanisms with Tiled SVD*.

We extract **tile-SVD mounts** \((v, u, \sigma)\) from linear sites in `google/gemma-2-2b`, score them with **full-write energy lift** on real forwards, and evaluate chunking, coverage saturation, and depth-dependent steer vs unembed alignment. Mount identity is the weight rule itself, not an SAE feature label.

| | |
|---|---|
| Paper | [`PAPER.md`](PAPER.md) |
| Organization | [Aquin Labs](https://aquin.app/) |
| Package | `usable-weight-mechanisms` v0.4.0 |
| Model / site | `google/gemma-2-2b` · `mlp.down` · layers 0-25 |
| Contact | aquin@aquin.app |

## Requirements

- Python >= 3.11
- CUDA recommended for the full 26-layer suite
- Accept the [Gemma license](https://huggingface.co/google/gemma-2-2b) on Hugging Face before the first download

## Install

```bash
pip install -e ".[dev]"
```

## Reproduce

Full depth (loads the model once):

```bash
python scripts/run_paper_experiments.py --layers all --device cuda \
  --texts data/corpus/train.jsonl \
  --out-dir data/eval/paper_experiments_all
```

Single-layer smoke test and depth band:

```bash
python scripts/run_paper_experiments.py --layer 12 --device cuda \
  --texts data/corpus/train.jsonl \
  --out-dir data/eval/paper_experiments

python scripts/run_paper_experiments.py --layers 0,6,12,18,25 --device cuda \
  --texts data/corpus/train.jsonl \
  --out-dir data/eval/paper_experiments_band
```

| Exp | Question |
|-----|----------|
| A | Does tile SVD beat whole-matrix SVD, column sampling, and random under a matched mount budget? Metric: **full-write** energy lift. |
| B | Does write coverage saturate with few modes per tile? |
| C | When does steering along \(u\) align with final-unembed(\(u\))? Required for GO only when layer >= 6. |

Methods, results, pass rules, and re-judge notes are in [`PAPER.md`](PAPER.md).

## Layout

| Path | Role |
|------|------|
| [`PAPER.md`](PAPER.md) | Paper (methods, results, reproducibility) |
| `scripts/run_paper_experiments.py` | Experiments A/B/C and GO/NO-GO judge |
| `scripts/mount_runtime.py` | Model load and MLP activation hooks |
| `src/atlas/mount/` | Tile SVD, scoring, coverage, lenses, judge |
| `tests/` | CPU unit tests |

```bash
python -m pytest -q
```

## License

MIT. Gemma weights remain under Google's terms.

## Citation

```bibtex
@misc{aquin2026usable,
  title        = {Finding Usable Weight Mechanisms with Tiled SVD},
  author       = {{Aquin Labs}},
  year         = {2026},
  howpublished = {\url{https://github.com/aquinlabs/usable-weight-mechanisms}},
  note         = {Technical report and software artifact}
}
```

Update the GitHub URL after you publish the remote.
