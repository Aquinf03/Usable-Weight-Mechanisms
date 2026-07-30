# Finding Usable Weight Mechanisms with Tiled SVD

**Aquin Labs** research code for the paper *Finding Usable Weight Mechanisms with Tiled SVD*.

We extract **tile-SVD mounts** \((v, u, \sigma)\) from linear sites in `google/gemma-2-2b`, score them with **full-write energy lift** on real forwards, and evaluate chunking, coverage saturation, and depth-dependent steer vs unembed alignment. Mount identity is the weight rule itself, not an SAE feature label. The paper suite covers **all seven linear maps** per layer; residual writes get A/B/C (steer after Gemma-2 post-sublayer RMSNorm), others get A/B only.

| | |
|---|---|
| Paper | [`PAPER.md`](PAPER.md) |
| Organization | [Aquin Labs](https://aquin.app/) |
| Package | `usable-weight-mechanisms` v0.5.0 |
| Model / sites | `google/gemma-2-2b` · 7 linear sites · layers 0-25 |
| Contact | aquin@aquin.app |

## Requirements

- Python >= 3.11
- CUDA recommended for the full suite (7 sites × 26 layers)
- Accept the [Gemma license](https://huggingface.co/google/gemma-2-2b) on Hugging Face before the first download

## Install

```bash
pip install -e ".[dev]"
```

## Reproduce

Build WikiText-2 corpus, then run all sites:

```powershell
pip install -e ".[dev]"
python scripts/build_corpus.py --out data/corpus/train.jsonl
python scripts/run_paper_experiments.py --layers all --sites all --device cuda `
  --texts data/corpus/train.jsonl `
  --out-dir data/eval/paper_experiments_all
```

You want `tokens=16384/... (subsampled)` in the log, not `157/157`.

Residual-only (faster):

```powershell
python scripts/run_paper_experiments.py --layers all --sites residual --device cuda `
  --texts data/corpus/train.jsonl `
  --out-dir data/eval/paper_experiments_residual
```

`--sites` presets: `all` / `paper` (7 maps), `residual` (`mlp.down,attn.o`), or a comma list.

| Exp | Question |
|-----|----------|
| A | Does tile SVD beat whole-matrix SVD, column sampling, and random under a matched mount budget? Metric: **full-write** energy lift. |
| B | Does write coverage saturate by 1-2 modes per tile? |
| C | When does steering along \(u\) (after post-sublayer RMSNorm) align with final-unembed(\(u\))? Required only on residual-write sites with layer >= 6. |

Methods, results, pass rules, and re-judge notes are in [`PAPER.md`](PAPER.md).

## Layout

| Path | Role |
|------|------|
| [`PAPER.md`](PAPER.md) | Paper (methods, results, reproducibility) |
| `scripts/build_corpus.py` | WikiText-2 -> `data/corpus/train.jsonl` |
| `scripts/run_paper_experiments.py` | Experiments A/B/C and pass-criteria judge |
| `scripts/rejudge_paper_results.py` | Re-score saved summaries without GPU |
| `scripts/mount_runtime.py` | Model load and site activation hooks |
| `src/atlas/mount/` | Tile SVD, sites, effective maps, scoring, judge |
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
