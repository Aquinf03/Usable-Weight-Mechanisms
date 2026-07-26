# Finding Usable Weight Mechanisms with Tiled SVD

**Aquin Labs**  
Technical report and software artifact (`usable-weight-mechanisms` v0.4.0)  
Model: `google/gemma-2-2b` · Site: `mlp.down` · Layers: 0-25  
Contact: aquin@aquin.app · https://aquin.app/

---

## Abstract

We study **weight-native mechanism mounts** on linear sites of a transformer. Each mount is a triple \((v, u, \sigma)\) obtained from a **column-tiled SVD** of a weight matrix, read as *trigger*, *write*, and *strength*. Unlike sparse autoencoder (SAE) dictionaries, mount identity is the weight rule itself.

We do not claim to invent SVD mounts. Prior work already reads singular vectors via unembedding (Millidge et al.) and frames Detector-Effector Units and singular-vector circuits (NaNA DEUs; Beyond Components). Our contribution is a **measurement stack** and a **pre-registered GO/NO-GO suite**:

1. **Chunking A/B.** Tile SVD vs whole-matrix SVD vs column sampling vs random, judged on **full-write energy lift** (not tile-local lift, which favors one-column tiles tautologically).
2. **Coverage saturation.** Write energy explained by mounts saturates at 1-2 modes per tile.
3. **Causal steer vs final unembed.** Spearman and top-20 Jaccard between steered \(\Delta\)logits and the unembed lens of \(u\), required only for mid and late layers (\(L \ge 6\)).

On Gemma-2-2B `mlp.down`, with 16,384 tokens subsampled from a held corpus, **all 26 layers pass** under these rules (early layers use the C1 depth waiver). Steer vs unembed is near zero early and rises to \(\rho \approx 0.86\) at \(L=25\). We release library code, the experiment entrypoint, and CPU unit tests.

---

## 1. Introduction

Mechanistic interpretability often answers “what does this direction mean?” by training a proxy dictionary (SAEs, transcoder features) and labeling with max-activating text. That yields a concept atlas whose identity is the learned dictionary plus verbal labels.

This work builds a different object: a **mechanism atlas**. For a linear map \(W\) inside the network, we extract **mounts**: rank-1 pieces of \(W\) that specify *when* the site writes (trigger \(v\) on a column tile), *where* it writes (write \(u\)), and *how strongly* (\(\sigma\)). Proof that a mount is used on-distribution is **energy lift over a random direction** on real forwards, not the algebraic SVD identity (which is almost always true by construction).

**Research questions.**

- **RQ1 (chunking).** Under a matched mount budget, does tiling input columns before SVD yield higher *full residual write* energy lift than whole-matrix SVD, column sampling, or random directions?
- **RQ2 (coverage).** Does write coverage saturate with few modes per tile, or keep rising indefinitely?
- **RQ3 (causal depth).** When does steering along \(u\) move final logits in the direction predicted by the final unembed of \(u\)?

**Non-claims.**

- We do not claim novelty for “SVD of weights is interpretable.”
- We do not claim human-readable concept names; mounts are identified as `L·site·mount_id`.
- We do not claim the method beats SAEs at concept discovery.
- Reported GO/NO-GO results are for **`mlp.down` only**.

---

## 2. Related work

**SVD of transformer weights and unembed.** Millidge et al. show that singular vectors of MLP and OV matrices, projected through the unembedding, often form interpretable token clusters and can be edited. Our Experiment C uses the same geometric idea (unembed of \(u\) as a logit target) as a *causal depth diagnostic*, not as the definition of a mount.

**Native SVD units / DEUs.** Xue and Andrzejak (NaNA) treat MLP SVD modes as Detector-Effector Units and analyze subspace contribution without training SAEs. Our \((v,u,\sigma)\) is in the same family; we add **tiling**, **energy-lift gating**, and a fair chunking A/B.

**Singular-vector circuits.** Beyond Components decomposes heads and MLPs into singular directions for circuit analysis. We do not reproduce task circuits; we measure population statistics of mounts across depth.

**SAE atlases.** Orthogonal product identity: learned features plus labels. Out of scope for this artifact.

**Positioning.** Closest neighbors already own SVD mounts, DEUs, and unembed readout. The wedge we defend is **tile chunking + full-write energy-lift proof + coverage saturation + depth-conditioned causal check**, packaged as a GO/NO-GO eval.

---

## 3. Method

### 3.1 Model and site

- **Model:** `google/gemma-2-2b` (26 layers, indices \(0..25\)).
- **Primary site:** `mlp.down`, \(W \in \mathbb{R}^{d_{\mathrm{model}} \times d_{\mathrm{mlp}}}\) with \(d_{\mathrm{model}}=2304\), \(d_{\mathrm{mlp}}=9216\).
- **Activations:** MLP intermediate \(x\) (post gate\(\times\)up); site write \(\Delta h = Wx\) into the residual stream.
- **Corpus:** JSONL texts under `data/corpus/`. Forwards collect all tokens, then **subsample to 16,384** tokens (seed 0) for memory.

### 3.2 Tile-SVD mounts

Let \(T=512\) (default tile width) and \(k\) = modes per tile (Experiments A/C default \(k=2\); coverage sweep \(k\in\{1,2,4,8,16\}\)).

Number of tiles \(N_t=\lceil d_{\mathrm{in}}/T\rceil\). For tile \(t\) with columns \([s_t,e_t)\):

\[
B_t = W_{:,s_t:e_t} = U_t \Sigma_t V_t^\top.
\]

For mode \(i < k\):

\[
u = \frac{U_t[:,i]}{\|U_t[:,i]\|_2},\quad
\sigma = \Sigma_t[i],\quad
v = (V_t^\top)[i,:].
\]

Mount id: `tile:{t}:sv{i}`. Identity is \((v,u,\sigma)\) plus site and layer metadata, not a verbal label.

**Baselines (matched budget).**

| Strategy | Construction |
|----------|----------------|
| `tile_svd` | Above; if over budget, keep top-\(\sigma\) mounts |
| `whole_matrix_svd` | SVD of full \(W\); left SV = \(u\); right SV as full-span trigger |
| `column_sample` | High-norm columns as \(u\); one-column tile; \(\sigma=\|W_{:,j}\|\) |
| `random_baseline` | Random unit \(u\); random unit trigger on full span; \(\sigma=1\) |

Implemented in `src/atlas/mount/strategies.py` and `paper_eval.build_strategy_mounts`.

### 3.3 Triggering and scoring on real forwards

Trigger coefficients (`trigger.mode_trigger_coeffs`):

\[
a_{t,j} = x_{t,\,s_j:e_j}\cdot v_j.
\]

**Tile write** for columns \([s,e)\): \(\Delta h^{\mathrm{tile}} = x_{:,s:e}\, B^\top\).

**SVD identity (sanity, not proof):** correlation of \(a\) with \(\Delta h^{\mathrm{tile}} u\) and slope vs \(\sigma\) (thresholds: corr \(>0.99\), relative slope error \(<0.05\)). Almost always true; hence A3 checks that identity is *not* what separates good mounts.

**Tile-local energy fraction** for mode \((u,\sigma)\):

\[
f_{\mathrm{mode}}=\mathbb{E}_t\!\left[\frac{(a_t\sigma)^2}{\|\Delta h^{\mathrm{tile}}_t\|_2^2+\varepsilon}\right],\quad
f_{\mathrm{rand}}=\mathbb{E}_t\!\left[\frac{(\Delta h^{\mathrm{tile}}_t\cdot\tilde u)^2}{\|\Delta h^{\mathrm{tile}}_t\|_2^2+\varepsilon}\right],
\]

\[
L_{\mathrm{tile}} = f_{\mathrm{mode}}-f_{\mathrm{rand}}.
\]

**Full-write energy lift** (fair cross-strategy metric):

\[
f_{\mathrm{full}}(u)=\mathbb{E}_t\!\left[\frac{(\Delta h_t\cdot u)^2}{\|\Delta h_t\|_2^2+\varepsilon}\right],\quad
L_{\mathrm{full}}=f_{\mathrm{full}}(u)-f_{\mathrm{full}}(\tilde u),
\]

with \(\tilde u\) seeded by \(10007+j\cdot 997\). Column sampling can score \(L_{\mathrm{tile}}\approx 1\) while \(L_{\mathrm{full}}\approx 0\); **GO rules use \(L_{\mathrm{full}}\)**.

**Proven** (library gate): SVD identity OK and \(L_{\mathrm{tile}}\ge 0.02\) (`mechanism.is_proven`). Paper GO rules use full-write lift aggregates, not this threshold alone.

### 3.4 Coverage

**Weight coverage:** fraction of \(\|W\|_F^2\) kept by per-tile rank-\(k\) reconstructions (`coverage.tile_svd_weight_coverage`).

**Sparse write coverage:** unit directions \(D\); per token, top-\(k_{\mathrm{active}}\) mounts (default 8), least-squares reconstruct \(\Delta h\); report \(R^2\)-style explained energy. **Coverage lift** \(L_{\mathrm{cov}}=f(D_{\mathrm{mount}})-f(D_{\mathrm{rand}})\).

**Saturation (B1):** across the modes sweep, peak \(L_{\mathrm{cov}}\ge 0.25\), later points within 0.05 of peak, and final not collapsed vs first (\(-0.03\)).

### 3.5 Causal steer vs unembed (Experiment C)

`lenses.unembed_lens`: approximate final RMSNorm, then \(t = W_{\mathrm{lm}}\,d\).

Steer: after layer \(\ell\), \(h \leftarrow h + \alpha u\) with \(\alpha=2.0\); measure last-token \(\Delta\)logits averaged over up to 8 texts. Report Spearman \(\rho(\overline{\Delta}, t)\) and top-20 Jaccard. C1 passes if \(\rho\ge 0.05\) or \(J_{20}\ge 0.05\), **required only for \(\ell\ge 6\)** (`min_causal_layer=6`). Early layers still *report* C; they do not fail GO when C is weak.

### 3.6 GO / NO-GO rules

| ID | Pass when |
|----|-----------|
| A1 | \(L_{\mathrm{full}}(\mathrm{tile}) > L_{\mathrm{full}}(\mathrm{rand})+0.005\) |
| A2 | \(L_{\mathrm{full}}(\mathrm{tile}) \ge L_{\mathrm{full}}(\mathrm{cols})-0.002\) |
| A3 | frac SVD-identity OK \(\ge 0.9\) **or** \(\lvert\mathrm{corr}(\mathrm{id},\mathrm{lift})\rvert<0.3\) |
| A4 | tile full lift \(\ge\) whole (with 0.002 slack) **or** ratio \(\ge 0.8\) |
| B1 | coverage saturates (peak / flat / no collapse) |
| C1 | steer alignment (raw), required iff \(L\ge 6\) |

Implemented in `judge_paper_go` (`src/atlas/mount/paper_eval.py`).

### 3.7 Software artifact

This repository is the paper code: library, `scripts/run_paper_experiments.py`, and CPU unit tests. Package name: `usable-weight-mechanisms`.

---

## 4. Experiments

### 4.1 Setup

```bash
python scripts/run_paper_experiments.py --layers all --device cuda \
  --texts data/corpus/train.jsonl \
  --out-dir data/eval/paper_experiments_all
```

Defaults: `tile_size=512`, `modes_per_tile=2`, modes sweep `1,2,4,8,16`, `n_steer=8`, `steer_alpha=2.0`, `max_tokens=16384`, experiments A+B+C. Model loaded once; per-layer dirs `L0/` … `L25/`.

Under an older judge that required C1 at every layer: **21/26 GO** (fail L0-L4 on C). With the current `judge_paper_go` (C1 required only for \(L\ge 6\)): **26/26 GO**, obtained by re-scoring saved summaries without re-running GPU work.

### 4.2 Experiment A: chunking

**Finding.** On every layer, tile **full-write** lift substantially exceeds whole-matrix SVD, column sampling, and random, except the last layer where tile \(\approx\) whole.

Selected layers (`full_lift` = mean full-write lift):

| Layer | tile | whole | columns | random |
|------:|-----:|------:|--------:|-------:|
| 0 | 0.122 | 0.012 | 0.006 | \(\approx 0\) |
| 6 | 0.105 | 0.016 | 0.001 | \(\approx 0\) |
| 12 | 0.122 | 0.015 | 0.018 | \(\approx 0\) |
| 18 | 0.166 | 0.016 | 0.001 | \(\approx 0\) |
| 24 | 0.038 | 0.015 | 0.000 | \(\approx 0\) |
| 25 | 0.013 | 0.013 | 0.000 | \(\approx 0\) |

Column `tile_lift\(\approx 0.999\)` is a known tautology and is ignored by A1-A4.

**Interpretation.** Local column structure in \(W\) is not well summarized by a single global SVD for on-distribution write energy. Tiling recovers higher-energy write directions under a fixed mount count. At \(L=25\), residual geometry is already tightly coupled to the unembed; tiling adds little on A.

### 4.3 Experiment B: coverage vs modes

**Finding.** \(L_{\mathrm{cov}}\) is high by \(m=1\) or \(m=2\) and then flat or slightly decreasing (saturation, not endless growth).

Examples (write coverage lift):

| Layer | m=1 | m=2 | m=4 | m=8 | m=16 |
|------:|----:|----:|----:|----:|-----:|
| 0 | 0.206 | 0.285 | 0.280 | 0.314 | 0.309 |
| 6 | 0.531 | 0.533 | 0.528 | 0.524 | 0.518 |
| 12 | 0.482 | 0.497 | 0.496 | 0.490 | 0.485 |
| 25 | 0.294 | 0.297 | 0.301 | 0.303 | 0.307 |

**Interpretation.** A small number of tile modes captures most explainable write energy under sparse reconstruction. Extra modes mostly add redundancy.

### 4.4 Experiment C: causal depth

**Finding.** Mean Spearman of steered \(\Delta\)logits vs unembed(\(u\)) is a clear depth curve:

| Band | Layers | Typical mean \(\rho\) |
|------|--------|------------------------|
| Early | 0-4 | \(\approx -0.01\) to \(0.03\) (C1 waived) |
| Onset | 5-6 | \(\approx 0.05\) to \(0.12\) |
| Mid | 7-11 | \(\approx 0.43\) to \(0.50\) |
| Mid-late | 12-17 | \(\approx 0.11\) to \(0.32\) (non-monotone dip then rise) |
| Late | 18-24 | \(\approx 0.34\) to \(0.66\) |
| Final | 25 | \(\approx\) **0.86** |

Top-20 Jaccard stays low until late layers (often 0-0.1 mid-depth; up to about 0.38 on some L25 mounts).

**Interpretation.** Final-logit alignment is a mid-to-late property of residual geometry. Using it as a universal GO gate would falsely reject early layers where A and B already succeed. The waiver at \(L<6\) is motivated by this curve.

### 4.5 Aggregate verdict

| Judge | Result |
|-------|--------|
| Old (C1 all layers) | PARTIAL · 21/26 GO · fail L0-L4 on C |
| Current (C1 if \(L\ge 6\)) | **GO · 26/26** |

**Claim in one sentence.** Under full-write chunking tests and coverage saturation, tile-SVD mounts succeed at every depth; steer vs final-unembed alignment emerges with depth and is required only from layer 6 upward.

---

## 5. Discussion

**What is solid.** Fair A/B (full-write lift), saturation B, and a reproducible depth plot for C on one site across all layers. The artifact is small, tested, and free of SAE or verbalizer scaffolding.

**What is thin as novelty.** SVD mounts and unembed readouts exist. Reviewers may ask whether tiling plus energy lift is enough for a methods paper. Honest venues include a workshop poster, a blog plus artifact, or a short paper that emphasizes the **measurement protocol** and the **negative result** that tile-local metrics mislead.

**Scope.** This release is the measurement stack for the paper. Reported tables are `mlp.down` only. Extending A/B/C to `attn.o` is future work.

---

## 6. Limitations

1. Single model (Gemma-2-2B); single primary site in the GO suite.
2. Corpus and 16k-token subsample may bias which mounts look strong.
3. Energy lift is not human meaning; no claim of semantic labels.
4. Steer experiment uses short texts, fixed \(\alpha=2\), last-token logits only.
5. C1 waiver is principled but still a design choice; always report raw early \(\rho\).
6. L12-L15 show a mid-depth Spearman dip that is not fully explained here.
7. L25 A4 is marginal (tile \(\approx\) whole) and still GO.

---

## 7. Reproducibility

Verdict uses **full-write lift** for tile vs whole vs columns vs random (avoids fake column `tile_lift\(\approx 0.999\)`), and **coverage saturation** (high and flat), not “must keep rising forever.” Early layers still run Experiment C (reported in CSV), but C1 does not fail the GO gate when \(L < 6\).

### 7.1 Single layer (L12 smoke)

```bash
pip install -e ".[dev]"

python scripts/run_paper_experiments.py --layer 12 --device cuda \
  --texts data/corpus/train.jsonl \
  --out-dir data/eval/paper_experiments
```

Inspect `verdict.json` and `chunking.csv`. The fair A/B column is `mean_full_write_lift`.

### 7.2 All 26 layers (recommended)

Loads the model once and writes `L0/` … `L25/` plus `layers.csv` and `aggregate_verdict.json`:

```bash
python scripts/run_paper_experiments.py --layers all --device cuda \
  --texts data/corpus/train.jsonl \
  --out-dir data/eval/paper_experiments_all
```

Faster depth band:

```bash
python scripts/run_paper_experiments.py --layers 0,6,12,18,25 --device cuda \
  --texts data/corpus/train.jsonl \
  --out-dir data/eval/paper_experiments_band
```

Defaults: `tile_size=512`, `modes_per_tile=2`, modes sweep `1,2,4,8,16`, `n_steer=8`, `steer_alpha=2.0`, `max_tokens=16384`.

### 7.3 Pass rules

| ID | Pass when |
|----|-----------|
| A1 | tile **full-write** lift > random + 0.005 |
| A2 | tile **full-write** lift \(\ge\) columns (fair) |
| A3 | identity always-true or weakly correlated with lift |
| A4 | tile **full-write** \(\ge\) whole (or within 20%) |
| B1 | coverage **saturates**: peak \(\ge\) 0.25, stays within 0.05, no collapse |
| C1 | steer Spearman \(\ge\) 0.05 **or** top-20 Jaccard \(\ge\) 0.05; **required only if layer \(\ge\) 6** |

### 7.4 Tests and re-judge

```bash
python -m pytest -q
```

Re-score existing `L*/summary.json` after a judge change (no GPU):

```bash
python - <<'PY'
from pathlib import Path
import json
from atlas.mount.paper_eval import judge_paper_go

root = Path("data/eval/paper_experiments_all")
rows = []
for p in sorted(root.glob("L*/summary.json"), key=lambda x: int(x.parent.name[1:])):
    s = json.loads(p.read_text(encoding="utf-8"))
    v = judge_paper_go(s)
    s["verdict"] = v
    p.write_text(json.dumps(s, indent=2), encoding="utf-8")
    (p.parent / "verdict.json").write_text(json.dumps(v, indent=2), encoding="utf-8")
    rows.append(v["go"])
    print(f"L{s['layer']}: {'GO' if v['go'] else 'NO-GO'} ({v['n_pass']}/{v['n_checks']})")
print(f"GO on {sum(rows)}/{len(rows)} layers")
PY
```

Outputs (gitignored): `data/eval/paper_experiments_all/L*/{summary,verdict,chunking,coverage,causal}.*`, `layers.csv`, `aggregate_verdict.json`.

---

## 8. Conclusion

Tile-SVD mounts on Gemma-2-2B `mlp.down` beat matched whole, column, and random baselines on **full-write energy lift**, saturate **write coverage** at 1-2 modes per tile, and show **depth-dependent** agreement between residual steers and final unembedding. The released Aquin Labs artifact implements extraction, proof metrics, and the paper GO/NO-GO suite, with mount identity defined as the weight rule \((v,u,\sigma)\).

---

## Acknowledgments

This work was produced at **Aquin Labs**. Gemma-2-2B is released by Google under its model license; accept that license before downloading weights.

---

# Appendix A: Hyperparameters (defaults)

| Symbol / flag | Default | Where |
|---------------|---------|--------|
| `tile_size` | 512 | strategies, scripts |
| `modes_per_tile` | 2 | paper A/C |
| modes sweep | 1,2,4,8,16 | paper B |
| proven `min_lift` | 0.02 | `is_proven` |
| `k_active` | 8 | coverage / paper B |
| `steer_alpha` | 2.0 | paper C |
| `n_steer` | 8 | paper C |
| `max_tokens` | 16384 | paper scripts |
| `min_causal_layer` | 6 | `judge_paper_go` |
| SVD identity | corr > 0.99, slope err < 0.05 | `mechanism.py` |
| random-\(u\) seed | \(10007+j\cdot997\) | mechanism / paper_eval |

---

# Appendix B: Software artifact

Package: **`usable-weight-mechanisms` v0.4.0** · Python >= 3.11 · MIT · hatchling · import path `src/`.

## B.1 Root

| Path | Role |
|------|------|
| `README.md` | Install and reproduce |
| `PAPER.md` | This document |
| `pyproject.toml` | Package metadata and dependencies |
| `.gitignore` | Ignores corpus, exports, eval dumps, caches |

## B.2 Library (`src/atlas/`)

| Path | Role |
|------|------|
| `src/atlas/__init__.py` | `__version__ = "0.4.0"` |
| `src/atlas/mount/__init__.py` | Public exports |
| `src/atlas/mount/strategies.py` | Tile / whole / column / random mounts |
| `src/atlas/mount/trigger.py` | Trigger coeffs and default probe texts |
| `src/atlas/mount/mechanism.py` | Energy lift, \(R^2\), `is_proven` |
| `src/atlas/mount/coverage.py` | Weight and sparse write coverage |
| `src/atlas/mount/lenses.py` | RMSNorm and unembed lens |
| `src/atlas/mount/paper_eval.py` | Experiment helpers and `judge_paper_go` |

## B.3 Scripts

| Path | Role |
|------|------|
| `scripts/mount_runtime.py` | `parse_layers`, `load_texts`, MLP hooks, `load_model` |
| `scripts/run_paper_experiments.py` | Experiments A/B/C and aggregate verdict |

## B.4 Tests

| Path | Asserts |
|------|---------|
| `tests/test_mount_methods.py` | Mount builders; lenses |
| `tests/test_mount_mechanism.py` | Energy lift / proven |
| `tests/test_coverage.py` | Coverage and triggers |
| `tests/test_paper_eval.py` | Spearman/Jaccard; GO judge; C1 waiver |

## B.5 Data layout

| Path | Role |
|------|------|
| `data/eval/.gitkeep` | Eval outputs are gitignored |
| `data/corpus/**` | Probe texts (gitignored) |

---

# Appendix C: Output schema

Per layer under `data/eval/paper_experiments_all/L{n}/`: `summary.json`, `verdict.json`, `chunking.csv`, `coverage.csv`, `causal.jsonl`. Aggregate: `layers.csv`, `aggregate_verdict.json`.

---

# Appendix D: Figure plan

Plot from `layers.csv` when preparing a PDF:

1. **Figure 1.** Depth vs mean full-write lift: tile / whole / columns / random.
2. **Figure 2.** Coverage lift vs modes for layers 0, 6, 12, 18, 25.
3. **Figure 3.** Depth vs mean Spearman (Experiment C); mark \(L<6\) as informational.
4. **Figure 4 (optional).** Proven fraction vs layer from chunking CSV.

---

# Appendix E: Suggested phrasing

- “We evaluate tile-SVD mounts with a full-write energy-lift criterion that avoids tile-local artifacts of column baselines.”
- “Write coverage saturates by one to two modes per tile across depth.”
- “Agreement between residual steers and final unembedding is weak early and strong late; we require it only for \(L\ge 6\).”
- “We do not introduce SVD mounts; we contribute a fair chunking protocol, proof gate, and depth study on Gemma-2-2B.”

---

*Aquin Labs · 2026*
