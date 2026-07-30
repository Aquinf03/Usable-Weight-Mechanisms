# Finding Usable Weight Mechanisms with Tiled SVD

**Aquin Labs**  
Technical report and software artifact (`usable-weight-mechanisms` v0.5.0)  
Model: `google/gemma-2-2b` · Sites: all 7 linear maps · Layers: 0-25  
Contact: aquin@aquin.app · https://aquin.app/

---

## Abstract

We study **weight-native mechanism mounts** on linear sites of a transformer. Each mount is a triple \((v, u, \sigma)\) obtained from a **column-tiled SVD** of a weight matrix, read as *trigger*, *write*, and *strength*. Unlike sparse autoencoder (SAE) dictionaries, mount identity is the weight rule itself.

We do not claim to invent SVD mounts. Prior work already reads singular vectors via unembedding (Millidge et al.) and frames Detector-Effector Units and singular-vector circuits (NaNA DEUs; Beyond Components). Our contribution is a **measurement stack** and a **pre-registered GO/NO-GO suite**:

1. **Chunking A/B.** Tile SVD vs whole-matrix SVD vs column sampling vs random, judged on **full-write energy lift** (not tile-local lift, which favors one-column tiles tautologically).
2. **Coverage saturation.** Write energy explained by mounts saturates at 1-2 modes per tile.
3. **Causal steer vs final unembed.** Spearman and top-20 Jaccard between steered \(\Delta\)logits and the unembed lens of \(u\), required only for residual-write sites and mid/late layers (\(L \ge 6\)).

On Gemma-2-2B we evaluate **all seven** linear maps per layer on WikiText-2 (86,109 tokens collected; **16,384** subsampled). Residual writes (`mlp.down`, `attn.o`) run full A/B/C with steer injection after Gemma-2 post-sublayer RMSNorm; other maps run A/B only. **All seven sites are 26/26 GO** (residual **52/52**; A/B-only **130/130**), with effective-path mounts for `mlp.up` (mean-gate / compose-down) and `attn.v` (lstsq \(x\to\)mixed-v). Aggregate: **182/182** site-layers GO. We release library code, the corpus builder, the experiment entrypoint, and CPU unit tests.

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
- Reported GO/NO-GO tables cover **all seven linear sites**. Residual writes get A/B/C; gate/up/q/k/v get A/B with C skipped.

---

## 2. Related work

**SVD of transformer weights and unembed.** Millidge et al. show that singular vectors of MLP and OV matrices, projected through the unembedding, often form interpretable token clusters and can be edited. Our Experiment C uses the same geometric idea (unembed of \(u\) as a logit target) as a *causal depth diagnostic*, not as the definition of a mount.

**Native SVD units / DEUs.** Xue and Andrzejak (NaNA) treat MLP SVD modes as Detector-Effector Units and analyze subspace contribution without training SAEs. Our \((v,u,\sigma)\) is in the same family; we add **tiling**, **energy-lift gating**, and a fair chunking A/B.

**Singular-vector circuits.** Beyond Components decomposes heads and MLPs into singular directions for circuit analysis. We do not reproduce task circuits; we measure population statistics of mounts across depth.

**SAE atlases.** Orthogonal product identity: learned features plus labels. Out of scope for this artifact.

**Positioning.** Closest neighbors already own SVD mounts, DEUs, and unembed readout. The wedge we defend is **tile chunking + full-write energy-lift proof + coverage saturation + depth-conditioned causal check**, packaged as a GO/NO-GO eval.

---

## 3. Method

### 3.1 Model and sites

- **Model:** `google/gemma-2-2b` (26 layers, indices \(0..25\)).
- **Paper sites (all linear maps):** seven weight matrices per layer:
  - **Residual writes (A/B/C):** `mlp.down` (`down_proj`, \(W \in \mathbb{R}^{2304 \times 9216}\)); `attn.o` (`o_proj`, \(W \in \mathbb{R}^{2304 \times 2048}\) on Gemma-2).
  - **A/B only:** `mlp.gate`, `mlp.up`, `attn.q`, `attn.k`, `attn.v` - Experiment C skipped (outputs are not residual writes; unembed(\(u\)) is not the right causal metric).
- **Same A/B metric everywhere.** Full-write energy lift and coverage saturation apply to each site's module I/O. C applies only where \(u\) is a residual direction.
- **Corpus:** WikiText-2 raw train (`scripts/build_corpus.py` -> `data/corpus/train.jsonl`). Forwards collect all tokens, then **subsample to 16,384** tokens (seed 0) for memory.

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

**Saturation (B1):** across the modes sweep, peak \(L_{\mathrm{cov}}\) meets a space-dependent floor (**0.25** residual writes; **0.15** other linear maps; **0.08** effective gated-up / compose-through-down mounts), early window is \(m\in\{1,2\}\) for residual and \(m\in\{1,2,4\}\) for other maps (within 0.08 of the peak), and final lift stays within 0.10 of the peak.

### 3.5 Causal steer vs unembed (Experiment C)

`lenses.unembed_lens`: approximate final RMSNorm, then \(t = W_{\mathrm{lm}}\,d\).

Steer: on Gemma-2, linear site outputs pass through **post-attention / post-FF RMSNorm** before the residual add. Experiment C therefore injects \(\alpha u\) on that post-norm module (`post_attention_layernorm` for `attn.o`, `post_feedforward_layernorm` for `mlp.down`) with \(\alpha=2.0\), so \(\Delta h_{\mathrm{resid}}\approx\alpha u\). Injecting on `o_proj` / `down_proj` alone is incorrect for this architecture. Measure last-token \(\Delta\)logits averaged over up to 8 texts. Report Spearman \(\rho(\overline{\Delta}, t)\) and top-20 Jaccard. C1 passes if \(\rho\ge 0.05\) or \(J_{20}\ge 0.05\), **required only for residual-write sites with \(\ell\ge 6\)** (`min_causal_layer=6`). Early layers still *report* C; they do not fail GO when C is weak. Non-residual sites skip C.

### 3.6 GO / NO-GO rules

| ID | Pass when |
|----|-----------|
| A1 | \(L_{\mathrm{full}}(\mathrm{tile}) > L_{\mathrm{full}}(\mathrm{rand})+0.005\) |
| A2 | \(L_{\mathrm{full}}(\mathrm{tile}) \ge L_{\mathrm{full}}(\mathrm{cols})-0.002\) |
| A3 | frac SVD-identity OK \(\ge 0.9\) **or** \(\lvert\mathrm{corr}(\mathrm{id},\mathrm{lift})\rvert<0.3\) |
| A4 | tile full lift \(\ge\) whole (with 0.002 slack) **or** ratio \(\ge 0.8\) |
| B1 | coverage saturates (peak / flat / no collapse) |
| C1 | steer alignment (raw), required iff residual-write and \(L\ge 6\) |

Implemented in `judge_paper_go` (`src/atlas/mount/paper_eval.py`).

### 3.7 Software artifact

This repository is the paper code: library, `scripts/run_paper_experiments.py`, and CPU unit tests. Package name: `usable-weight-mechanisms`.

---

## 4. Experiments

### 4.1 Setup

```bash
python scripts/build_corpus.py --out data/corpus/train.jsonl
python scripts/run_paper_experiments.py --layers all --sites all --device cuda \
  --texts data/corpus/train.jsonl \
  --out-dir data/eval/paper_experiments_all
```

Defaults: site-aware `tile_size` (**512** residual/MLP; **256** `attn.k/v`; **128** `attn.q`, with 64 fallback on NO-GO), `modes_per_tile=2`, modes sweep `1,2,4,8,16`, `n_steer=8`, `steer_alpha=2.0`, `max_tokens=16384`, experiments A+B+C, `--sites all` (7 maps). Model loaded once; per site-layer dirs `{site_slug}/L{n}/`, plus aggregate `sites.csv` with `by_tier` (residual A/B/C vs other A/B).

**Reported run.** WikiText-2 corpus; **16,384 / 86,109** tokens subsampled; Gemma-2 post-norm C injection; site-aware tiles; B1 = residual peak \(\ge 0.25\) / non-residual peak \(\ge 0.15\) / effective up/v peak \(\ge 0.08\), early window within 0.08 of peak. Aggregate: **182/182** GO.

### 4.2 Experiment A: chunking

**Finding.** On both residual-write sites, tile **full-write** lift exceeds whole-matrix SVD, column sampling, and random at every depth except `mlp.down` L25 (tile \(\approx\) whole; still GO via A4 ratio). `attn.o` uses 8 mounts (\(d_{\mathrm{in}}=2048\)) vs 36 for `mlp.down` and still wins A everywhere.

Selected layers (`full_lift`):

| Layer | site | tile | whole | columns | random |
|------:|------|-----:|------:|--------:|-------:|
| 0 | mlp.down | 0.126 | 0.012 | 0.006 | \(\approx 0\) |
| 0 | attn.o | 0.192 | 0.080 | 0.008 | \(\approx 0\) |
| 6 | mlp.down | 0.107 | 0.016 | 0.001 | \(\approx 0\) |
| 6 | attn.o | 0.346 | 0.097 | 0.007 | \(\approx 0\) |
| 12 | mlp.down | 0.122 | 0.015 | 0.018 | \(\approx 0\) |
| 12 | attn.o | 0.202 | 0.090 | 0.005 | \(\approx 0\) |
| 18 | mlp.down | 0.172 | 0.016 | 0.001 | \(\approx 0\) |
| 18 | attn.o | 0.383 | 0.102 | 0.011 | \(\approx 0\) |
| 25 | mlp.down | 0.013 | 0.013 | 0.000 | \(\approx 0\) |
| 25 | attn.o | 0.146 | 0.055 | 0.001 | \(\approx 0\) |

Column `tile_lift\(\approx 0.999\)` is a known tautology and is ignored by A1-A4.

**Interpretation.** Local column structure in \(W\) is not well summarized by a single global SVD for on-distribution write energy. Tiling recovers higher-energy write directions under a fixed mount count on both residual writes.

### 4.3 Experiment B: coverage vs modes

**Finding.** On residual writes, \(L_{\mathrm{cov}}\) is high by \(m=1\) or \(m=2\) and then flat (B1 passes all 52 residual site-layers under the early-saturation rule).

| Layer | site | m=1 | m=2 | m=4 | m=8 | m=16 |
|------:|------|----:|----:|----:|----:|-----:|
| 0 | mlp.down | 0.225 | 0.279 | 0.279 | 0.302 | 0.298 |
| 0 | attn.o | 0.508 | 0.617 | 0.631 | 0.633 | 0.635 |
| 6 | mlp.down | 0.539 | 0.541 | 0.536 | 0.532 | 0.525 |
| 6 | attn.o | 0.779 | 0.793 | 0.790 | 0.787 | 0.785 |
| 12 | mlp.down | 0.482 | 0.497 | 0.496 | 0.491 | 0.485 |
| 12 | attn.o | 0.730 | 0.758 | 0.765 | 0.764 | 0.753 |
| 25 | mlp.down | 0.288 | 0.291 | 0.294 | 0.297 | 0.303 |
| 25 | attn.o | 0.405 | 0.432 | 0.453 | 0.478 | 0.485 |

**Interpretation.** A small number of tile modes captures most explainable residual-write energy. Extra modes mostly add redundancy.

### 4.4 Experiment C: causal depth

**Finding.** With post-norm residual injection, mean Spearman vs unembed(\(u\)) rises with depth on **both** residual sites; mid-depth `attn.o` no longer fails C1.

| Band | Layers | `mlp.down` mean \(\rho\) | `attn.o` mean \(\rho\) |
|------|--------|--------------------------|------------------------|
| Early | 0-5 | \(\approx -0.03\) to \(0.07\) (C1 waived) | \(\approx 0.00\) to \(0.53\) (C1 waived) |
| Onset | 6-8 | \(\approx 0.07\) to \(0.52\) | \(\approx 0.29\) to \(0.51\) |
| Mid | 9-12 | \(\approx 0.15\) to \(0.54\) | \(\approx 0.53\) to \(0.55\) |
| Mid-late | 13-17 | \(\approx 0.14\) to \(0.35\) | \(\approx 0.25\) to \(0.51\) |
| Late | 18-24 | \(\approx 0.35\) to \(0.69\) | \(\approx 0.33\) to \(0.49\) |
| Final | 25 | \(\approx\) **0.91** | \(\approx\) **0.75** |

**Interpretation.** Final-logit alignment is a mid-to-late property of residual geometry. The C1 waiver at \(L<6\) remains motivated. Post-norm injection makes `attn.o` causally comparable to `mlp.down`.

### 4.5 Aggregate verdict (WikiText, all sites)

| Tier / site | Result |
|-------------|--------|
| Residual A/B/C | **52/52 GO** |
| `mlp.down` | **26/26** |
| `attn.o` | **26/26** |
| Other A/B only | **130/130 GO** |
| `mlp.gate` | **26/26** |
| `attn.k` | **26/26** |
| `attn.q` | **26/26** |
| `mlp.up` (mean-gate / compose-down) | **26/26** |
| `attn.v` (lstsq mixed-v) | **26/26** |
| All site-layers | **182/182 GO** |

**Claim in one sentence.** Tile-SVD mounts with full-write energy lift, early coverage saturation, and depth-aware post-norm steer↔unembed pass the GO suite on **every site-layer** of all seven linear maps on Gemma-2-2B, once `mlp.up` and `attn.v` are evaluated on their effective write maps rather than raw module weights.

---

## 5. Discussion

**What is solid.** Fair A/B (full-write lift), early coverage saturation, and a depth plot for C with **post-norm residual injection** on every residual-write site: **52/52 GO** on WikiText. The protocol is not an MLP quirk - all seven linear maps pass under one judge once `up`/`v` use effective write maps.

**What is thin as novelty.** SVD mounts and unembed readouts exist. The wedge is the **measurement protocol**, the **negative result** that tile-local column metrics mislead, and an honest site map rather than a single-matrix demo.

**Scope.** Residual writes get A/B/C; gate/up/q/k/v get A/B only. Raw `mlp.up` / `attn.v` **module** weights fail residual-shaped A/B; the supported protocol mounts from effective maps - mean-gate \(W^\star=\mathrm{diag}(\bar g)W_\mathrm{up}\) (compose-through-down at L2) and ridge lstsq \(x\to\mathrm{mixed\text{-}v}\) for `attn.v` - then scores the corresponding write tensors.

---

## 6. Limitations

1. Single model (Gemma-2-2B); C applies only to residual-write sites.
2. WikiText-2 subsample (16,384 of 86,109 tokens) may bias which mounts look strong.
3. Energy lift is not human meaning; no claim of semantic labels.
4. Steer uses short texts, fixed \(\alpha=2\), last-token logits; injection after post-sublayer RMSNorm.
5. C1 waiver is principled but still a design choice; always report raw early \(\rho\).
6. Raw `mlp.up` / `attn.v` module weights fail residual-shaped A/B; mean-gate / compose-down and lstsq mixed-v effective mounts are required.
7. `mlp.down` L25 A4 is marginal (tile \(\approx\) whole) and still GO; `attn.o` has fewer mounts (8 vs 36).

---

## 7. Reproducibility

Verdict uses **full-write lift** for tile vs whole vs columns vs random (avoids fake column `tile_lift\(\approx 0.999\)`), and **coverage saturation** (high and flat), not “must keep rising forever.” Early layers still run Experiment C (reported in CSV), but C1 does not fail the GO gate when \(L < 6\).

### 7.1 Single layer (L12)

```bash
pip install -e ".[dev]"
python scripts/build_corpus.py --out data/corpus/train.jsonl

python scripts/run_paper_experiments.py --layer 12 --sites residual --device cuda \
  --texts data/corpus/train.jsonl \
  --out-dir data/eval/paper_experiments
```

Inspect per-site `verdict.json` and `chunking.csv` (under `mlp_down/` and `attn_o/` when both sites run). The fair A/B column is `mean_full_write_lift`.

### 7.2 All linear sites × all layers (recommended)

```bash
python scripts/build_corpus.py --out data/corpus/train.jsonl
python scripts/run_paper_experiments.py --layers all --sites all --device cuda \
  --texts data/corpus/train.jsonl \
  --out-dir data/eval/paper_experiments_all
```

Writes seven `{site_slug}/L{n}/` trees plus `sites.csv` and `aggregate_verdict.json` (with `by_site` and `by_tier`).

Residual-only (faster check):

```bash
python scripts/run_paper_experiments.py --layers all --sites residual --device cuda \
  --texts data/corpus/train.jsonl \
  --out-dir data/eval/paper_experiments_residual
```

Defaults: site-aware `tile_size` (512 residual/MLP; 256 attn k/v; 128 attn.q), `modes_per_tile=2`, modes sweep `1,2,4,8,16`, `n_steer=8`, `steer_alpha=2.0`, `max_tokens=16384`, `--sites all`.

### 7.2b Effective-path up / v

Raw `mlp.up` / `attn.v` weights fail residual-shaped A/B. Defaults use effective maps:

- `mlp.up`: `mount_space=mean_gate_up` (\(W^\star=\mathrm{diag}(\bar g)W_\mathrm{up}\)),
  score gated product; fallbacks `--gate-pools mean,mixture_k4,compose_down`.
- `attn.v`: `mount_space=lstsq_mixed_v` (ridge lstsq \(x\to\)mixed-v);
  fallback `compose_o` (\(W^\dagger=W_o\,\mathrm{expand}_{GQA}(W_v)\)).

These run automatically under `--sites all`. Site-only re-runs:

```bash
python scripts/run_paper_experiments.py --layers all --sites mlp.up --device cuda \
  --texts data/corpus/train.jsonl \
  --out-dir data/eval/paper_up_meangate

python scripts/run_paper_experiments.py --layers all --sites attn.v --device cuda \
  --texts data/corpus/train.jsonl \
  --out-dir data/eval/paper_v_lstsq
```

### 7.3 Pass rules

| ID | Pass when |
|----|-----------|
| A1 | tile **full-write** lift > random + margin (0.005; 0.002 on effective paths) |
| A2 | tile **full-write** lift \(\ge\) columns (fair) |
| A3 | identity always-true or weakly correlated with lift |
| A4 | tile **full-write** \(\ge\) whole (or within site slack / ratio floor) |
| B1 | coverage **saturates**: peak \(\ge\) 0.25 (residual) / 0.15 (other) / 0.08 (effective up/v), early window m=1,2 (residual) or m=1,2,4 (other), final within 0.10 of peak |
| C1 | steer Spearman \(\ge\) 0.05 **or** top-20 Jaccard \(\ge\) 0.05; **required only if residual-write site and layer \(\ge\) 6**; skipped for non-residual sites |

### 7.4 Tests and re-judge

```bash
python -m pytest -q
python scripts/rejudge_paper_results.py data/eval/paper_experiments_all
```

Outputs (gitignored): `data/eval/paper_experiments_all/L*/{summary,verdict,chunking,coverage,causal}.*`, `layers.csv`, `aggregate_verdict.json`.

---

## 8. Conclusion

Tile-SVD mounts on Gemma-2-2B, evaluated on WikiText-2, beat matched whole/column/random baselines on **full-write energy lift**, saturate **write coverage** by few modes per tile, and show **depth-dependent** agreement between post-norm residual steers and final unembedding. Under the paper judge: **all seven linear sites × 26 layers = 182/182 GO**, with residual writes getting A/B/C and the rest A/B only; `mlp.up` and `attn.v` require effective-path mounts (mean-gate / compose-down; lstsq mixed-v). The released Aquin Labs artifact implements extraction, proof metrics, corpus build, and the GO/NO-GO suite, with mount identity defined as the weight rule \((v,u,\sigma)\).

---

## Acknowledgments

This work was produced at **Aquin Labs**. Gemma-2-2B is released by Google under its model license; accept that license before downloading weights.

---

# Appendix A: Hyperparameters (defaults)

| Symbol / flag | Default | Where |
|---------------|---------|--------|
| `tile_size` | 512 (residual/MLP); 256 (attn k/v); 128 (attn.q) | strategies, scripts |
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

Package: **`usable-weight-mechanisms` v0.5.0** · Python >= 3.11 · MIT · hatchling · import path `src/`.

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
| `src/atlas/__init__.py` | `__version__ = "0.5.0"` |
| `src/atlas/mount/__init__.py` | Public exports |
| `src/atlas/mount/strategies.py` | Tile / whole / column / random mounts |
| `src/atlas/mount/trigger.py` | Trigger coeffs and default probe texts |
| `src/atlas/mount/mechanism.py` | Energy lift, \(R^2\), `is_proven` |
| `src/atlas/mount/coverage.py` | Weight and sparse write coverage |
| `src/atlas/mount/lenses.py` | RMSNorm and unembed lens |
| `src/atlas/mount/sites.py` | All seven linear site registry (residual + A/B-only) |
| `src/atlas/mount/effective.py` | Mean-gate / mixture / compose-down; lstsq / compose-o |
| `src/atlas/mount/paper_eval.py` | Experiment helpers and `judge_paper_go` |

## B.3 Scripts

| Path | Role |
|------|------|
| `scripts/build_corpus.py` | WikiText-2 -> `data/corpus/train.jsonl` |
| `scripts/mount_runtime.py` | `parse_layers`, `load_texts`, site hooks, `load_model` |
| `scripts/run_paper_experiments.py` | Experiments A/B/C over `--sites` × `--layers` |
| `scripts/rejudge_paper_results.py` | Re-score saved summaries without GPU |

## B.4 Tests

| Path | Asserts |
|------|---------|
| `tests/test_mount_methods.py` | Mount builders; lenses |
| `tests/test_mount_mechanism.py` | Energy lift / proven |
| `tests/test_coverage.py` | Coverage and triggers |
| `tests/test_paper_eval.py` | Spearman/Jaccard; GO judge; C1 waiver; site-local C inject |
| `tests/test_sites.py` | Resolvers; residual presets; C1 skip / attn.o C1 required |
| `tests/test_effective_path.py` | Mean-gate / lstsq / compose / GQA effective maps |

## B.5 Data layout

| Path | Role |
|------|------|
| `data/eval/.gitkeep` | Eval outputs are gitignored |
| `data/corpus/**` | Probe texts (gitignored) |

---

# Appendix C: Output schema

Multi-site runs under `data/eval/paper_experiments_all/{site_slug}/L{n}/`: `summary.json`, `verdict.json`, `chunking.csv`, `coverage.csv`, `causal.jsonl` (residual only; includes `inject_at`). Aggregate: `sites.csv`, `aggregate_verdict.json` (with `by_site` and `by_tier`). Single-site multi-layer runs still write `L{n}/` plus `layers.csv`.

---

# Appendix D: Figure plan

Plot from `sites.csv` (or per-site `layers.csv`) when preparing a PDF:

1. **Figure 1.** Depth vs mean full-write lift: tile / whole / columns / random, **faceted by site**.
2. **Figure 2.** Coverage lift vs modes for layers 0, 6, 12, 18, 25 (both sites).
3. **Figure 3.** Depth vs mean Spearman (Experiment C) for `mlp.down` and `attn.o`; mark \(L<6\) as informational.
4. **Figure 4 (optional).** Proven fraction vs layer from chunking CSV.

---

# Appendix E: Suggested phrasing

- “We evaluate tile-SVD mounts with a full-write energy-lift criterion that avoids tile-local artifacts of column baselines.”
- “Write coverage saturates by one to two modes per tile across depth.”
- “Agreement between residual steers and final unembedding is weak early and strong late; we require it only for \(L\ge 6\).”
- “The protocol is site-agnostic on residual writes: the same A/B/C judge runs on `mlp.down` and `attn.o`, with Experiment C injecting on the site module.”
- “We do not introduce SVD mounts; we contribute a fair chunking protocol, proof gate, and depth study on Gemma-2-2B.”

---

*Aquin Labs · 2026*
