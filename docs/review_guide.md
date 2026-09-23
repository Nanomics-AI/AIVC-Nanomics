# Technical review guide

This document is a code-reading route for the core AIVC implementation. It is
not a performance report.

## End-to-end data flow

```text
Tahoe raw sparse expression
        ↓
sentinel removal → gene mapping → log1p → frozen global normalization
        ↓
our GeneJEPA Epoch25 EMA Teacher
        ↓
one signed 768-dimensional embedding per control cell
        ↓
ST-A + drug one-hot + standardized log-dose
        ↓
signed 768-dimensional predicted treated-cell embeddings
        ↓
Decoder v1
        ↓
predicted 5,000-gene expression vector in log1p(CP10K) space
```

## Step 1 — GeneJEPA configuration

Start with [`genejepa/configs.py`](../genejepa/configs.py).

The formal model configuration is:

- model width `d = 768`;
- `512` learned latent tokens;
- `12` latent transformer blocks;
- `6` attention heads;
- an EMA Teacher updated from the student during training.

The same file also holds the training, data-volume, checkpoint, and random-seed
configuration used by [`genejepa/train.py`](../genejepa/train.py).

## Step 2 — Tokenizer and encoder

Read these files next:

1. [`genejepa/tokenizer.py`](../genejepa/tokenizer.py)
2. [`genejepa/models.py`](../genejepa/models.py)

The tokenizer combines a mapped gene identity with its continuous normalized
expression value. The encoder cross-attends 512 learned latent queries to the
variable-length cell tokens, processes the latent array through 12 transformer
blocks, applies `final_norm`, and mean-pools the 512 internal tokens.

The public embedding returned by `get_embedding()` is therefore `[B, 768]`.
The 512 tokens are internal model state; the cache does not store a
`[512, 768]` array per cell.

## Step 3 — Tahoe preprocessing

Read [`genejepa/data.py`](../genejepa/data.py), especially
`Tahoe100MDataset.__iter__` and `Tahoe100MDataModule._collate_fn`.

For each Tahoe cell, the production path performs exactly:

```text
raw sparse counts
→ remove the leading sentinel entry when present
→ map Tahoe gene token IDs to the GeneJEPA vocabulary
→ log1p(count)
→ (value - global_mean) / (global_std + 1e-6)
```

There is no CP10K normalization in the GeneJEPA input path. CP10K is used only
for the Decoder target described below.

## Step 4 — GeneJEPA training and inference

Read [`genejepa/train.py`](../genejepa/train.py) and
[`genejepa/callbacks.py`](../genejepa/callbacks.py).

Training maintains a student encoder and an exponential-moving-average Teacher.
Formal embedding extraction calls `get_embedding(..., use_teacher=True)` on the
frozen Epoch25 checkpoint. The shared production helper is
[`perturbation_scripts/tahoe_genejepa_embedding.py`](../perturbation_scripts/tahoe_genejepa_embedding.py).

## Step 5 — Embedding cache

Read the cache path in this order:

1. [`prepare_tahoe_experiment1_manifests.py`](../perturbation_scripts/prepare_tahoe_experiment1_manifests.py) — freeze condition splits, drug vocabulary, and dose transform.
2. [`plan_tahoe_experiment1_full_cache.py`](../perturbation_scripts/plan_tahoe_experiment1_full_cache.py) — create deterministic stable cell locators and worker partitions.
3. [`tahoe_genejepa_embedding.py`](../perturbation_scripts/tahoe_genejepa_embedding.py) — reuse the GeneJEPA preprocessing and frozen EMA Teacher.
4. [`extract_tahoe_experiment1_full_cache_worker.py`](../perturbation_scripts/extract_tahoe_experiment1_full_cache_worker.py) — resumable independent extraction workers.
5. [`merge_tahoe_experiment1_full_cache.py`](../perturbation_scripts/merge_tahoe_experiment1_full_cache.py) — scatter worker parts into global `embedding_index` order.
6. [`tahoe_experiment1_latent_data.py`](../perturbation_scripts/tahoe_experiment1_latent_data.py) — memory-map the merged cache and sample cell sets.

Each physical cell is embedded once. The merged payload is a float32 array
`[N, 768]`, and cache row `i` corresponds to global `embedding_index == i`.
The cache consumer does not apply centering, whitening, L2 normalization, or
expression preprocessing to the stored latent.

## Step 6 — ST-A

Read:

1. [`tahoe_experiment1_latent_data.py`](../perturbation_scripts/tahoe_experiment1_latent_data.py)
2. [`run_tahoe_experiment1_st_a.py`](../perturbation_scripts/run_tahoe_experiment1_st_a.py)
3. [`patches/state_genejepa_st_a_compat.patch`](../patches/state_genejepa_st_a_compat.patch)

The DataLoader supplies:

```text
ctrl_cell_emb  [B, 256, 768]
pert_cell_emb  [B, 256, 768]   # training target
pert_emb       [B, 256, 380]
```

The 380 perturbation features are a 379-dimensional drug one-hot vector and one
standardized `log10(dose_uM)` value. The same vector is repeated across the 256
cells in a set.

ST-A uses the upstream STATE basal encoder, perturbation encoder, bidirectional
Llama backbone, and `project_out`. It predicts the absolute treated latent:

```text
Zpred = project_out(transformer_hidden)
```

Output is `[B, 256, 768]`. The final activation is identity because GeneJEPA
latents are signed. Training uses set-level Energy distance against the real
treated-cell set.

## Step 7 — Decoder v1

Read:

1. [`tahoe_decoder_v1_data.py`](../perturbation_scripts/tahoe_decoder_v1_data.py)
2. [`run_genejepa_decoder_v1.py`](../perturbation_scripts/run_genejepa_decoder_v1.py)
3. [`results/genejepa_decoder_v1_contract.json`](../results/genejepa_decoder_v1_contract.json)

The Decoder input is one signed 768-dimensional latent. Its supervised target is
a frozen 5,000-gene vector constructed from the same physical cell:

```text
all mapped raw gene counts
→ compute the full mapped-gene library size
→ CP10K normalization
→ log1p
→ select the frozen 5,000-gene panel
```

The 5,000-gene panel defines which targets are returned by the Decoder dataset;
it does not define the denominator used for CP10K normalization.

The architecture is:

```text
768 → 1024 → 1024 → 512 → 5000
```

The hidden stages use LayerNorm, GELU, and dropout `0.1`; the final output uses
Softplus. The frozen panel is stored in
[`results/genejepa_decoder_v1_gene_panel.csv`](../results/genejepa_decoder_v1_gene_panel.csv).

## Step 8 — Reconstructing the complete path

| Arrow | Implementation |
| --- | --- |
| sparse Tahoe cell → model-ready tokens | `genejepa/data.py` |
| tokens → 768-d cell embedding | `genejepa/tokenizer.py`, `genejepa/models.py`, `tahoe_genejepa_embedding.py` |
| physical cells → indexed embedding cache | `plan_tahoe_experiment1_full_cache.py`, extraction worker, merge script |
| control embedding set + drug/dose → predicted treated embedding set | `tahoe_experiment1_latent_data.py`, `run_tahoe_experiment1_st_a.py`, STATE patch |
| predicted treated embedding → predicted expression | `run_genejepa_decoder_v1.py` |

Data, checkpoints, and generated cache payloads are external to Git. See
[`provenance.md`](provenance.md) for exact expected paths and hashes.
