# AIVC Nanomics

> **For technical review**
>
> - English: [docs/review_guide.md](docs/review_guide.md)
> - 中文：[docs/review_guide_zh.md](docs/review_guide_zh.md)

This repository presents the core AIVC virtual-cell pipeline:

```text
Tahoe sparse single-cell expression
        ↓
GeneJEPA preprocessing and our frozen Epoch25 EMA Teacher
        ↓
one signed 768-dimensional embedding per cell
        ↓
ST-A with drug and dose information
        ↓
predicted treated-cell embeddings
        ↓
Decoder v1
        ↓
predicted treated gene expression in log1p(CP10K) space
```

## Core modules

### 1. GeneJEPA

`genejepa/` contains our 12-block, 6-head, 768-dimensional GeneJEPA with 512
learned latent tokens. Tahoe sparse counts are mapped to the training gene
vocabulary, transformed with `log1p`, and normalized with the frozen global
mean and standard deviation. Formal cell embeddings use the Epoch25 EMA
Teacher and mean-pool the 512 internal latent tokens to one 768-dimensional
vector.

### 2. ST-A

`perturbation_scripts/run_tahoe_experiment1_st_a.py` trains the absolute-output
STATE transition model on sets of 256 control and treated cell embeddings. The
perturbation input is a 379-dimensional drug one-hot vector plus one
standardized log-dose value. ST-A directly predicts signed treated latents.

### 3. Decoder v1

`perturbation_scripts/run_genejepa_decoder_v1.py` trains the latent-to-expression
decoder. It maps each 768-dimensional latent to a frozen 5,000-gene panel in
`log1p(CP10K)` expression space.

## Repository layout

- `genejepa/`: tokenizer, encoder, data pipeline, training, and EMA Teacher.
- `perturbation_scripts/`: main cache construction, embedding extraction,
  ST-A training, and Decoder v1 training.
- `patches/`: minimal signed-latent compatibility patch for upstream STATE.
- `results/`: lightweight frozen contracts and gene-panel metadata required by
  the code.
- `docs/`: English and Chinese code-reading guides plus artifact provenance.

Tahoe-100M is an external dataset and is not stored in this repository. Model
checkpoints, embedding caches, parquet shards, logs, and generated predictions
are also excluded; their expected locations and hashes are documented in
[docs/provenance.md](docs/provenance.md).

The complete pre-curation code history remains available on the
`project-code-audit-20260923` branch. No final biological or performance claim
is made in this code-review repository.
