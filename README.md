# AIVC Nanomics

This repository contains the project-owned code and lightweight frozen configuration for the current AIVC virtual-cell experiments.

The main workflow is:

```text
Tahoe single-cell expression
-> GeneJEPA cell embeddings
-> perturbation model (ST-A / baselines)
-> gene-expression Decoder
-> predicted treated expression and ARC7/Cell-Eval diagnostics
```

## GeneJEPA backbones

Two frozen backbones are used by the experiment code:

1. **Our half-size GeneJEPA**: 12 transformer blocks, 6 attention heads, Epoch25 EMA Teacher.
2. **Author GeneJEPA**: 24 transformer blocks, 12 attention heads, official Epoch49 checkpoint, EMA Teacher.

The Author model is an external dependency. Its upstream repository, exact commit, checkpoint filename, and SHA-256 are recorded in [docs/provenance.md](docs/provenance.md); the third-party repository and checkpoint are not vendored here.

## Perturbation and decoding

- ST-A predicts absolute signed GeneJEPA latents.
- ST-R is retained as an output-space residual comparison.
- B0/B1/B2 provide identity, mean-shift, and pooled-MLP baselines.
- Decoder v1 predicts a frozen 5,000-gene panel.
- HD100 Decoder predicts the high-detection Top100 panel.
- ARC7 scripts build and evaluate frozen same-cell diagnostics with Cell-Eval.

No final scientific performance claim is made in this code-delivery repository.

## Repository layout

- `genejepa/`: project GeneJEPA model, data, callbacks, and training entry point.
- `perturbation_scripts/`: Tahoe cache, baselines, ST-A/ST-R, Decoder, HD100, ARC7, and Author-backbone workflows.
- `benchmark_scripts/`: HLCA embedding and probe benchmarks.
- `download Tahoe datasets scripts/`: Tahoe metadata/download verification utilities.
- `experiment_configs/`: retained historical experiment configuration snapshots.
- `results/`: only small frozen protocols, contracts, and gene panels required by the code.
- `patches/`: the project-owned compatibility patch applied to upstream STATE.
- `docs/`: code inventory, external-artifact provenance, and selected project handoffs.

See [docs/code_inventory.md](docs/code_inventory.md) for the current entry points.

## Environments

GeneJEPA and STATE use separate Python environments.

```bash
# GeneJEPA environment
uv sync
```

STATE is installed from its upstream repository at the commit recorded in `docs/provenance.md`, then the compatibility patch in `patches/` is applied in that separate checkout. Do not upgrade the GeneJEPA environment to satisfy STATE dependencies.

## Deliberately excluded

Tahoe raw data, parquet shards, embedding caches, model checkpoints, training outputs, H5AD/NPY/NPZ payloads, logs, virtual environments, credentials, and downloaded third-party repositories are not stored in GitHub.
