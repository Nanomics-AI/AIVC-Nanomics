# AIVC Nanomics

> **For technical review**
>
> - English: [docs/review_guide.md](docs/review_guide.md)
> - 中文：[docs/review_guide_zh.md](docs/review_guide_zh.md)

This branch presents the source used by the current formal AIVC virtual-cell
pipeline:

```text
Tahoe single-cell data
        ↓
Author GeneJEPA Epoch49 EMA Teacher
        ↓
mean-pooled 768-dimensional cell embeddings
        ↓
ST-A v2
        ↓
predicted treated latent set
        ↓
D2 set-level delta decoder
        ↓
signed Top20 perturbation delta
```

- **Author GeneJEPA** is an external upstream dependency used as the frozen
  cell representation model.
- **ST-A v2** is the current perturbation model. It predicts an absolute,
  signed treated latent set from a control set and drug/dose conditioning.
- **D2** is the validation-selected set-level decoder/readout. It predicts a
  signed Top20 expression delta from paired control and treated latent sets.

The Python files in `perturbation_scripts/` and the retained preprocessing
files in `genejepa/` are byte-for-byte copies of the source used in the formal
Phase II and Phase III experiments. See
[docs/source_sha256.md](docs/source_sha256.md) for the source audit.

Large data, Tahoe parquet shards, embedding and expression caches, checkpoint
weights, and generated evaluation outputs are not stored in GitHub. Expected
local paths and hashes are documented in
[docs/provenance.md](docs/provenance.md).

Previous formal and experimental snapshots are preserved in the
`legacy-main-20261006` and `project-code-audit-20260923` branches.
