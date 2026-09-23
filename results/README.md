# Lightweight frozen artifacts

Only small, code-facing contracts are tracked in this directory:

- the Decoder v1 contract and frozen 5,000-gene panel;
- the frozen HD100 gene panel;
- Experiment 1 evaluation sampling and perturbation featurization;
- B2 v2 and ST training protocols.

All generated predictions, metrics, caches, checkpoints, logs, large manifests, and scientific result tables remain local and are ignored by Git. Expected external artifacts and hashes are documented in `docs/provenance.md`.
