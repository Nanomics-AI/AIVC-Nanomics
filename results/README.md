# Lightweight core-pipeline artifacts

Only small artifacts required to interpret or run the core pipeline are tracked:

- `genejepa_decoder_v1_contract.json`;
- the frozen Decoder v1 5,000-gene panel and its summary;
- the frozen drug/dose featurization specification;
- the Epoch25 embedding/preprocessing provenance contract;
- the ST-A-only training protocol.

Raw data, embedding caches, checkpoints, generated predictions, metrics, and
logs remain local and are excluded by `.gitignore`.
