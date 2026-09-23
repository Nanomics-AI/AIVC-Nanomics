# STATE compatibility patch for ST-A

`state_genejepa_st_a_compat.patch` applies to:

```text
https://github.com/ArcInstitute/state
commit f182478607c75f6b8f6256409cc0b4b902f991e9
```

Apply it from the root of a clean STATE checkout:

```bash
git checkout f182478607c75f6b8f6256409cc0b4b902f991e9
git apply /path/to/AIVC-Nanomics/patches/state_genejepa_st_a_compat.patch
uv sync
uv add tensorboard
```

The upstream model applies a final ReLU when no gene decoder is attached. Our
ST-A target is a signed GeneJEPA latent, so the patch adds an explicit
`final_activation="identity"` option while preserving the upstream default.

ST-A remains the upstream absolute-output path:

```text
Zpred = project_out(transformer_hidden)
```

The patch does not change the transformer, loss, hidden activations, or define
an alternate output branch.
