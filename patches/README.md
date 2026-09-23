# STATE compatibility patch

`state_genejepa_latent_compat.patch` applies to:

```text
https://github.com/ArcInstitute/state
commit f182478607c75f6b8f6256409cc0b4b902f991e9
```

Apply it from the root of a clean STATE checkout:

```bash
git checkout f182478607c75f6b8f6256409cc0b4b902f991e9
git apply /path/to/AIVC-Nanomics/patches/state_genejepa_latent_compat.patch
uv sync
```

The patch adds the signed final-output option required by GeneJEPA latents, the ST-R raw output-space residual, and the TensorBoard dependency used by the formal runners.
