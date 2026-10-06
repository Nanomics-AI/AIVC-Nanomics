# STATE compatibility patch for ST-A v2

`state_genejepa_st_a_compat.patch` applies to:

```text
https://github.com/ArcInstitute/state
commit f182478607c75f6b8f6256409cc0b4b902f991e9
arc-state 0.11.3
```

Apply it from the root of a clean STATE checkout:

```bash
git checkout f182478607c75f6b8f6256409cc0b4b902f991e9
git apply /path/to/AIVC-Nanomics/patches/state_genejepa_st_a_compat.patch
uv sync
uv add tensorboard
```

Patch SHA-256:

```text
c672fcdbca6cabf3415c75871e6dbb1dc8016af9e38e612fbc55182ff03ecd17
```

Upstream STATE applies a final ReLU when no gene decoder is attached. The
GeneJEPA latent target is signed, so this patch adds the explicit
`final_activation="identity"` option while retaining the upstream default for
other callers.

The current ST-A v2 uses the absolute-output path:

```text
Zpred = project_out(transformer_hidden)
```

It does not use the historical residual route. The patch does not change the
transformer, hidden activations, loss, or conditioning used by ST-A v2.
