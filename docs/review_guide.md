# Technical review guide

## 1. Current pipeline

```text
Tahoe physical cells
  → Author GeneJEPA Epoch49 EMA Teacher
  → final_norm latent tokens [B,512,768]
  → mean over the 512 tokens
  → cell latent [B,768]
  → ST-A v2(control set, drug, dose)
  → predicted treated latent set [B,256,768]
  → D2(control set, treated set)
  → signed Top20 expression delta [B,20]
```

This repository keeps the exact formal Phase II/III source files. It does not
contain a cleaner reimplementation of the pipeline.

## 2. Suggested reading order

1. **Author GeneJEPA integration**
   - `perturbation_scripts/phase2_author_genejepa.py`
   - `perturbation_scripts/phase1_author_genejepa.py`
   - `perturbation_scripts/author_genejepa_epoch49_hd100_cache.py`
2. **Tahoe preprocessing and physical-cell cache**
   - `perturbation_scripts/phase2_stav2_data.py`
   - `perturbation_scripts/extract_tahoe_experiment1_full_cache_worker.py`
   - `perturbation_scripts/extract_tahoe_latent_audit_embeddings.py`
   - `genejepa/configs.py`
   - `genejepa/data.py`
3. **ST-A v2 dataset and deterministic set sampling**
   - `perturbation_scripts/phase2_stav2_dataset.py`
   - `perturbation_scripts/tahoe_experiment1_latent_data.py`
4. **ST-A v2 model**
   - `perturbation_scripts/phase2_stav2_model.py`
5. **ST-A v2 formal runner**
   - `perturbation_scripts/run_phase2_stav2.py`
   - current formal subcommand: `train`
6. **D2 data and target**
   - `perturbation_scripts/phase3_set_decoder_data.py`
   - `perturbation_scripts/tahoe_decoder_v1_data.py`
7. **D2 model**
   - `perturbation_scripts/phase3_set_decoder_model.py`
   - current selected variant: `d2`
8. **D2 formal runner**
   - `perturbation_scripts/run_phase3_set_decoder.py`
   - current formal command: `train --variant d2`
9. **Formal full-pipeline evaluator**
   - `perturbation_scripts/evaluate_phase3_set_decoder.py`

## 3. Author GeneJEPA representation

The representation model is the external
[BiostateAI/GeneJEPA](https://github.com/BiostateAI/GeneJEPA) source pinned at
commit `a2f4d7218b17f2f52cc5f1cc94420c8ef1ae3265`. It is not presented as an
AIVC-authored GeneJEPA implementation.

The formal checkpoint is the Epoch49 EMA Teacher. The author architecture has
24 transformer blocks, 12 attention heads, hidden dimension 768, and 512
latent tokens. The extraction path captures:

```text
EMA Teacher final_norm: [B,512,768]
mean(dim=1):             [B,768]
```

Only the signed float32 `[B,768]` cell embedding is persisted. The 512 token
vectors are not stored. Tahoe preprocessing is performed once: sentinel
handling, gene-token mapping, `log1p`, and the frozen Tahoe global mean/std
normalization. No centering, whitening, L2 normalization, or rectification is
applied to the resulting latent.

## 4. Physical-cell pools and sampling

Each eligible condition and matched control pool is capped at 512 cached
physical cells. A training/evaluation set contains 256 cells. The shared
sampler in `tahoe_experiment1_latent_data.py` derives a deterministic seed from
`seed + epoch + pair_id + side` and samples without replacement inside each
set. Different epochs may overlap. Training changes the dataset epoch;
validation and the five formal test repeats use their frozen epochs.

The embedding arrays are memory-mapped directly. `phase2_cell_index` is the
row-alignment contract across locator plans, Author embeddings, and Phase III
Top20 expression targets.

## 5. ST-A v2

Input:

```text
control latent set: [B,256,768]
drug ID:            [B]
dose:               [B]
```

Drug conditioning is a learned `Embedding(379,768)`. Dose is:

```text
dose_scaled = log1p(dose_uM) / log1p(5)
```

The model conditions every control cell as:

```text
conditioned_control = control_latent
                    + drug_embedding(drug_id) * dose_scaled
```

The conditioned set passes through the official STATE body and the signed
output projection. The output is an **absolute treated latent set** of shape
`[B,256,768]`; ST-A v2 does not predict residuals. The training objective is
Energy distance (`geomloss.SamplesLoss(loss="energy", blur=0.05)`). The STATE
compatibility patch only enables identity final activation for signed latents.

Formal training entry:

```text
perturbation_scripts/run_phase2_stav2.py train
```

## 6. D2 set-level delta decoder

D2 is not a single-cell absolute-expression decoder. It consumes two latent
sets:

```text
control set: [B,256,768]
treated set: [B,256,768]
```

The same set encoder is applied to both sides:

```text
Linear 768→256
TransformerEncoder ×2
  d_model=256
  nhead=8
  feed-forward=1024
  dropout=0
  norm_first=True
  no positional encoding
scalar attention pooling
```

This gives `h_control` and `h_treated`. The signed prediction is:

```text
latent_delta = h_treated - h_control
readout: 256→256→128→20
output: signed Top20 delta [B,20]
```

`phase3_set_decoder_model.py` retains both D1 and D2 because it is the exact
formal source used in the validation-only architecture comparison. D2 is the
current selected route. No D2-only rewrite was made.

Formal D2 training entry:

```text
perturbation_scripts/run_phase3_set_decoder.py train --variant d2
```

## 7. Top20 target

The Phase III target reuses helpers from `tahoe_decoder_v1_data.py`:

```text
all mapped raw gene counts
  → library size across all mapped genes
  → CP10K
  → log1p
  → select the frozen Top20 panel
```

The Top20 panel does **not** define the CP10K denominator. The condition-level
ground truth is:

```text
GT delta = mean(real treated Top20) - mean(matched real control Top20)
```

## 8. D2 training versus full-pipeline inference

During D2 training, ST-A is not in the loop:

```text
real control latent set + real treated latent set → D2 → Top20 delta
```

The formal full pipeline reconnects ST-A at inference time:

```text
real control + drug/dose
  → ST-A v2
  → predicted treated latent set

real control + predicted treated
  → D2
  → Top20 delta
```

The formal evaluator is:

```text
perturbation_scripts/evaluate_phase3_set_decoder.py evaluate
```

It also computes D1 and a frozen Old Decoder reference under the same evaluator
contract. `run_phase1_top20_decoder.py` is retained only because the exact
formal evaluator imports its `build_decoder()` reference. The Old Decoder is
not the current decoder route.

## 9. Historical filenames retained for provenance

Some required files retain development-stage names such as `phase1`, `hd100`,
`audit`, `b0`, or `decoder_v1`. These filenames reflect development history.
They remain because the exact formal Phase II/III source imports functions from
them. They are intentionally unchanged so the reviewed GitHub source matches
the code used in the experiments.

Two retained files also contain dormant, command-specific ARC7 imports from
their earlier roles. Those branches are not called by the current Phase II/III
entry points and the independent ARC7 scripts are intentionally excluded. The
formal Author path uses the loader/configuration functions from
`author_genejepa_epoch49_hd100_cache.py`; the formal evaluator uses only
`build_decoder()` from `run_phase1_top20_decoder.py`. External
`genejepa.train` imports are resolved from the pinned Author GeneJEPA checkout
after `register_author_config_aliases()` changes the package source.

## 10. Artifact boundary

GitHub contains source and lightweight frozen contracts only. Tahoe parquet,
condition tables, checkpoint weights, physical-cell locator plans, embedding
caches, expression caches, logs, and generated evaluation results are external
runtime artifacts. Their paths and hashes are listed in `docs/provenance.md`.
