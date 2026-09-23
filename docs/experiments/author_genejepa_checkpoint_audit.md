# Author GeneJEPA epoch49 compatibility audit

- Status: **PASS_WITH_REQUIRED_LOADING_ADAPTATION**
- Created: `2026-09-20T03:15:46.314160+00:00`
- Scope: checkpoint/code/vocab/normalization/embedding contract plus fixed 8-cell smoke only
- No full-cache extraction, Decoder/ST training, or ARC7 evaluation was run.

## 1. Checkpoint identity and load behavior

The artifact is `external/author_genejepa/genejepa-epoch=49.ckpt` with SHA-256
`5db5c5750aeecc09955fefcaf143b349d9d8e1e877fe4a1073400ba66ef56f05`.  Its filename says epoch49,
while the Lightning payload stores zero-based `epoch=48`
and `global_step=66591`.

The standalone artifact needs two inference-only loading adaptations.  First, its config
dataclasses were pickled under `__main__`, so a separate script must register the four classes
imported from the **official** `genejepa.configs` module under those names.  Second, strict
loading against the downloaded official Git HEAD fails because the checkpoint has exactly
three additional keys: `teacher_center`, `model.local_desc_proj.weight`, and
`model.mask_desc_embed.weight`.  They do not exist in that code checkout and are outside both
the student encoder and `teacher_encoder.ema_model` used by `get_embedding()`.

The diagnostic smoke therefore uses official `load_from_checkpoint(strict=False)` only after
asserting `missing_keys=[]` and that the unexpected set is **exactly** those three keys.  Any
other mismatch remains fatal.  This does not edit the source, checkpoint, or any encoder weight,
but the task's original zero-unexpected-key strict-load requirement is not met.

Imported package: `external/author_genejepa_code/genejepa/__init__.py`  
Loaded class: `genejepa.train.JepaLightningModule`

## 2. Architecture recovered from checkpoint and official code

| Field | Author epoch49 |
|---|---:|
| Model | `genejepa.models.GenePerceiverJEPA` |
| Encoder | `genejepa.models.GenePerceiverEncoder` |
| Vocabulary size | 62710 |
| Latent tokens | 512 |
| Embedding dimension | 768 |
| Encoder blocks | 24 |
| Attention heads | 12 |
| Head dimension | 64 |
| Gene identity embedding | 384 |
| Fourier frequencies | 64 |
| Fourier range / scale | 0.1 .. 100.0 / 1.0 |
| Predictor depth / expansion | 3 / 4x |

The model-card claims 512 latents, 768 dimensions, 24 blocks, and 12 heads are an exact
match to the checkpoint.  The tokenizer splits 768 into 384 identity + 384 value channels;
64 frequencies create 128 sin/cos inputs before the value MLP, and the final tokenizer output
is 768-dimensional.

## 3. EMA teacher

`on_save_checkpoint` stores `model.teacher_encoder.state_dict()` as the top-level
`ema_state_dict`; `on_load_checkpoint` restores it.  `get_embedding(use_teacher=True)` calls
`model.teacher_encoder.ema_model`, not the student or EMA online model.

| Sampled parameter | max abs student-teacher diff | mean abs diff |
|---|---:|---:|
| `latents` | 0.0003323555 | 2.9444282e-05 |
| `tokenizer.value_encoder.0.weight` | 1.9334257e-05 | 3.3340491e-06 |
| `latent_blocks_seq.0.attn.in_proj_weight` | 2.6304275e-05 | 4.2455008e-06 |
| `latent_blocks_seq.23.ffn.0.weight` | 2.5909394e-05 | 4.4490785e-06 |
| `final_norm.weight` | 0.0001168251 | 5.5136781e-05 |

The sampled differences prove that the restored teacher is not merely an alias of current
student weights.  Formal author extraction should use **`use_teacher=True`**, following the
official inference method and README.

## 4. Author vocabulary and compatibility

- Rows / unique genes / unique token IDs: 62710 / 62710 / 62710
- Columns: `gene_symbol, ensembl_id, token_id`
- Raw foundation token IDs: 3..62712 (continuous; 0, 1, 2 absent/reserved)
- Actual model indices: 0..62709, obtained by sorting raw token ID and enumerating
- Missing values: {'gene_symbol': 0, 'ensembl_id': 0, 'token_id': 0}
- Duplicate gene symbols / Ensembl IDs / token IDs: 0 / 0 / 0
- Exact metadata file SHA match with current project: **True**
- Exact row/order and token mapping match: **True / True**
- Matched genes: 62710 (100.00%)
- Author-only / current-only: 0 / 0

This is compatibility **Case A**: existing physical
Tahoe token-to-GeneJEPA-index mapping can be reused directly.  Raw `token_id` must still pass
through the existing sorted-enumeration map; it must not be fed directly to the embedding table.

## 5. Preprocessing contract

The official order is:

```text
Tahoe sparse genes + raw expressions
→ remove the leading genes/expressions item when expressions[0] < 0
→ filter/map raw Tahoe token_id through author vocab to 0-based model index
→ log1p(raw expression)
→ (x - author_mean) / (author_std + 1e-6)
→ identity + Fourier-value tokenizer
→ GenePerceiver encoder
```

Implicit zeros are not materialized as tokens.  A stored zero, if present, remains a token:
`log1p(0)=0`, then global standardization is applied.  There is no CP10K/library-size
normalization in the encoder input path.

| Field | Our Epoch25 | Author epoch49 | Exact compatible? |
|---|---|---|---|
| Raw source | Tahoe `genes/expressions` | Tahoe `genes/expressions` | Yes |
| Sentinel | leading negative expression removed once | same | Yes |
| Vocab/token mapping | 62,710; sort token_id then enumerate | same artifact and rule | Yes |
| Transform | log1p then scalar global standardization | same | Yes |
| Mean | 0.82553487936616432 | 0.82553487936394987 | **No, abs diff 2.21e-12** |
| Std | 0.31346807453602465 | 0.31346807453597519 | **No, abs diff 4.95e-14** |
| Zero handling | sparse implicit zeros absent | same | Yes |
| Fourier N/min/max/scale | 64 / 0.1 / 100 / 1 | same | Yes |

The numerical difference is tiny, but formal author extraction must use the author stats file;
the two normalization artifacts are not bitwise or numerically exact.

## 6. Embedding readout contract

`get_embedding(use_teacher=True)` returns the EMA teacher encoder's final output.  Inside the
encoder, the 24th block output has shape `[B,512,768]`; `final_norm` is applied per latent token,
then `mean(dim=1)` yields `[B,768]`.  It is the final layer, not CLS or attention pooling.
There is no L2 normalization, centering, whitening, rectification, or predictor head in this
inference return value.

## 7. Comparison with our Epoch25 contract

| Field | Our Epoch25 | Author epoch49 |
|---|---|---|
| Checkpoint epoch label | 25 | 49 (payload epoch 48) |
| Model family/code | GenePerceiverJEPA | GenePerceiverJEPA |
| Vocab | 62710 | 62710 |
| Latents | 512 | 512 |
| Embedding dim | 768 | 768 |
| Blocks | 12 | 24 |
| Heads | 6 | 12 |
| Tokenizer | 384 identity + 384 Fourier-value | same |
| Normalization | current scalar stats | author scalar stats |
| Inference branch | EMA teacher | EMA teacher |
| Pooling | final_norm + latent mean | same |
| Post-normalization | none | none |

The 768-dimensional interface is compatible, but the representation is not interchangeable:
old Epoch25 embedding caches and Decoder weights cannot be reused as author-epoch49 latents or
Decoder weights.

## 8. Fixed 8-cell smoke

- Stable locators and metadata verified: **True**
- Author output: `[8, 768]`, `float32`, finite=True
- Range / mean / std: -2.77605 .. 3.10855 / 0.000235454 / 0.906313
- Negative fraction: 51.269531%
- Euclidean distance-matrix correlation with the same cells in our Epoch25 space: Pearson=-0.0679272, Spearman=-0.110564

The cross-check is descriptive only.  Coordinate-wise similarity is deliberately omitted because
independently trained latent spaces can rotate or reparameterize.

## 9. Compatibility answers

| Question | Status | Answer |
|---|---|---|
| A_checkpoint_load | NEEDS ADAPTATION | Register four official config classes as __main__ aliases, then use an exact three-key inference-only allowlist; the downloaded Git HEAD does not meet zero-unexpected-key strict loading. |
| B_embedding_shape | YES | Synthetic and real smoke both return [B,768]. |
| C_use_ema_teacher | YES | Use get_embedding(..., use_teacher=True); it calls teacher_encoder.ema_model. |
| D_vocab_gene_universe | YES | The author and current 62,710-row metadata parquet files are byte-identical. |
| E_token_id_and_order | YES | Raw token IDs, row order, and 0-based enumerated model mapping are identical. |
| F_reuse_tahoe_mapping | YES | Reuse the existing physical token_id -> 0-based GeneJEPA mapping directly. |
| G_global_stats_exact | NO | Values differ at ~1e-12/1e-14 scale; formal author extraction must use the author stats artifact. |
| H_preprocessing_exact | NEEDS ADAPTATION | Transform order and mapping logic match, but substitute author mean/std; do not silently reuse current stats. |
| I_tokenizer_fourier | YES | Tokenizer source bytes and Fourier configuration are identical. |
| J_embedding_pooling | YES | Model source bytes match: final LayerNorm then mean over 512 latents. |
| K_decoder_input_dimension_change | NO | Author-HD100 Decoder input remains 768; its weights must be retrained. |
| L_safe_to_start_author_hd100 | YES | Safe only with the audited exact-key loader guard plus author stats; create a new author embedding cache and train a new decoder. |

## 10. Correct future extraction recipe

1. Import only `external/author_genejepa_code` and assert `genejepa.__file__` points there.
2. Register the four official config dataclasses as `__main__` aliases for this artifact.
3. Load with official `JepaLightningModule.load_from_checkpoint(..., strict=False)` and abort
   unless missing keys are empty and unexpected keys equal the frozen three-key allowlist.
4. Freeze/eval and call `module.model.get_embedding(..., use_teacher=True)`.
5. Reuse stable Tahoe locators and the existing token mapping because vocab bytes/order match.
6. Apply sentinel removal once, mapping once, log1p once, and **author** scalar stats once.
7. Save the raw signed float32 `[N,768]` output without centering/whitening/L2/ReLU.

Reusable: Tahoe parquet data, stable locators, physical read logic, exact gene map, HD100 gene
panel and expression targets.  Must be regenerated/retrained: author embedding cache and the
Author-HD100 Decoder.  Existing Epoch25 embeddings and trained decoder weights are not reusable.
