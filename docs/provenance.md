# External artifact and source provenance

The files below are required only in the original runtime workspace. They are not stored in this Git repository.

## GeneJEPA backbones

| Artifact | Expected local path | SHA-256 / source |
| --- | --- | --- |
| Our half-size Epoch25 checkpoint | `checkpoints/genejepa_quarter_d12_h6_700k_e30_seed42_run1/scjepa-epoch=25-val_loss=0.179.ckpt` | `6c6e89bc6d9349519250908cc4d4742c6319007f4f147328338952fa6cb98e9b` |
| Author Epoch49 checkpoint | `external/author_genejepa/genejepa-epoch=49.ckpt` | `5db5c5750aeecc09955fefcaf143b349d9d8e1e877fe4a1073400ba66ef56f05` |
| Author gene metadata | `external/author_genejepa/gene_metadata.parquet` | `d6104d1ca570d94832be0d27ffe8b7e4e54cfd366c9080239219f9c406eb751c` |
| Author global stats | `external/author_genejepa/global_stats.json` | `083f7aa484e9751f1b57e9285058f3895b1622b5b17e74255d61383dfb954f72` |

Author GeneJEPA source:

- upstream: `https://github.com/BiostateAI/GeneJEPA`
- exact commit: `a2f4d7218b17f2f52cc5f1cc94420c8ef1ae3265`
- local audit: clean working tree; no project modifications are vendored here
- inference: `use_teacher=True`, branch `teacher_encoder.ema_model`

## STATE dependency

- upstream: `https://github.com/ArcInstitute/state`
- base commit: `f182478607c75f6b8f6256409cc0b4b902f991e9` (`arc-state` 0.11.3)
- project modification: `patches/state_genejepa_latent_compat.patch`
- environment separation: STATE uses its own Python/uv environment; it must not upgrade the GeneJEPA environment

The patch contains the only project source modification found against that upstream commit: explicit signed-output control and the raw output-space residual definition. It also records the TensorBoard dependency added to the separate STATE environment.

## Trained project artifacts

| Artifact | Expected local path | SHA-256 |
| --- | --- | --- |
| ST-A best | `results/tahoe_experiment1_st_formal_checkpoints_v2/st-a/best.pt` | `9bd0f2719f42dac4fa5a9aabc4fd2bb242a442fb652ae90a8ee52fa54ca03652` |
| Decoder v1 best | `results/genejepa_decoder_v1_checkpoints/best.pt` | `29772ddf20bc36dde956fcd0befecbc0b71aaf4f1a8dd07fdf7195c5e6b590d0` |
| HD100 Decoder best | `results/genejepa_decoder_hd100_checkpoints/best.pt` | `4e69581ddf40d57302389dbade539535a4cd3b4f74ba84ae09bb5f5cc7ac3a9b` |
| B2 v2 best | `results/tahoe_experiment1_b2_v2_formal_checkpoints/b2_v2_best.pt` | `8d23da4e6760e1165eeab216eeb0cda22111cfcd735af4aa1884599209fbcdd3` |

## Deliberately unversioned runtime data

- Tahoe parquet shards and metadata caches;
- the 30.84M-cell Epoch25 embedding cache;
- the planned 25.79M-cell Author Epoch49 cache and worker partial files;
- all checkpoint/optimizer state;
- H5AD predictions, NPY/NPZ arrays, large CSV result tables, TensorBoard/W&B data, and console logs.

The Author Epoch49 cache extraction was prepared and short-audited but its formal long run was paused before this code-upload task.
