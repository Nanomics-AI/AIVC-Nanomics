# Author GeneJEPA epoch49 -> HD100 handoff

## Current status

- Cache policy: train treated `22,941,936` + val treated `2,841,724` + frozen ARC7-C `6,144` = `25,789,804` cells.
- Author checkpoint: `external/author_genejepa/genejepa-epoch=49.ckpt`.
- Inference branch: frozen EMA teacher `teacher_encoder.ema_model`.
- Formal extraction batch size: `64` per independent worker.
- Plan, batch probe, 8-cell production parity, early resume test, and small scatter/merge rehearsal: PASS.
- Worker 0: `192 / 12,908,951`, paused, failed `0`.
- Worker 1: `192 / 12,880,853`, paused, failed `0`.
- The two `.npy.partial` files are preallocated to their final logical size. File size is therefore not a progress indicator; use the progress JSON/status command.
- Frozen extraction script SHA-256: `bb111848f2a4dbe8fd7b2df1f9b6c183391b681a406daf5a7864d77ab4296eae`. Do not edit that script until both workers and merge are complete.

## Resume the formal extraction

Run worker 0 in WSL terminal 1:

```bash
cd /mnt/c/SH/AIVC/GeneJEPA-main
set -o pipefail

CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python \
perturbation_scripts/author_genejepa_epoch49_hd100_cache.py \
run-worker \
  --worker-id 0 \
  --commit-every-batches 10 \
2>&1 | tee -a results/author_genejepa_epoch49_hd100_cache_worker0_console.log
```

Run worker 1 in WSL terminal 2:

```bash
cd /mnt/c/SH/AIVC/GeneJEPA-main
set -o pipefail

CUDA_VISIBLE_DEVICES=1 \
.venv/bin/python \
perturbation_scripts/author_genejepa_epoch49_hd100_cache.py \
run-worker \
  --worker-id 1 \
  --commit-every-batches 10 \
2>&1 | tee -a results/author_genejepa_epoch49_hd100_cache_worker1_console.log
```

These commands resume after the committed prefix; they do not restart from zero. Do not pass `--max-new-cells` for the full run.

The reused low-level worker may print a legacy line containing `Epoch25`. The wrapper line, checkpoint/loader provenance, and runtime manifests identify the actual Author epoch49 EMA teacher; the legacy wording is not a model switch.

## Monitor

This command is read-only with respect to extraction state:

```bash
cd /mnt/c/SH/AIVC/GeneJEPA-main

.venv/bin/python \
perturbation_scripts/author_genejepa_epoch49_hd100_cache.py \
status
```

Progress files:

```text
results/author_genejepa_epoch49_hd100_cache/cache_worker0_progress.json
results/author_genejepa_epoch49_hd100_cache/cache_worker1_progress.json
```

The short single-GPU probe measured about `69.75 cells/s`; ideal two-worker time is about 51 hours, while shared parquet I/O may make the real wall time closer to roughly 60-72 hours. Use the settled long-run rates, not the three-batch acceptance prefix, for the updated ETA.

After a worker reaches its cell total, allow its final rename/hash/manifest stage to finish even if GPU utilization falls to zero.

## After both workers finish

Do not start training or ARC7 evaluation yet. Return the two final console tails and the output of the status command for integrity audit. The next stages are:

1. full scatter/merge and cache manifest audit;
2. one real `[B,768] -> [B,100]` static smoke;
3. short two-GPU DDP smoke plus resume check;
4. formal Author-HD100 training (manual long run);
5. Decoder-only C inference and ARC7 seven-metric evaluation (manual formal runs).

