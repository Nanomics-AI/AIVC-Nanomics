#!/usr/bin/env bash

LOG_FILE="logs/genejepa_quarter_d12_h6_700k_e30_seed42_run1.log"

RETRY_DELAY=60
MAX_RETRIES=20
attempt=0

export GENEJEPA_TRAIN_WORKERS=2
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_CUMEM_ENABLE=0
export NCCL_CUMEM_HOST_ENABLE=0
export WANDB_MODE=offline

while true; do
    attempt=$((attempt + 1))

    echo "" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting training attempt $attempt" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"

    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" | tee -a "$LOG_FILE"

    uv run -m genejepa.train \
        2>&1 | tee -a "$LOG_FILE"

    exit_code=${PIPESTATUS[0]}

    if [ "$exit_code" -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training completed successfully." \
            | tee -a "$LOG_FILE"
        break
    fi

    echo "" | tee -a "$LOG_FILE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training crashed with exit code $exit_code." \
        | tee -a "$LOG_FILE"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU status after crash:" \
        | tee -a "$LOG_FILE"

    nvidia-smi 2>&1 | tee -a "$LOG_FILE"

    if [ "$attempt" -ge "$MAX_RETRIES" ]; then
        echo "Reached maximum retry count: $MAX_RETRIES" \
            | tee -a "$LOG_FILE"
        exit "$exit_code"
    fi

    echo "Waiting ${RETRY_DELAY}s before automatic resume..." \
        | tee -a "$LOG_FILE"

    sleep "$RETRY_DELAY"
done
