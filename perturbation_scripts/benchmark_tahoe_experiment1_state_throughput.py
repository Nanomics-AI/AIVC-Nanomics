#!/usr/bin/env python3
"""Benchmark real STATE training throughput across per-rank batch sizes."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from smoke_tahoe_experiment1_state import (
    build_model,
    compute_loss,
    environment_report,
    move_model_batch,
    write_json,
)
from tahoe_experiment1_latent_data import (
    SET_SIZE,
    TahoeExperiment0LatentSetDataset,
    make_dataloader,
)


DEFAULT_SIZES = (4, 8, 16, 32, 64)


def gradients_are_finite(model: torch.nn.Module) -> tuple[bool, int]:
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    return bool(gradients) and all(
        bool(torch.isfinite(gradient).all()) for gradient in gradients
    ), len(gradients)


def make_fixed_batch(
    dataset: TahoeExperiment0LatentSetDataset,
    *,
    batch_size: int,
    rank: int,
    world_size: int,
    seed: int,
) -> dict[str, Any]:
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=True,
        )
    loader = make_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        seed=seed,
        sampler=sampler,
    )
    try:
        return next(iter(loader))
    except StopIteration as error:
        raise ValueError(
            f"Dataset cannot provide batch_size={batch_size} on rank {rank}"
        ) from error


def training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    optimizer.zero_grad(set_to_none=True)
    prediction, loss = compute_loss(model, batch)
    loss.backward()
    optimizer.step()
    return prediction.detach(), loss.detach()


def run_trial(
    *,
    mode: str,
    batch_size: int,
    dataset: TahoeExperiment0LatentSetDataset,
    device: torch.device,
    rank: int,
    world_size: int,
    warmup_steps: int,
    timed_steps: int,
    learning_rate: float,
    seed: int,
) -> dict[str, Any] | None:
    distributed = world_size > 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cpu_batch = make_fixed_batch(
        dataset,
        batch_size=batch_size,
        rank=rank,
        world_size=world_size,
        seed=seed,
    )
    batch = move_model_batch(cpu_batch, device)
    raw_model, _model_kwargs = build_model(mode)
    raw_model.to(device)
    model = (
        DistributedDataParallel(raw_model, device_ids=[device.index])
        if distributed
        else raw_model
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()

    losses: list[torch.Tensor] = []
    final_prediction: torch.Tensor | None = None
    try:
        for _ in range(warmup_steps):
            final_prediction, _loss = training_step(model, optimizer, batch)
        if distributed:
            dist.barrier()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for _ in range(timed_steps):
            final_prediction, loss = training_step(model, optimizer, batch)
            losses.append(loss)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if distributed:
            dist.barrier()
        loss_finite = bool(torch.isfinite(torch.stack(losses)).all())
        prediction_finite = bool(torch.isfinite(final_prediction).all())
        gradient_finite, gradient_tensor_count = gradients_are_finite(model)
        local = {
            "rank": rank,
            "elapsed_seconds": elapsed,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "loss_first": float(losses[0]),
            "loss_last": float(losses[-1]),
            "loss_finite": loss_finite,
            "prediction_finite": prediction_finite,
            "gradients_finite": gradient_finite,
            "gradient_tensor_count": gradient_tensor_count,
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_total_bytes": torch.cuda.get_device_properties(device).total_memory,
        }
    except torch.OutOfMemoryError as error:
        local = {
            "rank": rank,
            "oom": True,
            "error": f"{type(error).__name__}: {error}",
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_total_bytes": torch.cuda.get_device_properties(device).total_memory,
        }

    if distributed:
        gathered: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(gathered, local)
    else:
        gathered = [local]

    result = None
    if rank == 0:
        runtimes = [item for item in gathered if item is not None]
        oom = any(item.get("oom", False) for item in runtimes)
        result = {
            "mode": mode,
            "world_size": world_size,
            "batch_size_per_rank": batch_size,
            "global_batch_size": batch_size * world_size,
            "warmup_steps": warmup_steps,
            "timed_steps": timed_steps,
            "precision": "float32",
            "oom": oom,
            "rank_runtime": runtimes,
        }
        if not oom:
            wall_seconds = max(item["elapsed_seconds"] for item in runtimes)
            global_sets = timed_steps * batch_size * world_size
            result.update(
                {
                    "step_wall_seconds": wall_seconds / timed_steps,
                    "global_sets_per_second": global_sets / wall_seconds,
                    "global_cells_per_second": global_sets * SET_SIZE / wall_seconds,
                    "peak_allocated_bytes_max_rank": max(
                        item["peak_allocated_bytes"] for item in runtimes
                    ),
                    "peak_reserved_bytes_max_rank": max(
                        item["peak_reserved_bytes"] for item in runtimes
                    ),
                    "peak_reserved_fraction_max_rank": max(
                        item["peak_reserved_bytes"] / item["gpu_total_bytes"]
                        for item in runtimes
                    ),
                    "finite_loss": all(item["loss_finite"] for item in runtimes),
                    "finite_prediction": all(
                        item["prediction_finite"] for item in runtimes
                    ),
                    "finite_gradients": all(
                        item["gradients_finite"] for item in runtimes
                    ),
                }
            )

    del optimizer, model, raw_model, batch, cpu_batch, losses, final_prediction
    gc.collect()
    torch.cuda.empty_cache()
    if distributed:
        dist.barrier()
    return result


def write_comparison_csv(
    path: Path,
    single: dict[str, Any],
    ddp: dict[str, Any],
) -> None:
    baseline = {
        (row["mode"], row["batch_size_per_rank"]): row
        for row in single["results"]
    }
    columns = [
        "mode",
        "batch_size_per_rank",
        "single_global_batch",
        "ddp_global_batch",
        "single_step_wall_seconds",
        "ddp_step_wall_seconds",
        "single_global_sets_per_second",
        "ddp_global_sets_per_second",
        "single_global_cells_per_second",
        "ddp_global_cells_per_second",
        "ddp_speedup_vs_single",
        "single_peak_allocated_GiB",
        "ddp_peak_allocated_GiB_max_rank",
        "single_peak_reserved_GiB",
        "ddp_peak_reserved_GiB_max_rank",
        "single_finite",
        "ddp_finite",
        "single_oom",
        "ddp_oom",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in ddp["results"]:
            other = baseline[(row["mode"], row["batch_size_per_rank"])]
            speedup = None
            if not row["oom"] and not other["oom"]:
                speedup = (
                    row["global_cells_per_second"]
                    / other["global_cells_per_second"]
                )
            writer.writerow(
                {
                    "mode": row["mode"],
                    "batch_size_per_rank": row["batch_size_per_rank"],
                    "single_global_batch": other["global_batch_size"],
                    "ddp_global_batch": row["global_batch_size"],
                    "single_step_wall_seconds": other.get("step_wall_seconds"),
                    "ddp_step_wall_seconds": row.get("step_wall_seconds"),
                    "single_global_sets_per_second": other.get(
                        "global_sets_per_second"
                    ),
                    "ddp_global_sets_per_second": row.get(
                        "global_sets_per_second"
                    ),
                    "single_global_cells_per_second": other.get(
                        "global_cells_per_second"
                    ),
                    "ddp_global_cells_per_second": row.get(
                        "global_cells_per_second"
                    ),
                    "ddp_speedup_vs_single": speedup,
                    "single_peak_allocated_GiB": other.get(
                        "peak_allocated_bytes_max_rank", 0
                    )
                    / 2**30,
                    "ddp_peak_allocated_GiB_max_rank": row.get(
                        "peak_allocated_bytes_max_rank", 0
                    )
                    / 2**30,
                    "single_peak_reserved_GiB": other.get(
                        "peak_reserved_bytes_max_rank", 0
                    )
                    / 2**30,
                    "ddp_peak_reserved_GiB_max_rank": row.get(
                        "peak_reserved_bytes_max_rank", 0
                    )
                    / 2**30,
                    "single_finite": all(
                        other.get(key, False)
                        for key in (
                            "finite_loss",
                            "finite_prediction",
                            "finite_gradients",
                        )
                    ),
                    "ddp_finite": all(
                        row.get(key, False)
                        for key in (
                            "finite_loss",
                            "finite_prediction",
                            "finite_gradients",
                        )
                    ),
                    "single_oom": other["oom"],
                    "ddp_oom": row["oom"],
                }
            )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", choices=("st-a", "st-r"), default=["st-a", "st-r"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=list(DEFAULT_SIZES))
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--timed-steps", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--single-baseline", type=Path)
    parser.add_argument("--comparison-csv", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if (
        any(size < 1 for size in args.batch_sizes)
        or args.warmup_steps < 1
        or args.timed_steps < 1
        or args.learning_rate <= 0
    ):
        parser.error("batch sizes, step counts, and learning rate must be positive")
    if args.comparison_csv is not None and args.single_baseline is None:
        parser.error("--comparison-csv requires --single-baseline")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if (
            os.environ.get("NCCL_CUMEM_ENABLE") != "0"
            or os.environ.get("NCCL_CUMEM_HOST_ENABLE") != "0"
        ):
            raise RuntimeError("Both NCCL CUMEM workaround variables must be 0")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    else:
        torch.cuda.set_device(0)
    device = torch.device("cuda", local_rank if distributed else 0)

    try:
        environment = environment_report(world_size)
        if args.output.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists; inspect it or pass --overwrite: {args.output}")
        dataset = TahoeExperiment0LatentSetDataset(split="train")
        results = []
        for mode_index, mode in enumerate(args.modes):
            for batch_size in args.batch_sizes:
                result = run_trial(
                    mode=mode,
                    batch_size=batch_size,
                    dataset=dataset,
                    device=device,
                    rank=rank,
                    world_size=world_size,
                    warmup_steps=args.warmup_steps,
                    timed_steps=args.timed_steps,
                    learning_rate=args.learning_rate,
                    seed=args.seed + 1000 * mode_index + batch_size,
                )
                if rank == 0:
                    results.append(result)
                    print(json.dumps(result, ensure_ascii=False), flush=True)

        if rank == 0:
            payload = {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "pass"
                if all(
                    not row["oom"]
                    and row["finite_loss"]
                    and row["finite_prediction"]
                    and row["finite_gradients"]
                    for row in results
                )
                else "fail",
                "scope": (
                    "Training-throughput plumbing benchmark only; global batch differs "
                    "between single GPU and DDP and results do not compare convergence."
                ),
                "environment": environment,
                "dataset": {
                    "source": "Experiment 0 audited raw signed Epoch25 latent cache",
                    "sets": len(dataset),
                    "cell_set_len": SET_SIZE,
                    "expression_transforms": [],
                },
                "benchmark": {
                    "warmup_steps": args.warmup_steps,
                    "timed_steps": args.timed_steps,
                    "optimizer": "AdamW",
                    "learning_rate": args.learning_rate,
                    "precision": "float32",
                    "timed_region": "forward + real Energy loss + backward + optimizer step",
                    "data_transfer_in_timed_region": False,
                    "checkpoint_saved": False,
                },
                "results": results,
            }
            if args.single_baseline is not None:
                baseline = json.loads(args.single_baseline.read_text(encoding="utf-8"))
                baseline_rows = {
                    (row["mode"], row["batch_size_per_rank"]): row
                    for row in baseline["results"]
                }
                for row in payload["results"]:
                    other = baseline_rows[(row["mode"], row["batch_size_per_rank"])]
                    row["throughput_speedup_vs_single"] = (
                        None
                        if row["oom"] or other["oom"]
                        else row["global_cells_per_second"]
                        / other["global_cells_per_second"]
                    )
                if args.comparison_csv is not None:
                    write_comparison_csv(args.comparison_csv, baseline, payload)
            write_json(args.output, payload)
            print(f"Wrote {args.output}", flush=True)
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
