#!/usr/bin/env python3
"""Run real single-GPU or torchrun DDP STATE smoke checks on Experiment 0 latents."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from tahoe_experiment1_latent_data import (
    LATENT_DIM,
    PERT_DIM,
    PROJECT_ROOT,
    SET_SIZE,
    TahoeExperiment0LatentSetDataset,
    make_dataloader,
)


RESULTS = PROJECT_ROOT / "results" / "tahoe_experiment1_state_smoke"
STATE_ROOT = PROJECT_ROOT.parent / "state-main"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def version_tuple(text: str) -> tuple[int, ...]:
    numeric = text.split("+", 1)[0].split(".")
    return tuple(int(part) for part in numeric[:3])


def environment_report(world_size: int) -> dict[str, Any]:
    import geomloss
    import hydra
    import omegaconf
    import peft
    import state
    import transformers
    from state.tx.models.state_transition import StateTransitionPerturbationModel

    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"STATE smoke requires Python 3.11, found {sys.version}")
    if version_tuple(torch.__version__) < (2, 7, 0):
        raise RuntimeError(f"torch>=2.7.0 is required, found {torch.__version__}")
    if version_tuple(transformers.__version__) < (4, 52, 3):
        raise RuntimeError(
            f"transformers>=4.52.3 is required, found {transformers.__version__}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")
    if torch.cuda.device_count() != world_size:
        raise RuntimeError(
            f"Visible CUDA devices ({torch.cuda.device_count()}) != world size ({world_size})"
        )
    cuda_probe = torch.arange(16, device="cuda", dtype=torch.float32).square().sum()
    torch.cuda.synchronize()
    if float(cuda_probe) != 1240.0:
        raise RuntimeError("CUDA arithmetic probe returned an unexpected value")
    minimum_versions = {
        "geomloss": (0, 2, 6),
        "hydra-core": (1, 3, 2),
        "peft": (0, 11, 0),
    }
    for distribution, minimum in minimum_versions.items():
        installed = importlib.metadata.version(distribution)
        if version_tuple(installed) < minimum:
            raise RuntimeError(
                f"{distribution}>={'.'.join(map(str, minimum))} is required, found {installed}"
            )

    state_source = Path(state.__file__).resolve()
    expected_source = (STATE_ROOT / "src").resolve()
    if not state_source.is_relative_to(expected_source):
        raise RuntimeError(
            f"arc-state is not imported from editable local source: {state_source}"
        )
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": True,
        "cuda_arithmetic_probe": float(cuda_probe),
        "visible_cuda_devices": torch.cuda.device_count(),
        "visible_gpu_names": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
        "transformers": transformers.__version__,
        "geomloss": importlib.metadata.version("geomloss"),
        "hydra_core": importlib.metadata.version("hydra-core"),
        "omegaconf": omegaconf.__version__,
        "peft": peft.__version__,
        "arc_state": importlib.metadata.version("arc-state"),
        "arc_state_source": str(state_source),
        "editable_local_state_source": True,
        "state_transition_model_import": StateTransitionPerturbationModel.__name__,
        "nccl_cumem_enable": os.environ.get("NCCL_CUMEM_ENABLE"),
        "nccl_cumem_host_enable": os.environ.get("NCCL_CUMEM_HOST_ENABLE"),
        "geomloss_module": str(Path(geomloss.__file__).resolve()),
        "hydra_module": str(Path(hydra.__file__).resolve()),
    }


def build_model(mode: str):
    from state.tx.models.state_transition import StateTransitionPerturbationModel

    if mode not in {"st-a", "st-r"}:
        raise ValueError(f"Unknown mode: {mode}")
    kwargs = {
        "input_dim": LATENT_DIM,
        "hidden_dim": 768,
        "output_dim": LATENT_DIM,
        "pert_dim": PERT_DIM,
        "predict_residual": mode == "st-r",
        "residual_mode": "output",
        "final_activation": "identity",
        "distributional_loss": "energy",
        "transformer_backbone_key": "llama",
        "transformer_backbone_kwargs": {
            "bidirectional_attention": True,
            "max_position_embeddings": SET_SIZE,
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_hidden_layers": 8,
            "num_attention_heads": 12,
            "num_key_value_heads": 12,
            "head_dim": 64,
            "use_cache": False,
            "attention_dropout": 0.0,
            "hidden_dropout": 0.0,
            "layer_norm_eps": 1e-6,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "tie_word_embeddings": False,
            "rotary_dim": 0,
            "use_rotary_embeddings": False,
        },
        "output_space": "embedding",
        "embed_key": "X_genejepa_epoch25",
        "gene_decoder_bool": False,
        "cell_set_len": SET_SIZE,
        "n_encoder_layers": 1,
        "n_decoder_layers": 1,
        "activation": "gelu",
        "dropout": 0.0,
        "loss": "energy",
        "blur": 0.05,
        "mmd_num_chunks": 1,
        "randomize_mmd_chunks": False,
        "extra_tokens": 0,
        "batch_encoder": False,
        "batch_predictor": False,
        "use_batch_token": False,
        "confidence_token": False,
        "finetune_vci_decoder": False,
        "log1p_from_raw_counts": False,
        "lora": {"enable": False},
    }
    # The official constructor prints the full module. Keep smoke JSON readable.
    with contextlib.redirect_stdout(io.StringIO()):
        model = StateTransitionPerturbationModel(**kwargs)
    return model, kwargs


def model_core(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def move_model_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: batch[key].to(device, non_blocking=True)
        for key in ("ctrl_cell_emb", "pert_cell_emb", "pert_emb")
    }


def compute_loss(model, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = model(batch).reshape(-1, SET_SIZE, LATENT_DIM)
    target = batch["pert_cell_emb"].reshape(-1, SET_SIZE, LATENT_DIM)
    loss = model_core(model)._compute_distribution_loss(prediction, target).mean()
    return prediction, loss


def finite_gradient_norm(model) -> float:
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=float("inf"), error_if_nonfinite=True
    )
    value = float(norm.detach())
    if not value > 0:
        raise AssertionError("Gradient norm is zero")
    return value


def full_stack_contract(
    model,
    batch: dict[str, torch.Tensor],
    mode: str,
) -> dict[str, Any]:
    from geomloss import SamplesLoss

    core = model_core(model)
    if type(core.transformer_backbone).__name__ != "LlamaBidirectionalModel":
        raise AssertionError("Smoke must use the official bidirectional Llama backbone")
    if not isinstance(core.loss_fn, SamplesLoss) or core.distributional_loss != "energy":
        raise AssertionError("Smoke must use real geomloss.SamplesLoss(loss='energy')")
    projected: list[torch.Tensor] = []
    hook = core.project_out.register_forward_hook(
        lambda _module, _inputs, output: projected.append(output.detach())
    )
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    prediction, loss = compute_loss(model, batch)
    if tuple(prediction.shape) != (batch["ctrl_cell_emb"].shape[0], SET_SIZE, LATENT_DIM):
        raise AssertionError(f"Unexpected prediction shape: {tuple(prediction.shape)}")
    if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
        raise AssertionError("Forward output or Energy loss is non-finite")
    if not (prediction < 0).any():
        raise AssertionError("No negative output coordinates survived")
    expected = projected[-1]
    if mode == "st-r":
        expected = batch["ctrl_cell_emb"] + expected
    torch.testing.assert_close(prediction, expected, rtol=1e-6, atol=1e-6)
    loss.backward()
    gradient_norm = finite_gradient_norm(model)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    hook.remove()
    model.zero_grad(set_to_none=True)
    return {
        "forward_finite": True,
        "output_shape": list(prediction.shape),
        "output_dtype": str(prediction.dtype).removeprefix("torch."),
        "output_min": float(prediction.detach().min()),
        "output_max": float(prediction.detach().max()),
        "output_negative_ratio": float((prediction.detach() < 0).float().mean()),
        "energy_loss": float(loss.detach()),
        "energy_loss_class": type(core.loss_fn).__name__,
        "energy_loss_module": type(core.loss_fn).__module__,
        "transformer_class": type(core.transformer_backbone).__name__,
        "backward_finite": True,
        "gradient_norm": gradient_norm,
        "residual_contract": (
            "Zpred = raw_Zctrl + project_out(ST_hidden)"
            if mode == "st-r"
            else "Zpred = project_out(ST_hidden)"
        ),
        "residual_contract_exact": True,
        "forward_backward_wall_seconds": elapsed,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }


def next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def parameter_sync_error(model, world_size: int) -> float:
    moments = torch.zeros(
        3,
        device=torch.device("cuda", torch.cuda.current_device()),
        dtype=torch.float64,
    )
    for parameter in model_core(model).parameters():
        values = parameter.detach().double()
        moments[0] += values.sum()
        moments[1] += values.square().sum()
        moments[2] += values.abs().sum()
    gathered = [torch.empty_like(moments) for _ in range(world_size)]
    dist.all_gather(gathered, moments)
    return max(float((value - gathered[0]).abs().max()) for value in gathered)


def save_checkpoint(
    model,
    optimizer: torch.optim.Optimizer,
    path: Path,
    mode: str,
    steps: int,
    model_kwargs: dict[str, Any],
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "format": "tahoe_experiment1_state_smoke_v1",
            "mode": mode,
            "steps": steps,
            "model_kwargs": model_kwargs,
            "model_state": model_core(model).state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        temporary,
    )
    temporary.replace(path)
    restored = torch.load(path, map_location="cpu", weights_only=False)
    if restored["mode"] != mode or restored["steps"] != steps:
        raise AssertionError("Checkpoint round-trip metadata mismatch")
    if set(restored["model_state"]) != set(model_core(model).state_dict()):
        raise AssertionError("Checkpoint round-trip state_dict keys mismatch")
    del restored
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "round_trip_load": True,
        "contains_optimizer_state": True,
    }


def run_mode(
    *,
    mode: str,
    dataset: TahoeExperiment0LatentSetDataset,
    device: torch.device,
    rank: int,
    world_size: int,
    batch_size: int,
    steps: int,
    learning_rate: float,
    seed: int,
    save_checkpoint_flag: bool,
    checkpoint_dir: Path,
) -> dict[str, Any] | None:
    distributed = world_size > 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=True,
        )
        if distributed
        else None
    )
    loader = make_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=not distributed,
        seed=seed,
        sampler=sampler,
    )
    iterator = iter(loader)
    first_cpu_batch, iterator = next_batch(iterator, loader)
    first_batch = move_model_batch(first_cpu_batch, device)

    raw_model, model_kwargs = build_model(mode)
    raw_model.to(device)
    model = (
        DistributedDataParallel(raw_model, device_ids=[device.index])
        if distributed
        else raw_model
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()

    contract = full_stack_contract(model, first_batch, mode)
    if distributed:
        dist.barrier()
    torch.cuda.reset_peak_memory_stats()
    losses = []
    gradient_norms = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _step in range(steps):
        # Repeat one fixed batch per rank: this is an overfit/plumbing smoke and
        # keeps the single-vs-DDP throughput denominator comparable.
        batch = first_batch
        optimizer.zero_grad(set_to_none=True)
        _prediction, loss = compute_loss(model, batch)
        if not torch.isfinite(loss):
            raise AssertionError("Non-finite training loss")
        loss.backward()
        gradient_norms.append(finite_gradient_norm(model))
        optimizer.step()
        report_loss = loss.detach().clone()
        if distributed:
            dist.all_reduce(report_loss, op=dist.ReduceOp.SUM)
            report_loss /= world_size
        losses.append(float(report_loss))
    overfit_decreased = min(losses[1:], default=losses[0]) < losses[0]
    if not distributed and steps >= 2 and not overfit_decreased:
        raise AssertionError("Fixed-batch overfit loss did not decrease within the smoke run")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    local_runtime = {
        "rank": rank,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
    }
    if distributed:
        runtimes: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(runtimes, local_runtime)
        sync_error = parameter_sync_error(model, world_size)
        if sync_error != 0:
            raise AssertionError(f"DDP parameter moments differ across ranks: {sync_error}")
    else:
        runtimes = [local_runtime]
        sync_error = 0.0

    checkpoint = None
    if rank == 0 and save_checkpoint_flag:
        checkpoint = save_checkpoint(
            model,
            optimizer,
            checkpoint_dir / f"{mode}_world{world_size}.pt",
            mode,
            steps,
            model_kwargs,
        )
    if distributed:
        dist.barrier()

    if rank != 0:
        return None
    global_sets = steps * batch_size * world_size
    return {
        "mode": mode,
        "world_size": world_size,
        "batch_size_per_rank": batch_size,
        "steps": steps,
        "model_parameters": sum(parameter.numel() for parameter in model_core(model).parameters()),
        "model_trainable_parameters": sum(
            parameter.numel()
            for parameter in model_core(model).parameters()
            if parameter.requires_grad
        ),
        "contract": contract,
        "training_loss": {
            "first": losses[0],
            "last": losses[-1],
            "minimum": min(losses),
            "all_finite": all(math_isfinite(value) for value in losses),
            "fixed_batch_overfit_decreased": overfit_decreased
            if not distributed
            else None,
            "values": losses,
        },
        "gradient_norm": {
            "minimum": min(gradient_norms),
            "maximum": max(gradient_norms),
            "all_finite": all(math_isfinite(value) for value in gradient_norms),
        },
        "training_wall_seconds": elapsed,
        "mean_step_wall_seconds": elapsed / steps,
        "global_sets_per_second": global_sets / elapsed,
        "global_cells_per_second": global_sets * SET_SIZE / elapsed,
        "rank_runtime": runtimes,
        "ddp_parameter_moment_max_abs_error": sync_error,
        "checkpoint": checkpoint,
        "single_gpu_data_behavior": "same fixed batch repeated for overfit smoke"
        if not distributed
        else None,
        "ddp_data_behavior": "one fixed rank-distinct batch per rank, selected by DistributedSampler"
        if distributed
        else None,
    }


def math_isfinite(value: float) -> bool:
    return not (value != value or value in {float("inf"), float("-inf")})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", choices=("st-a", "st-r"), default=["st-a", "st-r"])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--check-environment", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=RESULTS / "checkpoints")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--single-baseline", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("steps, batch-size, and learning-rate must be positive")
    return args


def main() -> None:
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if os.environ.get("NCCL_CUMEM_ENABLE") != "0" or os.environ.get(
            "NCCL_CUMEM_HOST_ENABLE"
        ) != "0":
            raise RuntimeError("This host requires both NCCL CUMEM workaround variables set to 0")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    else:
        torch.cuda.set_device(0)
    device = torch.device("cuda", local_rank if distributed else 0)

    try:
        environment = environment_report(world_size)
        if args.check_environment:
            if rank == 0:
                payload = {"status": "pass", "environment": environment}
                if args.output is not None:
                    if args.output.exists() and not args.overwrite:
                        raise FileExistsError(
                            f"Environment audit exists; inspect it or pass --overwrite: {args.output}"
                        )
                    write_json(args.output, payload)
                print(json.dumps(payload, indent=2))
            return
        output = args.output or RESULTS / (
            "ddp_full_stack.json" if distributed else "single_full_stack.json"
        )
        occupied = [output]
        if args.save_checkpoint:
            occupied.extend(
                args.checkpoint_dir / f"{mode}_world{world_size}.pt"
                for mode in args.modes
            )
        existing = [path for path in occupied if path.exists()]
        if existing and not args.overwrite:
            raise FileExistsError(
                "Smoke output exists; inspect it or pass --overwrite: "
                + ", ".join(map(str, existing))
            )
        dataset = TahoeExperiment0LatentSetDataset(split="train")
        modes: dict[str, Any] = {}
        for mode in args.modes:
            result = run_mode(
                mode=mode,
                dataset=dataset,
                device=device,
                rank=rank,
                world_size=world_size,
                batch_size=args.batch_size,
                steps=args.steps,
                learning_rate=args.learning_rate,
                seed=args.seed + (0 if mode == "st-a" else 1),
                save_checkpoint_flag=args.save_checkpoint,
                checkpoint_dir=args.checkpoint_dir,
            )
            if rank == 0:
                modes[mode] = result

        if rank == 0:
            speedup = None
            if args.single_baseline is not None:
                baseline = json.loads(args.single_baseline.read_text(encoding="utf-8"))
                speedup = {
                    mode: modes[mode]["global_cells_per_second"]
                    / baseline["modes"][mode]["global_cells_per_second"]
                    for mode in modes
                    if mode in baseline["modes"]
                }
            speedup_pass = (
                None if speedup is None else bool(speedup) and all(value > 1 for value in speedup.values())
            )
            payload = {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "fail" if speedup_pass is False else "pass",
                "scope": "real STATE Llama + real geomloss Energy smoke",
                "environment": environment,
                "dataset": {
                    "source": "Experiment 0 audited Epoch25 latent cache and fixed set plan",
                    "sets": len(dataset),
                    "conditions": int(dataset.cohort["condition_id"].nunique()),
                    "raw_signed_latent": True,
                    "expression_transforms": [],
                },
                "modes": modes,
                "ddp_speedup_vs_single": speedup,
                "ddp_throughput_improved": speedup_pass,
            }
            write_json(output, payload)
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            if speedup_pass is False:
                raise RuntimeError("DDP throughput did not exceed the single-GPU baseline")
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
