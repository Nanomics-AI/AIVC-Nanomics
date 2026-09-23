#!/usr/bin/env python3
"""Print a lightweight combined status for the two formal cache workers."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
PREFIX = RESULTS / "tahoe_experiment1_cache_cap512_all_dmso"
OUTPUT = Path(str(PREFIX) + "_extraction_status.json")
REFERENCE_SINGLE_RATE = 85.47608052384246


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    workers = []
    for worker_id in (0, 1):
        path = Path(str(PREFIX) + f"_worker{worker_id}_progress.json")
        progress = read_json(path)
        total = int(progress.get("total_cells", 0))
        processed = int(progress.get("processed_cells", 0))
        remaining = int(progress.get("remaining_cells", total - processed))
        rate = float(progress.get("cells_per_active_second", 0.0))
        workers.append(
            {
                "worker_id": worker_id,
                "status": progress.get("status"),
                "processed_cells": processed,
                "remaining_cells": remaining,
                "cells_per_second": rate,
                "eta_hours": remaining / rate / 3600 if rate else None,
                "failed_attempts": int(progress.get("failed_attempts", 0)),
                "permanent_failed_cells": int(
                    progress.get("permanent_failed_cells", 0)
                ),
                "current": progress.get("current"),
                "gpu": progress.get("gpu"),
                "partial_logical_GiB": (
                    progress.get("partial_logical_bytes") / 2**30
                    if progress.get("partial_logical_bytes") is not None
                    else None
                ),
                "partial_allocated_GiB": (
                    progress.get("partial_allocated_bytes") / 2**30
                    if progress.get("partial_allocated_bytes") is not None
                    else None
                ),
                "progress_path": path.relative_to(PROJECT_ROOT).as_posix(),
            }
        )
    combined_rate = sum(worker["cells_per_second"] for worker in workers)
    worker_etas = [
        worker["remaining_cells"] / worker["cells_per_second"]
        for worker in workers
        if worker["cells_per_second"] > 0
    ]
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "pass"
            if all(worker["permanent_failed_cells"] == 0 for worker in workers)
            else "fail"
        ),
        "workers": workers,
        "combined": {
            "processed_cells": sum(worker["processed_cells"] for worker in workers),
            "remaining_cells": sum(worker["remaining_cells"] for worker in workers),
            "cells_per_second_sum": combined_rate,
            "effective_total_speedup_vs_experiment0_single": (
                combined_rate / REFERENCE_SINGLE_RATE if combined_rate else 0.0
            ),
            "parallel_wall_eta_hours": (
                max(worker_etas) / 3600 if len(worker_etas) == 2 else None
            ),
            "eta_note": "Parallel ETA is valid after both workers have measured rates.",
        },
    }
    atomic_write_json(OUTPUT, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
