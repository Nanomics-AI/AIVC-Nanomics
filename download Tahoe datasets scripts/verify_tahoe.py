#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path

EXPECTED_GENEJEPA_DATA_FILES = 3388


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def human_size(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024


def resolve_path(p: str, base: Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def main():
    parser = argparse.ArgumentParser(
        description="Validate locally downloaded Tahoe-100M files for GeneJEPA."
    )
    parser.add_argument(
        "--cache-dir",
        default="hf_data_cache",
        help="GeneJEPA cache directory (default: ./hf_data_cache)",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Deep check: read every Parquet row group. This can take a long time.",
    )
    args = parser.parse_args()

    cache = Path(args.cache_dir).resolve()
    manifest_path = cache / "local_file_manifest.json"
    plan_path = cache / "tahoe_download_plan.json"
    state_path = cache / "tahoe_download_state.json"

    print("=" * 78)
    print("Tahoe-100M / GeneJEPA 本地数据完整性检查")
    print(f"缓存目录: {cache}")
    print(f"模式: {'深度检查（读取所有 row groups）' if args.deep else '标准检查（大小 + Parquet footer）'}")
    print("=" * 78)

    errors = []
    warnings = []

    if not manifest_path.exists():
        print(f"\n❌ 缺少 GeneJEPA manifest: {manifest_path}")
        print("   GeneJEPA 官方 prepare_data() 只有在全部文件完成后才会写这个文件。")
        return 2

    manifest = load_json(manifest_path)
    data_files = manifest.get("data_files") or []
    metadata_file = manifest.get("metadata_file") or ""

    print("\n[1/5] 检查 manifest")
    print(f"  data_files 数量: {len(data_files)}")
    print(f"  metadata_file: {'有' if metadata_file else '无'}")

    if len(data_files) != EXPECTED_GENEJEPA_DATA_FILES:
        errors.append(
            f"manifest 中 data_files={len(data_files)}，"
            f"GeneJEPA 当前代码按 {EXPECTED_GENEJEPA_DATA_FILES} 个 shard 设计。"
        )
    if not metadata_file:
        errors.append("manifest 中 metadata_file 为空。")

    plan = load_json(plan_path) if plan_path.exists() else None
    state = load_json(state_path) if state_path.exists() else None

    print("\n[2/5] 检查 v3 下载记录")
    if plan:
        plan_data = plan.get("data_files") or []
        print(f"  download plan: 有，data files={len(plan_data)}")
        if len(plan_data) != len(data_files):
            errors.append(
                f"plan 中 data_files={len(plan_data)}，manifest 中={len(data_files)}。"
            )
    else:
        print("  download plan: 无（不是致命问题）")
        warnings.append("没有 tahoe_download_plan.json，无法使用下载时保存的远端 size 做逐文件比对。")

    if state:
        next_idx = int(state.get("next_data_index", -1))
        state_paths = state.get("data_local_paths") or []
        print(f"  download state: 有，next_data_index={next_idx}")
        if plan:
            if next_idx != len(plan.get("data_files") or []):
                errors.append(
                    f"state 的 next_data_index={next_idx}，"
                    f"但 plan 共有 {len(plan.get('data_files') or [])} 个 data files。"
                )
        if len(state_paths) != len(data_files):
            errors.append(
                f"state 中记录 {len(state_paths)} 个 data 文件，manifest 中为 {len(data_files)}。"
            )

        manifest_set = {str(resolve_path(p, cache)) for p in data_files}
        state_set = {str(resolve_path(p, cache)) for p in state_paths}
        if manifest_set != state_set:
            errors.append("manifest 与 v3 state 记录的 data 文件集合不一致。")
    else:
        print("  download state: 无（不是致命问题）")
        warnings.append("没有 tahoe_download_state.json，无法交叉核对 v3 的完成进度。")

    print("\n[3/5] 检查文件存在性与文件大小")

    resolved_data = [resolve_path(p, cache) for p in data_files]
    resolved_meta = resolve_path(metadata_file, cache) if metadata_file else None

    total_bytes = 0
    missing = []

    for path in resolved_data:
        if not path.is_file():
            missing.append(str(path))
            continue
        try:
            total_bytes += path.stat().st_size
        except OSError as e:
            errors.append(f"无法 stat 文件 {path}: {e}")

    if resolved_meta:
        if not resolved_meta.is_file():
            missing.append(str(resolved_meta))
        else:
            total_bytes += resolved_meta.stat().st_size

    if missing:
        errors.append(f"有 {len(missing)} 个 manifest 文件不存在。")
        for p in missing[:10]:
            print(f"  ❌ missing: {p}")
        if len(missing) > 10:
            print(f"  ... 另外还有 {len(missing) - 10} 个")
    else:
        print("  ✅ manifest 中所有文件都存在")
        print(f"  本次涉及总大小: {human_size(total_bytes)}")

    size_mismatches = []
    if plan and state:
        plan_data = plan.get("data_files") or []
        state_paths = state.get("data_local_paths") or []
        if len(plan_data) == len(state_paths):
            for i, (item, pstr) in enumerate(zip(plan_data, state_paths)):
                expected = item.get("size")
                if expected is None:
                    continue
                path = resolve_path(pstr, cache)
                if not path.is_file():
                    continue
                actual = path.stat().st_size
                try:
                    expected = int(expected)
                except (TypeError, ValueError):
                    continue
                if actual != expected:
                    size_mismatches.append(
                        (i, item.get("path"), expected, actual, str(path))
                    )

            meta_item = plan.get("metadata_file") or {}
            expected = meta_item.get("size")
            state_meta = state.get("metadata_local_path") or ""
            if expected is not None and state_meta:
                path = resolve_path(state_meta, cache)
                if path.is_file():
                    actual = path.stat().st_size
                    try:
                        expected = int(expected)
                    except (TypeError, ValueError):
                        expected = None
                    if expected is not None and actual != expected:
                        size_mismatches.append(
                            ("metadata", meta_item.get("path"), expected, actual, str(path))
                        )

            if size_mismatches:
                errors.append(f"发现 {len(size_mismatches)} 个文件大小与下载计划不一致。")
                for x in size_mismatches[:10]:
                    print(
                        f"  ❌ size mismatch: {x[1]} "
                        f"expected={x[2]} actual={x[3]}"
                    )
            else:
                print("  ✅ 文件大小与 v3 下载计划记录一致")
        else:
            warnings.append("plan/state 文件数不一致，跳过逐文件远端 size 比对。")

    print("\n[4/5] 检查 Parquet 结构")

    try:
        import pyarrow.parquet as pq
    except Exception as e:
        errors.append(f"无法 import pyarrow.parquet: {e}")
        pq = None

    parquet_failures = []
    total_rows = 0
    checked = 0
    start = time.time()

    parquet_paths = list(resolved_data)
    if resolved_meta:
        parquet_paths.append(resolved_meta)

    if pq is not None:
        for idx, path in enumerate(parquet_paths, start=1):
            if not path.is_file():
                continue
            try:
                pf = pq.ParquetFile(path)
                md = pf.metadata
                total_rows += md.num_rows
                checked += 1

                if args.deep:
                    for rg in range(md.num_row_groups):
                        pf.read_row_group(rg)

            except Exception as e:
                parquet_failures.append((str(path), repr(e)))

            if idx % 100 == 0 or idx == len(parquet_paths):
                elapsed = time.time() - start
                print(
                    f"\r  已检查 {idx}/{len(parquet_paths)} 个文件 | "
                    f"失败 {len(parquet_failures)} | "
                    f"{elapsed/60:.1f} min",
                    end="",
                    flush=True,
                )

        print()

        if parquet_failures:
            errors.append(f"有 {len(parquet_failures)} 个 Parquet 文件无法正常读取。")
            for p, e in parquet_failures[:10]:
                print(f"  ❌ parquet failure: {p}")
                print(f"     {e}")
        else:
            print(f"  ✅ {checked} 个 Parquet 文件均可读取")
            print(f"  可读取到的总 row 数（含 gene metadata）: {total_rows:,}")

    print("\n[5/5] 检查残留 .incomplete 文件")
    incomplete = []
    if cache.exists():
        for root, _, files in os.walk(cache):
            for name in files:
                if name.endswith(".incomplete"):
                    incomplete.append(Path(root) / name)

    if incomplete:
        warnings.append(f"发现 {len(incomplete)} 个残留 .incomplete 文件。")
        print(f"  ⚠️ 发现 {len(incomplete)} 个 .incomplete 文件")
        for p in incomplete[:10]:
            try:
                print(f"     {p} ({human_size(p.stat().st_size)})")
            except OSError:
                print(f"     {p}")
        if len(incomplete) > 10:
            print(f"     ... 另外还有 {len(incomplete) - 10} 个")
        print("  先不要删除；若所有正式文件均通过检查，它们通常只是旧下载残留。")
    else:
        print("  ✅ 没有发现 .incomplete 残留")

    print("\n" + "=" * 78)
    if errors:
        print("❌ 验证未通过")
        print("\n错误：")
        for e in errors:
            print(f"  - {e}")
        if warnings:
            print("\n警告：")
            for w in warnings:
                print(f"  - {w}")
        print("=" * 78)
        return 1

    print("✅ 验证通过")
    print("  - GeneJEPA manifest 完整")
    print("  - 所有 manifest 文件存在")
    if plan and state:
        print("  - v3 下载进度记录一致，并完成 size 校验")
    if pq is not None:
        print(
            "  - 所有 Parquet 文件均可正常读取"
            + ("（已读取全部 row groups）" if args.deep else "（footer/metadata 检查）")
        )
    if warnings:
        print("\n附加警告：")
        for w in warnings:
            print(f"  - {w}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
