import base64
import hashlib
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ============================================================
# Tahoe-100M 智能下载监控器 v3
#
# 目标：
# 1. 强制使用 HF-Mirror
# 2. 禁用 hf-xet，下载超时设为 300 秒
# 3. 第一次建立本地下载计划；后续重连不再从第 1 个文件
#    逐个调用 hf_hub_download() 检查
# 4. 只有当 hf_hub_download() 正常返回，且最终文件存在、
#    文件大小与远端计划中的 size 一致（若 size 可用）时，
#    才把该文件记录为“已完整下载”
# 5. 下载异常退出时自动重启，并从“下一个未完成文件”继续
# 6. 速度监控规则：
#    - 每 15 秒单独计算一次这一段的平均下载速度
#    - 只有连续 20 个 15 秒区间（共 5 分钟）
#      每一个区间都 < 1 MiB/s 时，才主动重连
#    - 任意一个 15 秒区间 >= 1 MiB/s，连续低速计数立即清零
# 7. 最终自动生成 GeneJEPA 官方 setup() 需要的
#    hf_data_cache/local_file_manifest.json
# ============================================================

REPO_ID = "vevotx/Tahoe-100M"

CHECK_INTERVAL = 15
SPEED_THRESHOLD = 1 * 1024 * 1024          # 1 MiB/s
LOW_SPEED_DURATION = 5 * 60                 # 5 min
LOW_INTERVALS_REQUIRED = LOW_SPEED_DURATION // CHECK_INTERVAL  # 20
RESTART_DELAY = 5

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "hf_data_cache"

PLAN_PATH = CACHE_DIR / "tahoe_download_plan.json"
STATE_PATH = CACHE_DIR / "tahoe_download_state.json"
STATUS_PATH = CACHE_DIR / "tahoe_download_status.json"
MANIFEST_PATH = CACHE_DIR / "local_file_manifest.json"


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """原子写 JSON，避免程序中断时留下半个 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def human_size(num_bytes: float) -> str:
    if num_bytes >= 1024 ** 3:
        return f"{num_bytes / (1024 ** 3):.2f} GiB"
    if num_bytes >= 1024 ** 2:
        return f"{num_bytes / (1024 ** 2):.2f} MiB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.1f} KiB"
    return f"{num_bytes:.0f} B"


def human_speed(bytes_per_second: float) -> str:
    return f"{human_size(bytes_per_second)}/s"


def status_update(**kwargs: Any) -> None:
    payload = {
        "updated_at": time.time(),
        **kwargs,
    }
    atomic_write_json(STATUS_PATH, payload)


def local_dir_for_repo_path(repo_path: str) -> Path:
    """
    与 GeneJEPA 官方 data.py 保持同样的 local_dir 规则：
    local_dir = hf_data_cache / dirname(repo_path)
    """
    dirname = os.path.dirname(repo_path)
    return CACHE_DIR / dirname if dirname else CACHE_DIR


def expected_final_path(repo_path: str) -> Path:
    """
    huggingface_hub 的 local_dir 模式会把 filename 再拼到 local_dir 下。
    """
    local_dir = local_dir_for_repo_path(repo_path)
    sanitized = Path(*repo_path.split("/"))
    return local_dir / sanitized


def expected_size_ok(path: Path, expected_size: Optional[int]) -> bool:
    if not path.is_file():
        return False
    try:
        actual = path.stat().st_size
    except OSError:
        return False

    if actual <= 0:
        return False

    if expected_size is None:
        return True

    try:
        expected = int(expected_size)
    except (TypeError, ValueError):
        return True

    return actual == expected


def build_final_file_index() -> Dict[str, List[Path]]:
    """
    仅在需要兼容旧脚本已下载的数据时扫描一次本地最终 parquet。
    跳过 .cache；不会访问网络。
    """
    index: Dict[str, List[Path]] = {}

    if not CACHE_DIR.exists():
        return index

    for root, dirs, files in os.walk(CACHE_DIR):
        # 不进入 Hugging Face 的隐藏缓存目录
        dirs[:] = [d for d in dirs if d != ".cache"]

        for filename in files:
            if not filename.endswith(".parquet"):
                continue
            path = Path(root) / filename
            index.setdefault(filename, []).append(path)

    return index


def find_existing_final(
    item: Dict[str, Any],
    fallback_index: Optional[Dict[str, List[Path]]] = None,
) -> Optional[Path]:
    """
    先检查按当前 huggingface_hub local_dir 规则推导的最终路径。
    若旧脚本布局不同，再按 basename 在本地索引中查找。
    """
    repo_path = item["path"]
    expected_size = item.get("size")

    direct = expected_final_path(repo_path)
    if expected_size_ok(direct, expected_size):
        return direct.resolve()

    if fallback_index is not None:
        basename = Path(repo_path).name
        for candidate in fallback_index.get(basename, []):
            if expected_size_ok(candidate, expected_size):
                return candidate.resolve()

    return None


def load_or_create_plan() -> Dict[str, Any]:
    """
    第一次运行时访问仓库一次，保存全部 3388 个 data parquet
    和 gene_metadata.parquet 的路径/大小。
    后续重连直接读本地 plan，不再重新列仓库。
    """
    existing = read_json(PLAN_PATH)
    if (
        existing
        and existing.get("repo_id") == REPO_ID
        and existing.get("data_files")
        and existing.get("metadata_file")
    ):
        print(
            f"📋 使用本地下载计划：{len(existing['data_files'])} 个数据文件",
            flush=True,
        )
        return existing

    status_update(phase="planning", message="正在从 Hugging Face 获取一次性文件清单")

    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    print("📋 首次运行：正在获取 Tahoe-100M 文件清单...", flush=True)

    api = HfApi()

    try:
        repo_files = list(
            api.list_repo_tree(
                REPO_ID,
                repo_type="dataset",
                recursive=True,
            )
        )
    except HfHubHTTPError:
        raise

    data_files = []
    metadata_file = None

    for f in repo_files:
        path = getattr(f, "path", None)
        if not path:
            continue

        size = getattr(f, "size", None)

        if path.startswith("data/") and path.endswith(".parquet"):
            data_files.append({"path": path, "size": size})
        elif path.endswith("gene_metadata.parquet"):
            metadata_file = {"path": path, "size": size}

    data_files.sort(key=lambda x: x["path"])

    if not data_files or metadata_file is None:
        raise FileNotFoundError(
            f"在 {REPO_ID} 中没有找到所需 parquet 文件。"
        )

    plan = {
        "version": 1,
        "repo_id": REPO_ID,
        "created_at": time.time(),
        "data_files": data_files,
        "metadata_file": metadata_file,
    }

    atomic_write_json(PLAN_PATH, plan)

    print(
        f"✅ 下载计划已保存：{len(data_files)} 个 data parquet + 1 个 metadata",
        flush=True,
    )

    return plan


def fresh_state() -> Dict[str, Any]:
    return {
        "version": 1,
        "repo_id": REPO_ID,
        "next_data_index": 0,
        "data_local_paths": [],
        "metadata_local_path": "",
        "completed_bytes": 0,
        "last_completed_repo_path": "",
        "updated_at": time.time(),
    }


def save_state(state: Dict[str, Any]) -> None:
    state["updated_at"] = time.time()
    atomic_write_json(STATE_PATH, state)


def load_or_bootstrap_state(plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    优先读取我们自己的 state。
    若第一次切换到 v3，则用本地最终文件做一次快速引导，
    找到“从 00000 开始连续完整存在”的最长前缀。

    注意：这是纯本地文件检查，不会对每个已完成文件调用
    hf_hub_download()，因此不会再产生约 1 秒/文件的网络往返。
    """
    data_items = plan["data_files"]
    state = read_json(STATE_PATH)

    if not state or state.get("repo_id") != REPO_ID:
        state = fresh_state()

    # 基本结构修复
    next_index = int(state.get("next_data_index", 0))
    paths = list(state.get("data_local_paths", []))

    if next_index < 0 or next_index > len(data_items) or len(paths) != next_index:
        print("⚠️ 本地进度文件结构不一致，将重新从本地最终文件建立进度。")
        state = fresh_state()
        next_index = 0
        paths = []

    # 如果已有 v3 state，先验证“最后一个已记录文件”仍在
    # 如果最后一个不在，则保守地重新 bootstrap。
    if next_index > 0:
        last_item = data_items[next_index - 1]
        last_path = Path(paths[-1])

        if not expected_size_ok(last_path, last_item.get("size")):
            print("⚠️ 已记录的最后一个文件无法通过本地完整性检查，重新建立进度。")
            state = fresh_state()
            next_index = 0
            paths = []

    fallback_index = None

    # 第一次迁移旧下载成果时，必要时扫描一次本地 parquet。
    if next_index == 0:
        fallback_index = build_final_file_index()

    fast_forwarded = 0

    while next_index < len(data_items):
        item = data_items[next_index]

        found = find_existing_final(item, fallback_index)

        if found is None:
            break

        size = found.stat().st_size

        state["data_local_paths"].append(str(found))
        state["completed_bytes"] = int(state.get("completed_bytes", 0)) + size
        state["last_completed_repo_path"] = item["path"]
        next_index += 1
        state["next_data_index"] = next_index
        fast_forwarded += 1

    save_state(state)

    if next_index > 0:
        print(
            f"✅ 已确认本地连续完整文件：{next_index}/{len(data_items)}"
            f"（最后一个：{state['last_completed_repo_path']}）",
            flush=True,
        )
    else:
        print("ℹ️ 尚未发现可确认的连续完整 data parquet。", flush=True)

    if fast_forwarded > 0:
        print(
            f"   本次通过本地检查快速前进 {fast_forwarded} 个文件，"
            "没有逐个访问 Hugging Face。",
            flush=True,
        )

    return state


def verify_downloaded_file(
    local_path: str,
    item: Dict[str, Any],
) -> Path:
    path = Path(local_path)

    if not path.is_file():
        raise IOError(
            f"hf_hub_download 返回成功，但最终文件不存在：{path}"
        )

    actual_size = path.stat().st_size
    expected_size = item.get("size")

    if expected_size is not None:
        try:
            expected = int(expected_size)
        except (TypeError, ValueError):
            expected = None

        if expected is not None and actual_size != expected:
            raise IOError(
                f"文件大小校验失败：{item['path']}，"
                f"实际 {actual_size} bytes，预期 {expected} bytes"
            )

    if actual_size <= 0:
        raise IOError(f"下载文件大小为 0：{item['path']}")

    return path.resolve()


def download_one(item: Dict[str, Any]) -> Path:
    from huggingface_hub import hf_hub_download

    repo_path = item["path"]
    local_dir = local_dir_for_repo_path(repo_path)

    status_update(
        phase="downloading",
        repo_path=repo_path,
        local_dir=str(local_dir),
        expected_size=item.get("size"),
    )

    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=repo_path,
        repo_type="dataset",
        cache_dir=str(CACHE_DIR),
        local_dir=str(local_dir),
    )

    return verify_downloaded_file(local_path, item)


def worker() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # 如果 GeneJEPA 官方 manifest 已存在，说明下载工作已经完成。
        if MANIFEST_PATH.exists():
            print(
                f"✅ 已存在 {MANIFEST_PATH}，GeneJEPA 会直接使用它，无需继续下载。",
                flush=True,
            )
            status_update(phase="complete")
            return

        plan = load_or_create_plan()
        state = load_or_bootstrap_state(plan)

        data_items = plan["data_files"]
        total = len(data_items)

        # ------------------------------------------------------------
        # 下载 data/*.parquet
        # ------------------------------------------------------------
        while state["next_data_index"] < total:
            idx = int(state["next_data_index"])
            item = data_items[idx]

            # 极小概率：文件已经完整落盘，但上次进程恰好在 state 写入前崩掉。
            # 这里仅做本地检查，不访问网络。
            existing = find_existing_final(item)
            if existing is not None:
                size = existing.stat().st_size
                state["data_local_paths"].append(str(existing))
                state["completed_bytes"] += size
                state["last_completed_repo_path"] = item["path"]
                state["next_data_index"] = idx + 1
                save_state(state)
                print(
                    f"⏭️  本地已完整存在，直接记录："
                    f"({idx + 1}/{total}) {item['path']}",
                    flush=True,
                )
                continue

            print(
                f"\n⬇️  ({idx + 1}/{total}) 下载 {item['path']} ...",
                flush=True,
            )

            local_path = download_one(item)
            size = local_path.stat().st_size

            # 只有到这里，才算“确认完整下载”
            state["data_local_paths"].append(str(local_path))
            state["completed_bytes"] += size
            state["last_completed_repo_path"] = item["path"]
            state["next_data_index"] = idx + 1
            save_state(state)

            print(
                f"✅ 完整确认：{item['path']} "
                f"({human_size(size)})；"
                f"下次将直接从 #{idx + 2} 开始",
                flush=True,
            )

        # ------------------------------------------------------------
        # 下载 gene_metadata.parquet
        # ------------------------------------------------------------
        metadata_item = plan["metadata_file"]

        if not state.get("metadata_local_path"):
            existing_metadata = find_existing_final(metadata_item)

            if existing_metadata is not None:
                state["metadata_local_path"] = str(existing_metadata)
                state["completed_bytes"] += existing_metadata.stat().st_size
                save_state(state)
            else:
                print(
                    f"\n⬇️  下载 metadata：{metadata_item['path']} ...",
                    flush=True,
                )

                metadata_path = download_one(metadata_item)
                state["metadata_local_path"] = str(metadata_path)
                state["completed_bytes"] += metadata_path.stat().st_size
                save_state(state)

        # ------------------------------------------------------------
        # 写 GeneJEPA 官方兼容 manifest
        # ------------------------------------------------------------
        manifest = {
            "data_files": sorted(state["data_local_paths"]),
            "metadata_file": state["metadata_local_path"],
        }

        if len(manifest["data_files"]) != total:
            raise RuntimeError(
                f"准备写 manifest 时发现 data 文件数量不对："
                f"{len(manifest['data_files'])} != {total}"
            )

        if not manifest["metadata_file"]:
            raise RuntimeError("metadata_file 为空，不能写 GeneJEPA manifest。")

        atomic_write_json(MANIFEST_PATH, manifest)

        status_update(phase="complete")

        print("\n" + "=" * 72)
        print("✅ Tahoe-100M 全部下载完成。")
        print(f"✅ GeneJEPA manifest 已生成：{MANIFEST_PATH}")
        print("=" * 72, flush=True)

    except KeyboardInterrupt:
        status_update(phase="stopped")
        print("\n下载子进程已停止。", flush=True)
        raise SystemExit(130)

    except Exception as exc:
        status_update(
            phase="error",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise


# ============================================================
# 下面是 watchdog 监控逻辑
# ============================================================

def run_worker() -> subprocess.Popen:
    env = os.environ.copy()

    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["HF_HUB_DISABLE_XET"] = "1"
    env["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"

    return subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
        ],
        env=env,
        cwd=str(BASE_DIR),
    )


def stop_worker(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return

    print("\n⚠️ 正在停止当前下载连接...", flush=True)

    try:
        process.send_signal(signal.SIGINT)
        process.wait(timeout=10)
        return
    except Exception:
        pass

    try:
        process.terminate()
        process.wait(timeout=5)
        return
    except Exception:
        pass

    try:
        process.kill()
        process.wait(timeout=5)
    except Exception:
        pass


def short_hash(filename: str) -> str:
    """
    与 huggingface_hub 本地下载缓存对 incomplete 文件名的短 hash 规则一致。
    """
    digest = hashlib.sha1(filename.encode()).digest()
    return base64.urlsafe_b64encode(digest).decode()


def current_incomplete_size(status: Dict[str, Any]) -> int:
    """
    只定位“当前正在下载文件”对应的 .incomplete 临时文件，
    不扫描全部几千个 parquet。
    """
    if status.get("phase") != "downloading":
        return 0

    repo_path = status.get("repo_path")
    local_dir_str = status.get("local_dir")

    if not repo_path or not local_dir_str:
        return 0

    local_dir = Path(local_dir_str)
    sanitized = Path(*repo_path.split("/"))

    metadata_path = (
        local_dir
        / ".cache"
        / "huggingface"
        / "download"
        / Path(str(sanitized) + ".metadata")
    )

    parent = metadata_path.parent
    if not parent.exists():
        return 0

    prefix = short_hash(metadata_path.name)
    total = 0

    try:
        for candidate in parent.glob(f"{prefix}.*.incomplete"):
            try:
                total += candidate.stat().st_size
            except OSError:
                pass
    except OSError:
        pass

    return total


def measured_total_bytes() -> int:
    """
    用于测速的累计字节数 =
        已被 worker 确认完整的文件总大小
        + 当前 .incomplete 文件已经写入的大小

    这样文件从 .incomplete 原子移动成最终文件时，
    completed_bytes 会接替它，整体计数基本连续。
    """
    state = read_json(STATE_PATH) or {}
    status = read_json(STATUS_PATH) or {}

    completed = int(state.get("completed_bytes", 0))
    incomplete = current_incomplete_size(status)

    return completed + incomplete


def watchdog() -> None:
    print("=" * 78)
    print("Tahoe-100M 智能下载监控器 v3")
    print("进度：只从“下一个未完成文件”继续，不再从第 1 个远端检查")
    print(
        f"低速规则：每 {CHECK_INTERVAL}s 独立测速；"
        f"连续 {LOW_INTERVALS_REQUIRED} 个区间（5min）"
        f"都 < {human_speed(SPEED_THRESHOLD)} 才重连"
    )
    print("任意一个 15s 区间 >= 1 MiB/s，连续低速计数立即清零")
    print(f"镜像： https://hf-mirror.com")
    print(f"缓存： {CACHE_DIR}")
    print("=" * 78)

    restart_count = 0
    process: Optional[subprocess.Popen] = None

    try:
        while True:
            label = "" if restart_count == 0 else f"（第 {restart_count} 次重连）"
            print(f"\n🚀 启动下载进程{label}...", flush=True)

            process = run_worker()

            # 每次新 worker 都重新开始监控状态。
            low_streak = 0
            last_phase = None
            last_measure_time = time.monotonic()
            last_measure_bytes = measured_total_bytes()

            while True:
                time.sleep(CHECK_INTERVAL)

                code = process.poll()

                if code == 0:
                    print("\n✅ 下载程序正常结束。")
                    return

                if code is not None:
                    print(
                        f"\n❌ 下载进程异常退出，return code = {code}。"
                    )
                    print("   → 5 秒后从下一个未完成文件自动重启。")
                    break

                status = read_json(STATUS_PATH) or {}
                phase = status.get("phase", "unknown")

                now = time.monotonic()
                current_bytes = measured_total_bytes()

                # 只有 phase=downloading 才进行低速连续计数。
                # planning / 本地快速恢复 / finalizing 等阶段不测速。
                if phase != "downloading":
                    low_streak = 0
                    last_phase = phase
                    last_measure_time = now
                    last_measure_bytes = current_bytes

                    print(
                        f"[等待] phase={phase} | "
                        "当前不是实际文件传输阶段，不进行低速判断",
                        flush=True,
                    )
                    continue

                # 刚从其他阶段切换到 downloading：
                # 从这一刻重新建立 15 秒基线，不把前面的等待时间算进去。
                if last_phase != "downloading":
                    low_streak = 0
                    last_phase = "downloading"
                    last_measure_time = now
                    last_measure_bytes = current_bytes

                    current_file = status.get("repo_path", "?")
                    print(
                        f"[测速启动] 当前文件：{current_file}",
                        flush=True,
                    )
                    continue

                elapsed = now - last_measure_time

                # 如果在文件完成/原子移动的极短窗口里计数出现回退，
                # 不把这一段错误地计成低速，直接重置基线。
                if current_bytes < last_measure_bytes:
                    low_streak = 0
                    last_measure_time = now
                    last_measure_bytes = current_bytes
                    print(
                        "[测速] 检测到文件完成/计数切换，重置本次 15s 基线。",
                        flush=True,
                    )
                    continue

                delta = current_bytes - last_measure_bytes
                interval_speed = delta / elapsed if elapsed > 0 else 0

                if interval_speed < SPEED_THRESHOLD:
                    low_streak += 1
                    verdict = (
                        f"低于阈值，连续低速 "
                        f"{low_streak}/{LOW_INTERVALS_REQUIRED}"
                    )
                else:
                    low_streak = 0
                    verdict = "速度正常，连续低速计数清零"

                current_file = status.get("repo_path", "?")

                print(
                    f"[15s测速] {current_file} | "
                    f"新增 {human_size(delta)} / {elapsed:.1f}s | "
                    f"{human_speed(interval_speed)} | {verdict}",
                    flush=True,
                )

                last_measure_time = now
                last_measure_bytes = current_bytes
                last_phase = "downloading"

                # 用户要求的是“连续 5 分钟内，每一个 15s 区间都低于 1 MiB/s”
                if low_streak >= LOW_INTERVALS_REQUIRED:
                    print("\n🚨 已满足主动重连条件：")
                    print(
                        f"   连续 {LOW_INTERVALS_REQUIRED} 个 "
                        f"{CHECK_INTERVAL}s 区间全部低于 "
                        f"{human_speed(SPEED_THRESHOLD)}"
                    )
                    print("   → 主动断开当前连接并重新建立下载。")

                    stop_worker(process)
                    break

            restart_count += 1
            print(
                f"\n🔄 {RESTART_DELAY} 秒后进行第 {restart_count} 次重连...\n",
                flush=True,
            )
            time.sleep(RESTART_DELAY)

    except KeyboardInterrupt:
        print("\n\n🛑 收到用户停止命令，正在退出...", flush=True)
        if process is not None:
            stop_worker(process)
        print("已停止。")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker()
    else:
        watchdog()
