"""Run identity and the JSON metadata that makes a run reproducible."""

import json
import os
import platform
import socket
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from .scheduler import scheduler_info

SCHEMA_VERSION = 1

_TRACKED_PACKAGES = ("enerfl", "flwr", "codecarbon", "protobuf")
_GIT_TIMEOUT_SECONDS = 5


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def generate_run_id() -> str:
    """Sortable and unique: ``YYYYMMDDTHHMMSSZ-<8 hex>``."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def default_metadata_path(csv_path: str) -> Path:
    """``results/run.csv`` -> ``results/run.meta.json``."""
    return Path(csv_path).with_suffix(".meta.json")


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def machine_info() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "system": platform.system(),
        "release": platform.release(),
        "architecture": platform.machine(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "memory_total_bytes": _memory_total_bytes(),
    }


def git_info(cwd: str | None = None) -> dict[str, Any] | None:
    """Commit of the repository the process runs from (the experiment code).

    ``None`` when ``git`` or a repository is not available.
    """
    try:
        commit = _git(["rev-parse", "HEAD"], cwd)
        dirty = bool(_git(["status", "--porcelain"], cwd))
    except (OSError, subprocess.SubprocessError):
        return None
    return {"commit": commit, "dirty": dirty}


def collect_run_metadata(
    *,
    run_id: str,
    algorithm_name: str,
    started_at_utc: str,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "algorithm_name": algorithm_name,
        "started_at_utc": started_at_utc,
        "finished_at_utc": None,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "packages": package_versions(),
        "machine": machine_info(),
        "git": git_info(),
        "scheduler": scheduler_info(),
        "configuration": configuration,
        "global_metrics": {},
    }


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, sort_keys=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _git(args: list[str], cwd: str | None) -> str:
    out = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT_SECONDS,
        check=True,
    )
    return out.stdout.strip()


def _cpu_model() -> str | None:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                key, _, value = line.partition(":")
                if key.strip() in ("model name", "Model", "Hardware"):
                    return value.strip() or None
    except OSError:
        pass
    return platform.processor() or None


def _memory_total_bytes() -> int | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None
