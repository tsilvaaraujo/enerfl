"""Batch-scheduler context (SLURM, OAR) read from the job's environment."""

import itertools
import os
import re
from collections import Counter
from collections.abc import Mapping
from typing import Any

# Safety net against absurd inputs; real node lists are far smaller.
_MAX_NODES = 100_000

_SLURM_FIELDS = {
    "job_id": "SLURM_JOB_ID",
    "job_name": "SLURM_JOB_NAME",
    "partition": "SLURM_JOB_PARTITION",
    "account": "SLURM_JOB_ACCOUNT",
    "qos": "SLURM_JOB_QOS",
    "cluster": "SLURM_CLUSTER_NAME",
    "array_job_id": "SLURM_ARRAY_JOB_ID",
    "array_task_id": "SLURM_ARRAY_TASK_ID",
    "step_id": "SLURM_STEP_ID",
    "tasks_per_node": "SLURM_TASKS_PER_NODE",
}

_OAR_FIELDS = {
    "job_id": "OAR_JOB_ID",
    "job_name": "OAR_JOB_NAME",
    "array_job_id": "OAR_ARRAY_ID",
    "array_task_id": "OAR_ARRAY_INDEX",
}


def scheduler_info(env: Mapping[str, str] | None = None) -> dict[str, Any] | None:
    """Describe the batch job this process runs in, or ``None`` outside one.

    Only identifiers of the job and its nodes are collected: no user names
    or working directories.
    """
    env = os.environ if env is None else env
    if env.get("SLURM_JOB_ID"):
        return _slurm_info(env)
    if env.get("OAR_JOB_ID"):
        return _oar_info(env)
    return None


def expand_slurm_nodelist(nodelist: str) -> list[str]:
    """Expand ``node[01-03,07],gpu5`` into the individual host names.

    Raises ``ValueError`` on input it cannot parse.
    """
    hosts: list[str] = []
    for item in _split_top_level(nodelist):
        choices = []
        for part in re.split(r"(\[[^\]]*\])", item):
            if not part:
                continue
            if part.startswith("["):
                choices.append(_expand_range_group(part[1:-1]))
            elif "[" in part or "]" in part:
                raise ValueError(f"unbalanced brackets in {item!r}")
            else:
                choices.append([part])
        for combo in itertools.product(*choices):
            hosts.append("".join(combo))
            if len(hosts) > _MAX_NODES:
                raise ValueError("node list too large")
    return hosts


def _split_top_level(text: str) -> list[str]:
    items: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in text:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth < 0:
                raise ValueError("unbalanced brackets")
        if ch == "," and depth == 0:
            items.append("".join(current))
            current = []
        else:
            current.append(ch)
    if depth != 0:
        raise ValueError("unbalanced brackets")
    items.append("".join(current))
    return [i for i in items if i]


def _expand_range_group(group: str) -> list[str]:
    values: list[str] = []
    for piece in group.split(","):
        start, sep, end = piece.partition("-")
        if not start.isdigit() or (sep and not end.isdigit()):
            raise ValueError(f"bad range {piece!r}")
        if not sep:
            values.append(start)
            continue
        width = len(start)
        lo, hi = int(start), int(end)
        if hi < lo or hi - lo >= _MAX_NODES:
            raise ValueError(f"bad range {piece!r}")
        values.extend(str(n).zfill(width) for n in range(lo, hi + 1))
    return values


def _slurm_info(env: Mapping[str, str]) -> dict[str, Any]:
    info: dict[str, Any] = {"name": "slurm"}
    info.update({key: env.get(var) for key, var in _SLURM_FIELDS.items()})
    info["num_nodes"] = _int(env.get("SLURM_JOB_NUM_NODES") or env.get("SLURM_NNODES"))
    info["node_id"] = _int(env.get("SLURM_NODEID"))
    info["proc_id"] = _int(env.get("SLURM_PROCID"))
    info["cpus_on_node"] = _int(env.get("SLURM_CPUS_ON_NODE"))

    raw = env.get("SLURM_JOB_NODELIST") or env.get("SLURM_NODELIST")
    info["node_list_raw"] = raw
    info["nodes"] = None
    if raw:
        try:
            info["nodes"] = expand_slurm_nodelist(raw)
        except ValueError:
            pass  # keep the raw form; the expansion is best effort
    return info


def _oar_info(env: Mapping[str, str]) -> dict[str, Any]:
    info: dict[str, Any] = {"name": "oar"}
    info.update({key: env.get(var) for key, var in _OAR_FIELDS.items()})

    nodes = None
    slots = None
    node_file = env.get("OAR_NODEFILE") or env.get("OAR_NODE_FILE")
    if node_file:
        try:
            with open(node_file, encoding="utf-8") as f:
                counts = Counter(line.strip() for line in f if line.strip())
        except OSError:
            counts = Counter()
        if counts:
            # One line per reserved core: unique hosts keep first-seen order.
            nodes = list(counts)
            slots = dict(counts)
    info["node_file"] = node_file
    info["nodes"] = nodes
    info["slots_per_node"] = slots
    info["num_nodes"] = len(nodes) if nodes else None
    return info


def _int(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None
