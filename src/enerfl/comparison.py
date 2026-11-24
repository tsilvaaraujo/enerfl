import csv
from pathlib import Path
from typing import Any

from .csv_writer import _safe_csv_path
from .server import Profiler


def compare_runs(
    *profilers: Profiler,
    metric_keys: list[str] | None = None,
) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    per_round: list[list[dict[str, Any]]] = []

    for p in profilers:
        glb = dict(p._metrics.global_metrics)
        name = p._algorithm_name or f"run_{len(summaries)}"
        glb["algorithm"] = name

        all_round_metrics: dict[str, list[float]] = {}
        round_rows: list[dict[str, Any]] = []
        for rn in sorted(p._metrics.rounds.keys()):
            row: dict[str, Any] = {"algorithm": name, "round": rn}
            for phase_metrics in p._metrics.rounds[rn].values():
                for k, v in phase_metrics.items():
                    if isinstance(v, (int, float)) and v is not None:
                        all_round_metrics.setdefault(k, []).append(float(v))
                        row[k] = v
            round_rows.append(row)

        for k, vals in all_round_metrics.items():
            if metric_keys is None or k in metric_keys:
                glb[f"mean_{k}"] = sum(vals) / len(vals)
        summaries.append(glb)
        per_round.append(round_rows)

    best = _find_best(summaries, metric_keys)
    return {"summary": summaries, "per_round": per_round, "best": best}


def compare_runs_to_csv(*profilers: Profiler, path: str) -> None:
    all_rows: list[dict[str, Any]] = []
    for p in profilers:
        name = p._algorithm_name or f"run_{len(all_rows)}"
        for rn in sorted(p._metrics.rounds.keys()):
            row: dict[str, Any] = {"algorithm": name, "round": rn}
            for phase_metrics in p._metrics.rounds[rn].values():
                row.update(phase_metrics)
            row.update(p._metrics.global_metrics)
            all_rows.append(row)

    if not all_rows:
        return

    all_keys: list[str] = []
    seen: set[str] = set()
    for row in all_rows:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    with open(_safe_csv_path(path), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)


def _find_best(
    summaries: list[dict[str, Any]],
    metric_keys: list[str] | None,
) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    if len(summaries) < 2:
        return best

    shared_keys = set(summaries[0].keys())
    for s in summaries[1:]:
        shared_keys &= set(s.keys())

    lower_is_better = {"time", "bytes", "overhead", "idle", "duration"}

    for k in sorted(shared_keys):
        vals = [
            (s["algorithm"], s[k])
            for s in summaries
            if isinstance(s[k], (int, float))
        ]
        if not vals:
            continue
        if metric_keys is not None and k not in metric_keys:
            continue
        should_minimize = any(t in k for t in lower_is_better)
        best_entry = (
            min(vals, key=lambda x: x[1])
            if should_minimize
            else max(vals, key=lambda x: x[1])
        )
        best[k] = {"algorithm": best_entry[0], "value": best_entry[1]}

    return best
