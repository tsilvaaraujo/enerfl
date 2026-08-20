import time
from contextlib import contextmanager
from typing import Any, Generator

from .carbon import CarbonTracker, get_global_tracker
from .csv_writer import MetricsCSVWriter

FL_CLIENT_STAGES = (
    "data_loading",
    "model_deserialization",
    "local_training",
    "local_evaluation",
    "model_serialization",
    "send_update",
)

_CLIENT_CARBON_RENAME = {
    "total_energy_joules": "client_energy_joules",
    "total_emissions_kg": "client_emissions_kg",
    "total_carbon_duration": "client_carbon_duration",
}


class ClientProfiler:
    def __init__(
        self,
        enable_carbon: bool = False,
        algorithm_name: str = "",
        enabled: bool = True,
        **carbon_kwargs: Any,
    ) -> None:
        self._enabled = enabled
        self._timings: dict[str, float] = {}
        self._payload_bytes: dict[str, int] = {}
        self._extras: dict[str, dict[str, Any]] = {}
        self._algorithm_name = algorithm_name
        self._carbon_tracker: CarbonTracker | None = None
        self._carbon_snapshot: dict[str, float] = {}
        self._carbon_baseline: dict[str, float] = {}
        if enabled and enable_carbon:
            self._carbon_tracker = get_global_tracker(**carbon_kwargs)
            self._carbon_baseline = self._carbon_tracker.read()

    @contextmanager
    def stage(
        self,
        name: str,
        *,
        payload_bytes: int | None = None,
        **extra: Any,
    ) -> Generator[None, None, None]:
        if not self._enabled:
            yield
            return
        if payload_bytes is not None:
            self._payload_bytes[name] = payload_bytes
        if extra:
            self._extras.setdefault(name, {}).update(extra)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._timings[name] = time.perf_counter() - t0

    measure = stage

    def record(self, name: str, **metrics: Any) -> None:
        self._extras.setdefault(name, {}).update(metrics)

    def get_metrics(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for stage_name, timing in self._timings.items():
            metrics[f"{stage_name}_time"] = timing
        for stage_name, nbytes in self._payload_bytes.items():
            metrics[f"{stage_name}_payload_bytes"] = nbytes
        for stage_name, extra in self._extras.items():
            for key, value in extra.items():
                metrics[f"{stage_name}_{key}"] = value
        return metrics

    def get_total_time(self) -> float:
        return sum(self._timings.values())

    def stop_carbon(self) -> dict[str, float]:
        if self._carbon_tracker is None:
            return {}
        if not self._carbon_snapshot:
            end = self._carbon_tracker.read()
            self._carbon_snapshot = {
                key: max(0.0, end.get(key, 0.0) - self._carbon_baseline.get(key, 0.0))
                for key in (
                    "total_energy_joules",
                    "total_emissions_kg",
                    "total_carbon_duration",
                )
            }
        return self._carbon_snapshot

    def to_metric_record(self) -> dict[str, float]:
        if not self._enabled:
            return {}
        out: dict[str, float] = {}
        for stage_name, timing in self._timings.items():
            out[f"{stage_name}_time"] = timing
        out["client_total_time"] = self.get_total_time()
        for key, value in self.stop_carbon().items():
            out[_CLIENT_CARBON_RENAME.get(key, key)] = float(value)
        return out

    def to_csv(self, path: str) -> None:
        row = self.get_metrics()
        if self._algorithm_name:
            row["algorithm_name"] = self._algorithm_name
        writer = MetricsCSVWriter(path, base_fields=["algorithm_name"])
        writer.rewrite_rows([row])

    @classmethod
    def from_metric_record(cls, metrics: dict[str, Any]) -> "ClientProfiler":
        cp = cls()
        for key, value in metrics.items():
            if key.endswith("_time") and isinstance(value, (int, float)):
                stage = key.removesuffix("_time")
                if stage != "client_total":
                    cp._timings[stage] = float(value)
        return cp

    def get_timing(self, stage_name: str) -> float | None:
        return self._timings.get(stage_name)

    def get_payload_bytes(self, stage_name: str) -> int | None:
        return self._payload_bytes.get(stage_name)

    def summary(self) -> None:
        if not self._enabled:
            return
        print("\n" + "-" * 50)
        title = "CLIENT PROFILER SUMMARY"
        if self._algorithm_name:
            title += f" [{self._algorithm_name}]"
        print(title)
        print("-" * 50)

        for stage_name in FL_CLIENT_STAGES:
            self._print_stage(stage_name)
        for stage_name in self._timings:
            if stage_name not in FL_CLIENT_STAGES:
                self._print_stage(stage_name)
        print("-" * 50)

    def _print_stage(self, stage_name: str) -> None:
        t = self._timings.get(stage_name)
        b = self._payload_bytes.get(stage_name)
        if t is None and b is None:
            return
        parts = []
        if t is not None:
            parts.append(f"{t:.4f}s")
        if b is not None:
            parts.append(f"{b:,} bytes")
        print(f"  {stage_name}: {' | '.join(parts)}")
