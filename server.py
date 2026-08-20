import re
import time
from contextlib import contextmanager
from typing import Any, Generator

from .carbon import CarbonTracker, get_global_tracker
from .collectors import (
    CompressionCollector,
    MessageCountCollector,
    MetricsCollector,
    PayloadSizeCollector,
    TimingCollector,
    UpdateCountCollector,
    WeightCountCollector,
)
from .csv_writer import MetricsCSVWriter
from .metrics import ProfilerMetrics

_PER_CLIENT_RE = re.compile(r"^client_\d+_")


class Profiler:
    def __init__(
        self,
        enable_timing: bool = True,
        enable_message_count: bool = False,
        enable_update_count: bool = True,
        enable_weights_count: bool = False,
        enable_payload_size: bool = True,
        enable_compression: bool = False,
        enable_carbon: bool = False,
        carbon_tracker_kwargs: dict[str, Any] | None = None,
        stream_csv_path: str | None = None,
        network_interface: str | None = None,
        algorithm_name: str = "",
        round_num: int = 0,
        enabled: bool = True,
    ) -> None:
        self._enabled = enabled
        self._collectors: list[MetricsCollector] = []
        self._metrics = ProfilerMetrics()
        self._current_round = round_num
        self._algorithm_name = algorithm_name
        self._cumulative_bytes_sent: int = 0
        self._cumulative_bytes_received: int = 0
        self._csv_writer: MetricsCSVWriter | None = None
        self._network_interface = network_interface
        self._carbon_tracker: CarbonTracker | None = None

        if not enabled:
            return

        if algorithm_name:
            self._metrics.record_global("algorithm_name", algorithm_name)

        collector_map = [
            (enable_timing, TimingCollector),
            (enable_message_count, MessageCountCollector),
            (enable_update_count, UpdateCountCollector),
            (enable_weights_count, WeightCountCollector),
            (enable_payload_size, PayloadSizeCollector),
            (enable_compression, CompressionCollector),
        ]
        for enabled_flag, cls in collector_map:
            if enabled_flag:
                self._collectors.append(cls())

        if enable_carbon:
            self._carbon_tracker = get_global_tracker(**(carbon_tracker_kwargs or {}))
        if stream_csv_path:
            self.enable_csv_stream(stream_csv_path)

    def start_carbon(self) -> "Profiler":
        if self._carbon_tracker is not None:
            self._carbon_tracker.start()
        return self

    def stop_carbon(self) -> "Profiler":
        if self._carbon_tracker is not None:
            self._carbon_tracker.stop()
            for name, value in self._carbon_tracker.snapshot().items():
                self._metrics.record_global(name, value)
        return self

    def add_collector(self, collector: MetricsCollector) -> "Profiler":
        self._collectors.append(collector)
        return self

    def remove_collector(self, name: str) -> "Profiler":
        self._collectors = [c for c in self._collectors if c.name() != name]
        return self

    def set_round(self, round_num: int) -> "Profiler":
        self._current_round = round_num
        return self

    @contextmanager
    def measure(self, phase: str, **context: Any) -> Generator[None, None, None]:
        t0 = time.perf_counter()
        start_states = {
            c.name(): c.on_start(phase, self._current_round, t0=t0, **context)
            for c in self._collectors
        }
        yield
        for collector in self._collectors:
            metrics = collector.on_end(
                phase,
                self._current_round,
                start_states.get(collector.name(), {}),
                **context,
            )
            self._metrics.record(self._current_round, phase, metrics)

    def record(self, phase: str, metrics: dict[str, Any]) -> None:
        self._metrics.record(self._current_round, phase, metrics)

    def record_global(self, name: str, value: Any) -> None:
        self._metrics.record_global(name, value)

    def enable_csv_stream(self, path: str) -> "Profiler":
        self._csv_writer = MetricsCSVWriter(path, base_fields=["algorithm", "round"])
        return self

    def write_round_to_csv(self, round_num: int | None = None) -> None:
        if self._csv_writer is None:
            return
        rn = self._current_round if round_num is None else round_num
        self._csv_writer.append_row(self._build_round_row(rn))

    def rewrite_csv_from_metrics(self, path: str | None = None) -> None:
        if path is not None:
            self.enable_csv_stream(path)
        if self._csv_writer is None:
            return
        rows = [
            self._build_round_row(round_num)
            for round_num in sorted(self._metrics.rounds.keys())
        ]
        self._csv_writer.rewrite_rows(rows)

    @property
    def algorithm_name(self) -> str:
        return self._algorithm_name

    def get_metrics(self) -> ProfilerMetrics:
        return self._metrics

    def get_metrics_dict(self, round_num: int | None = None) -> dict[str, Any]:
        return self._metrics.get_flat_metrics(
            round_num if round_num is not None else self._current_round
        )

    def summary(
        self, show_phases: bool = False, show_per_client: bool = False
    ) -> None:
        print("\n" + "=" * 65)
        title = "PROFILER SUMMARY"
        if self._algorithm_name:
            title += f" [{self._algorithm_name}]"
        print(title)
        print("=" * 65)

        if self._metrics.global_metrics:
            print("\nGlobal:")
            for name, value in self._metrics.global_metrics.items():
                _print_metric(name, value, indent=2)

        for round_num in sorted(self._metrics.rounds.keys()):
            round_data = self._metrics.rounds[round_num]
            print(f"\nRound {round_num}:")

            if show_phases:
                for phase, metrics in round_data.items():
                    print(f"  {phase}:")
                    for name, value in metrics.items():
                        if not show_per_client and _PER_CLIENT_RE.match(name):
                            continue
                        _print_metric(name, value, indent=4)
            else:
                all_metrics = {}
                for metrics in round_data.values():
                    all_metrics.update(metrics)
                for name, value in sorted(all_metrics.items()):
                    if not show_per_client and _PER_CLIENT_RE.match(name):
                        continue
                    _print_metric(name, value, indent=2)

        print("=" * 65)

    def to_dataframe(self) -> "pd.DataFrame":
        import pandas as pd

        rows: list[dict[str, Any]] = []
        for round_num in sorted(self._metrics.rounds.keys()):
            row: dict[str, Any] = {
                "algorithm": self._algorithm_name,
                "round": round_num,
            }
            for phase_metrics in self._metrics.rounds[round_num].values():
                row.update(phase_metrics)
            row.update(self._metrics.global_metrics)
            rows.append(row)
        return pd.DataFrame(rows)

    def to_csv(self, path: str) -> None:
        self.rewrite_csv_from_metrics(path)

    def reset(self) -> None:
        self._metrics = ProfilerMetrics()
        self._current_round = 0
        self._cumulative_bytes_sent = 0
        self._cumulative_bytes_received = 0
        if self._csv_writer is not None:
            self._csv_writer.reset()
        if self._carbon_tracker is not None:
            self._carbon_tracker.stop()
            self._carbon_tracker = None

    def _build_round_row(self, round_num: int) -> dict[str, Any]:
        row: dict[str, Any] = {
            "algorithm": self._algorithm_name,
            "round": round_num,
        }
        for phase_metrics in self._metrics.rounds.get(round_num, {}).values():
            row.update(phase_metrics)
        row.update(self._metrics.global_metrics)
        return row


def _print_metric(name: str, value: Any, indent: int = 0) -> None:
    prefix = " " * indent
    if isinstance(value, float):
        print(f"{prefix}{name}: {value:.4f}")
    else:
        print(f"{prefix}{name}: {value}")
