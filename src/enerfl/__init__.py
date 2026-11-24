from .carbon import CarbonTracker
from .client import ClientProfiler, FL_CLIENT_STAGES
from .collectors import (
    CompressionCollector,
    MessageCountCollector,
    MetricsCollector,
    PayloadSizeCollector,
    TimingCollector,
    UpdateCountCollector,
    WeightCountCollector,
)
from .comparison import compare_runs, compare_runs_to_csv
from .csv_writer import MetricsCSVWriter
from .metrics import ProfilerMetrics
from .server import Profiler

try:
    from .strategy import profiled
except ModuleNotFoundError as exc:
    if exc.name is None or exc.name.split(".")[0] != "flwr":
        raise
    _flwr_import_error = exc

    def profiled(*_args, **_kwargs):  # type: ignore[no-redef]
        """Stand-in used when flwr is not installed."""
        raise ImportError(
            "enerfl.profiled() requires Flower. "
            "Install it with: pip install 'enerfl[flower]'"
        ) from _flwr_import_error

__all__ = [
    "Profiler",
    "ClientProfiler",
    "ProfilerMetrics",
    "MetricsCSVWriter",
    "MetricsCollector",
    "TimingCollector",
    "MessageCountCollector",
    "UpdateCountCollector",
    "WeightCountCollector",
    "PayloadSizeCollector",
    "CompressionCollector",
    "CarbonTracker",
    "FL_CLIENT_STAGES",
    "profiled",
    "compare_runs",
    "compare_runs_to_csv",
]
