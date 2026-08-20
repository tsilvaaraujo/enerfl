import atexit
import inspect
import time
from logging import WARNING
from typing import Any

from flwr.common import log

_JOULES_PER_KWH = 3.6e6
_GLOBAL_TRACKER: "CarbonTracker | None" = None


def _finite(value: Any) -> float:
    """Coerce to a finite float; non-finite (NaN/Inf) values become 0.0.

    Short FL rounds can make CodeCarbon emit degenerate energy/emission values,
    which would otherwise poison Flower's weighted metric aggregation and drop
    the whole client MetricRecord.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return v if v == v and v not in (float("inf"), float("-inf")) else 0.0


def _filter_tracker_kwargs(klass: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(klass.__init__)
    except (ValueError, TypeError):
        return kwargs

    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs

    accepted = {
        name
        for name, p in params.items()
        if p.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    filtered = {}
    for key, value in kwargs.items():
        if key in accepted:
            filtered[key] = value
        else:
            log(WARNING, "CarbonTracker: dropping unsupported kwarg %r", key)
    return filtered


class CarbonTracker:
    def __init__(self, **tracker_kwargs: Any) -> None:
        try:
            from codecarbon import EmissionsTracker
        except ImportError as exc:
            raise ImportError(
                "CarbonTracker requires 'codecarbon'. Install: pip install codecarbon"
            ) from exc

        safe_kwargs = _filter_tracker_kwargs(
            EmissionsTracker,
            {
                "log_level": "error",
                "save_to_file": False,
                "measure_power_secs": 1,
                **tracker_kwargs,
            },
        )
        self._tracker = EmissionsTracker(**safe_kwargs)
        self._started = False
        self._stopped = False

    def start(self) -> "CarbonTracker":
        if not self._started:
            self._tracker.start()
            self._started = True
        return self

    def _energy_joules(self) -> float:
        energy = getattr(self._tracker, "_total_energy", None)
        return float(energy.kWh) * _JOULES_PER_KWH if energy is not None else 0.0

    def read(self) -> dict[str, float]:
        if not self._started:
            return {
                "total_energy_joules": 0.0,
                "total_emissions_kg": 0.0,
                "total_carbon_duration": 0.0,
            }
        emissions = float(getattr(self._tracker, "final_emissions", 0.0) or 0.0)
        start_time = getattr(self._tracker, "_start_time", None)
        duration_s = time.time() - start_time if start_time is not None else 0.0
        return {
            "total_energy_joules": _finite(self._energy_joules()),
            "total_emissions_kg": _finite(emissions),
            "total_carbon_duration": _finite(duration_s),
        }

    def snapshot(self) -> dict[str, float]:
        return self.read()

    def stop(self) -> "CarbonTracker":
        if self._started and not self._stopped:
            self._tracker.stop()
            self._stopped = True
        return self


def get_global_tracker(**kwargs: Any) -> CarbonTracker:
    global _GLOBAL_TRACKER
    if _GLOBAL_TRACKER is None:
        _GLOBAL_TRACKER = CarbonTracker(**kwargs).start()
        atexit.register(_GLOBAL_TRACKER.stop)
    return _GLOBAL_TRACKER
