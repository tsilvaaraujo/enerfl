import sys
import types
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _install_flwr_stub() -> None:
    try:
        import flwr

        return
    except ImportError:
        pass

    flwr = types.ModuleType("flwr")
    common = types.ModuleType("flwr.common")
    serverapp = types.ModuleType("flwr.serverapp")
    strategy_pkg = types.ModuleType("flwr.serverapp.strategy")
    strategy_mod = types.ModuleType("flwr.serverapp.strategy.strategy")

    def log(*_args, **_kwargs):
        return None

    class ArrayRecord:
        def __init__(self, *_args, **_kwargs):
            pass

    class ConfigRecord(dict):
        pass

    class MetricRecord(dict):
        pass

    class Message:
        def __init__(self, content=None, reply_to=None):
            self.content = content if content is not None else {}
            self.reply_to = reply_to

        def has_error(self):
            return False

    class Grid:
        pass

    class Strategy:
        pass

    class Result:
        def __init__(self):
            self.arrays = None
            self.train_metrics_clientapp = {}
            self.evaluate_metrics_clientapp = {}
            self.evaluate_metrics_serverapp = {}

    def log_strategy_start_info(*_args, **_kwargs):
        return None

    common.log = log
    common.ArrayRecord = ArrayRecord
    common.ConfigRecord = ConfigRecord
    common.MetricRecord = MetricRecord
    common.Message = Message
    serverapp.Grid = Grid
    serverapp.strategy = strategy_pkg
    strategy_pkg.Strategy = Strategy
    strategy_pkg.strategy = strategy_mod
    strategy_mod.Result = Result
    strategy_mod.log_strategy_start_info = log_strategy_start_info
    flwr.common = common
    flwr.serverapp = serverapp

    sys.modules["flwr"] = flwr
    sys.modules["flwr.common"] = common
    sys.modules["flwr.serverapp"] = serverapp
    sys.modules["flwr.serverapp.strategy"] = strategy_pkg
    sys.modules["flwr.serverapp.strategy.strategy"] = strategy_mod


def _install_codecarbon_stub() -> None:
    try:
        import codecarbon  # noqa: F401

        return
    except ImportError:
        pass

    codecarbon = types.ModuleType("codecarbon")

    class _Energy:
        def __init__(self, kwh=0.0):
            self.kWh = kwh

    class EmissionsTracker:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._total_energy = _Energy(0.0)
            self._start_time = None
            self.final_emissions = 0.0
            self._flush_value = 0.0
            self.started = False
            self.stopped = False

        def start(self):
            self.started = True
            self._start_time = 1000.0

        def flush(self):
            return self._flush_value

        def stop(self):
            self.stopped = True

    codecarbon.EmissionsTracker = EmissionsTracker
    codecarbon._Energy = _Energy
    sys.modules["codecarbon"] = codecarbon


_install_flwr_stub()
_install_codecarbon_stub()
