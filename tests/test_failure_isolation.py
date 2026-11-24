import builtins
from unittest.mock import mock_open

import pytest
from flwr.common import ArrayRecord

import enerfl.client as client_mod
import enerfl.server as server_mod
import enerfl.strategy as st
from enerfl import _safe
from enerfl.client import ClientProfiler
from enerfl.collectors import MetricsCollector, TimingCollector
from enerfl.server import Profiler
from enerfl.strategy import _ProfiledStrategy, _read_network_counters


@pytest.fixture(autouse=True)
def _clean_failures(monkeypatch):
    monkeypatch.setattr(st, "_flwr_send_and_receive", None)
    _safe.reset_failures()
    yield
    _safe.reset_failures()


class _Boom(MetricsCollector):
    def __init__(self, on_start=False, on_end=False):
        self._on_start, self._on_end = on_start, on_end

    def name(self):
        return "boom"

    def on_start(self, phase, round_num, **context):
        if self._on_start:
            raise RuntimeError("on_start failed")
        return {}

    def on_end(self, phase, round_num, start_state, **context):
        if self._on_end:
            raise RuntimeError("on_end failed")
        return {"x": 1}


class _Reply:
    def __init__(self, total_time):
        self.content = {"metrics": {"client_total_time": total_time}}

    def has_error(self):
        return False


class _Grid:
    def __init__(self, replies=None, error=None):
        self._replies, self._error = replies or [], error

    def send_and_receive(self, messages, timeout):
        if self._error:
            raise self._error
        return self._replies


class _Inner:
    def summary(self):
        pass

    def configure_train(self, server_round, arrays, config, grid):
        return [object()]

    def aggregate_train(self, server_round, replies):
        return (None, {"loss": 0.1})

    def configure_evaluate(self, server_round, arrays, config, grid):
        return [object()]

    def aggregate_evaluate(self, server_round, replies):
        return {"accuracy": 0.9}


def test_guard_swallows_and_counts():
    with _safe.guard("site"):
        raise ValueError("x")
    with _safe.guard("site"):
        raise ValueError("y")
    assert _safe.failure_sites() == {"site": 2}
    assert _safe.failure_summary() == {
        "profiler_failures": 2,
        "profiler_failure_sites": "site=2",
    }


@pytest.mark.parametrize("exc", [KeyboardInterrupt, SystemExit])
def test_guard_does_not_swallow_signals(exc):
    with pytest.raises(exc):
        with _safe.guard("site"):
            raise exc()
    assert _safe.failure_count() == 0


def test_measure_isolates_collector_failures():
    p = Profiler()
    p.add_collector(_Boom(on_end=True))
    p.set_round(1)
    with p.measure("phase"):
        pass
    assert "phase_time" in p.get_metrics().rounds[1]["phase"]
    assert _safe.failure_sites() == {"collector.boom.on_end": 1}


def test_measure_skips_on_end_when_on_start_failed():
    p = Profiler()
    p.add_collector(_Boom(on_start=True, on_end=True))
    p.set_round(1)
    with p.measure("phase"):
        pass
    assert _safe.failure_sites() == {"collector.boom.on_start": 1}


def test_measure_still_propagates_errors_from_the_measured_block():
    p = Profiler()
    with pytest.raises(RuntimeError, match="experiment"):
        with p.measure("phase"):
            raise RuntimeError("experiment")
    assert _safe.failure_count() == 0


def test_profiler_survives_carbon_init_failure(monkeypatch):
    def fail(**_kw):
        raise RuntimeError("no network for geolocation")

    monkeypatch.setattr(server_mod, "get_global_tracker", fail)
    p = Profiler(enable_carbon=True)
    p.start_carbon().stop_carbon()
    assert _safe.failure_sites() == {"carbon.init": 1}


def test_profiler_survives_carbon_runtime_failures(monkeypatch):
    class Tracker:
        def start(self):
            raise RuntimeError("start")

        def stop(self):
            raise RuntimeError("stop")

        def snapshot(self):
            raise RuntimeError("snapshot")

    monkeypatch.setattr(server_mod, "get_global_tracker", lambda **_kw: Tracker())
    p = Profiler(enable_carbon=True)
    p.start_carbon().stop_carbon()
    assert _safe.failure_sites() == {
        "carbon.start": 1,
        "carbon.stop": 1,
        "carbon.snapshot": 1,
    }


def test_client_profiler_survives_carbon_init_failure(monkeypatch):
    def fail(**_kw):
        raise RuntimeError("codecarbon broken")

    monkeypatch.setattr(client_mod, "get_global_tracker", fail)
    cp = ClientProfiler(enable_carbon=True)
    with cp.stage("local_training"):
        pass
    record = cp.to_metric_record()
    assert "local_training_time" in record
    assert record["client_total_time"] >= 0


def test_client_profiler_survives_carbon_read_failure(monkeypatch):
    class Tracker:
        def read(self):
            if getattr(self, "armed", False):
                raise RuntimeError("read")
            self.armed = True
            return {}

    monkeypatch.setattr(client_mod, "get_global_tracker", lambda **_kw: Tracker())
    cp = ClientProfiler(enable_carbon=True)
    with cp.stage("local_training"):
        pass
    record = cp.to_metric_record()
    assert "local_training_time" in record
    assert _safe.failure_sites() == {"client.carbon.read": 1}


def test_csv_open_oserror_degrades(monkeypatch, tmp_path):
    def fail(*_a, **_k):
        raise PermissionError("read-only filesystem")

    monkeypatch.setattr(server_mod, "MetricsCSVWriter", fail)
    p = Profiler(stream_csv_path=str(tmp_path / "m.csv"))
    p.record("x", {"a": 1})
    p.write_round_to_csv(1)
    assert _safe.failure_sites() == {"csv.open": 1}


def test_csv_bad_extension_fails_fast(tmp_path):
    with pytest.raises(ValueError):
        Profiler(stream_csv_path=str(tmp_path / "m.txt"))


def test_csv_write_failures_do_not_raise(tmp_path):
    p = Profiler(stream_csv_path=str(tmp_path / "m.csv"))

    class Broken:
        def append_row(self, _row):
            raise OSError("disk full")

        def rewrite_rows(self, _rows):
            raise OSError("disk full")

    p._csv_writer = Broken()
    p.set_round(1)
    p.record("x", {"a": 1})
    p.write_round_to_csv(1)
    p.rewrite_csv_from_metrics()
    assert _safe.failure_sites() == {"csv.write_round": 1, "csv.rewrite": 1}


def test_malformed_proc_net_dev_is_unobservable(monkeypatch):
    content = "h1\nh2\n  eth0: 100 0 0 0 0 0 0 0 notanint 0 0 0 0 0 0 0\n"
    monkeypatch.setattr(builtins, "open", mock_open(read_data=content))
    assert _read_network_counters() is None


def _run(p, grid):
    strat = _ProfiledStrategy(_Inner(), p)
    return strat.start(
        grid=grid, initial_arrays=ArrayRecord(), num_rounds=2, timeout=1.0
    )


def test_clean_run_reports_zero_failures(monkeypatch):
    monkeypatch.setattr(st, "_read_network_counters", lambda _i=None: None)
    p = Profiler()
    _run(p, _Grid([_Reply(1.0)]))
    assert p.get_metrics().global_metrics["profiler_failures"] == 0
    assert p.get_metrics().global_metrics["profiler_failure_sites"] == ""


def test_run_completes_when_profiling_code_fails(monkeypatch):
    monkeypatch.setattr(st, "_read_network_counters", lambda _i=None: None)

    def broken(_replies):
        raise RuntimeError("bad reply shape")

    monkeypatch.setattr(st, "_client_total_times", broken)
    p = Profiler()
    result = _run(p, _Grid([_Reply(1.0)]))

    assert result is not None
    assert sorted(p.get_metrics().rounds) == [1, 2]
    g = p.get_metrics().global_metrics
    assert g["profiler_failures"] == 4
    assert "communication_train.derived=2" in g["profiler_failure_sites"]


def test_experiment_errors_are_not_swallowed(monkeypatch):
    monkeypatch.setattr(st, "_read_network_counters", lambda _i=None: None)
    p = Profiler()
    with pytest.raises(TimeoutError, match="grid down"):
        _run(p, _Grid(error=TimeoutError("grid down")))
    assert _safe.failure_count() == 0


def test_timing_collector_is_default_and_unaffected():
    p = Profiler()
    assert any(isinstance(c, TimingCollector) for c in p._collectors)


def test_start_returns_final_arrays_and_initial_arrays_without_aggregation(
    monkeypatch,
):
    monkeypatch.setattr(st, "_read_network_counters", lambda _i=None: None)
    initial = ArrayRecord()
    result = _ProfiledStrategy(_Inner(), Profiler()).start(
        grid=_Grid([_Reply(1.0)]), initial_arrays=initial, num_rounds=1, timeout=1.0
    )
    assert result.arrays is initial


def test_should_stop_ends_the_run_early_and_records_rounds_run(monkeypatch):
    monkeypatch.setattr(st, "_read_network_counters", lambda _i=None: None)
    seen = []

    def should_stop(round_num, result):
        seen.append(round_num)
        return round_num == 2

    p = Profiler()
    _ProfiledStrategy(_Inner(), p).start(
        grid=_Grid([_Reply(1.0)]),
        initial_arrays=ArrayRecord(),
        num_rounds=5,
        timeout=1.0,
        should_stop=should_stop,
    )
    assert seen == [1, 2]
    assert sorted(p.get_metrics().rounds) == [1, 2]
    assert p.get_metrics().global_metrics["num_rounds"] == 2


def test_uses_flwr_send_and_receive_when_available(monkeypatch):
    monkeypatch.setattr(st, "_read_network_counters", lambda _i=None: None)
    calls = []

    def fake(grid, messages, timeout):
        calls.append(timeout)
        return [_Reply(1.0)]

    monkeypatch.setattr(st, "_flwr_send_and_receive", fake)
    _run(Profiler(), _Grid(error=AssertionError("grid.send_and_receive used")))
    assert len(calls) == 4
