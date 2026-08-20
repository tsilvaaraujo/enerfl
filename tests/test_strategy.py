import builtins
import time
from unittest.mock import mock_open

import pytest

import enerfl.strategy as st
from enerfl.server import Profiler
from enerfl.strategy import (
    _ProfiledStrategy,
    _SendReceiveResult,
    _client_total_times,
    _read_network_counters,
    _round_efficiency,
    _scalar_metrics,
)


class _Reply:
    def __init__(self, total_time=None, error=False):
        self.content = {"metrics": {}}
        if total_time is not None:
            self.content["metrics"]["client_total_time"] = total_time
        self._error = error

    def has_error(self):
        return self._error


class _FakeResult:
    def __init__(self):
        self.arrays = None
        self.train_metrics_clientapp = {}
        self.evaluate_metrics_clientapp = {}


class _FakeInner:
    def summary(self):
        pass

    def configure_train(self, server_round, arrays, config, grid):
        return [object(), object()]

    def aggregate_train(self, server_round, replies):
        return (None, {"loss": 0.1})

    def configure_evaluate(self, server_round, arrays, config, grid):
        return [object()]

    def aggregate_evaluate(self, server_round, replies):
        return {"accuracy": 0.9}


# --- pure helpers -----------------------------------------------------------


def test_round_efficiency():
    assert _round_efficiency(10.0, 6.0, 2.0) == 0.8
    assert _round_efficiency(10.0, None, None) is None
    assert _round_efficiency(0.0, 5.0) is None
    assert _round_efficiency(10.0, True, 2.0) == 0.2  # bool ignored


def test_client_total_times_filters_errors_and_nonnumeric():
    replies = [
        _Reply(6.0),
        _Reply(error=True),
        _Reply(None),
        _Reply(True),
        _Reply(8.0),
    ]
    assert _client_total_times(replies) == [6.0, 8.0]


def test_scalar_metrics_excludes_bool_and_nonnumeric():
    out = _scalar_metrics({"a": 1, "b": 2.5, "flag": True, "name": "x"})
    assert out == {"a": 1, "b": 2.5}


def test_read_network_counters_missing_file(monkeypatch):
    def boom(*_a, **_k):
        raise OSError()

    monkeypatch.setattr(builtins, "open", boom)
    assert _read_network_counters() is None


def test_read_network_counters_parses_and_skips_loopback(monkeypatch):
    content = (
        "hdr1\n"
        "hdr2\n"
        "  eth0: 100 0 0 0 0 0 0 0 200 0 0 0 0 0 0 0\n"
        "    lo: 999 0 0 0 0 0 0 0 999 0 0 0 0 0 0 0\n"
    )
    monkeypatch.setattr(builtins, "open", mock_open(read_data=content))
    tx_total, rx_total = _read_network_counters()
    assert (tx_total, rx_total) == (200, 100)


# --- derived metrics through the phase methods ------------------------------


def test_train_phase_derived_metrics(monkeypatch):
    p = Profiler(algorithm_name="T", enable_carbon=False)
    strat = _ProfiledStrategy(_FakeInner(), p)

    sr = _SendReceiveResult(
        replies=[_Reply(6.0), _Reply(8.0)],
        elapsed=10.0,
        tx_delta=100,
        rx_delta=200,
        observed=True,
    )
    monkeypatch.setattr(
        st, "_measure_send_receive_with_network_counters", lambda **_k: sr
    )

    p.set_round(1)
    ctx = strat._profile_train_phase(
        grid=None,
        current_round=1,
        arrays=None,
        train_config=None,
        timeout=1.0,
        _t=time.perf_counter,
        _rec=p.record,
        result=_FakeResult(),
    )

    comm = p.get_metrics().rounds[1]["communication_train"]
    assert comm["client_computation_train"] == 8.0
    assert comm["network_overhead_train"] == 2.0
    assert comm["straggler_index_train"] == 2.0
    assert comm["comm_to_compute_ratio_train"] == 0.25
    assert comm["bandwidth_utilization_bytes_per_sec_train"] == 30.0
    assert ctx["client_computation"] == 8.0


def test_evaluate_phase_derived_metrics(monkeypatch):
    p = Profiler(algorithm_name="E", enable_carbon=False)
    strat = _ProfiledStrategy(_FakeInner(), p)

    sr = _SendReceiveResult(
        replies=[_Reply(4.0), _Reply(10.0)],
        elapsed=12.0,
        tx_delta=60,
        rx_delta=40,
        observed=True,
    )
    monkeypatch.setattr(
        st, "_measure_send_receive_with_network_counters", lambda **_k: sr
    )

    p.set_round(2)
    ctx = strat._profile_evaluate_phase(
        grid=None,
        current_round=2,
        arrays=None,
        evaluate_config=None,
        timeout=1.0,
        _t=time.perf_counter,
        _rec=p.record,
        result=_FakeResult(),
    )

    comm = p.get_metrics().rounds[2]["communication_evaluate"]
    assert comm["client_computation_evaluate"] == 10.0
    assert comm["network_overhead_evaluate"] == 2.0
    assert comm["straggler_index_evaluate"] == 6.0
    assert comm["comm_to_compute_ratio_evaluate"] == pytest.approx(0.2)
    assert comm["bandwidth_utilization_bytes_per_sec_evaluate"] == pytest.approx(
        100 / 12
    )
    assert ctx["client_computation"] == 10.0


def test_no_derived_metrics_without_client_times(monkeypatch):
    p = Profiler(algorithm_name="T", enable_carbon=False)
    strat = _ProfiledStrategy(_FakeInner(), p)

    sr = _SendReceiveResult(
        replies=[_Reply(None), _Reply(error=True)],
        elapsed=5.0,
        tx_delta=0,
        rx_delta=0,
        observed=False,
    )
    monkeypatch.setattr(
        st, "_measure_send_receive_with_network_counters", lambda **_k: sr
    )

    p.set_round(1)
    ctx = strat._profile_train_phase(
        grid=None,
        current_round=1,
        arrays=None,
        train_config=None,
        timeout=1.0,
        _t=time.perf_counter,
        _rec=p.record,
        result=_FakeResult(),
    )

    comm = p.get_metrics().rounds[1]["communication_train"]
    assert "client_computation_train" not in comm
    assert "bandwidth_utilization_bytes_per_sec_train" not in comm
    assert ctx["client_computation"] is None
