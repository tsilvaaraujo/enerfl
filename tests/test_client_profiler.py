import time

import enerfl.client as client_mod
from enerfl.client import ClientProfiler


class _FakeTracker:
    def __init__(self):
        self.values = {
            "total_energy_joules": 0.0,
            "total_emissions_kg": 0.0,
            "total_carbon_duration": 0.0,
        }

    def read(self):
        return dict(self.values)

    def start(self):
        return self

    def stop(self):
        return self


def test_stage_timing_and_total():
    cp = ClientProfiler(enable_carbon=False)
    with cp.stage("data_loading"):
        time.sleep(0.01)
    with cp.stage("local_training"):
        time.sleep(0.01)

    assert cp.get_timing("data_loading") >= 0.01
    assert cp.get_total_time() >= 0.02

    record = cp.to_metric_record()
    assert "data_loading_time" in record
    assert "local_training_time" in record
    assert record["client_total_time"] >= 0.02


def test_disabled_profiler_is_noop():
    cp = ClientProfiler(enabled=False, enable_carbon=False)
    with cp.stage("data_loading"):
        pass
    assert cp.to_metric_record() == {}


def test_payload_and_extra_metrics():
    cp = ClientProfiler(enable_carbon=False)
    with cp.stage("model_serialization", payload_bytes=2048):
        pass
    cp.record("local_training", epochs=3)
    metrics = cp.get_metrics()
    assert metrics["model_serialization_payload_bytes"] == 2048
    assert metrics["local_training_epochs"] == 3


def test_carbon_delta_over_baseline(monkeypatch):
    fake = _FakeTracker()
    monkeypatch.setattr(client_mod, "get_global_tracker", lambda **_k: fake)

    cp = ClientProfiler(enable_carbon=True)
    fake.values = {
        "total_energy_joules": 12.0,
        "total_emissions_kg": 3.0e-3,
        "total_carbon_duration": 5.0,
    }

    record = cp.to_metric_record()
    assert record["client_energy_joules"] == 12.0
    assert record["client_emissions_kg"] == 3.0e-3
    assert record["client_carbon_duration"] == 5.0


def test_carbon_delta_never_negative(monkeypatch):
    fake = _FakeTracker()
    fake.values["total_energy_joules"] = 100.0
    monkeypatch.setattr(client_mod, "get_global_tracker", lambda **_k: fake)

    cp = ClientProfiler(enable_carbon=True)
    fake.values["total_energy_joules"] = 40.0

    assert cp.stop_carbon()["total_energy_joules"] == 0.0


def test_from_metric_record_roundtrip():
    src = {
        "data_loading_time": 1.5,
        "local_training_time": 2.5,
        "client_total_time": 4.0,
        "num-examples": 10,
    }
    cp = ClientProfiler.from_metric_record(src)
    assert cp.get_timing("data_loading") == 1.5
    assert cp.get_timing("local_training") == 2.5
    assert cp.get_timing("client_total") is None
