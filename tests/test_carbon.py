import time

from codecarbon import EmissionsTracker, OfflineEmissionsTracker

import enerfl.carbon as carbon_mod
from enerfl.carbon import CarbonTracker, _finite, get_global_tracker
from enerfl.client import ClientProfiler


def test_finite_coerces_bad_values():
    assert _finite(2.5) == 2.5
    assert _finite(float("nan")) == 0.0
    assert _finite(float("inf")) == 0.0
    assert _finite("not-a-number") == 0.0
    assert _finite(None) == 0.0


def test_read_before_start_is_zero():
    t = CarbonTracker()
    reading = t.read()
    assert reading == {
        "total_energy_joules": 0.0,
        "total_emissions_kg": 0.0,
        "total_carbon_duration": 0.0,
    }


def test_energy_converted_to_joules():
    t = CarbonTracker().start().stop()
    t._tracker._total_energy.kWh = 2.0
    assert t.read()["total_energy_joules"] == 2.0 * 3.6e6


def test_defaults_reach_the_tracker():
    t = CarbonTracker()
    assert t._tracker._save_to_file is False
    assert t._tracker._log_level == "error"
    assert t._tracker._measure_power_secs == 1


def test_user_kwargs_override_defaults():
    t = CarbonTracker(measure_power_secs=5)
    assert t._tracker._measure_power_secs == 5


def test_country_iso_code_reaches_the_tracker():
    t = CarbonTracker(country_iso_code="FRA")
    assert isinstance(t._tracker, OfflineEmissionsTracker)
    assert t._tracker._country_iso_code == "FRA"


def test_without_country_the_online_tracker_is_used():
    t = CarbonTracker()
    assert type(t._tracker) is EmissionsTracker


def test_global_tracker_is_singleton():
    carbon_mod._GLOBAL_TRACKER = None
    try:
        a = get_global_tracker(country_iso_code="FRA")
        b = get_global_tracker()
        assert a is b
    finally:
        carbon_mod._GLOBAL_TRACKER = None


class _Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


def test_duration_while_running_is_the_elapsed_time(monkeypatch):
    clock = _Clock(100.0)
    monkeypatch.setattr(carbon_mod, "perf_counter", clock)
    t = CarbonTracker().start()
    clock.now = 103.0
    assert t.read()["total_carbon_duration"] == 3.0
    t.stop()


def test_duration_is_frozen_at_stop(monkeypatch):
    clock = _Clock(100.0)
    monkeypatch.setattr(carbon_mod, "perf_counter", clock)
    t = CarbonTracker().start()
    clock.now = 105.0
    t.stop()
    clock.now = 200.0
    assert t.read()["total_carbon_duration"] == 5.0


def _busy(seconds):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


def test_emissions_are_reported_while_running():
    t = CarbonTracker(country_iso_code="FRA").start()
    _busy(0.3)
    reading = t.read()
    t.stop()
    assert reading["total_emissions_kg"] > 0


def test_emissions_after_stop_match_the_final_value():
    t = CarbonTracker(country_iso_code="FRA").start()
    _busy(0.3)
    t.stop()
    assert t.read()["total_emissions_kg"] == t._tracker.final_emissions > 0


def test_client_reports_its_emissions(monkeypatch):
    monkeypatch.setattr(carbon_mod, "_GLOBAL_TRACKER", None)
    cp = ClientProfiler(enable_carbon=True, country_iso_code="FRA")
    _busy(0.3)
    record = cp.to_metric_record()
    carbon_mod._GLOBAL_TRACKER.stop()
    assert record["client_emissions_kg"] > 0
