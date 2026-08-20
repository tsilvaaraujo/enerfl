import enerfl.carbon as carbon_mod
from enerfl.carbon import CarbonTracker, _finite, get_global_tracker


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
    t = CarbonTracker()
    t.start()
    t._tracker._total_energy.kWh = 2.0
    reading = t.read()
    assert reading["total_energy_joules"] == 2.0 * 3.6e6
    assert reading["total_emissions_kg"] == 0.0


def test_kwargs_forwarded_to_backend():
    t = CarbonTracker(country_iso_code="FRA", measure_power_secs=1)
    assert t._tracker.kwargs["country_iso_code"] == "FRA"
    assert t._tracker.kwargs["save_to_file"] is False
    assert t._tracker.kwargs["log_level"] == "error"


def test_global_tracker_is_singleton():
    carbon_mod._GLOBAL_TRACKER = None
    try:
        a = get_global_tracker(country_iso_code="FRA")
        b = get_global_tracker()
        assert a is b
    finally:
        carbon_mod._GLOBAL_TRACKER = None
