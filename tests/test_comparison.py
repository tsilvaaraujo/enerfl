from enerfl.comparison import compare_runs, compare_runs_to_csv
from enerfl.server import Profiler
import pytest


def _make(name, times, accs):
    p = Profiler(algorithm_name=name, enable_carbon=False)
    for i, (t, a) in enumerate(zip(times, accs), start=1):
        p.set_round(i)
        p.record("round_total", {"round_total_time": t})
        p.record("client_evaluate", {"accuracy": a})
    return p


def test_compare_runs_means_and_best():
    fast = _make("fast", times=[10.0, 20.0], accs=[0.80, 0.90])
    slow = _make("slow", times=[30.0, 50.0], accs=[0.70, 0.72])

    result = compare_runs(fast, slow)

    by_name = {s["algorithm"]: s for s in result["summary"]}
    assert by_name["fast"]["mean_round_total_time"] == 15.0
    assert by_name["slow"]["mean_round_total_time"] == 40.0
    assert by_name["fast"]["mean_accuracy"] == pytest.approx(0.85)

    best = result["best"]
    assert best["mean_round_total_time"]["algorithm"] == "fast"
    assert best["mean_accuracy"]["algorithm"] == "fast"


def test_compare_runs_to_csv(tmp_path):
    a = _make("A", times=[1.0], accs=[0.5])
    b = _make("B", times=[2.0], accs=[0.6])
    path = tmp_path / "cmp.csv"

    compare_runs_to_csv(a, b, path=str(path))

    text = path.read_text()
    assert "algorithm" in text
    assert "A" in text and "B" in text
