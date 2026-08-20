from enerfl.metrics import ProfilerMetrics


def test_record_and_flat_metrics():
    m = ProfilerMetrics()
    m.record(1, "train", {"a": 1.0})
    m.record(1, "train", {"b": 2.0})
    m.record(1, "eval", {"c": 3.0})
    m.record(2, "train", {"a": 9.0})

    flat_r1 = m.get_flat_metrics(1)
    assert flat_r1 == {"a": 1.0, "b": 2.0, "c": 3.0}

    flat_all = m.get_flat_metrics()
    assert flat_all["a"] == 9.0
    assert flat_all["c"] == 3.0


def test_record_global_and_to_dict():
    m = ProfilerMetrics()
    m.record_global("algorithm_name", "FedAvg")
    m.record(1, "train", {"a": 1.0})

    assert m.global_metrics["algorithm_name"] == "FedAvg"
    d = m.to_dict()
    assert d["global"]["algorithm_name"] == "FedAvg"
    assert d["rounds"][1]["train"]["a"] == 1.0


def test_record_merges_into_existing_phase():
    m = ProfilerMetrics()
    m.record(1, "comm", {"x": 1})
    m.record(1, "comm", {"y": 2})
    assert m.rounds[1]["comm"] == {"x": 1, "y": 2}
