import time

from enerfl.collectors import (
    CompressionCollector,
    MessageCountCollector,
    PayloadSizeCollector,
    TimingCollector,
    UpdateCountCollector,
    WeightCountCollector,
)


def test_timing_collector_uses_precaptured_t0():
    c = TimingCollector()
    t0 = time.perf_counter() - 0.5
    out = c.on_end("train", 1, {"t0": t0})
    assert "train_time" in out
    assert out["train_time"] >= 0.5


def test_update_count_collector():
    c = UpdateCountCollector()
    out = c.on_end("train", 1, {}, updates_sent=4, updates_received=3)
    assert out == {"train_updates_sent": 4, "train_updates_received": 3}


def test_message_count_collector_handles_iterables():
    c = MessageCountCollector()
    out = c.on_end("train", 1, {}, messages=[0, 1, 2], replies=iter([0, 1]))
    assert out["train_sent_count"] == 3
    assert out["train_received_count"] == 2


def test_payload_size_collector():
    c = PayloadSizeCollector()
    out = c.on_end(
        "train", 1, {}, payload_bytes_sent=100, payload_bytes_received=250
    )
    assert out["train_payload_bytes_sent"] == 100
    assert out["train_payload_bytes_received"] == 250


def test_weight_count_collector():
    c = WeightCountCollector()
    assert c.on_end("train", 1, {}, weights_sent=12) == {"train_weights_sent": 12}


def test_compression_collector_ratios():
    c = CompressionCollector()
    out = c.on_end(
        "train",
        1,
        {},
        original_bytes=1000,
        compressed_bytes=250,
        total_params=100,
        nonzero_params=25,
    )
    assert out["train_compression_ratio"] == 4.0
    assert out["train_sparsity_ratio"] == 0.75


def test_compression_collector_zero_guard():
    c = CompressionCollector()
    out = c.on_end("train", 1, {}, original_bytes=1000, compressed_bytes=0)
    assert out["train_compression_ratio"] == float("inf")
