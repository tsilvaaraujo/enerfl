import builtins
from unittest.mock import mock_open

import pytest

from enerfl.network import read_network_counters

_PROC_NET_DEV = (
    "hdr1\n"
    "hdr2\n"
    "  eth0: 100 0 0 0 0 0 0 0 200 0 0 0 0 0 0 0\n"
    "  eth1: 10 0 0 0 0 0 0 0 20 0 0 0 0 0 0 0\n"
    "    lo: 999 0 0 0 0 0 0 0 999 0 0 0 0 0 0 0\n"
)


def _fake_proc(monkeypatch, content):
    monkeypatch.setattr(builtins, "open", mock_open(read_data=content))


def test_missing_file_is_unobservable(monkeypatch):
    def boom(*_a, **_k):
        raise OSError()

    monkeypatch.setattr(builtins, "open", boom)
    assert read_network_counters() is None


def test_sums_all_interfaces_and_skips_loopback(monkeypatch):
    _fake_proc(monkeypatch, _PROC_NET_DEV)
    assert read_network_counters() == (220, 110)  # (tx, rx)


def test_named_interface_only(monkeypatch):
    _fake_proc(monkeypatch, _PROC_NET_DEV)
    assert read_network_counters("eth1") == (20, 10)


def test_unknown_or_idle_interface_is_unobservable(monkeypatch):
    _fake_proc(monkeypatch, _PROC_NET_DEV)
    assert read_network_counters("wlan9") is None


def test_malformed_content_is_unobservable(monkeypatch):
    content = "h1\nh2\n  eth0: 100 0 0 0 0 0 0 0 notanint 0 0 0 0 0 0 0\n"
    _fake_proc(monkeypatch, content)
    assert read_network_counters() is None
