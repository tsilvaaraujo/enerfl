import csv
import json
import re
import subprocess

import pytest
from flwr.common import ArrayRecord

import enerfl.metadata as md
import enerfl.strategy as st
from enerfl import _safe
from enerfl.server import Profiler


@pytest.fixture(autouse=True)
def _clean_failures():
    _safe.reset_failures()
    yield
    _safe.reset_failures()


def test_run_id_format_and_uniqueness():
    ids = {md.generate_run_id() for _ in range(50)}
    assert len(ids) == 50
    assert all(re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{8}", i) for i in ids)


def test_profiler_generates_or_keeps_the_run_id():
    assert Profiler().run_id
    assert Profiler(run_id="campaign-7").run_id == "campaign-7"
    assert Profiler(run_id="x").get_metrics().global_metrics["run_id"] == "x"


def test_run_id_survives_reset():
    p = Profiler(run_id="x")
    p.reset()
    assert p.get_metrics().global_metrics["run_id"] == "x"


def test_run_id_is_a_column_of_the_csv(tmp_path):
    p = Profiler(run_id="abc", stream_csv_path=str(tmp_path / "m.csv"))
    p.set_round(1)
    p.record("x", {"a": 1})
    p.write_round_to_csv(1)
    with open(tmp_path / "m.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["run_id"] == "abc"


def test_metadata_file_sits_next_to_the_csv_and_is_written_at_start(tmp_path):
    p = Profiler(
        run_id="abc",
        algorithm_name="FedAvg",
        network_interface="eth0",
        carbon_tracker_kwargs=None,
        stream_csv_path=str(tmp_path / "out" / "run.csv"),
    )
    path = tmp_path / "out" / "run.meta.json"
    assert p.metadata_path == path
    data = json.loads(path.read_text())

    assert data["schema_version"] == md.SCHEMA_VERSION
    assert data["run_id"] == "abc"
    assert data["algorithm_name"] == "FedAvg"
    assert data["started_at_utc"] and data["finished_at_utc"] is None
    assert data["python"]["version"]
    assert set(data["packages"]) == {"enerfl", "flwr", "codecarbon", "protobuf"}
    assert data["machine"]["hostname"] and data["machine"]["cpu_count"]
    assert data["configuration"]["network_interface"] == "eth0"
    assert "timing" in data["configuration"]["collectors"]


def test_explicit_metadata_path_wins(tmp_path):
    target = tmp_path / "custom.json"
    Profiler(
        stream_csv_path=str(tmp_path / "run.csv"), metadata_path=str(target)
    )
    assert target.exists()
    assert not (tmp_path / "run.meta.json").exists()


def test_no_path_means_no_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Profiler()
    assert list(tmp_path.iterdir()) == []


def test_disabled_profiler_writes_nothing(tmp_path):
    Profiler(enabled=False, metadata_path=str(tmp_path / "m.json"))
    assert not (tmp_path / "m.json").exists()


def test_unwritable_metadata_path_never_raises(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    Profiler(metadata_path=str(blocker / "sub" / "m.json"))
    assert _safe.failure_sites() == {"metadata.write": 1}


def test_git_info_outside_a_repository_is_none(tmp_path):
    assert md.git_info(cwd=str(tmp_path)) is None


def test_git_info_reads_commit_and_dirty_flag(tmp_path):
    def git(*args):
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
            cwd=tmp_path, check=True, capture_output=True,
        )

    git("init", "-q")
    (tmp_path / "f").write_text("1")
    git("add", "f")
    git("commit", "-q", "-m", "c")
    info = md.git_info(cwd=str(tmp_path))
    assert re.fullmatch(r"[0-9a-f]{40}", info["commit"])
    assert info["dirty"] is False
    (tmp_path / "f").write_text("2")
    assert md.git_info(cwd=str(tmp_path))["dirty"] is True


def test_missing_git_binary_is_none(monkeypatch):
    def boom(*_a, **_k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(md.subprocess, "run", boom)
    assert md.git_info() is None


def test_finished_run_stamps_end_time_and_final_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "read_network_counters", lambda _i=None: None)
    monkeypatch.setattr(st, "_flwr_send_and_receive", None)

    class Inner:
        def summary(self):
            pass

        def configure_train(self, *a):
            return [object()]

        def aggregate_train(self, r, replies):
            return (None, {"loss": 0.1})

        def configure_evaluate(self, *a):
            return [object()]

        def aggregate_evaluate(self, r, replies):
            return {"accuracy": 0.9}

    class Grid:
        def send_and_receive(self, messages, timeout):
            return []

    p = Profiler(run_id="r1", stream_csv_path=str(tmp_path / "run.csv"))
    st.profiled(Inner(), p).start(
        grid=Grid(), initial_arrays=ArrayRecord(), num_rounds=2, timeout=1.0
    )
    data = json.loads((tmp_path / "run.meta.json").read_text())
    assert data["finished_at_utc"] is not None
    assert data["algorithm_name"] == "Inner"
    assert data["global_metrics"]["num_rounds"] == 2
    assert data["global_metrics"]["profiler_failures"] == 0
    assert data["global_metrics"]["run_id"] == "r1"
